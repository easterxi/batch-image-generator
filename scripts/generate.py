#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, csv, json, random, re, sys, time
import urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from controlnet_aux import OpenposeDetector

ROOT = Path(__file__).resolve().parents[1]
NODE = {"checkpoint":"1","positive":"2","negative":"3","load_image":"4","controlnet":"5","apply":"6","latent":"7","sampler":"8","save":"10"}
REQUIRED = {"CheckpointLoaderSimple","CLIPTextEncode","LoadImage","ControlNetLoader","ControlNetApplyAdvanced","EmptyLatentImage","KSampler","VAEDecode","SaveImage"}

def project_path(v):
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p

def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def http_json(url,payload=None,timeout=60):
    data=None; headers={}
    if payload is not None:
        data=json.dumps(payload).encode("utf-8")
        headers["Content-Type"]="application/json"
    req=urllib.request.Request(url,data=data,headers=headers)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.loads(r.read().decode())

def http_bytes(url,timeout=120):
    with urllib.request.urlopen(url,timeout=timeout) as r:
        return r.read()

def validate_server(server):
    info=http_json(server.rstrip("/")+"/object_info",timeout=30)
    missing=sorted(REQUIRED-set(info.keys()))
    if missing:
        raise RuntimeError("Missing ComfyUI node types: "+", ".join(missing))

def expand_prompt(template,traits,rng):
    used={}
    def repl(m):
        k=m.group(1)
        if k not in traits: return m.group(0)
        vals=traits[k]
        if not isinstance(vals,list) or not vals:
            raise ValueError(f"Trait {k} must be a non-empty list")
        v=rng.choice(vals); used[k]=v; return str(v)
    return re.sub(r"\{([A-Za-z0-9_]+)\}",repl,template),used

def prep_reference(src,dst,w,h,mode,low=100,high=200):
    if not src.exists():
        raise FileNotFoundError(f"Reference image not found: {src}")
    with Image.open(src) as source:
        im=ImageOps.fit(source.convert("RGB"),(w,h),method=Image.Resampling.LANCZOS)

        if mode=="canny":
            arr=np.array(im)
            gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
            edge=cv2.Canny(gray,int(low),int(high))
            im=Image.fromarray(cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB))

        elif mode=="openpose":
            print("Loading OpenPose detector...", flush=True)
            detector=OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            print("Detecting pose...", flush=True)
            im=detector(
                im,
                detect_resolution=max(w,h),
                image_resolution=max(w,h),
                include_body=True,
                include_hand=False,
                include_face=False,
                output_type="pil",
            ).convert("RGB")
            im=ImageOps.fit(im,(w,h),method=Image.Resampling.LANCZOS)

        elif mode=="none":
            pass

        else:
            raise ValueError("reference_preprocess must be 'canny', 'openpose', or 'none'.")

        dst.parent.mkdir(parents=True,exist_ok=True)
        im.save(dst)
        print(f"Prepared control image: {dst}", flush=True)

def seed_for(mode,base,index,rng):
    if mode=="fixed": return int(base)
    if mode=="sequential": return int(base)+index-1
    if mode=="random": return rng.randrange(0,2**63-1)
    raise ValueError("seed_mode must be random, sequential, or fixed")

def queue(server,wf):
    x=http_json(server.rstrip("/")+"/prompt",{"prompt":wf},60)
    if "prompt_id" not in x:
        raise RuntimeError(f"No prompt_id returned: {x}")
    return x["prompt_id"]

def wait_history(server,pid,poll,timeout):
    start=time.time()
    url=server.rstrip("/")+f"/history/{pid}"
    while True:
        if time.time()-start>timeout:
            raise TimeoutError(f"Timed out waiting for {pid}")
        try:
            h=http_json(url,timeout=30)
        except Exception:
            time.sleep(poll); continue
        if pid in h:
            entry=h[pid]
            st=entry.get("status",{})
            if st.get("completed") is True or st.get("status_str") in {"success","error"}:
                return entry
        time.sleep(poll)

def saved_image(entry):
    outs=entry.get("outputs",{})
    imgs=outs.get(NODE["save"],{}).get("images",[])
    if imgs: return imgs[0]
    for o in outs.values():
        for im in o.get("images",[]):
            if im.get("type")=="output":
                return im
    raise RuntimeError("No output image found in history")

def fetch_image(server,meta):
    q=urllib.parse.urlencode({
        "filename":meta["filename"],
        "subfolder":meta.get("subfolder",""),
        "type":meta.get("type","output"),
    })
    return http_bytes(server.rstrip("/")+"/view?"+q)

def manifest_done(path):
    done=set()
    if not path.exists(): return done
    with path.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status")=="completed":
                try: done.add(int(r["index"]))
                except Exception: pass
    return done

def log(path,row):
    fields=["index","status","filename","seed","prompt_id","timestamp_utc","traits_json","prompt","error"]
    exists=path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow({k:row.get(k,"") for k in fields})

def build(template,cfg,pos,neg,seed,index,ckpt,cn):
    wf=copy.deepcopy(template)
    wf[NODE["checkpoint"]]["inputs"]["ckpt_name"]=ckpt
    wf[NODE["positive"]]["inputs"]["text"]=pos
    wf[NODE["negative"]]["inputs"]["text"]=neg
    wf[NODE["load_image"]]["inputs"]["image"]="control_reference.png"
    wf[NODE["controlnet"]]["inputs"]["control_net_name"]=cn
    a=wf[NODE["apply"]]["inputs"]
    a["strength"]=float(cfg["controlnet_strength"])
    a["start_percent"]=float(cfg["controlnet_start"])
    a["end_percent"]=float(cfg["controlnet_end"])
    l=wf[NODE["latent"]]["inputs"]
    l["width"]=int(cfg["width"]); l["height"]=int(cfg["height"]); l["batch_size"]=1
    s=wf[NODE["sampler"]]["inputs"]
    s.update({"seed":int(seed),"steps":int(cfg["steps"]),"cfg":float(cfg["cfg"]),"sampler_name":cfg["sampler_name"],"scheduler":cfg["scheduler"],"denoise":float(cfg["denoise"])})
    wf[NODE["save"]]["inputs"]["filename_prefix"]=f"batch/{cfg['filename_prefix']}_{index:06d}"
    return wf

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config/config.json")
    ap.add_argument("--count",type=int,required=True)
    ap.add_argument("--reference")
    ap.add_argument("--checkpoint")
    ap.add_argument("--controlnet")
    ap.add_argument("--output-dir")
    ap.add_argument("--seed-mode",choices=["random","sequential","fixed"])
    ap.add_argument("--base-seed",type=int)
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()

    if args.count<1:
        raise SystemExit("--count must be >=1")

    cfg=jload(project_path(args.config))
    for key,val in [("reference_image",args.reference),("checkpoint_name",args.checkpoint),("controlnet_name",args.controlnet),("output_dir",args.output_dir),("seed_mode",args.seed_mode)]:
        if val is not None: cfg[key]=val
    if args.base_seed is not None: cfg["base_seed"]=args.base_seed

    comfy=Path(cfg["comfyui_dir"]).expanduser()
    server=cfg["server_url"].rstrip("/")
    if not comfy.exists():
        raise SystemExit(f"ComfyUI directory not found: {comfy}")

    ckpt=cfg["checkpoint_name"]
    cn=cfg["controlnet_name"]
    if ckpt.startswith("YOUR_") or cn.startswith("YOUR_"):
        raise SystemExit("Set checkpoint/controlnet filenames in config or CLI")

    ref=project_path(cfg["reference_image"])
    out=Path(cfg["output_dir"]).expanduser()
    out=out if out.is_absolute() else ROOT/out
    out.mkdir(parents=True,exist_ok=True)

    manifest=out/"manifest.csv"
    pos_t=project_path(cfg["positive_prompt_file"]).read_text(encoding="utf-8").strip()
    neg=project_path(cfg["negative_prompt_file"]).read_text(encoding="utf-8").strip()
    traits=jload(project_path(cfg["traits_file"]))
    template=jload(project_path(cfg["workflow"]))

    print("Checking ComfyUI server...", flush=True)
    validate_server(server)
    print("Preparing reference...", flush=True)
    prep_reference(
        ref,
        comfy/"input"/"control_reference.png",
        int(cfg["width"]),
        int(cfg["height"]),
        cfg["reference_preprocess"],
        cfg.get("canny_low_threshold",100),
        cfg.get("canny_high_threshold",200),
    )

    done=manifest_done(manifest)
    rng=random.Random()
    print(f"Target count: {args.count}\nOutput: {out}\nReference: {ref}\nCheckpoint: {ckpt}\nControlNet: {cn}\n", flush=True)

    for i in range(1,args.count+1):
        target=out/f"{cfg['filename_prefix']}_{i:06d}.png"
        if not args.force and (i in done or target.exists()):
            print(f"[{i:06d}/{args.count:06d}] skip", flush=True)
            continue

        prompt,used=expand_prompt(pos_t,traits,rng)
        seed=seed_for(cfg["seed_mode"],cfg["base_seed"],i,rng)
        wf=build(template,cfg,prompt,neg,seed,i,ckpt,cn)
        pid=""
        print(f"[{i:06d}/{args.count:06d}] seed={seed} traits={json.dumps(used,ensure_ascii=False)}", flush=True)

        try:
            pid=queue(server,wf)
            hist=wait_history(server,pid,float(cfg["poll_seconds"]),float(cfg["timeout_seconds"]))
            st=hist.get("status",{})
            if st.get("status_str")=="error":
                raise RuntimeError(f"ComfyUI execution error: {st}")

            data=fetch_image(server,saved_image(hist))
            tmp=target.with_suffix(".png.partial")
            tmp.write_bytes(data)
            tmp.replace(target)

            log(manifest,{
                "index":i,
                "status":"completed",
                "filename":target.name,
                "seed":seed,
                "prompt_id":pid,
                "timestamp_utc":datetime.now(timezone.utc).isoformat(),
                "traits_json":json.dumps(used,ensure_ascii=False),
                "prompt":prompt,
                "error":"",
            })
            print("  saved ->",target, flush=True)

        except KeyboardInterrupt:
            print("Interrupted. Rerun to resume.", flush=True)
            raise

        except Exception as e:
            log(manifest,{
                "index":i,
                "status":"error",
                "filename":target.name,
                "seed":seed,
                "prompt_id":pid,
                "timestamp_utc":datetime.now(timezone.utc).isoformat(),
                "traits_json":json.dumps(used,ensure_ascii=False),
                "prompt":prompt,
                "error":repr(e),
            })
            print("  ERROR:",e,file=sys.stderr, flush=True)

    print("Done. Manifest:",manifest, flush=True)

if __name__=="__main__":
    main()
