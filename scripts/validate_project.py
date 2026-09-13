#!/usr/bin/env python3
from pathlib import Path
import json, py_compile
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'config/config.json',ROOT/'config/traits.json',ROOT/'workflows/controlnet_canny_api.json']:
    json.loads(p.read_text()); print('JSON OK:',p.relative_to(ROOT))
py_compile.compile(str(ROOT/'scripts/generate.py'),doraise=True); print('Python OK: scripts/generate.py')
wf=json.loads((ROOT/'workflows/controlnet_canny_api.json').read_text())
need={'CheckpointLoaderSimple','CLIPTextEncode','LoadImage','ControlNetLoader','ControlNetApplyAdvanced','EmptyLatentImage','KSampler','VAEDecode','SaveImage'}
have={v.get('class_type') for v in wf.values()}; miss=need-have
if miss: raise SystemExit('Missing nodes: '+', '.join(sorted(miss)))
print('Workflow node set OK\nProject validation passed.')
