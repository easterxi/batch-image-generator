# ComfyUI ControlNet Batch Generator

GitHub-ready starter for generating **1 / 10 / 100 / 1,000 / 10,000+ images** from one prompt/template using **ComfyUI + ControlNet**.

## Included

- fixed ControlNet reference image
- automatic Canny preprocessing in Python
- standard/core ComfyUI nodes only
- one prompt or one randomized prompt template
- random / sequential / fixed seeds
- deterministic sequential filenames
- CSV manifest
- crash/session resume
- Google Drive persistent output
- Google Colab notebook
- GitHub Actions validation

GitHub stores your code. Colab/Kaggle/your GPU performs inference.

## Architecture

```text
GitHub repo
   ↓ clone
GPU runtime
   ├─ ComfyUI localhost:8188
   ├─ reference image → Canny → ControlNet
   ├─ master prompt + traits
   └─ scripts/generate.py
          ├─ POST /prompt
          ├─ GET /history/{prompt_id}
          ├─ GET /view
          ├─ output PNG
          └─ manifest.csv
```

## Why Canny first?

For your use case, Canny is useful for holding **pose outline, head/body position, framing, and scale** while allowing hair, clothes and other traits to vary. It also avoids installing custom preprocessor nodes.

## Quick start

1. Create a GitHub repo and upload this folder.
2. Put your reference image in Google Drive, e.g.:
   `MyDrive/comfy-batch/reference.png`
3. Open `notebooks/colab_controlnet_batch.ipynb` in Colab.
4. Select a GPU runtime.
5. Fill in your GitHub repo URL and model URLs.
6. Test with `COUNT = 10`.
7. Increase to `100`, `1000`, or `10000`.

## Models

Use a checkpoint and Canny ControlNet from the **same model family**:

```text
SD1.5 checkpoint + SD1.5 Canny ControlNet
SDXL checkpoint  + SDXL Canny ControlNet
```

Do not mix families.

The notebook does not hard-code a third-party checkpoint URL so you can choose models with licenses you accept.

## One prompt

Edit `prompts/positive.txt`.

It can be a plain prompt, or use variables such as:

```text
one {gender}, age {age}, {hair_color} {hair_style}, wearing {outfit}
```

Values come from `config/traits.json`. If you remove all `{variables}`, every image uses the exact same prompt and only the seed changes.

## ControlNet strength

Edit `config/config.json`:

```json
"controlnet_strength": 0.8
```

Typical starting ranges:

```text
0.55–0.70  more freedom
0.75–0.90  stronger pose/composition lock
0.90–1.00  very strict, sometimes over-constrained
```

For your portrait consistency goal, start around **0.80**.

## Resume after Colab disconnects

If you run:

```bash
python scripts/generate.py --count 10000
```

and the session stops after `portrait_002347.png`, reconnect and run the same command. Existing completed images are skipped, so it resumes with missing indexes.

The safest output location on Colab is Google Drive, for example:

```text
/content/drive/MyDrive/comfy-batch/outputs
```

## Commands

```bash
python scripts/generate.py --count 10
```

```bash
python scripts/generate.py --count 10000 \
  --reference /content/drive/MyDrive/comfy-batch/reference.png \
  --checkpoint your_model.safetensors \
  --controlnet your_canny_controlnet.safetensors \
  --output-dir /content/drive/MyDrive/comfy-batch/outputs
```

Fixed seed:

```bash
python scripts/generate.py --count 100 --seed-mode fixed --base-seed 123456
```

Sequential seeds:

```bash
python scripts/generate.py --count 100 --seed-mode sequential --base-seed 100000
```

## Project structure

```text
.github/workflows/validate.yml
config/config.json
config/traits.json
notebooks/colab_controlnet_batch.ipynb
prompts/positive.txt
prompts/negative.txt
references/README.md
scripts/generate.py
scripts/validate_project.py
workflows/controlnet_canny_api.json
```

## Upgrade path

Once Canny works, you can swap in an exported API workflow using:

- OpenPose — body pose
- Depth — geometry/volume
- Lineart — outlines
- SoftEdge — softer structure
- IP-Adapter — appearance/identity
- ControlNet + IP-Adapter — pose plus appearance

The batch/resume engine can remain largely the same.
