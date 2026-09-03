# InfiniteTalk (via ComfyUI, GGUF low-VRAM) — Setup Runbook

## What this is

**Model:** [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) (MeiGen-AI) — an audio-driven
talking-avatar model. It takes **one still portrait photo + one audio file** and generates a video
of that portrait speaking, with the mouth/motion driven directly by the audio (no separate driving
video and no motion prompt needed for the lip-sync itself).

**Backbone:** built on [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
(Wan-AI), a 14-billion-parameter image-to-video diffusion model. The full backbone is 82.3GB — too
big for this laptop's ~74GB free disk and 6GB VRAM. This runbook instead uses a **GGUF-quantized**
copy of the same backbone (~8GB) plus a matching quantized InfiniteTalk adapter, run through
**ComfyUI** (the only inference stack that currently loads these GGUF files) instead of
InfiniteTalk's own native `generate_infinitetalk.py` script.

**Hardware target:** RTX 4050 Laptop GPU (6GB VRAM), ~16GB system RAM, ~74GB free disk at time of
writing.

**Known risk:** even with quantization, this class of model (14B params) is right at the edge of
what a 6GB/16GB machine can run. First run may OOM or thrash on the page file — that's a real,
useful result, not necessarily a setup mistake. See Troubleshooting at the bottom.

**Total download size:** ~18GB (breakdown in Step 3).

---

## Step 1 — Clone ComfyUI

Reuses the existing `myenv` Python environment (already has torch + CUDA installed), instead of
creating a second multi-GB env from scratch.

```bash
git clone https://github.com/comfyanonymous/ComfyUI C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI
```

```bash
cd C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI && "C:\Users\GHANSHYAM\Desktop\project\ai-interview\myenv\Scripts\python.exe" -m pip install -r requirements.txt
```

---

## Step 2 — Install custom nodes

- **ComfyUI-Manager**: gives a UI that flags any model file a workflow can't find, and where it
  expects it — the most reliable way to confirm the folder paths in Step 3 are right.
- **ComfyUI-WanVideoWrapper** (by Kijai): provides the actual InfiniteTalk/WanVideo loader and
  sampler nodes, and ships the example workflow used in Step 5.

```bash
git clone https://github.com/ltdrdata/ComfyUI-Manager C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\custom_nodes\ComfyUI-Manager
```

```bash
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper
```

```bash
"C:\Users\GHANSHYAM\Desktop\project\ai-interview\myenv\Scripts\python.exe" -m pip install -r "C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt"
```

---

## Step 3 — Download model weights (~18GB total)

Quant level picked deliberately small (`Q3_K_S`, not `Q4`) to leave headroom for both the 6GB VRAM
and the 16GB system RAM.

| # | File | Size | Destination folder |
|---|------|------|---------------------|
| 1 | `wan2.1-i2v-14b-480p-Q3_K_S.gguf` | 7.93 GB | `models\diffusion_models\` |
| 2 | `Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf` | 1.4 GB | `models\diffusion_models\` |
| 3 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.74 GB | `models\text_encoders\` |
| 4 | `wan_2.1_vae.safetensors` | 254 MB | `models\vae\` |
| 5 | `clip_vision_h.safetensors` | 1.26 GB | `models\clip_vision\` |
| 6 | `chinese-wav2vec2-base` (config + weights) | ~380 MB | `models\wav2vec2\chinese-wav2vec2-base\` |

```bash
huggingface-cli download city96/Wan2.1-I2V-14B-480P-gguf wan2.1-i2v-14b-480p-Q3_K_S.gguf --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\diffusion_models
```

```bash
huggingface-cli download Kijai/WanVideo_comfy_GGUF InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\diffusion_models
```

```bash
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\text_encoders
```

```bash
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\vae
```

```bash
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/clip_vision/clip_vision_h.safetensors --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\clip_vision
```

```bash
huggingface-cli download TencentGameMate/chinese-wav2vec2-base config.json preprocessor_config.json pytorch_model.bin --local-dir C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI\models\wav2vec2\chinese-wav2vec2-base
```

**Note:** the last three downloads may land nested one level deeper (e.g.
`models\text_encoders\split_files\text_encoders\umt5_xxl_....safetensors`) because that's the
path inside the source repo. If so, move the file up so it sits directly in `text_encoders\`,
`vae\`, or `clip_vision\` — ComfyUI's model dropdowns scan those folders non-recursively.

---

## Step 4 — Launch ComfyUI with low-VRAM flags

```bash
cd C:\Users\GHANSHYAM\Desktop\lipsync_models\ComfyUI && "C:\Users\GHANSHYAM\Desktop\project\ai-interview\myenv\Scripts\python.exe" main.py --lowvram
```

This opens the UI at `http://127.0.0.1:8188` in your browser.

---

## Step 5 — Load the workflow and run it

1. In the ComfyUI menu, load the workflow file:
   `custom_nodes\ComfyUI-WanVideoWrapper\example_workflows\wanvideo_I2V_InfiniteTalk_example_03.json`
2. ComfyUI-Manager will highlight (in red) any loader node that can't find its model file — if a
   path from Step 3 was slightly off, this tells you exactly where it actually needs to go.
3. Swap the image-loader node's file to this project's test portrait:
   `Phase 0.2\testing_engine_models\avatar\assets\source_face.jpg`
4. Swap the `LoadAudio` node's file to this project's TTS test clip:
   `Phase 0.2\testing_engine_models\text_to_speech\tts_output\round_2.wav`
5. Click **Queue Prompt**.

---

## Troubleshooting

- **`CUDA out of memory`**: expected risk given the 6GB card. Look for a "block swap" /
  "num_blocks_to_swap" or similar VRAM-offload setting on the WanVideo model-loader node in the
  workflow and increase it, trading speed for VRAM headroom.
- **Very slow, but no crash, and Task Manager shows RAM near 100% / high disk activity**: this is
  the system-RAM bottleneck flagged before starting this — offloaded weights are being swapped to
  the page file. Real, just slow; not a broken setup.
- **A model dropdown is empty / red in the UI**: the file didn't land in the folder that node
  scans. Use ComfyUI-Manager's "missing models" list, or check the node's dropdown for the exact
  folder name it's reading from, and move the file there.
- **`ModuleNotFoundError` on launch**: a dependency from `ComfyUI-WanVideoWrapper/requirements.txt`
  didn't install cleanly — re-run that pip install command and read the actual error rather than
  retrying blindly.
