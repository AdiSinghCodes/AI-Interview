"""
TEST 23: InfiniteTalk — audio-driven avatar (Wan2.1-I2V-14B backbone)
====================================================
What it shows:
  - Whether the InfiniteTalk repo + its 3 weight sets are present locally
  - Real end-to-end inference time driving a single still photo with
    this project's TTS audio, in low-VRAM mode
  - Realtime factor, comparable to MuseTalk (test_19), Wav2Lip (test_20),
    LivePortrait (test_21), SadTalker (test_22)

WHY THIS ONE, AND THE REAL RISK: unlike the other 4 models here,
InfiniteTalk is audio-native end to end (image + audio in, no separate
driving video and no hand-authored motion prompt needed for the lip/
motion sync itself — see project notes on evaluating Runway-style
prompt-driven video gen for this role). But it's built on
Wan2.1-I2V-14B-480P, a 14B-parameter video diffusion backbone. Low-VRAM
mode (`--num_persistent_param_in_dit 0`, optionally `--quant fp8`) is
what makes this conceivable on a 6GB card at all, but public reports for
this whole class of Wan-14B-derived low-VRAM workflows (this model,
Wan2.2 GGUF, FramePack) consistently note 24-32GB of SYSTEM RAM as the
real bottleneck once the GPU-side quantization/offload kicks in — this
machine has ~16GB. RESULT: FAIL here most likely means "ran out of RAM/
VRAM", not "broken setup" — see the traceback before assuming the latter.

Shells out to the repo's own `generate_infinitetalk.py`, same as running
it by hand. If your checkout's flags differ from what's hardcoded below,
check `python generate_infinitetalk.py --help` in the repo and adjust
INFINITETALK_ARGS.

Requirements (all multi-GB — read sizes before running these yourself):
  conda create -n multitalk python=3.10 && conda activate multitalk
  git clone https://github.com/MeiGen-AI/InfiniteTalk C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\InfiniteTalk
  cd C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\InfiniteTalk
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
  pip install misaki[en] ninja psutil packaging wheel flash_attn==2.7.4.post1
  pip install -r requirements.txt
  # flash_attn has no prebuilt Windows wheel for most setups -- it compiles
  # from source (needs matching CUDA toolkit + MSVC). This is the step
  # most likely to fail first on native Windows (no WSL). If it does,
  # that's the actual finding of this test, not something to silently
  # work around.
  conda install -c conda-forge librosa ffmpeg

  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P   # ~14GB+
  huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base
  huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir weights/InfiniteTalk
  # For the fp8 quantized DiT (try this if fp16 OOMs on 6GB even with
  # --num_persistent_param_in_dit 0): the fp8 file lives inside the
  # MeiGen-AI/InfiniteTalk download above, under quant_models/.

  Drop a still portrait photo at
    avatar/assets/source_face.jpg
  Generate driving audio once via text_to_speech/test_09_tts_kokoro.py

Run:
  python avatar/test_23_avatar_infinitetalk.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated
from _avatar_common import (
    MODELS_ROOT, OUTPUT_DIR, DRIVING_AUDIO,
    check_repo, check_driving_audio, check_source_face,
    gpu_free_mb, run_inference_subprocess, report_result, stage_asset,
)

REPO_DIR = MODELS_ROOT / "InfiniteTalk"
WEIGHTS_DIR = REPO_DIR / "weights"
WAN_CKPT_DIR = WEIGHTS_DIR / "Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = WEIGHTS_DIR / "chinese-wav2vec2-base"
INFINITETALK_CKPT = WEIGHTS_DIR / "InfiniteTalk" / "single" / "infinitetalk.safetensors"

# No scene/appearance description available for this test's stock portrait,
# so this stays generic — it only steers style/background, since the
# audio (not the prompt) drives the actual lip motion.
FALLBACK_PROMPT = ("A person sitting still, facing the camera, speaking "
                    "naturally during a video call.")


def check_infinitetalk_weights() -> bool:
    missing = []
    if not any(WAN_CKPT_DIR.glob("*.safetensors")) and not any(WAN_CKPT_DIR.glob("*.bin")):
        missing.append(f"Wan2.1-I2V-14B-480P weights under {WAN_CKPT_DIR}")
    if not any(WAV2VEC_DIR.glob("*.safetensors")) and not any(WAV2VEC_DIR.glob("*.bin")):
        missing.append(f"chinese-wav2vec2-base weights under {WAV2VEC_DIR}")
    if not INFINITETALK_CKPT.exists():
        missing.append(f"InfiniteTalk adapter checkpoint at {INFINITETALK_CKPT}")
    if missing:
        print("FAILED: missing weight set(s):")
        for m in missing:
            print(f"  - {m}")
        print("  -> see this file's docstring for the 3 huggingface-cli download commands")
        return False
    print(f"OK: all 3 InfiniteTalk weight sets found under {WEIGHTS_DIR}")
    return True


def run_test():
    print("TEST 23: InfiniteTalk (audio-driven avatar, Wan2.1-I2V-14B backbone)")
    print("=" * 50)
    print("NOTE: 14B-param backbone in low-VRAM mode on a 6GB card, ~16GB "
          "system RAM -- likely to hit an OOM wall. See docstring.\n")

    if not check_repo(
        REPO_DIR,
        f"git clone https://github.com/MeiGen-AI/InfiniteTalk {REPO_DIR}",
    ):
        return
    if not check_infinitetalk_weights():
        return

    source_face = check_source_face(need_video=False)
    if not source_face:
        return
    if not check_driving_audio():
        return

    # Same dot-in-path defense as the other avatar tests (this project's
    # own path contains "Phase 0.2").
    staged_face = stage_asset(source_face, REPO_DIR, "source_face" + source_face.suffix)
    staged_audio = stage_asset(DRIVING_AUDIO, REPO_DIR, "driving_audio.wav")

    OUTPUT_DIR.mkdir(exist_ok=True)
    result_dir = OUTPUT_DIR / "infinitetalk"
    result_dir.mkdir(exist_ok=True)

    input_json_path = REPO_DIR / "_avatar_test_input" / "infinitetalk_input.json"
    input_json_path.write_text(json.dumps({
        "prompt": FALLBACK_PROMPT,
        "cond_video": str(staged_face),
        "cond_audio": {"person1": str(staged_audio)},
    }, indent=2), encoding="utf-8")

    vram_before = gpu_free_mb()
    if vram_before is not None:
        print(f"Free VRAM before: {vram_before:.0f} MB")

    save_file = result_dir / "infinitetalk_res_lowvram"
    cmd = [
        sys.executable, "generate_infinitetalk.py",
        "--ckpt_dir", str(WAN_CKPT_DIR),
        "--wav2vec_dir", str(WAV2VEC_DIR),
        "--infinitetalk_dir", str(INFINITETALK_CKPT),
        "--input_json", str(input_json_path),
        "--size", "infinitetalk-480",
        "--sample_steps", "40",
        "--num_persistent_param_in_dit", "0",
        "--mode", "streaming",
        "--motion_frame", "9",
        "--save_file", str(save_file),
    ]
    rc, output, elapsed = run_inference_subprocess(cmd, cwd=REPO_DIR, timeout=1800)

    vram_after = gpu_free_mb()
    vram_note = ""
    if vram_before is not None and vram_after is not None:
        # Same caveat as the other avatar tests: post-exit only, not peak.
        vram_note = (f"Free VRAM before={vram_before:.0f}MB after={vram_after:.0f}MB "
                     "(post-exit only, NOT peak usage -- watch nvidia-smi live for that)")

    if rc is None:
        report_result(False, elapsed, extra="FAILED: timed out.\n" + output[-2000:])
        return
    if rc != 0:
        extra = "FAILED: non-zero exit.\n" + output[-2000:]
        if "CUDA out of memory" in output or "MemoryError" in output or "Killed" in output:
            extra += ("\n\n-> looks like the VRAM/RAM wall from the docstring. "
                      "Try adding --quant fp8 --quant_dir "
                      "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors "
                      "if you haven't, or watch Task Manager's RAM graph during a re-run "
                      "to confirm system RAM (not VRAM) is what's maxing out.")
        report_result(False, elapsed, extra=extra)
        return

    outputs = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outputs:
        report_result(False, elapsed, extra="FAILED: process exited 0 but no .mp4 found under "
                      f"{result_dir}.\n" + output[-1500:])
        return

    report_result(True, elapsed, out_path=outputs[0], extra=vram_note)


if __name__ == "__main__":
    run_gated(run_test)
