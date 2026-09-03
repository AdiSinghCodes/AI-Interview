"""
TEST 21: LivePortrait — Portrait animation (video-driven, NOT audio-native)
====================================================
What it shows:
  - Whether the LivePortrait repo + pretrained weights are present locally
  - Real end-to-end inference time driving a still portrait with a
    separate talking-motion video

IMPORTANT caveat this test exists to confirm: stock LivePortrait takes a
DRIVING VIDEO (motion to copy), not raw audio — there's no built-in
audio-to-motion step. That's the reasoning this project already used to
rule it out as the direct audio-driven avatar (see project notes) for a
TTS-only pipeline. This test still exercises the video-driven path
honestly, in case a driving video ever comes from somewhere else (e.g. a
webcam feed, or another model's output) — it does NOT prove audio-driven
lip-sync, and RESULT: PASS here should not be read as "LivePortrait
solves audio-driven avatar."

Shells out to the repo's own `inference.py`, same as running it by hand.
If your checkout's flags differ from what's hardcoded below, check
`python inference.py --help` in the repo and adjust LIVEPORTRAIT_ARGS.

Requirements:
  git clone https://github.com/KwaiVGI/LivePortrait C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\LivePortrait
  cd C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\LivePortrait
  pip install -r requirements.txt
  huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights
  Drop a still portrait photo at
    avatar/assets/source_face.jpg
  Drop a short talking-motion video (any face moving/talking — this is
  what gets copied onto the portrait) at
    avatar/assets/driving_video.mp4

Run:
  python avatar/test_21_avatar_liveportrait.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated
from _avatar_common import (
    MODELS_ROOT, OUTPUT_DIR, ASSETS_DIR, SOURCE_FACE_IMAGE,
    check_repo, check_files, gpu_free_mb, run_inference_subprocess,
    report_result, stage_asset,
)

REPO_DIR = MODELS_ROOT / "LivePortrait"
WEIGHTS_DIR = REPO_DIR / "pretrained_weights"
DRIVING_VIDEO = ASSETS_DIR / "driving_video.mp4"


def check_weights_dir() -> bool:
    # The repo ships pretrained_weights/ with just a .gitkeep placeholder,
    # so a plain .exists() on the directory is always true — check for an
    # actual weight file inside instead.
    if not WEIGHTS_DIR.exists():
        weight_files = []
    else:
        weight_files = [
            p for p in WEIGHTS_DIR.rglob("*")
            if p.is_file() and p.suffix in {".pth", ".safetensors", ".onnx", ".bin"}
        ]
    if not weight_files:
        print(f"FAILED: no weight files found under {WEIGHTS_DIR}")
        print(f"  -> run: huggingface-cli download KwaiVGI/LivePortrait "
              f"--local-dir {WEIGHTS_DIR}")
        return False
    print(f"OK: {len(weight_files)} weight file(s) found in {WEIGHTS_DIR}")
    return True


def run_test():
    print("TEST 21: LivePortrait (video-driven portrait animation)")
    print("=" * 50)
    print("NOTE: this is video-driven, not audio-driven — see docstring.\n")

    if not check_repo(
        REPO_DIR,
        f"git clone https://github.com/KwaiVGI/LivePortrait {REPO_DIR}",
    ):
        return
    if not check_weights_dir():
        return
    if not check_files(
        "LivePortrait test assets", [SOURCE_FACE_IMAGE, DRIVING_VIDEO],
        f"drop a still portrait at {SOURCE_FACE_IMAGE} and a short "
        f"talking-motion video at {DRIVING_VIDEO}",
    ):
        return

    # This project's own path contains "Phase 0.2" — a literal dot outside
    # any extension, which trips up naive extension-guessing code in some
    # of these repos (confirmed on Wav2Lip; staging here defensively).
    staged_source = stage_asset(SOURCE_FACE_IMAGE, REPO_DIR, "source_face" + SOURCE_FACE_IMAGE.suffix)
    staged_driving = stage_asset(DRIVING_VIDEO, REPO_DIR, "driving_video.mp4")

    OUTPUT_DIR.mkdir(exist_ok=True)
    result_dir = OUTPUT_DIR / "liveportrait"
    result_dir.mkdir(exist_ok=True)

    vram_before = gpu_free_mb()
    if vram_before is not None:
        print(f"Free VRAM before: {vram_before:.0f} MB")

    cmd = [
        sys.executable, "inference.py",
        "-s", str(staged_source),
        "-d", str(staged_driving),
        "-o", str(result_dir),
    ]
    rc, output, elapsed = run_inference_subprocess(cmd, cwd=REPO_DIR)

    vram_after = gpu_free_mb()
    vram_note = ""
    if vram_before is not None and vram_after is not None:
        # NOTE: vram_before/after are both measured in this wrapper's own
        # process, and "after" is only read once the subprocess above has
        # already exited and released its memory -- so this can never show
        # real peak usage, only confirm nothing leaked. Watch Task Manager
        # / `nvidia-smi -l 2` DURING the run above for the real number.
        vram_note = (f"Free VRAM before={vram_before:.0f}MB after={vram_after:.0f}MB "
                     "(post-exit only, NOT peak usage -- watch nvidia-smi live for that)")

    if rc is None:
        report_result(False, elapsed, extra="FAILED: timed out.\n" + output[-2000:])
        return
    if rc != 0:
        report_result(False, elapsed, extra="FAILED: non-zero exit.\n" + output[-2000:])
        return

    outputs = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outputs:
        report_result(False, elapsed, extra="FAILED: process exited 0 but no .mp4 found under "
                      f"{result_dir}.\n" + output[-1500:])
        return

    report_result(True, elapsed, out_path=outputs[0], extra=vram_note)


if __name__ == "__main__":
    run_gated(run_test)
