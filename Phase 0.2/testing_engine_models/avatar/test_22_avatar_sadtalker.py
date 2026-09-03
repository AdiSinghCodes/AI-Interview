"""
TEST 22: SadTalker — Avatar lip-sync (diffusion, audio-driven head pose + lips)
====================================================
What it shows:
  - Whether the SadTalker repo + checkpoint set are present locally
  - Real end-to-end inference time driving a single still photo with
    this project's TTS audio — SadTalker also animates head pose, not
    just the mouth, so quality can look better than Wav2Lip at the cost
    of being the slowest of the four models here (moderate-high VRAM,
    diffusion-based)
  - Realtime factor, to compare directly against MuseTalk (test_19) and
    Wav2Lip (test_20) on the same audio clip

Shells out to the repo's own `inference.py`, same as running it by hand.
If your checkout's flags differ from what's hardcoded below, check
`python inference.py --help` in the repo and adjust SADTALKER_ARGS.

Requirements:
  git clone https://github.com/OpenTalker/SadTalker C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\SadTalker
  cd C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\SadTalker
  pip install -r requirements.txt
  # Windows: run the repo's download_models script (or follow its README
  # to manually place checkpoints into checkpoints/ and gfpgan/weights/)
  Drop a source face photo at
    avatar/assets/source_face.jpg
  Generate driving audio once via text_to_speech/test_09_tts_kokoro.py

Run:
  python avatar/test_22_avatar_sadtalker.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated
from _avatar_common import (
    MODELS_ROOT, OUTPUT_DIR, DRIVING_AUDIO,
    check_repo, check_driving_audio, check_source_face,
    gpu_free_mb, run_inference_subprocess, report_result, stage_asset,
)

REPO_DIR = MODELS_ROOT / "SadTalker"
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"


def check_checkpoints_dir() -> bool:
    if not CHECKPOINTS_DIR.exists():
        print(f"FAILED: no checkpoints dir at {CHECKPOINTS_DIR}")
        print("  -> run the repo's download_models script (see its README)")
        return False
    files = list(CHECKPOINTS_DIR.glob("*.safetensors")) + list(CHECKPOINTS_DIR.glob("*.pth")) + \
        list(CHECKPOINTS_DIR.glob("*.pth.tar"))
    if not files:
        print(f"FAILED: {CHECKPOINTS_DIR} exists but has no checkpoint files.")
        print("  -> run the repo's download_models script (see its README)")
        return False
    print(f"OK: {len(files)} checkpoint file(s) found in {CHECKPOINTS_DIR}")
    return True


def run_test():
    print("TEST 22: SadTalker (avatar lip-sync + head pose)")
    print("=" * 50)

    if not check_repo(
        REPO_DIR,
        f"git clone https://github.com/OpenTalker/SadTalker {REPO_DIR}",
    ):
        return
    if not check_checkpoints_dir():
        return

    source_face = check_source_face(need_video=False)
    if not source_face:
        return
    if not check_driving_audio():
        return

    # This project's own path contains "Phase 0.2" — a literal dot outside
    # any extension, which trips up naive extension-guessing code in some
    # of these repos (confirmed on Wav2Lip; staging here defensively).
    staged_face = stage_asset(source_face, REPO_DIR, "source_face" + source_face.suffix)
    staged_audio = stage_asset(DRIVING_AUDIO, REPO_DIR, "driving_audio.wav")

    OUTPUT_DIR.mkdir(exist_ok=True)
    result_dir = OUTPUT_DIR / "sadtalker"
    result_dir.mkdir(exist_ok=True)

    vram_before = gpu_free_mb()
    if vram_before is not None:
        print(f"Free VRAM before: {vram_before:.0f} MB")

    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", str(staged_audio),
        "--source_image", str(staged_face),
        "--result_dir", str(result_dir),
        "--still",
        "--preprocess", "full",
    ]
    rc, output, elapsed = run_inference_subprocess(cmd, cwd=REPO_DIR, timeout=1200)

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
