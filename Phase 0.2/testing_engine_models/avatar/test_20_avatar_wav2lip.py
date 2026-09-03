"""
TEST 20: Wav2Lip — Avatar lip-sync (older GAN, lightweight fallback)
====================================================
What it shows:
  - Whether the Wav2Lip repo + GAN checkpoint are present locally
  - Real end-to-end inference time on this project's driving audio —
    Wav2Lip only needs ~2GB VRAM, so this is the fallback if MuseTalk
    (test_19) turns out too tight alongside the interviewer LLM on this
    6GB card
  - Realtime factor, so you can compare head-to-head against MuseTalk's
    number from test_19 on the exact same audio clip

Shells out to the repo's own `inference.py`, same as running it by hand.
If your checkout's flags differ from what's hardcoded below, check
`python inference.py --help` in the repo and adjust WAV2LIP_ARGS.

Requirements:
  git clone https://github.com/Rudrabha/Wav2Lip C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\Wav2Lip
  cd C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\Wav2Lip
  pip install -r requirements.txt
  # download wav2lip_gan.pth from the repo README's model links into
  # checkpoints/wav2lip_gan.pth
  # download the face detector s3fd.pth into
  # face_detection/detection/sfd/s3fd.pth (repo README has the link)
  Drop a source face (image or short video) at
    avatar/assets/source_face.jpg  (or source_face.mp4)
  Generate driving audio once via text_to_speech/test_09_tts_kokoro.py

Run:
  python avatar/test_20_avatar_wav2lip.py
"""

import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated
from _avatar_common import (
    MODELS_ROOT, OUTPUT_DIR, DRIVING_AUDIO,
    check_repo, check_files, check_driving_audio, check_source_face,
    gpu_free_mb, run_inference_subprocess, report_result, stage_asset,
)

REPO_DIR = MODELS_ROOT / "Wav2Lip"
CHECKPOINT = REPO_DIR / "checkpoints" / "wav2lip_gan.pth"
FACE_DETECTOR = REPO_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"


def run_test():
    print("TEST 20: Wav2Lip (avatar lip-sync)")
    print("=" * 50)

    if not check_repo(
        REPO_DIR,
        f"git clone https://github.com/Rudrabha/Wav2Lip {REPO_DIR}",
    ):
        return
    if not check_files(
        "Wav2Lip checkpoints", [CHECKPOINT, FACE_DETECTOR],
        "download wav2lip_gan.pth and s3fd.pth from the links in the "
        "repo's README into the paths above",
    ):
        return

    source_face = check_source_face(need_video=False)
    if not source_face:
        return
    if not check_driving_audio():
        return

    # This project's own path contains "Phase 0.2" — a dot AND a space
    # outside any extension. Wav2Lip's image/video detection does a naive
    # path.split('.')[1] (breaks on the dot), and its final audio+video
    # mux shells out to ffmpeg via an UNQUOTED formatted string (breaks on
    # the space, since Windows CreateProcess parses that string itself
    # with no shell involved). Stage every path Wav2Lip touches — input
    # and output — under the repo's own directory, which has neither.
    staged_face = stage_asset(source_face, REPO_DIR, "source_face" + source_face.suffix)
    staged_audio = stage_asset(DRIVING_AUDIO, REPO_DIR, "driving_audio.wav")
    local_out_dir = REPO_DIR / "_avatar_test_output"
    local_out_dir.mkdir(exist_ok=True)
    local_out_path = local_out_dir / "wav2lip_result.mp4"

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "wav2lip_result.mp4"

    vram_before = gpu_free_mb()
    if vram_before is not None:
        print(f"Free VRAM before: {vram_before:.0f} MB")

    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", str(CHECKPOINT),
        "--face", str(staged_face),
        "--audio", str(staged_audio),
        "--outfile", str(local_out_path),
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
    if not local_out_path.exists():
        report_result(False, elapsed, extra=f"FAILED: process exited 0 but {local_out_path} "
                      "wasn't created.\n" + output[-1500:])
        return

    shutil.copyfile(local_out_path, out_path)
    report_result(True, elapsed, out_path=out_path, extra=vram_note)


if __name__ == "__main__":
    run_gated(run_test)
