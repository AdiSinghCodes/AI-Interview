"""
TEST 19: MuseTalk — Avatar lip-sync (GAN-based, on existing video)
====================================================
What it shows:
  - Whether the MuseTalk repo + its (several, separately-downloaded)
    checkpoint sets are present locally
  - Real end-to-end inference time: driving a short source video with
    the TTS audio this project already generates, producing a lip-synced
    output video
  - Realtime factor — MuseTalk is the current pick for this project
    specifically because it's claimed real-time-capable at ~4-6GB VRAM,
    on a 6GB card with nothing else loaded. This is what confirms or
    kills that assumption on THIS machine.

This does not reimplement MuseTalk's inference — it shells out to the
repo's own `scripts.inference` entrypoint (via a generated YAML config,
which is how MuseTalk expects video/audio pairs), same as you'd run it
by hand. If your checkout's CLI differs from what's hardcoded below,
check `python -m scripts.inference --help` in the repo and adjust
MUSETALK_ARGS.

Requirements:
  git clone https://github.com/TMElyralab/MuseTalk C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\MuseTalk
  cd C:\\Users\\GHANSHYAM\\Desktop\\lipsync_models\\MuseTalk
  pip install -r requirements.txt
  # then run the repo's own weight-download script (download_weights.bat
  # on Windows, or follow its README) — pulls musetalk, sd-vae (into
  # models/sd-vae, despite the HF repo being named sd-vae-ft-mse),
  # whisper, dwpose, syncnet, and face-parse-bisent checkpoints into
  # models/. If pip/hf commands in that script fail with "Fatal error in
  # launcher", your venv was moved after creation — fix with:
  #   python -m pip install --force-reinstall --no-deps pip "huggingface_hub[hf_xet]"
  Drop a short (a few seconds) face video at
    avatar/assets/source_face.mp4
  Generate driving audio once via text_to_speech/test_09_tts_kokoro.py
    (saves avatar's driving audio at
     text_to_speech/tts_output/round_2.wav)

Run:
  python avatar/test_19_avatar_musetalk.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import run_gated
from _avatar_common import (
    MODELS_ROOT, OUTPUT_DIR, DRIVING_AUDIO,
    check_repo, check_files, check_driving_audio, check_source_face,
    gpu_free_mb, run_inference_subprocess, report_result, stage_asset,
)

REPO_DIR = MODELS_ROOT / "MuseTalk"
MODELS_DIR = REPO_DIR / "models"

# Matches the repo's own download_weights.bat/.sh — note it downloads SD
# VAE into models/sd-vae (NOT models/sd-vae-ft-mse, despite that being the
# HF repo name and the folder mkdir'd earlier in the same script).
EXPECTED_CHECKPOINTS = [
    MODELS_DIR / "musetalk" / "pytorch_model.bin",
    MODELS_DIR / "sd-vae" / "diffusion_pytorch_model.bin",
    MODELS_DIR / "whisper" / "pytorch_model.bin",
    MODELS_DIR / "dwpose" / "dw-ll_ucoco_384.pth",
    MODELS_DIR / "syncnet" / "latentsync_syncnet.pt",
    MODELS_DIR / "face-parse-bisent" / "79999_iter.pth",
]


def write_inference_config(config_path: Path, video_path: Path, audio_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "task_0:\n"
        f"  video_path: \"{video_path.as_posix()}\"\n"
        f"  audio_path: \"{audio_path.as_posix()}\"\n",
        encoding="utf-8",
    )


def run_test():
    print("TEST 19: MuseTalk (avatar lip-sync)")
    print("=" * 50)

    if not check_repo(
        REPO_DIR,
        f"git clone https://github.com/TMElyralab/MuseTalk {REPO_DIR}",
    ):
        return
    if not check_files(
        "MuseTalk checkpoints", EXPECTED_CHECKPOINTS,
        "run the repo's download_weights script (see its README) to "
        "pull musetalk/sd-vae-ft-mse/whisper/dwpose/face-parse-bisent",
    ):
        return

    source_video = check_source_face(need_video=True)
    if not source_video:
        return
    if not check_driving_audio():
        return

    # This project's own path contains "Phase 0.2" — a literal dot outside
    # any extension, which trips up naive extension-guessing code in some
    # of these repos (confirmed on Wav2Lip; staging here defensively).
    staged_video = stage_asset(source_video, REPO_DIR, "source_face" + source_video.suffix)
    staged_audio = stage_asset(DRIVING_AUDIO, REPO_DIR, "driving_audio.wav")

    OUTPUT_DIR.mkdir(exist_ok=True)
    config_path = REPO_DIR / "configs" / "inference" / "test_19_avatar_check.yaml"
    write_inference_config(config_path, staged_video, staged_audio)
    result_dir = OUTPUT_DIR / "musetalk"
    result_dir.mkdir(exist_ok=True)

    vram_before = gpu_free_mb()
    if vram_before is not None:
        print(f"Free VRAM before: {vram_before:.0f} MB")

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", str(config_path),
        "--result_dir", str(result_dir),
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
