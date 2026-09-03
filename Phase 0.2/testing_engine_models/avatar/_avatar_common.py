"""
Shared helpers for the avatar/lip-sync test_1N_*.py scripts.

These 4 models (MuseTalk, Wav2Lip, LivePortrait, SadTalker) aren't pip
packages like the rest of this suite — each is a GitHub repo with its own
inference CLI and its own multi-GB checkpoint set. So unlike e.g.
test_04's "is it pulled in Ollama?" check, these scripts check whether the
repo + checkpoints exist locally, and if so, shell out to that repo's own
inference script (subprocess) rather than reimplementing its internals —
that avoids drifting out of sync with upstream on every repo update.

Repos and checkpoints live OUTSIDE this git repo (multi-GB, shouldn't be
committed) — same convention as ENGINE_ENV_FILE in _harness.py pointing
at the separate interview-engine project.
"""

import shutil
import subprocess
import time
from pathlib import Path

MODELS_ROOT = Path(r"C:\Users\GHANSHYAM\Desktop\lipsync_models")

_THIS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = _THIS_DIR / "assets"
SOURCE_FACE_IMAGE = ASSETS_DIR / "source_face.jpg"
SOURCE_FACE_VIDEO = ASSETS_DIR / "source_face.mp4"
DRIVING_AUDIO = _THIS_DIR.parent / "text_to_speech" / "tts_output" / "round_2.wav"
OUTPUT_DIR = _THIS_DIR / "avatar_output"


def check_repo(repo_dir: Path, clone_cmd: str) -> bool:
    if not repo_dir.exists():
        print(f"FAILED: repo not found at {repo_dir}")
        print(f"  -> {clone_cmd}")
        return False
    print(f"OK: repo present at {repo_dir}")
    return True


def check_files(label: str, paths: list, how_to_get: str) -> bool:
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"FAILED: missing {label}:")
        for p in missing:
            print(f"  - {p}")
        print(f"  -> {how_to_get}")
        return False
    print(f"OK: all {label} present.")
    return True


def check_driving_audio() -> bool:
    if not DRIVING_AUDIO.exists():
        print(f"FAILED: no driving audio at {DRIVING_AUDIO}")
        print("  -> run text_to_speech/test_09_tts_kokoro.py once to generate "
              "one (it saves round_2.wav there), or drop any WAV file at "
              "that exact path.")
        return False
    print(f"OK: driving audio present at {DRIVING_AUDIO}")
    return True


def check_source_face(need_video: bool = False):
    """Returns the source face Path to use, or None (and prints FAILED)
    if neither an image nor a video is available."""
    ASSETS_DIR.mkdir(exist_ok=True)
    preferred = SOURCE_FACE_VIDEO if need_video else SOURCE_FACE_IMAGE
    fallback = SOURCE_FACE_IMAGE if need_video else SOURCE_FACE_VIDEO
    if preferred.exists():
        print(f"OK: source face at {preferred}")
        return preferred
    if fallback.exists():
        print(f"OK: source face at {fallback} (using fallback type)")
        return fallback
    kind = "a short face video" if need_video else "a face photo (.jpg) or short face video (.mp4)"
    print(f"FAILED: no source face found in {ASSETS_DIR}")
    print(f"  -> drop {kind} at {preferred}")
    return None


def stage_asset(src: Path, repo_dir: Path, name: str) -> Path:
    """Copies src into repo_dir under `name` and returns that path.

    This project's own path contains "Phase 0.2" — a literal dot outside
    any file extension. Several of these repos' inference scripts guess
    image-vs-video by naively doing `path.split('.')[1]` instead of
    os.path.splitext, which silently picks the wrong branch when the dot
    in "Phase 0.2" becomes part [1] instead of the real extension (this is
    exactly what broke Wav2Lip: it fell through to its video-decode
    branch on a .jpg). repo_dir (under lipsync_models\\<Repo>) has no
    dots in its path, so staging inputs there sidesteps the bug instead
    of trying to patch every repo's parsing.
    """
    dest_dir = repo_dir / "_avatar_test_input"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / name
    shutil.copyfile(src, dest)
    return dest


def gpu_free_mb():
    """Free VRAM in MB via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def run_inference_subprocess(cmd: list, cwd: Path, timeout: int = 900):
    """Runs a model repo's own inference script, streaming its output live
    (these can run for minutes with no other feedback otherwise) while
    still capturing it for the final PASS/FAIL check. Returns
    (returncode_or_None_on_timeout, combined_stdout_stderr, elapsed_s)."""
    print(f"\nRunning:\n  {' '.join(str(c) for c in cmd)}\n  (cwd={cwd})\n")
    t0 = time.perf_counter()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
            if time.perf_counter() - t0 > timeout:
                proc.kill()
                proc.wait()
                return None, "".join(lines), time.perf_counter() - t0
        proc.wait(timeout=max(0.0, timeout - (time.perf_counter() - t0)))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None, "".join(lines), time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    return proc.returncode, "".join(lines), elapsed


def video_duration_seconds(path: Path):
    """Reads output video duration via OpenCV, for a realtime-factor stat."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps if fps else None
    except Exception:
        return None


def report_result(ok: bool, elapsed: float, out_path=None, extra: str = ""):
    print("\n" + "=" * 50)
    if ok:
        dur = video_duration_seconds(out_path) if out_path else None
        rtf = (dur / elapsed) if (dur and elapsed) else None
        if out_path:
            print(f"Output saved to:   {out_path}")
        if dur is not None:
            print(f"Output duration:   {dur:.2f}s")
        print(f"Wall-clock time:   {elapsed:.2f}s")
        if rtf is not None:
            print(f"Realtime factor:   {rtf:.2f}x "
                  f"({'faster than realtime — good' if rtf > 1 else 'SLOWER than realtime — will lag a live turn'})")
        if extra:
            print(extra)
        print("RESULT: PASS")
    else:
        if extra:
            print(extra)
        print("RESULT: FAIL")
