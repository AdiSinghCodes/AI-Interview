#!/usr/bin/env python3
"""
Quick installer for all proctoring model test dependencies.
Run this once before running any test scripts.

Usage:  python install_deps.py
"""

import subprocess
import sys

def pip(pkg, extra=""):
    cmd = [sys.executable, "-m", "pip", "install", pkg]
    if extra:
        cmd.append(extra)
    print(f"  Installing {pkg}…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ {pkg}")
    else:
        print(f"  ⚠  {pkg} — {result.stderr.strip()[-100:]}")

print("=" * 55)
print("  Proctoring Test Suite — Dependency Installer")
print("=" * 55)

print("\n[1/6] Core vision libraries")
pip("opencv-python")
pip("numpy")
pip("Pillow")

print("\n[2/6] YOLOv8 (Test 1)")
pip("ultralytics")

print("\n[3/6] MediaPipe (Tests 2, 4, 5)")
pip("mediapipe==0.10.14")

print("\n[4/6] InsightFace (Test 3)")
pip("insightface")
pip("onnxruntime")   # CPU inference
pip("onnxruntime-gpu")  # uncomment if you have GPU

print("\n[5/6] Audio (Tests 6 & 7)")
pip("sounddevice")
pip("silero-vad")

print("\n[6/6] PyAnnote (Test 7 — optional, needs HuggingFace token)")
pip("pyannote.audio")

print("\n" + "=" * 55)
print("  Installation complete!")
print("\nOptional GPU support:")
print("  pip install onnxruntime-gpu torch torchvision")
print("\nFor PyAnnote (Test 7):")
print("  1. Sign up at huggingface.co")
print("  2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
print("     and click 'Agree and access repository'")
print("  3. Get token at https://huggingface.co/settings/tokens")
print("  4. export HF_TOKEN=hf_your_token")
print("=" * 55)