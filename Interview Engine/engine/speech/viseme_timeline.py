"""
Viseme timeline generator (plan p6, module #6). Turns TTS phoneme +
duration output into a time-stamped ARKit-blendshape track the browser
Three.js mesh plays in sync with the audio. Amplitude-only lip sync
"reads as a fish opening and closing its mouth" — this is what avoids that.

Runs server-side, ships down the socket alongside the audio chunk. Costs
zero VRAM (the mesh renders on the browser GPU).

STUB — build-order pass 5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisemeFrame:
    t_ms: float          # offset from the start of the utterance
    viseme: str          # ARKit blendshape group, e.g. "viseme_PP", "viseme_AA"
    weight: float        # 0-1


def build_timeline(phoneme_durations: list[tuple[str, float]]) -> list[VisemeFrame]:
    raise NotImplementedError("viseme timeline lands in build-order pass 5")
