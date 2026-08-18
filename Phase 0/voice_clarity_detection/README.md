# Voice Clarity Detection Model

A real-time voice clarity coach using Silero VAD (Voice Activity Detection) and RMS loudness analysis. Unlike the other Phase 0 modules, this one isn't about catching cheating — it makes sure the candidate's *answer* can actually be heard, and nudges them to speak up when it can't.

## Features

✅ **Speech Detection** - Silero VAD tells us when the candidate is speaking vs silent (falls back to a volume-based detector if Silero isn't installed)
✅ **Loudness Analysis** - Measures RMS loudness (dBFS) of speech, independent of whether speech is present
✅ **Low Volume Coaching** - If speech stays too quiet for >1.5s, shows: *"Your voice is not clear. Please speak a little louder."*
✅ **No Audio / Frozen Mic Coaching** - If nothing is heard for >8s, shows: *"We can't hear you. Please check your microphone."*
✅ **Cooldown** - Coaching messages don't repeat more than once every 5 seconds, so they don't spam the candidate

## Project Structure

```
voice_clarity_detection/
├── voice_clarity_model.py   # VoiceClarityDetector - VAD + loudness logic
├── demo.py                  # Real-time microphone demo
└── requirements.txt         # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Silero VAD + torch are optional — if they aren't installed (or fail to load), the detector automatically falls back to a simple RMS-based voice detector, so the module still works out of the box.

## Usage

### Run the Demo

```bash
cd voice_clarity_detection
python demo.py
```

**Controls:**
- Press `q` to quit

Speak normally, then try mumbling or speaking far from the mic — a message banner should appear telling you to speak louder. Stop talking for 8+ seconds and it will ask you to check your microphone.

## How It Works

1. Microphone audio is captured in ~32ms chunks (512 samples @ 16kHz, the size Silero VAD expects).
2. Each chunk is scored for `voice_prob` (0-1) by Silero VAD — this is speech-vs-silence, not loudness.
3. While speech is detected, the chunk's RMS loudness is separately measured in dBFS.
4. If loudness stays below `QUIET_DBFS` (-32 dBFS) for `LOW_VOLUME_HOLD_SEC` (1.5s), the "speak louder" message fires.
5. If no speech is detected at all for `LONG_SILENCE_SEC` (8s), the "check your microphone" message fires.
6. A `MESSAGE_COOLDOWN_SEC` (5s) prevents either message from re-firing too often.

## Integrating with the interview flow

`VoiceClarityDetector.process_chunk(chunk)` returns:

```python
{
    'voice_prob': 0.83,        # 0..1, speech probability
    'is_speaking': True,
    'rms_dbfs': -28.4,         # loudness of this chunk
    'clarity': 'CLEAR',        # 'CLEAR' | 'LOW_VOLUME' | 'SILENT'
    'message': None,           # coaching string, or None if nothing to say
}
```

Wire the microphone stream (e.g. via `sounddevice.InputStream`, as in `demo.py`) into `process_chunk()` per chunk, and surface `message` to the candidate (banner, TTS prompt, etc.) whenever it's non-`None`. This mirrors how `eye_contact_detection`, `face_orientation_detection`, etc. expose a `process_frame()` per-frame API for the video side — this module is the audio-side equivalent, ready to plug into `integrated_system` alongside the video detectors.

## Tuning

All thresholds are class constants on `VoiceClarityDetector` so they're easy to tune per-microphone/room:

| Constant | Default | Meaning |
|---|---|---|
| `VAD_THRESHOLD` | 0.5 | Min voice probability to count as "speaking" |
| `QUIET_DBFS` | -32.0 | Below this loudness, speech is considered too quiet |
| `LOW_VOLUME_HOLD_SEC` | 1.5 | How long quiet speech must persist before coaching fires |
| `LONG_SILENCE_SEC` | 8.0 | How long total silence must persist before "can't hear you" fires |
| `MESSAGE_COOLDOWN_SEC` | 5.0 | Minimum time between coaching messages |

## Troubleshooting

**Issue**: `sounddevice not installed`
- Solution: `pip install sounddevice` (on Windows this pulls in PortAudio automatically)

**Issue**: Silero VAD not available
- Solution: `pip install silero-vad torch` — the module still runs on the volume fallback without this

**Issue**: Coaching message fires even when speaking normally
- Solution: your mic gain may be low — lower `QUIET_DBFS` (e.g. to -38) to match your setup

## License

Project for interview practice avatar system
