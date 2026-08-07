import numpy as np
import re
from models.schemas import VoiceAnalysisResult

class VoiceAnalyzer:
    def __init__(self):
        self.filler_words = r'\b(um|uh|like|you know|basically|actually|literally|so|well|i mean)\b'
        
    def analyze_audio(self, audio_bytes: bytes, transcript: str = None, duration_sec: float = 1.0) -> VoiceAnalysisResult:
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_array) > 0:
                rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
                volume_db = 20 * np.log10(rms) if rms > 0 else -100
                silence_ratio = np.sum(np.abs(audio_array) < 500) / len(audio_array)
            else:
                volume_db = -100
                silence_ratio = 1.0
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            volume_db = -50
            silence_ratio = 0.0
            
        filler_count = 0
        fillers_detected = []
        speaking_rate_wpm = 0.0
        
        if transcript:
            words = transcript.split()
            duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.1
            speaking_rate_wpm = len(words) / duration_min
            
            matches = re.finditer(self.filler_words, transcript.lower())
            for match in matches:
                fillers_detected.append(match.group(0))
            filler_count = len(fillers_detected)
            
        feedback_messages = []
        if volume_db < -30:
            feedback_messages.append("Your voice is too quiet, please speak louder")
        elif volume_db > -5:
            feedback_messages.append("You are speaking too loudly")
            
        if transcript:
            if speaking_rate_wpm > 180:
                feedback_messages.append("You are speaking too fast, try to slow down")
            elif speaking_rate_wpm < 80 and speaking_rate_wpm > 0:
                feedback_messages.append("Try to speak more fluently")
                
            if filler_count > 5:
                feedback_messages.append("Try to reduce filler words like um, uh")
                
        if silence_ratio > 0.6:
            feedback_messages.append("Please provide your response")
            
        return VoiceAnalysisResult(
            volume_db=float(volume_db),
            speaking_rate_wpm=float(speaking_rate_wpm),
            filler_count=filler_count,
            fillers_detected=fillers_detected,
            silence_ratio=float(silence_ratio),
            feedback_messages=feedback_messages
        )
