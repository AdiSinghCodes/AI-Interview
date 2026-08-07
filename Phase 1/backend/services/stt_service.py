import io

class STTService:
    def __init__(self, model_size="base"):
        self.model = None
        self.model_size = model_size
    
    def _load_model(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                print(f"Whisper model '{self.model_size}' loaded")
            except Exception as e:
                print(f"Failed to load Whisper: {e}")
    
    def transcribe(self, audio_bytes: bytes) -> dict:
        if self.model is None:
            self._load_model()
            if self.model is None:
                return {"transcript": "STT Service unavailable.", "words": []}
                
        try:
            # faster_whisper can process file-like objects or byte buffers,
            # but sometimes it requires a path or a specific format.
            # Using io.BytesIO
            segments, info = self.model.transcribe(io.BytesIO(audio_bytes), word_timestamps=True)
            text = ""
            words = []
            for segment in segments:
                text += segment.text + " "
                if segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
            
            return {
                "transcript": text.strip(),
                "words": words
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"transcript": f"Transcription error: {e}", "words": []}
