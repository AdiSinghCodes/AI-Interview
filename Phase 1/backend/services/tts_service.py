import edge_tts

class TTSService:
    def __init__(self):
        self.default_voice = 'en-US-GuyNeural'
        
    async def synthesize(self, text: str, voice: str = 'en-US-GuyNeural') -> bytes:
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            return bytes(audio_data)
        except Exception as e:
            print(f"TTS error: {e}")
            return b""

    async def get_voices(self) -> list:
        try:
            voices = await edge_tts.list_voices()
            return [v for v in voices if v["Locale"].startswith("en")]
        except Exception as e:
            print(f"Error getting voices: {e}")
            return []
