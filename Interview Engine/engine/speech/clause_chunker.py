"""
Clause chunker (plan p5, module #4). Splits a streaming token feed into
speakable fragments at the first clause boundary once a fragment is long
enough — roughly 8-10 words — instead of waiting for a full sentence.
Worth 300-500 ms per turn on its own.

    chunker = ClauseChunker()
    for delta in llm_stream:
        for clause in chunker.feed(delta):
            tts.synthesize(clause)
    for clause in chunker.flush():
        tts.synthesize(clause)

Not wired into the live turn until build-order pass 3 (TTS lands); tested
now because it is pure text and cheap to get wrong.
"""

from __future__ import annotations

_CLAUSE_BOUNDARY = frozenset(",;:—")   # comma, semicolon, colon, em dash
_SENTENCE_END = frozenset(".!?")


class ClauseChunker:
    def __init__(self, min_words: int = 8, max_words: int = 14):
        self.min_words = min_words
        self.max_words = max_words
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        out: list[str] = []
        while True:
            cut = self._find_cut(self._buffer)
            if cut is None:
                break
            head, self._buffer = self._buffer[:cut], self._buffer[cut:].lstrip()
            head = head.strip()
            if head:
                out.append(head)
        return out

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

    def _find_cut(self, text: str) -> int | None:
        words = 0
        for i, ch in enumerate(text):
            if ch.isspace() and i > 0 and not text[i - 1].isspace():
                words += 1
            if ch in _SENTENCE_END and words >= 1:
                return i + 1
            if ch in _CLAUSE_BOUNDARY and words >= self.min_words:
                return i + 1
            if words >= self.max_words and ch.isspace():
                return i
        return None
