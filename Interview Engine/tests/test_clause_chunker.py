from engine.speech.clause_chunker import ClauseChunker


def test_emits_on_sentence_end():
    c = ClauseChunker()
    assert c.feed("Okay that makes sense. ") == ["Okay that makes sense."]


def test_holds_a_short_leading_clause_until_it_is_long_enough():
    c = ClauseChunker(min_words=8)
    assert c.feed("Right, ") == []                 # comma after 1 word — too short
    assert c.feed("so how would you ") == []
    out = c.feed("actually scale that write path, and then ")
    assert len(out) == 1
    assert out[0].endswith("write path,")


def test_max_words_forces_a_cut_without_punctuation():
    c = ClauseChunker(min_words=8, max_words=10)
    out = c.feed("one two three four five six seven eight nine ten eleven ")
    assert out == ["one two three four five six seven eight nine ten"]


def test_flush_returns_the_remainder():
    c = ClauseChunker()
    c.feed("trailing words with no period")
    assert c.flush() == ["trailing words with no period"]
    assert c.flush() == []
