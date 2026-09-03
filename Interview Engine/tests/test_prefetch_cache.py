import numpy as np

from engine.orchestrator.prefetch_cache import PrefetchCache
from engine.plans import get_plan


class _FakeTTS:
    sample_rate = 24_000

    async def synthesize(self, text):
        yield np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)


async def test_warm_populates_everything_and_reports_progress():
    plan = get_plan("hr", num_questions=5)
    cache = PrefetchCache()
    assert not cache.ready

    await cache.warm(plan, _FakeTTS())

    assert cache.ready
    done, total = cache.progress
    assert done == total == 5 + 2 + 4          # questions + opening/closing + 4 acks
    assert cache.opening and cache.opening.pcm
    assert cache.closing and cache.closing.pcm
    for q in plan.questions:
        assert cache.get_question(q.id) is not None
    assert cache.random_ack().pcm


async def test_missing_question_is_none():
    cache = PrefetchCache()
    await cache.warm(get_plan("hr", num_questions=3), _FakeTTS())
    assert cache.get_question("nope") is None
