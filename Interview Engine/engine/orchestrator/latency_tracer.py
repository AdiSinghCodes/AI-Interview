"""
LatencyTracer — accumulates one `TurnTrace` per turn and answers the two
questions the loop actually needs:

  1. per-stage p50 / p95 across the session (where the time goes)
  2. is TTFT p95 over the danger line? (the degrade-ladder trigger)

The plan (p9, module #11) wants this in from turn 1. `dump()` prints a
table you can read at the end of a run.
"""

from __future__ import annotations

from engine.schemas.latency_trace import Stage, TurnTrace
from engine.settings import get_models


class LatencyTracer:
    def __init__(self) -> None:
        self._traces: list[TurnTrace] = []

    def new_turn(self, turn_index: int, *, cache_hit: bool = False) -> TurnTrace:
        trace = TurnTrace(turn_index=turn_index, cache_hit=cache_hit)
        self._traces.append(trace)
        return trace

    def adopt(self, trace: TurnTrace) -> None:
        """Register a trace created elsewhere (ws_live starts one at the VAD
        endpoint, before the orchestrator turn begins)."""
        if not any(t is trace for t in self._traces):
            self._traces.append(trace)

    # -- stats ----------------------------------------------------------
    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
        return ordered[k]

    def _stage_values(self, stage: Stage | str) -> list[float]:
        key = stage.value if isinstance(stage, Stage) else stage
        return [t.deltas()[key] for t in self._traces if key in t.deltas()]

    def p50(self, stage: Stage | str) -> float:
        return self._percentile(self._stage_values(stage), 50)

    def p95(self, stage: Stage | str) -> float:
        return self._percentile(self._stage_values(stage), 95)

    def p95_ttft(self) -> float:
        """Time from answer-received to first token, p95 — the danger-line metric."""
        vals = [
            t.ms(Stage.TTFT)
            for t in self._traces
            if t.ms(Stage.TTFT) is not None
        ]
        return self._percentile([v for v in vals if v is not None], 95)

    def over_danger_line(self) -> bool:
        danger = get_models().latency_budget_ms.danger_line
        return len(self._traces) >= 3 and self.p95_ttft() > danger

    # -- reporting ----------------------------------------------------
    def summary(self) -> dict:
        stages = [s for s in Stage.order() if self._stage_values(s)]
        return {
            "turns": len(self._traces),
            "cache_hits": sum(1 for t in self._traces if t.cache_hit),
            "p95_ttft_ms": round(self.p95_ttft(), 1),
            "danger_line_ms": get_models().latency_budget_ms.danger_line,
            "stages": {
                s: {"p50": round(self.p50(s), 1), "p95": round(self.p95(s), 1)}
                for s in stages
            },
        }

    def dump(self) -> str:
        s = self.summary()
        rows = [
            f"  latency over {s['turns']} turn(s)  "
            f"({s['cache_hits']} cache hit(s))",
            f"  {'stage':<20}{'p50 ms':>10}{'p95 ms':>10}",
            f"  {'-' * 40}",
        ]
        for stage, v in s["stages"].items():
            rows.append(f"  {stage:<20}{v['p50']:>10.1f}{v['p95']:>10.1f}")
        flag = "  OVER" if s["p95_ttft_ms"] > s["danger_line_ms"] else "  ok"
        rows.append(f"  {'-' * 40}")
        rows.append(
            f"  TTFT p95 {s['p95_ttft_ms']:.1f} ms  "
            f"(danger line {s['danger_line_ms']} ms){flag}"
        )
        return "\n".join(rows)
