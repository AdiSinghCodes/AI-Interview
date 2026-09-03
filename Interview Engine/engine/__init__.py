"""NTRVSTA Phase 2 — the interview engine.

One live low-latency turn loop (LISTENING → THINKING → SPEAKING) driven by
`engine.orchestrator.session_orchestrator`. Every model it uses is named only
in `config/models.yaml`; `engine.settings` is the loader.
"""

__version__ = "0.1.0"
