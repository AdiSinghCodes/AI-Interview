"""Python backend developer round."""

from __future__ import annotations

from engine.schemas.interview_plan import Persona

from ._build import plan, q

PLAN = plan(
    round_type="technical",
    persona=Persona(
        name="Ravi",
        role_title="senior Python engineer",
        demeanour="probing but warm",
        opening_line=(
            "Hi, I'm Ravi, I'll be running your Python round today. Talk me "
            "through your reasoning as you answer, I care more about how you "
            "think than a perfect recall. First up, tell me about a Python "
            "service you've built and what it does."
        ),
        system_prompt_extra="Push once for specifics when an answer is vague. Stay on Python and its ecosystem.",
    ),
    topics=["language internals", "data model", "concurrency", "web / APIs", "testing and packaging"],
    questions=[
        q("p1", "data model",
          "What's the difference between a list and a tuple, beyond mutability? When does the choice actually matter?",
          ["hashability", "dict keys", "memory / caching", "intent signalling"], difficulty=2),
        q("p2", "language internals",
          "Explain how a Python decorator works. Walk me through what happens when the module is imported.",
          ["functions are objects", "closure over the wrapped fn", "runs at def time", "functools.wraps"]),
        q("p3", "data model",
          "What is the difference between __str__ and __repr__, and what happens if you only define one?",
          ["repr fallback", "developer vs user audience", "containers use repr", "eval round-trip ideal"], difficulty=2),
        q("p4", "language internals",
          "What is the GIL, and how does it affect CPU-bound versus IO-bound work?",
          ["one bytecode at a time", "threads fine for IO", "multiprocessing for CPU", "C extensions release it"], difficulty=4),
        q("p5", "concurrency",
          "When would you use asyncio over threads? What's the failure mode of mixing blocking calls into an async function?",
          ["single-threaded event loop", "blocking starves the loop", "run_in_executor", "cooperative scheduling"], difficulty=4),
        q("p6", "data model",
          "A default argument like `def f(x, items=[])` bites people. What happens and why?",
          ["default evaluated once at def time", "shared mutable default", "use None sentinel"]),
        q("p7", "web / APIs",
          "In a FastAPI or Flask app, how do you keep a slow external call from blocking other requests?",
          ["async + httpx", "worker count", "timeouts", "background tasks / queue"]),
        q("p8", "testing and packaging",
          "How do you structure tests for code that talks to a database or an external API?",
          ["fixtures", "dependency injection / seams", "fakes vs mocks", "transaction rollback", "contract tests"]),
        q("p9", "language internals",
          "What's a generator, and how is it different from returning a list? Give a case where it clearly wins.",
          ["lazy evaluation", "constant memory", "streaming large data", "yield / state machine"]),
        q("p10", "testing and packaging",
          "What goes in a requirements file versus pyproject, and why do lock files exist?",
          ["direct vs transitive deps", "version pinning", "reproducible installs", "editable installs"], difficulty=2),
    ],
    criteria=[
        "accuracy on Python semantics and internals",
        "awareness of real-world failure modes",
        "clear explanation with concrete examples",
        "judgement about when to use what",
    ],
)
