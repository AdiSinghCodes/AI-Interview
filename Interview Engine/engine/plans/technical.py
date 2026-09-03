"""General technical round — core CS + backend fundamentals, stack-agnostic."""

from __future__ import annotations

from engine.schemas.interview_plan import Persona

from ._build import plan, q

PLAN = plan(
    round_type="technical",
    persona=Persona(
        name="Ravi",
        role_title="senior engineer",
        demeanour="probing but warm",
        opening_line=(
            "Hi, thanks for making the time. I'm Ravi, and I'll be taking your "
            "technical round today. There's no trick questions here, just talk me "
            "through your thinking as you go. To start, can you walk me through a "
            "project you've built that you're proud of, end to end?"
        ),
        system_prompt_extra=(
            "Favour depth over breadth. When an answer stays shallow, probe once "
            "with a concrete follow-up before moving to the next topic."
        ),
    ),
    topics=["fundamentals", "APIs and HTTP", "databases", "concurrency", "system reliability"],
    questions=[
        q("t1", "fundamentals",
          "What's the difference between a process and a thread, and when would you reach for one over the other?",
          ["memory isolation", "context-switch cost", "GIL / shared state", "IPC"]),
        q("t2", "APIs and HTTP",
          "How would you design a REST endpoint that returns a paginated list over millions of rows?",
          ["cursor vs offset", "index on the sort key", "cost of an exact total count", "page size limits"]),
        q("t3", "APIs and HTTP",
          "What does idempotency mean for an HTTP API, and which methods should be idempotent?",
          ["safe vs idempotent", "retries", "PUT/DELETE idempotent, POST not", "idempotency keys"]),
        q("t4", "databases",
          "Explain the difference between the READ COMMITTED and SERIALIZABLE isolation levels. When would you pick each?",
          ["dirty / phantom reads", "lost updates", "throughput cost", "default in Postgres"], difficulty=4),
        q("t5", "databases",
          "You have a query that's suddenly slow in production. Walk me through how you'd diagnose it.",
          ["EXPLAIN ANALYZE", "missing / unused index", "row estimates vs actual", "N+1", "lock contention"]),
        q("t6", "concurrency",
          "Two workers pull from the same job queue and occasionally process the same job. How do you make it exactly-once, or explain why you can't?",
          ["at-least-once + dedupe", "idempotency keys", "visibility timeout", "distributed locks"], difficulty=4),
        q("t7", "system reliability",
          "A downstream service you depend on starts taking 5 seconds instead of 50 milliseconds. What happens to your service, and what do you put in place?",
          ["connection / thread pool exhaustion", "timeouts", "circuit breaker", "bulkhead", "graceful degradation"]),
        q("t8", "system reliability",
          "What's the difference between a load balancer doing round-robin versus least-connections, and when does it matter?",
          ["uneven request cost", "long-lived connections", "slow-start", "health checks"]),
        q("t9", "fundamentals",
          "What is a hash collision and how do hash maps handle it? What's the worst-case lookup?",
          ["chaining vs open addressing", "load factor / resize", "O(n) worst case", "hash quality"]),
        q("t10", "APIs and HTTP",
          "How do you version a public HTTP API without breaking existing clients?",
          ["URL vs header versioning", "additive-only changes", "deprecation window", "contract tests"], difficulty=2),
    ],
    criteria=[
        "correctness of core concepts",
        "depth of reasoning under follow-up",
        "clear communication and structure",
        "practical judgement and tradeoff awareness",
    ],
)
