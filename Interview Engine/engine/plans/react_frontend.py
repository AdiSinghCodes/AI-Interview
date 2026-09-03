"""React frontend developer round."""

from __future__ import annotations

from engine.schemas.interview_plan import Persona

from ._build import plan, q

PLAN = plan(
    round_type="technical",
    persona=Persona(
        name="Maya",
        role_title="senior frontend engineer",
        demeanour="friendly and curious",
        opening_line=(
            "Hey, I'm Maya, I'll be doing your React round. Feel free to think "
            "out loud. Let's start easy, tell me about a React app you've worked "
            "on and what your role was."
        ),
        system_prompt_extra="Keep it practical, focus on hooks, rendering, and state. Probe on performance when it comes up.",
    ),
    topics=["rendering", "hooks", "state management", "performance", "architecture"],
    questions=[
        q("r1", "rendering",
          "What actually triggers a re-render in React, and what's the difference between a re-render and a DOM update?",
          ["state / props / parent render", "virtual DOM diff", "reconciliation", "commit vs render phase"]),
        q("r2", "hooks",
          "Explain the dependency array of useEffect. What are the common ways people get it wrong?",
          ["runs after render", "stale closures", "missing deps", "objects / functions as deps", "cleanup"], difficulty=3),
        q("r3", "hooks",
          "When would you use useMemo or useCallback, and when is reaching for them a mistake?",
          ["referential stability", "expensive compute", "premature optimisation", "profiler first"], difficulty=3),
        q("r4", "state management",
          "How do you decide between local state, lifting state up, context, and a store like Redux or Zustand?",
          ["locality of state", "prop drilling", "context re-render cost", "server vs client state"]),
        q("r5", "rendering",
          "What's the point of keys in a list, and what breaks if you use the array index as the key?",
          ["identity across renders", "state / DOM reuse bugs", "reorder / insert / delete"], difficulty=2),
        q("r6", "performance",
          "A page feels janky when typing in an input that filters a big list. How do you diagnose and fix it?",
          ["controlled input re-renders", "debounce", "virtualisation", "memoised rows", "React Profiler / flame chart"], difficulty=4),
        q("r7", "state management",
          "How do you handle data fetching, caching, and loading states in a React app today?",
          ["React Query / SWR", "cache keys", "stale-while-revalidate", "race conditions", "error boundaries"]),
        q("r8", "architecture",
          "What is server-side rendering solving, and what does it cost you?",
          ["first paint / SEO", "hydration", "server load", "client-server code split", "waterfalls"]),
        q("r9", "hooks",
          "Why can't you call hooks inside a condition or a loop?",
          ["call order identity", "linked list of hook state", "rules of hooks", "eslint plugin"], difficulty=3),
        q("r10", "architecture",
          "How would you structure a medium-sized React codebase so it stays maintainable?",
          ["feature folders", "component / container split", "shared UI library", "typed API layer", "avoiding god components"], difficulty=2),
    ],
    criteria=[
        "understanding of React's rendering model",
        "correct mental model of hooks",
        "performance-debugging instinct",
        "sensible architecture judgement",
    ],
)
