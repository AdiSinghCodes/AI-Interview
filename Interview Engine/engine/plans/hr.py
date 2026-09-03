"""HR / behavioural round."""

from __future__ import annotations

from engine.schemas.interview_plan import Persona

from ._build import plan, q

PLAN = plan(
    round_type="hr",
    persona=Persona(
        name="Priya",
        role_title="HR partner",
        demeanour="warm and encouraging",
        opening_line=(
            "Hi, I'm Priya from the people team. This is a relaxed conversation, "
            "just want to get to know how you work and what you're looking for. "
            "So to begin, tell me a bit about yourself and what drew you to apply "
            "here."
        ),
        system_prompt_extra=(
            "Be warm. Ask for a specific situation when an answer is generic. "
            "Never sound like an interrogation."
        ),
    ),
    topics=["motivation", "teamwork", "conflict", "ownership", "growth"],
    questions=[
        q("h1", "teamwork",
          "Tell me about a time you disagreed with a teammate on an approach. How did it play out?",
          ["specific situation", "how they raised it", "listening to the other side", "resolution and outcome"]),
        q("h2", "ownership",
          "Describe something that went wrong that was your responsibility. What did you do?",
          ["owns the mistake", "immediate response", "communication", "what changed afterwards"]),
        q("h3", "motivation",
          "What kind of work makes you lose track of time, and what kind drains you?",
          ["self-awareness", "concrete examples", "honest about weaknesses"], difficulty=2),
        q("h4", "conflict",
          "Have you worked with someone difficult? How did you handle it without it affecting the work?",
          ["stays professional", "separates person from problem", "escalation judgement"]),
        q("h5", "growth",
          "Tell me about feedback that was hard to hear. What did you do with it?",
          ["takes it seriously", "specific behaviour change", "no defensiveness"]),
        q("h6", "ownership",
          "Give me an example of going beyond what was asked of you. What made you do it?",
          ["initiative", "judgement about when it's worth it", "impact"], difficulty=2),
        q("h7", "teamwork",
          "How do you help a new person get up to speed on a team?",
          ["empathy", "structure / documentation", "pairing", "psychological safety"], difficulty=2),
        q("h8", "motivation",
          "Where do you want to be in a few years, and how does this role fit that?",
          ["has a direction", "realistic", "connects to the role"], difficulty=2),
        q("h9", "conflict",
          "Tell me about a deadline you were not going to make. How did you handle it?",
          ["raised it early", "options / tradeoffs offered", "no last-minute surprise"]),
        q("h10", "growth",
          "What's something you've changed your mind about in how you work?",
          ["reflective", "concrete before / after", "openness"], difficulty=2),
    ],
    criteria=[
        "gives specific situations, not generalities",
        "self-awareness and honesty",
        "communication and structure (situation, action, result)",
        "maturity in handling conflict and feedback",
    ],
)
