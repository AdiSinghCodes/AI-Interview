"""DSA / problem-solving round — spoken reasoning, no editor in this pass."""

from __future__ import annotations

from engine.schemas.interview_plan import Persona

from ._build import plan, q

PLAN = plan(
    round_type="dsa",
    persona=Persona(
        name="Arjun",
        role_title="engineer",
        demeanour="quiet, lets you think",
        opening_line=(
            "Hi, I'm Arjun. This round is about how you approach problems, so "
            "please think out loud, state your assumptions, and talk through the "
            "tradeoffs. There's a code editor for later rounds, today it's just "
            "the two of us talking it through. Ready? Here's the first one."
        ),
        system_prompt_extra=(
            "Let silence sit. Prompt with 'what's the time complexity?' or 'can "
            "you do better?' rather than giving hints early."
        ),
    ),
    topics=["arrays and strings", "hashing", "two pointers", "trees", "complexity"],
    questions=[
        q("d1", "hashing",
          "Given an array of integers and a target, describe how you'd find two numbers that add up to the target.",
          ["brute force O(n^2)", "hash map complement O(n)", "space tradeoff"], difficulty=2),
        q("d2", "complexity",
          "For that solution, what's the time and space complexity, and what changes if the array is already sorted?",
          ["O(n) time / O(n) space", "sorted -> two pointers O(1) space", "sort cost if unsorted"]),
        q("d3", "two pointers",
          "How would you check if a string is a palindrome, ignoring case and non-letters?",
          ["two pointers from both ends", "skip non-alphanumeric", "O(n) / O(1)"], difficulty=2),
        q("d4", "arrays and strings",
          "You're given stock prices by day. Describe how to find the maximum profit from one buy and one sell.",
          ["track running min", "max of price - min so far", "single pass O(n)"], difficulty=3),
        q("d5", "hashing",
          "How would you find the first non-repeating character in a string?",
          ["count pass then scan", "ordered counts", "O(n) time"], difficulty=2),
        q("d6", "trees",
          "Describe how you'd check whether a binary tree is a valid binary search tree.",
          ["in-order is sorted", "min/max bounds recursion", "off-by-one on equal keys"], difficulty=4),
        q("d7", "arrays and strings",
          "Given an array, describe how to move all zeros to the end while keeping the order of the rest.",
          ["write pointer", "swap or overwrite", "in-place O(1) space"], difficulty=3),
        q("d8", "two pointers",
          "How would you merge two sorted arrays into one sorted array?",
          ["two pointers", "compare heads", "O(n+m)", "merge from the back if in-place"], difficulty=3),
        q("d9", "complexity",
          "Explain the difference between O(n log n) and O(n^2) with a real example of each.",
          ["merge/quick sort vs nested loops", "growth at scale", "constant factors caveat"], difficulty=2),
        q("d10", "trees",
          "Describe breadth-first versus depth-first traversal of a tree. When would you pick each?",
          ["queue vs stack / recursion", "shortest path -> BFS", "memory: BFS wide, DFS deep", "level order"], difficulty=3),
    ],
    criteria=[
        "states assumptions and clarifies before coding",
        "reaches a correct approach, improves it when pushed",
        "accurate complexity analysis",
        "communicates the reasoning clearly while thinking",
    ],
)
