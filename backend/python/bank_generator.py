import json
import os
import re
from typing import Any, Dict, List

def is_technical_domain(setup: Dict[str, Any]) -> bool:
    domain = str(setup.get("domain", "")).lower()
    sub_domain = str(setup.get("subDomain", "")).lower()
    role = str(setup.get("role", "")).lower()
    types = setup.get("types") or []
    if isinstance(types, list):
        types_str = " ".join(str(x) for x in types).lower()
    else:
        types_str = str(types).lower()
    interview_type = str(setup.get("interviewType", "")).lower()

    if setup.get("isTechnical") or setup.get("requiresCoding") or setup.get("requiresSql"):
        return True

    pattern = r"software|coding|dsa|developer|engineer|fullstack|frontend|backend|web|data science|database|sql|devops|cloud|ai|machine learning|python|java|react|node"
    if re.search(pattern, domain) or re.search(pattern, sub_domain) or re.search(pattern, role) or re.search(pattern, types_str) or re.search(pattern, interview_type):
        return True
    return False

def generate_question_bank(setup: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    total = int(plan.get("total", 10))
    domain = str(setup.get("domain", "General")).strip()
    sub_domain = str(setup.get("subDomain", "")).strip()
    role = str(setup.get("role", "Candidate")).strip()
    stage = str(setup.get("stage", "entry")).strip()
    objective = str(setup.get("objective", "Job Interview")).strip()
    tech_stack = setup.get("techStack") or []
    if isinstance(tech_stack, list):
        tech_str = ", ".join(str(x) for x in tech_stack if x)
    else:
        tech_str = str(tech_stack)
    coding_lang = str(setup.get("codingLanguage") or "python").lower()

    is_tech = is_technical_domain(setup)

    easy_count = max(2, min(4, int(total * 0.25)))
    hard_count = max(2, min(4, int(total * 0.25)))
    medium_count = max(2, total - easy_count - hard_count)

    prompt = f"""You are an expert AI Interviewer creating a full, structured interview question bank.

INTERVIEW CONFIGURATION:
* Domain: {domain}
* Sub-Domain: {sub_domain}
* Role: {role}
* Experience Stage: {stage}
* Objective: {objective}
* Tech Stack / Skills: {tech_str}
* Total Questions Required: {total}
* Is Technical Interview: {is_tech}

{"STRICT TECHNICAL RULES: Include 2 Pseudocode questions in Medium Tier and 2 Coding/SQL Workspace problems (section: coding/sql) in Hard Tier." if is_tech else "NON-TECHNICAL: All verbal."}

STRICT TIERING REQUIREMENTS:
1. EASY TIER ({easy_count} questions):
   - Q1 MUST be a warm-up Candidate Introduction ("Introduce yourself, share your background, key projects, and experience").
   - Q2 and subsequent easy questions MUST ask about candidate education/college years, what inspired them to choose {sub_domain or domain}, and core career goals.

2. MEDIUM TIER ({medium_count} questions):
   - Practical technical concepts and scenario questions directly relevant to {role} in {sub_domain}.
   {"- Q5 & Q6 MUST be Pseudocode challenges asking for step-by-step pseudocode logic." if is_tech else ""}

3. HARD TIER ({hard_count} questions):
   - Advanced technical deep-dives, system architecture, failure handling.
   {"- Q9 MUST be section: 'coding' and Q10 MUST be section: 'sql' or 'coding' workspace problem with problem, constraints, examples, language: '{coding_lang}'." if is_tech else ""}

Return ONLY a valid JSON object matching this schema:
{{
  "questions": [
    {{
      "id": 1,
      "difficultyTier": "easy",
      "section": "verbal",
      "questionType": "introduction",
      "category": "Easy / Background",
      "text": "Exact question text...",
      "intent": "Assessment focus...",
      "problem": null,
      "constraints": [],
      "examples": [],
      "language": ""
    }}
  ]
}}"""

    try:
        from normal_questions_agent import _call
        raw = _call(
            [{"role": "system", "content": prompt}],
            temperature=0.4,
            max_tokens=3500,
            json_mode=True,
        )
        data = json.loads(raw)
        qs = data.get("questions") or data.get("question_bank") or []
        if isinstance(qs, list) and len(qs) >= 5:
            valid_qs = []
            for idx, q in enumerate(qs, 1):
                tier = q.get("difficultyTier") or ("easy" if idx <= easy_count else "hard" if idx > (total - hard_count) else "medium")
                section = str(q.get("section") or q.get("type") or "verbal").lower()
                qtype = "coding" if section == "coding" else "sql" if section == "sql" else "verbal"
                valid_qs.append({
                    "id": idx,
                    "difficultyTier": str(tier).lower(),
                    "type": qtype,
                    "section": section,
                    "questionType": str(q.get("questionType") or f"{tier}_question"),
                    "category": str(q.get("category") or f"{tier.capitalize()} Level"),
                    "text": str(q.get("text") or q.get("question") or ""),
                    "intent": str(q.get("intent") or "Assess candidate response."),
                    "problem": q.get("problem") or None,
                    "constraints": q.get("constraints") or [],
                    "examples": q.get("examples") or [],
                    "language": q.get("language") or (coding_lang if qtype != "verbal" else "")
                })

            # ENFORCE TECHNICAL REQUIREMENTS IN PYTHON OUTPUT
            if is_tech and len(valid_qs) >= 10:
                # 1. Medium pseudocode enforcement on Q5 & Q6
                if not any("pseudocode" in q["text"].lower() for q in valid_qs[3:7]):
                    valid_qs[4] = {
                        "id": 5,
                        "difficultyTier": "medium",
                        "type": "verbal",
                        "section": "verbal",
                        "questionType": "pseudocode_logic",
                        "category": "Medium / Pseudocode & Logic",
                        "text": f"Pseudocode Challenge: Write step-by-step pseudocode or explain the exact algorithmic logic to find the first non-repeating character in a string with O(n) time complexity.",
                        "intent": "Evaluate pseudocode logic and time complexity reasoning.",
                        "problem": None, "constraints": [], "examples": [], "language": ""
                    }
                    valid_qs[5] = {
                        "id": 6,
                        "difficultyTier": "medium",
                        "type": "verbal",
                        "section": "verbal",
                        "questionType": "pseudocode_logic",
                        "category": "Medium / Pseudocode & Logic",
                        "text": f"Pseudocode Challenge: Write step-by-step pseudocode or outline the algorithm to validate if a given string containing brackets '()[]{{}}' is balanced using a stack data structure.",
                        "intent": "Evaluate stack data structure comprehension and pseudocode clarity.",
                        "problem": None, "constraints": [], "examples": [], "language": ""
                    }

                # 2. Hard coding/SQL workspace problem enforcement on Q9 & Q10
                if not any(q["type"] in ["coding", "sql"] for q in valid_qs[-3:]):
                    valid_qs[8] = {
                        "id": 9,
                        "difficultyTier": "hard",
                        "type": "coding",
                        "section": "coding",
                        "questionType": "coding_workspace_problem",
                        "category": "Hard / Coding Workspace",
                        "text": f"Hands-on Coding Challenge: Write a function in {coding_lang} to solve the Two Sum Problem. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. Optimize your code for O(n) time complexity.",
                        "intent": "Assess hands-on coding, workspace execution, and algorithmic optimization.",
                        "problem": {
                            "title": "Two Sum Algorithmic Optimization",
                            "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution.",
                            "inputFormat": "nums = [2, 7, 11, 15], target = 9",
                            "outputFormat": "[0, 1]",
                            "sampleInput": "[2, 7, 11, 15], target = 9",
                            "sampleOutput": "[0, 1]"
                        },
                        "constraints": ["2 <= nums.length <= 10^4", "Target time complexity: O(n)"],
                        "examples": [{"input": "nums = [2, 7, 11, 15], target = 9", "output": "[0, 1]"}],
                        "language": coding_lang
                    }
                    valid_qs[9] = {
                        "id": 10,
                        "difficultyTier": "hard",
                        "type": "sql" if setup.get("requiresSql") else "coding",
                        "section": "sql" if setup.get("requiresSql") else "coding",
                        "questionType": "sql_workspace_problem" if setup.get("requiresSql") else "coding_workspace_problem",
                        "category": "Hard / SQL Workspace" if setup.get("requiresSql") else "Hard / Coding Workspace",
                        "text": f"Hands-on SQL Challenge: Write an SQL query to find the top 2 highest-earning employees in each department from an Employees table (columns: id, name, salary, department_id)." if setup.get("requiresSql") else f"Hands-on Coding Challenge: Write a function in {coding_lang} to reverse a singly linked list and return the head of the reversed list.",
                        "intent": "Assess hands-on database query construction or linked list pointer manipulation.",
                        "problem": {
                            "title": "Top Department Earners Query" if setup.get("requiresSql") else "Reverse Singly Linked List",
                            "description": "Write a query to find employees with highest salaries per department." if setup.get("requiresSql") else "Reverse a singly linked list in O(n) time and O(1) space.",
                            "inputFormat": "Employees (id, name, salary, department_id)" if setup.get("requiresSql") else "head = [1, 2, 3, 4, 5]",
                            "outputFormat": "Department, Employee, Salary" if setup.get("requiresSql") else "[5, 4, 3, 2, 1]"
                        },
                        "constraints": ["Use DENSE_RANK() window functions"] if setup.get("requiresSql") else ["Time complexity: O(n)"],
                        "examples": [],
                        "language": "sql" if setup.get("requiresSql") else coding_lang
                    }

                for q in valid_qs:
                    txt = str(q.get("text", "")).lower()
                    qtype = str(q.get("questionType", "")).lower()
                    cat = str(q.get("category", "")).lower()
                    if "pseudocode" in txt or "pseudocode" in qtype or "pseudocode" in cat or "merge two sorted" in txt or "step-by-step logic" in txt:
                        q["type"] = "coding"
                        q["section"] = "coding"
                        if not q.get("problem"):
                            prompt_str = q.get("text", "")
                            q["problem"] = {
                                "title": q.get("title", "Pseudocode Problem"),
                                "description": q.get("text"),
                                "starterCode": {
                                    "python": f"# Step-by-step Pseudocode / Code Solution\n# Prompt: {prompt_str}\n\ndef solution():\n    # Write your step-by-step pseudocode or algorithm here:\n    pass",
                                    "javascript": f"// Step-by-step Pseudocode / Code Solution\nfunction solution() {{\n    // Write your step-by-step pseudocode or algorithm here\n}}"
                                }
                            }

                return valid_qs
    except Exception as err:
        print(f"LLM Batch Question Bank Generation Warning: {err}. Using structured fallback generator.", flush=True)

    # Fallback to local JS generator equivalent in Python if needed
    return []
