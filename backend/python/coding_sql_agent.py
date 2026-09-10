import json
import os
from groq import Groq

MODELS = [
    os.getenv('INTERVIEW_LLM_MODEL', 'openai/gpt-oss-120b'),
    'qwen/qwen3.8-27b',
    'qwen/qwen3.6-27b',
]


def _client():
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        raise RuntimeError('GROQ_API_KEY is not configured')
    return Groq(api_key=key)


def _call(messages, temperature, max_tokens):
    last = None
    for model in MODELS:
        try:
            response = _client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={'type': 'json_object'},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            last = exc
    raise last or RuntimeError('No coding interview model is available')


def ask(setup, history, index, section):
    language = setup.get('codingLanguage', 'python')
    prompt = f"""You are VIVA's coding/SQL interviewer.

AUTHORITATIVE SETUP:
{json.dumps(setup, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-8:], ensure_ascii=False)}

QUESTION NUMBER: {index}
SECTION: {section}
WORKSPACE LANGUAGE: {language}

Rules:
- Generate exactly one {section} interview task.
- Respect domain, subdomain, role, objective, difficulty and custom topics.
- For SQL, generate a realistic SQL/database problem with schema/sample rows.
- For coding, generate a role-appropriate programming/algorithm problem in the selected language.
- Do not use the candidate's resume as the topic unless the setup says Resume Based or useResume=true and it is relevant.
- Do not ask a generic software-engineering question if the selected role/domain is something else.
- Do not solve the problem.

Return JSON with section, questionType, category, text, problem, constraints, examples, language."""
    return _call([{'role': 'system', 'content': prompt}], 0.4, 1300)


def evaluate(payload):
    problem = payload.get('codingSubmission') or {}
    prompt = f"""Evaluate a coding/SQL interview answer rigorously.

SETUP:
{json.dumps(payload.get('setup', {}), ensure_ascii=False)}
QUESTION:
{payload.get('currentQuestion', '')}
CANDIDATE ANSWER:
{payload.get('answer', '')}
SUBMISSION:
{json.dumps(problem, ensure_ascii=False)}
HISTORY:
{json.dumps(payload.get('history', [])[-6:], ensure_ascii=False)}

Return JSON:
score 0-10, correctness, approach, complexity, codeQuality, edgeCases, sqlQuality,
strengths, missingPoints, followUp, followUpReason, followUpQuestion, endInterview, summary.

Set followUp=true when the candidate's reasoning/code/query is incomplete or has a specific flaw worth probing. If true, create one precise cross-question in followUpQuestion. Never invent runtime/test results that were not supplied."""
    return _call([{'role': 'system', 'content': prompt}], 0.12, 1300)
