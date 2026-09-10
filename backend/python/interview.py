import os
import tempfile

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)


# ============================================================
# LAZY-LOADED AI AGENTS
# ============================================================
#
# IMPORTANT:
# Do NOT import the normal/coding agents at startup.
#
# The normal RAG agent can load SentenceTransformer and other
# dependencies, which can take several seconds.
#
# Flask should start first on port 5100.
# The agents are loaded only when they are actually needed.
# ============================================================

ask_verbal = None
eval_verbal = None

ask_code = None
eval_code = None


def load_verbal_agent():
    global ask_verbal
    global eval_verbal

    if ask_verbal is None or eval_verbal is None:
        print("Loading normal-question RAG agent...", flush=True)

        from normal_questions_agent import (
            ask,
            evaluate,
        )

        ask_verbal = ask
        eval_verbal = evaluate

        print(
            "Normal-question RAG agent loaded.",
            flush=True
        )

    return ask_verbal, eval_verbal


def load_coding_agent():
    global ask_code
    global eval_code

    if ask_code is None or eval_code is None:
        print("Loading coding/SQL agent...", flush=True)

        from coding_sql_agent import (
            ask,
            evaluate,
        )

        ask_code = ask
        eval_code = evaluate

        print(
            "Coding/SQL agent loaded.",
            flush=True
        )

    return ask_code, eval_code


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_file(file_storage):
    key = os.getenv(
        "GROQ_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured"
        )

    suffix = (
        os.path.splitext(
            file_storage.filename or "answer.webm"
        )[1]
        or ".webm"
    )

    fd, path = tempfile.mkstemp(
        suffix=suffix
    )

    os.close(fd)

    try:
        file_storage.save(path)

        with open(
            path,
            "rb"
        ) as f:

            result = Groq(
                api_key=key
            ).audio.transcriptions.create(
                file=(
                    os.path.basename(path),
                    f
                ),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="json",
            )

        return result.text

    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "interview-agent",
    })


# ============================================================
# TRANSCRIBE
# ============================================================

@app.post("/transcribe")
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({
                "message": "audio is required"
            }), 400

        text = transcribe_file(
            request.files["audio"]
        )

        return jsonify({
            "text": text
        })

    except Exception as e:
        print(
            "Transcription error:",
            e,
            flush=True
        )

        return jsonify({
            "message": str(e)
        }), 500


# ============================================================
# GENERATE QUESTION
# ============================================================

@app.post("/question")
def question():
    try:
        p = request.get_json(
            force=True
        )

        setup = p.get(
            "setup",
            {}
        )

        history = p.get(
            "history",
            []
        )

        index = int(
            p.get(
                "questionIndex",
                1
            )
        )

        plan = p.get(
            "plan",
            {}
        )

        # ----------------------------------------------------
        # DETERMINE INTERVIEW TYPE
        # ----------------------------------------------------

        coding_interview = (
            plan.get("coding", 0)
            + plan.get("sql", 0)
        ) > 0

        used_coding = sum(
            1
            for h in history
            if h.get("section")
            in (
                "coding",
                "sql"
            )
        )

        used_verbal = sum(
            1
            for h in history
            if h.get("section")
            == "verbal"
        )

        remaining_coding = (
            plan.get("coding", 0)
            + plan.get("sql", 0)
            - used_coding
        )

        remaining_verbal = (
            plan.get("verbal", 0)
            - used_verbal
        )

        # ----------------------------------------------------
        # VERBAL QUESTION
        # ----------------------------------------------------

        if (
            not coding_interview
            or (
                remaining_verbal
                >= remaining_coding
                and remaining_verbal
                > 0
            )
        ):

            ask_verbal_fn, _ = (
                load_verbal_agent()
            )

            result = ask_verbal_fn(
                setup,
                history,
                index,
                p.get(
                    "lastEvaluation"
                )
            )

        # ----------------------------------------------------
        # CODING / SQL QUESTION
        # ----------------------------------------------------

        else:

            ask_code_fn, _ = (
                load_coding_agent()
            )

            sql_left = (
                plan.get("sql", 0)
                - sum(
                    1
                    for h in history
                    if h.get("section")
                    == "sql"
                )
            )

            section = (
                "sql"
                if sql_left > 0
                else "coding"
            )

            result = ask_code_fn(
                setup,
                history,
                index,
                section
            )

        return jsonify(
            result
        )

    except Exception as e:
        print(
            "Question generation error:",
            e,
            flush=True
        )

        return jsonify({
            "message": str(e)
        }), 500


# ============================================================
# EVALUATE ANSWER
# ============================================================

@app.post("/evaluate")
def evaluate():
    try:
        p = request.get_json(
            force=True
        )

        section = p.get(
            "section",
            "verbal"
        )

        # ----------------------------------------------------
        # CODING / SQL EVALUATION
        # ----------------------------------------------------

        if section in (
            "coding",
            "sql"
        ):

            _, eval_code_fn = (
                load_coding_agent()
            )

            result = eval_code_fn(
                p
            )

        # ----------------------------------------------------
        # NORMAL / VERBAL EVALUATION
        # ----------------------------------------------------

        else:

            _, eval_verbal_fn = (
                load_verbal_agent()
            )

            result = eval_verbal_fn(
                p
            )

        return jsonify({
            "evaluation": result,

            "score": result.get(
                "score",
                0
            ),

            "followUp": result.get(
                "followUp",
                False
            ),

            "endInterview": result.get(
                "endInterview",
                False
            ),

            "summary": result.get(
                "summary"
            ),
        })

    except Exception as e:
        print(
            "Answer evaluation error:",
            e,
            flush=True
        )

        return jsonify({
            "message": str(e)
        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "INTERVIEW_AGENT_PORT",
            "5100"
        )
    )

    print(
        "",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    print(
        "       VIVA AI INTERVIEW AGENT",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    print(
        f"Starting Flask on port {port}...",
        flush=True
    )

    print(
        "AI agents will load on first request.",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        threaded=True,
    )