"""
The REST/WS surface. The engine must never know about the UI — the host
app talks to it only through these endpoints (requirements §13):

    POST /v1/sessions              create from config + resume
    POST /v1/sessions/{id}/rounds  generate a plan for a round
    WS   /v1/rounds/{id}/live      audio in, audio + visemes out
    GET  /v1/rounds/{id}/report    scores + feedback

STUB package — build-order pass 2 onward.
"""
