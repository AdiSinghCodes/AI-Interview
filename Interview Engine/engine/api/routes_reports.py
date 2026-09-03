"""GET /v1/rounds/{id}/report — scores + feedback for a finished round.
STUB — build-order pass 10."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1/rounds", tags=["reports"])


@router.get("/{round_id}/report")
async def get_report(round_id: str) -> dict:
    raise HTTPException(501, "reports API lands in build-order pass 10")
