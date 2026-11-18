"""Clinical Chat API: Doctor-style grounded Q&A with memory, safety escalation, and streaming."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from logger import logger
from modules.chat_memory import memory_manager
from modules.query_handlers import handle_chat_query, handle_chat_query_stream

router = APIRouter()


class ChatSourceItem(BaseModel):
    test: str
    value: str
    unit: Optional[str] = ""
    range: str = ""
    reference_range: Optional[str] = ""
    page: int
    snippet: str = ""


class ChatPriorityFlag(BaseModel):
    test: str
    flag: str
    urgency: str


class ChatQueryResponse(BaseModel):
    answer: str
    response: str
    sources: list[ChatSourceItem]
    priority_flags: list[ChatPriorityFlag]
    emergency_warning: Optional[str] = None
    disclaimer: str
    session_id: str


@router.post(
    "/ask/",
    summary="Ask medical question grounded in lab report with memory and doctor persona",
)
async def ask_question_endpoint(
    request: Request,
    question: Optional[str] = Form(None),
    report_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Doctor-style clinical explainer endpoint:
    - Grounded only in the patient's retrieved lab report
    - Formats clinical explanations: What it means -> Why it matters -> What you can do -> Urgency
    - Always includes exact evidence citation [Test: Name = Value Unit (ref Ref), page Page]
    - Preserves multi-turn conversation memory (~6 turns)
    - Detects safety emergencies and critical findings
    - Returns answer, sources, priority_flags, and disclaimer
    """
    try:
        # Support both JSON and Form post data
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                question = body.get("question") or question
                report_id = body.get("report_id") or report_id
                session_id = body.get("session_id") or session_id
            except Exception:
                pass

        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid 'question' parameter is required.",
            )

        logger.info(
            "Chat question received: '%s' (report_id=%s, session_id=%s)",
            question[:60],
            report_id or "global",
            session_id or "default",
        )

        result = handle_chat_query(
            question=question.strip(),
            report_id=report_id,
            session_id=session_id,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing chat question")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to generate doctor explanation: {str(exc)}"},
        )


@router.post(
    "/ask/stream",
    summary="Stream doctor-style grounded chat response via Server-Sent Events",
)
async def ask_question_stream_endpoint(
    request: Request,
    question: Optional[str] = Form(None),
    report_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Streaming chat endpoint emitting Server-Sent Events (SSE):
    - type 'emergency': sent if critical safety escalation is triggered
    - type 'chunk': incremental response tokens
    - type 'done': final payload with full answer, sources, priority flags, and disclaimer
    """
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                question = body.get("question") or question
                report_id = body.get("report_id") or report_id
                session_id = body.get("session_id") or session_id
            except Exception:
                pass

        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid 'question' parameter is required.",
            )

        logger.info(
            "Streaming chat question: '%s' (report_id=%s)",
            question[:60],
            report_id or "global",
        )

        return StreamingResponse(
            handle_chat_query_stream(
                question=question.strip(),
                report_id=report_id,
                session_id=session_id,
            ),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error initiating chat stream")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to initiate chat stream: {str(exc)}"},
        )


@router.delete(
    "/chat/history/{session_id}",
    summary="Clear conversation memory for a session",
)
async def clear_chat_history(session_id: str):
    """Clear conversational memory turns for the given session ID."""
    memory_manager.clear_history(session_id)
    return {"message": f"Conversation history for session '{session_id}' cleared successfully."}
