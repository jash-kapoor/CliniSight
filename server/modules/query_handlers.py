"""Clinical Chat Query Handlers: retrieval, safety escalation, memory, sources extraction, and streaming."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain.docstore.document import Document

from logger import logger
from modules.chat_memory import memory_manager
from modules.lab_indexer import is_general_overview_query, retrieve_report_context
from modules.llm import (
    MANDATORY_DISCLAIMER,
    clean_chat_answer,
    generate_chat_response,
    stream_chat_response,
)
from modules.vectorstore import similarity_search

# Critical symptom keywords indicating acute medical emergencies
EMERGENCY_SYMPTOM_KEYWORDS = [
    "chest pain",
    "chest pressure",
    "chest tightness",
    "heart attack",
    "shortness of breath",
    "breathlessness",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "suffocating",
    "fainting",
    "fainted",
    "loss of consciousness",
    "passed out",
    "syncope",
    "collapsed",
    "severe dizziness",
    "coughing blood",
    "vomiting blood",
    "sudden numbness",
    "sudden paralysis",
    "slurred speech",
    "stroke",
    "unbearable headache",
    "worst headache",
    "seizure",
    "convulsion",
]


def detect_safety_escalation(
    question: str,
    docs: List[Document],
) -> Tuple[bool, Optional[str]]:
    """
    Check if query mentions severe emergency symptoms or involves critical lab flags.
    Returns (is_emergency, advisory_text).
    """
    q_lower = question.lower()
    matched_symptoms = [s for s in EMERGENCY_SYMPTOM_KEYWORDS if s in q_lower]
    has_critical_flag = any(
        (doc.metadata or {}).get("flag", "").lower() == "critical"
        for doc in docs
    )

    if matched_symptoms or has_critical_flag:
        triggers = []
        if matched_symptoms:
            triggers.append(f"severe symptom(s): {', '.join(matched_symptoms)}")
        if has_critical_flag:
            triggers.append("one or more critical lab values requiring immediate medical evaluation")

        warning = (
            f"🚨 URGENT MEDICAL ADVISORY: Your question or report indicates {'; '.join(triggers)}. "
            "Please seek immediate emergency medical care (call emergency services like 102/108/911 or go to the nearest emergency room). "
            "Do not delay urgent clinical care."
        )
        return True, warning

    return False, None


def extract_sources_and_flags(
    docs: List[Document],
    question: str = "",
    answer: str = "",
    is_emergency_symptom: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract structured sources and priority flags from retrieved clinical documents.
    Format:
    sources: [{test, value, unit, range, reference_range, page, snippet}]
    priority_flags: [{test, flag, urgency}]
    """
    if is_emergency_symptom:
        return [], []

    sources: List[Dict[str, Any]] = []
    priority_flags: List[Dict[str, Any]] = []
    seen_tests = set()
    seen_flags = set()

    is_general = is_general_overview_query(question) if question else False
    q_lower = question.lower() if question else ""
    a_lower = answer.lower() if answer else ""

    for doc in docs:
        meta = doc.metadata or {}
        test_name = meta.get("test_name") or ""
        flag = (meta.get("flag") or "normal").lower()
        page = int(meta.get("page", 1))

        if not test_name:
            continue

        # Relevance filter
        t_lower = test_name.lower()
        is_relevant = (
            not question  # Legacy / default testing
            or is_general
            or t_lower in q_lower
            or any(part in q_lower for part in t_lower.split() if len(part) > 3)
            or t_lower in a_lower
            or (flag in {"low", "high", "critical"} and any(w in q_lower for w in ["abnormal", "critical", "urgent", "flag", "high", "low"]))
        )

        if not is_relevant:
            continue

        val = ""
        range_val = ""
        snippet = ""
        unit = ""
        for line in doc.page_content.splitlines():
            line_str = line.strip()
            if line_str.startswith("- Observed Value:"):
                val = line_str.replace("- Observed Value:", "").strip()
            elif line_str.startswith("- Biological Reference Range:"):
                range_val = line_str.replace("- Biological Reference Range:", "").strip()
            elif line_str.startswith("- Report Source Snippet:"):
                snippet = line_str.replace("- Report Source Snippet:", "").strip()

        val_clean = val
        if val and " " in val:
            parts = val.split(" ", 1)
            if any(c.isalpha() or c in "%/" for c in parts[1]):
                val_clean = parts[0]
                unit = parts[1]

        if test_name not in seen_tests:
            seen_tests.add(test_name)
            sources.append({
                "test": test_name,
                "value": val or "See report",
                "unit": unit or "",
                "range": range_val or "Not specified",
                "reference_range": range_val or "Not specified",
                "page": page,
                "snippet": snippet or doc.page_content[:150],
            })

        if flag in {"low", "high", "critical"} and test_name not in seen_flags:
            seen_flags.add(test_name)
            urgency = (
                "🔴 High Concern"
                if flag == "critical"
                else ("🟠 Medium Concern" if flag == "high" else "🟡 Low Concern")
            )
            priority_flags.append({
                "test": test_name,
                "flag": flag,
                "urgency": urgency,
            })

    return sources, priority_flags


def handle_chat_query(
    question: str,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute grounded clinical RAG chat with conversation memory and safety checks."""
    effective_session = session_id or report_id or "default_session"

    # 1. Retrieve report-isolated or general documents
    if report_id:
        docs = retrieve_report_context(report_id, question)
    else:
        docs = similarity_search(question)

    # 2. Check safety escalation & acute symptoms
    is_emergency, emergency_warning = detect_safety_escalation(question, docs)
    is_emergency_symptom = any(s in question.lower() for s in EMERGENCY_SYMPTOM_KEYWORDS)

    # 3. Retrieve conversation history & session turn count
    history_str = memory_manager.format_history_for_prompt(effective_session, limit=6)
    existing_history = memory_manager.get_history(effective_session)
    is_ongoing_turn = len(existing_history) > 0

    # 4. Format context text (omit lab details on emergency symptoms to avoid walkthrough)
    context_str = "" if is_emergency_symptom else "\n\n".join(d.page_content for d in docs)

    # 5. Generate grounded answer
    answer = generate_chat_response(
        context_text=context_str,
        history_text=history_str,
        question=question,
        emergency_advisory=emergency_warning,
        is_emergency_symptom=is_emergency_symptom,
        is_ongoing_turn=is_ongoing_turn,
    )

    # 6. Extract sources and priority flags (limited strictly to relevant tests)
    sources, priority_flags = extract_sources_and_flags(
        docs=docs,
        question=question,
        answer=answer,
        is_emergency_symptom=is_emergency_symptom,
    )

    # 7. Record turn in conversation memory
    memory_manager.add_turn(effective_session, question, answer)

    return {
        "answer": answer,
        "response": answer,  # Backwards compatibility
        "sources": sources,
        "priority_flags": priority_flags,
        "emergency_warning": emergency_warning,
        "disclaimer": MANDATORY_DISCLAIMER,
        "session_id": effective_session,
    }


def handle_chat_query_stream(
    question: str,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream grounded chat response via Server-Sent Events (SSE)."""
    effective_session = session_id or report_id or "default_session"

    if report_id:
        docs = retrieve_report_context(report_id, question)
    else:
        docs = similarity_search(question)

    is_emergency, emergency_warning = detect_safety_escalation(question, docs)
    is_emergency_symptom = any(s in question.lower() for s in EMERGENCY_SYMPTOM_KEYWORDS)

    history_str = memory_manager.format_history_for_prompt(effective_session, limit=6)
    existing_history = memory_manager.get_history(effective_session)
    is_ongoing_turn = len(existing_history) > 0

    context_str = "" if is_emergency_symptom else "\n\n".join(d.page_content for d in docs)

    full_response_parts: List[str] = []

    # If emergency, emit notification first
    if emergency_warning:
        init_event = json.dumps({"type": "emergency", "warning": emergency_warning})
        yield f"data: {init_event}\n\n"

    for token in stream_chat_response(
        context_text=context_str,
        history_text=history_str,
        question=question,
        emergency_advisory=emergency_warning,
        is_emergency_symptom=is_emergency_symptom,
        is_ongoing_turn=is_ongoing_turn,
    ):
        full_response_parts.append(token)
        chunk_event = json.dumps({"type": "chunk", "text": token})
        yield f"data: {chunk_event}\n\n"

    raw_answer = "".join(full_response_parts)
    full_answer = clean_chat_answer(
        raw_answer,
        is_ongoing_turn=is_ongoing_turn,
        is_emergency_symptom=is_emergency_symptom,
        emergency_advisory=emergency_warning,
    )

    sources, priority_flags = extract_sources_and_flags(
        docs=docs,
        question=question,
        answer=full_answer,
        is_emergency_symptom=is_emergency_symptom,
    )

    # Update conversation memory
    memory_manager.add_turn(effective_session, question, full_answer)

    # Final metadata event with sources and flags
    done_event = json.dumps({
        "type": "done",
        "answer": full_answer,
        "sources": sources,
        "priority_flags": priority_flags,
        "emergency_warning": emergency_warning,
        "disclaimer": MANDATORY_DISCLAIMER,
        "session_id": effective_session,
    })
    yield f"data: {done_event}\n\n"


def query_chain(chain: Any, user_input: str) -> Dict[str, Any]:
    """Legacy query chain adapter for backwards-compatibility."""
    try:
        logger.debug("Running legacy query_chain")
        if hasattr(chain, "invoke"):
            result = chain.invoke({"query": user_input})
        else:
            result = chain({"query": user_input})
        sources = []
        for doc in result.get("source_documents") or []:
            metadata = doc.metadata or {}
            source = metadata.get("test_name") or metadata.get("source") or ""
            page = metadata.get("page")
            if page is not None and source:
                sources.append(f"{source}#page={page}")
            elif source:
                sources.append(source)
        ans = result.get("result") or result.get("answer") or ""
        return {
            "response": ans,
            "answer": ans,
            "sources": sources,
            "priority_flags": [],
            "disclaimer": MANDATORY_DISCLAIMER,
        }
    except Exception:
        logger.exception("Error in query_chain")
        raise
