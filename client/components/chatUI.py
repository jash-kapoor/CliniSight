"""Grounded clinical chat UI component for patient Q&A.

Implements doctor-style grounded conversational explainer with:
- Centered layout (max width ~800px; full width on mobile)
- Clear conversation control and patient/date metadata chip
- 4 outlined suggestion buttons during empty state
- Full message history rendered in one loop
- Streaming assistant reply with st.write_stream (/ask/stream fallback to spinner)
- Collapsed "Sources (n)" expander with test details
- Calm emergency advisory banner (no emojis)
- Smooth scrolling and state persistence
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

try:
    from config import API_URL
except (ImportError, AttributeError):
    try:
        from client.config import API_URL
    except (ImportError, ModuleNotFoundError):
        API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

try:
    from utils.api import ask_question_api, clear_chat_history_api
    from utils.compat import get_stretch_kwarg
    from utils.emoji import strip_emoji
except ImportError:
    from client.utils.api import ask_question_api, clear_chat_history_api
    from client.utils.compat import get_stretch_kwarg
    from client.utils.emoji import strip_emoji


def get_report_chip_text() -> str:
    """Format the report metadata chip showing patient name and report date."""
    report_data = st.session_state.get("report_data") or {}
    upload_res = st.session_state.get("upload_response") or {}
    patient = report_data.get("patient") or upload_res.get("patient") or {}

    p_name = strip_emoji(patient.get("name") or "Patient")
    p_date = strip_emoji(patient.get("report_date") or patient.get("date") or "Recent")
    return f"Report: {p_name} | {p_date}"


def scroll_to_bottom() -> None:
    """Scroll the chat view to the newest message."""
    st.components.v1.html(
        """
        <script>
            setTimeout(() => {
                const msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                if (msgs && msgs.length > 0) {
                    msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }, 100);
        </script>
        """,
        height=0,
    )


def render_sources_expander(sources: List[Dict[str, Any]]) -> None:
    """Render clean evidence chips and an expandable drawer showing exact source details."""
    if not sources:
        return

    # Render clean chip badges for each source test
    chips_html_list = []
    for s in sources:
        t_name = strip_emoji(s.get("test", "Test"))
        val = strip_emoji(s.get("value", "-"))
        unit = strip_emoji(s.get("unit") or "")
        ref = strip_emoji(s.get("reference_range") or s.get("range") or "-")
        page = s.get("page", 1)
        val_disp = f"{val} {unit}".strip() if unit and unit not in str(val) else str(val)
        chip = (
            f'<span class="evidence-chip">'
            f'<strong>{t_name}</strong>: {val_disp} '
            f'<span style="opacity: 0.75;">(Ref: {ref}, Page {page})</span>'
            f'</span>'
        )
        chips_html_list.append(chip)

    st.markdown(
        f'<div class="evidence-chips-container">{"".join(chips_html_list)}</div>',
        unsafe_allow_html=True,
    )

    expander_title = f"Evidence details ({len(sources)} test{'s' if len(sources) != 1 else ''})"
    with st.expander(expander_title, expanded=False):
        for src in sources:
            test = strip_emoji(src.get("test", "Test"))
            val = strip_emoji(src.get("value", "-"))
            unit = strip_emoji(src.get("unit") or "")
            ref = strip_emoji(src.get("reference_range") or src.get("range") or "-")
            page = src.get("page", 1)
            val_display = f"{val} {unit}".strip() if unit and unit not in str(val) else str(val)
            snippet = strip_emoji(src.get("snippet") or src.get("text") or "")
            st.markdown(
                f"- **{test}**: `{val_display}` | Reference Range: `{ref}` | Page {page}"
            )
            if snippet:
                st.caption(f"Context: {snippet}")


def render_chat_header(session_id: str) -> None:
    """Render the slim header with title, clear button, and metadata chip."""
    col_title, col_btn = st.columns([3, 1], vertical_alignment="center")
    with col_title:
        st.markdown(
            '<h3 style="margin: 0; font-weight: 700; font-size: 1.35rem;">Ask about your results</h3>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button(
            "Clear conversation",
            key="btn_clear_conversation",
            type="secondary",
            icon=":material/refresh:",
            help="Reset chat history and start a fresh consultation",
            **get_stretch_kwarg(st.button),
        ):
            clear_chat_history_api(session_id)
            st.session_state["chat_session_id"] = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state["chat_messages"] = []
            st.session_state["pending_chat_prompt"] = None
            st.rerun()

    chip_label = get_report_chip_text()
    st.markdown(
        f'<div class="report-chip"><span style="opacity: 0.7;">Report:</span> <strong>{chip_label.replace("Report: ", "")}</strong></div>',
        unsafe_allow_html=True,
    )


def render_empty_state_suggestions() -> Optional[str]:
    """Render the 4 outlined suggestion buttons during empty state."""
    st.markdown(
        """
        <div style="margin: 16px 0 10px 0; font-size: 0.9rem; font-weight: 600; color: var(--text-color, #475569);">
            Suggested questions to begin:
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    selected_prompt: Optional[str] = None

    with col1:
        if st.button(
            "Explain my abnormal results",
            key="sugg_abnormal",
            type="secondary",
            **get_stretch_kwarg(st.button),
        ):
            selected_prompt = "Can you explain all my abnormal test results in simple terms?"
        if st.button(
            "What can I change in my diet?",
            key="sugg_diet",
            type="secondary",
            **get_stretch_kwarg(st.button),
        ):
            selected_prompt = "What diet and lifestyle modifications can I make based on these results?"

    with col2:
        if st.button(
            "Are any results urgent?",
            key="sugg_urgent",
            type="secondary",
            **get_stretch_kwarg(st.button),
        ):
            selected_prompt = "Are any of my laboratory findings urgent or critical?"
        if st.button(
            "What should I ask my doctor?",
            key="sugg_doctor",
            type="secondary",
            **get_stretch_kwarg(st.button),
        ):
            selected_prompt = "What important questions should I ask my doctor about these results?"

    return selected_prompt


def render_chat_followup_chips() -> Optional[str]:
    """Render 3-4 clickable suggestion pills right above the input."""
    report_data = st.session_state.get("report_data") or {}
    tests = report_data.get("tests") or []

    # Check for specific test findings to personalize chips
    has_creatinine = any("creatinine" in (t.get("test_name") or "").lower() for t in tests)
    has_mcv = any("mcv" in (t.get("test_name") or "").lower() for t in tests)

    chips = []
    if has_creatinine:
        chips.append(("Explain my Creatinine", "Can you explain my Serum Creatinine level and what it indicates?"))
    else:
        chips.append(("Explain key results", "Can you explain the most significant results from my report?"))

    if has_mcv:
        chips.append(("What does low MCV mean?", "What does a low MCV level indicate on my blood report?"))
    else:
        chips.append(("Are any results urgent?", "Are any of my laboratory findings urgent or critical?"))

    chips.append(("Show all abnormal tests", "Please list all abnormal tests with their values and normal ranges."))
    chips.append(("Questions for my doctor", "What are the most crucial questions I should ask my doctor about these results?"))

    st.markdown('<div class="chat-followup-container">', unsafe_allow_html=True)
    cols = st.columns(len(chips))
    selected_prompt = None

    for idx, (label, prompt) in enumerate(chips):
        with cols[idx]:
            if st.button(
                label,
                key=f"followup_chip_{idx}",
                type="secondary",
                **get_stretch_kwarg(st.button),
            ):
                selected_prompt = prompt

    st.markdown('</div>', unsafe_allow_html=True)
    return selected_prompt


def stream_assistant_reply(
    question: str,
    report_id: str,
    session_id: str,
) -> tuple[str, list[dict], str | None]:
    """Stream assistant response from /ask/stream or fallback to synchronous /ask/."""
    full_answer = ""
    sources: list[dict] = []
    emergency_warning: str | None = None
    stream_success = False

    # Try streaming from /ask/stream endpoint
    try:
        payload = {"question": question, "report_id": report_id, "session_id": session_id}
        resp = requests.post(
            f"{API_URL}/ask/stream",
            json=payload,
            stream=True,
            timeout=90,
        )
        if resp.status_code == 200:
            def token_generator():
                nonlocal full_answer, sources, emergency_warning
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data:"):
                        event_str = decoded[5:].strip()
                        if not event_str:
                            continue
                        try:
                            event = json.loads(event_str)
                            etype = event.get("type")
                            if etype == "chunk":
                                chunk = strip_emoji(event.get("text", ""))
                                full_answer += chunk
                                yield chunk
                            elif etype == "emergency":
                                emergency_warning = strip_emoji(event.get("warning"))
                            elif etype == "done":
                                sources = event.get("sources", [])
                                if not emergency_warning and event.get("emergency_warning"):
                                    emergency_warning = strip_emoji(event.get("emergency_warning"))
                                if not full_answer and event.get("answer"):
                                    ans = strip_emoji(event.get("answer"))
                                    full_answer = ans
                                    yield ans
                        except json.JSONDecodeError:
                            continue

            # Render streamed reply
            full_answer = st.write_stream(token_generator()) or full_answer
            stream_success = bool(full_answer.strip())
    except Exception:
        stream_success = False

    # Fallback to spinner with standard /ask/ endpoint
    if not stream_success:
        with st.spinner("Reviewing your report"):
            ok, data = ask_question_api(
                question=question,
                report_id=report_id,
                session_id=session_id,
            )
            if ok and isinstance(data, dict):
                raw_ans = data.get("response") or data.get("answer") or "No response generated."
                full_answer = strip_emoji(raw_ans)
                sources = data.get("sources", [])
                if data.get("emergency_warning"):
                    emergency_warning = strip_emoji(data.get("emergency_warning"))
                st.markdown(full_answer)
            else:
                err_msg = (
                    "We were unable to analyze this question right now. "
                    "Please check your network connection or try rephrasing your question."
                )
                st.error(err_msg, icon=":material/error:")
                full_answer = err_msg

    return full_answer, sources, emergency_warning


def handle_incoming_chat_turn(user_prompt: str) -> None:
    """Handle appending and generating a new conversation turn."""
    report_id = st.session_state.get("report_id")
    session_id = st.session_state.get("chat_session_id")
    clean_prompt = strip_emoji(user_prompt.strip())

    if not clean_prompt or not report_id or not session_id:
        return

    # 1. Append user message to state and display
    st.session_state["chat_messages"].append({"role": "user", "content": clean_prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(clean_prompt)

    # 2. Lock input and render assistant reply
    st.session_state["chat_generating"] = True
    with st.chat_message("assistant", avatar=":material/medical_services:"):
        full_answer, sources, emergency_warning = stream_assistant_reply(
            question=clean_prompt,
            report_id=report_id,
            session_id=session_id,
        )

        if emergency_warning:
            st.markdown(
                f'<div class="chat-emergency-banner"><strong>URGENT MEDICAL ADVISORY:</strong> {emergency_warning}</div>',
                unsafe_allow_html=True,
            )

        render_sources_expander(sources)

    # 3. Save assistant message to session state
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "emergency_warning": emergency_warning,
        }
    )
    st.session_state["chat_generating"] = False
    scroll_to_bottom()


def render_chat() -> Optional[str]:
    """
    Renders the grounded chat screen within a centered column.
    Returns any preset question clicked from suggestion buttons.
    """
    report_id = st.session_state.get("report_id")
    if not report_id:
        st.info("Please upload and analyze a lab report first to begin clinical Q&A.", icon=":material/info:")
        return None

    # Persistent session memory
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = f"session_{uuid.uuid4().hex[:8]}"

    session_id = st.session_state["chat_session_id"]

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    pending_selection: Optional[str] = None

    # Center conversation in column with max width ~800px; full width on mobile
    st.markdown('<div class="chat-layout-wrapper">', unsafe_allow_html=True)

    # Slim header and metadata chip
    render_chat_header(session_id)

    has_user_messages = any(m.get("role") == "user" for m in st.session_state["chat_messages"])

    # Suggestion buttons: empty state only (or compact popover when messages exist)
    if not has_user_messages:
        selected = render_empty_state_suggestions()
        if selected:
            pending_selection = selected
    else:
        # Subtle popover when conversation is active
        with st.popover("Suggested questions", icon=":material/lightbulb:"):
            if st.button("Explain abnormal results", key="pop_abnormal", **get_stretch_kwarg(st.button)):
                pending_selection = "Can you explain all my abnormal test results in simple terms?"
            if st.button("Are any results urgent?", key="pop_urgent", **get_stretch_kwarg(st.button)):
                pending_selection = "Are any of my laboratory findings urgent or critical?"
            if st.button("Diet and lifestyle advice", key="pop_diet", **get_stretch_kwarg(st.button)):
                pending_selection = "What diet and lifestyle modifications can I make based on these results?"
            if st.button("Questions for my doctor", key="pop_doctor", **get_stretch_kwarg(st.button)):
                pending_selection = "What important questions should I ask my doctor about these results?"

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Render full chat history in one loop
    with st.container(height=560, border=False):
        for msg in st.session_state["chat_messages"]:
            role = msg.get("role", "assistant")
            avatar = ":material/person:" if role == "user" else ":material/medical_services:"
            with st.chat_message(role, avatar=avatar):
                if msg.get("emergency_warning"):
                    st.markdown(
                        f'<div class="chat-emergency-banner"><strong>URGENT MEDICAL ADVISORY:</strong> {strip_emoji(msg["emergency_warning"])}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(strip_emoji(msg.get("content", "")))
                if msg.get("sources"):
                    render_sources_expander(msg.get("sources"))

    st.markdown('</div>', unsafe_allow_html=True)  # Close .chat-layout-wrapper

    return pending_selection
