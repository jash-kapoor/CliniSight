import streamlit as st
try:
    from utils.compat import get_stretch_kwarg
    from utils.emoji import strip_emoji
except ImportError:
    from client.utils.compat import get_stretch_kwarg
    from client.utils.emoji import strip_emoji


def render_doctor_questions():
    """Renders tailored questions to ask the physician with contextual reasoning, citations, and theme contrast."""
    insights = st.session_state.get("insights")

    if not insights:
        st.info("Please upload and analyze a lab report first to view recommended questions for your doctor.")
        return

    questions = insights.get("questions_to_ask_doctor", [])

    st.markdown("### Targeted Questions to Ask Your Doctor")
    st.caption(
        "Preparing specific, data-backed questions helps you make the most of your consultation. "
        "Each question below directly correlates with out-of-range parameters found in your report."
    )

    if not questions:
        st.write("All tests are within normal parameters. You may ask your doctor routine maintenance questions.")
        return

    # Render Question Cards using native containers for full theme contrast and no indentation issues
    for idx, q in enumerate(questions, 1):
        q_text = strip_emoji(q.get("question", ""))
        reason = strip_emoji(q.get("context_reason", ""))
        citation = strip_emoji(q.get("evidence_citation", ""))

        with st.container(border=True):
            st.markdown(f"#### {idx}. {q_text}")
            if reason:
                st.markdown(f"**Why Ask This:** {reason}")
            if citation:
                st.caption(f"Grounding Evidence: `{citation}`")

    # Downloadable Consultation Checklist
    checklist_text = "CLINICAL CONSULTATION QUESTIONS FOR DOCTOR\n"
    checklist_text += f"Report ID: {st.session_state.get('report_id', 'N/A')}\n"
    checklist_text += f"Patient: {strip_emoji(insights.get('patient_name', 'Patient'))}\n"
    checklist_text += "=" * 60 + "\n\n"

    for idx, q in enumerate(questions, 1):
        q_clean = strip_emoji(q.get('question', ''))
        reason_clean = strip_emoji(q.get('context_reason', ''))
        ev_clean = strip_emoji(q.get('evidence_citation', ''))
        checklist_text += f"{idx}. {q_clean}\n"
        checklist_text += f"   Reason: {reason_clean}\n"
        checklist_text += f"   Evidence: {ev_clean}\n\n"

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="Download Doctor Consultation Checklist (TXT)",
        data=checklist_text,
        file_name="doctor_consultation_questions.txt",
        mime="text/plain",
        icon=":material/download:",
        **get_stretch_kwarg(st.download_button),
    )
