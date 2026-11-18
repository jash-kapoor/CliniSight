import streamlit as st
try:
    from utils.compat import get_stretch_kwarg
    from utils.emoji import strip_emoji
except ImportError:
    from client.utils.compat import get_stretch_kwarg
    from client.utils.emoji import strip_emoji


def render_history_download():
    """Renders export options for conversational history and clinical report summaries."""
    chat_messages = st.session_state.get("chat_messages", [])
    
    # Filter out initial greeting if user has chatted
    user_turns = [m for m in chat_messages if m.get("role") == "user"]
    
    if user_turns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Export Chat History")
        
        chat_text = "AI MEDICAL ASSISTANT - CLINICAL CHAT TRANSCRIPT\n"
        chat_text += f"Report ID: {st.session_state.get('report_id', 'N/A')}\n"
        chat_text += "=" * 60 + "\n\n"
        
        for m in chat_messages:
            role = "PATIENT" if m.get("role") == "user" else "CLINICAL ASSISTANT"
            cleaned_content = strip_emoji(m.get('content', ''))
            chat_text += f"[{role}]:\n{cleaned_content}\n"
            if m.get("sources"):
                chat_text += "\nEvidence Sources:\n"
                for s in m["sources"]:
                    s_test = strip_emoji(s.get('test', ''))
                    s_val = strip_emoji(s.get('value', ''))
                    s_range = strip_emoji(s.get('range', ''))
                    chat_text += f"  - {s_test}: {s_val} (Ref: {s_range}, Page {s.get('page')})\n"
            chat_text += "\n" + "-" * 40 + "\n\n"

        st.sidebar.download_button(
            "Download Chat Transcript",
            chat_text,
            file_name="clinical_chat_transcript.txt",
            mime="text/plain",
            icon=":material/download:",
            **get_stretch_kwarg(st.download_button),
        )