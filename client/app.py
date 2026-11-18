import sys
from pathlib import Path

_client_dir = Path(__file__).resolve().parent
_project_root = _client_dir.parent
if str(_client_dir) not in sys.path:
    sys.path.insert(0, str(_client_dir))
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

import streamlit as st
from components.chatUI import (
    handle_incoming_chat_turn,
    render_chat,
    render_chat_followup_chips,
)
from components.dashboard import render_dashboard
from components.diet_precautions import render_diet_and_precautions
from components.disclaimer import render_disclaimer_banner
from components.doctor_questions import render_doctor_questions
from components.history_download import render_history_download
from components.upload import render_uploader_sidebar
from components.verification import render_verification_screen
from theme import get_theme_css


# Page Configuration
st.set_page_config(
    page_title="AI Clinical Lab Assistant",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Central Theme Styling (CSS)
st.markdown(get_theme_css(), unsafe_allow_html=True)


def main():
    # Top Header with Theme-Adaptive Gradient Title
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h1 class="main-title">
                AI Clinical Lab Explainer
            </h1>
            <p class="main-subtitle">
                Doctor-style intelligent lab report analysis, physiological clustering, Indian diet recommendations & grounded chat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Mandatory Disclaimer Banner
    render_disclaimer_banner()

    # 2. Sidebar Navigation & Upload Management
    render_uploader_sidebar()
    render_history_download()

    # 3. Main Workspace: Navigation or Hero Welcome
    report_id = st.session_state.get("report_id")

    if not report_id:
        st.markdown(
            """
            <div style="background-color: var(--secondary-background-color, rgba(148, 163, 184, 0.08)); border: 2px dashed rgba(148, 163, 184, 0.35); border-radius: 12px; padding: 40px 28px; text-align: center; margin-top: 20px;">
                <h3 style="color: var(--text-color, #1e293b); font-weight: 700; margin-bottom: 8px;">Upload a Medical Lab Report PDF</h3>
                <p style="color: var(--text-color, #64748b); opacity: 0.85; max-width: 600px; margin: 0 auto 20px auto; font-size: 0.95rem; line-height: 1.6;">
                    Upload a diagnostic blood test, CBC, lipid panel, metabolic profile, or general lab report in the sidebar. 
                    The clinical engine will extract parameters, validate flags, generate priority insights, and prepare your consultation guide.
                </p>
                <div style="display: inline-flex; gap: 16px; flex-wrap: wrap; justify-content: center; font-size: 0.85rem; color: var(--text-color, #475569); opacity: 0.85;">
                    <span>Deterministic Unit Normalization</span>
                    <span>&bull;</span>
                    <span>Isolated Vector Store per Report</span>
                    <span>&bull;</span>
                    <span>Indian Diet & Meal Planning</span>
                    <span>&bull;</span>
                    <span>Complete Privacy Purge</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sample_pdf = _project_root / "Report.pdf"
        if sample_pdf.exists():
            st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
            with col_b2:
                if st.button(
                    "Load Sample Report (Report.pdf)",
                    key="btn_load_sample_main",
                    type="primary",
                    icon=":material/description:",
                    use_container_width=True,
                ):
                    with st.spinner("Extracting parameters & indexing domain chunks..."):
                        with open(sample_pdf, "rb") as f:
                            file_bytes = f.read()
                        try:
                            from utils.api import get_report_api, get_report_summary_api, upload_report_api
                        except ImportError:
                            from client.utils.api import get_report_api, get_report_summary_api, upload_report_api

                        ok, res = upload_report_api(file_bytes, "Report.pdf")
                        if ok:
                            r_id = res["report_id"]
                            st.session_state["report_id"] = r_id
                            st.session_state["uploaded_filename"] = "Report.pdf"
                            st.session_state["uploaded_filenames"] = ["Report.pdf"]
                            st.session_state["upload_response"] = res

                            ok_rep, rep_data = get_report_api(r_id)
                            if ok_rep:
                                st.session_state["report_data"] = rep_data

                            is_verified = rep_data.get("verified_by_user", False) if ok_rep else False
                            st.session_state["verified_by_user"] = is_verified

                            if is_verified:
                                ok_sum, sum_data = get_report_summary_api(r_id)
                                if ok_sum:
                                    st.session_state["insights"] = sum_data
                            st.rerun()
                        else:
                            st.error(f"Analysis Failed: {res}")
    else:
        # Stage E: Human-In-The-Loop Verification Guard
        is_verified = st.session_state.get("verified_by_user", False)
        if not is_verified and st.session_state.get("report_data"):
            is_verified = st.session_state["report_data"].get("verified_by_user", False)

        if not is_verified:
            render_verification_screen()
        else:
            # Navigation control: Overview, Ask questions, Diet and lifestyle, Doctor visit
            nav_sections = ["Overview", "Ask questions", "Diet and lifestyle", "Doctor visit"]

            # Prefer st.segmented_control; fallback to horizontal st.radio if unavailable
            if hasattr(st, "segmented_control"):
                active_section = st.segmented_control(
                    "Clinical Sections",
                    options=nav_sections,
                    default="Overview",
                    label_visibility="collapsed",
                    key="main_clinical_nav",
                )
            else:
                active_section = st.radio(
                    "Clinical Sections",
                    options=nav_sections,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="main_clinical_nav",
                )

            if not active_section:
                active_section = "Overview"

            # Render ONLY the selected section
            if active_section == "Overview":
                render_dashboard()
            elif active_section == "Ask questions":
                pending_query = render_chat()

                # Suggested follow-up chips right above the chat input
                followup_query = render_chat_followup_chips()

                # Caption directly above pinned input
                st.markdown(
                    '<p class="chat-input-caption">AI-generated explanation, not a diagnosis.</p>',
                    unsafe_allow_html=True,
                )

                # TOP LEVEL CALL: st.chat_input called ONLY when "Ask questions" is active
                # and ONLY at the top level of the main script body (not inside any containers)
                chat_is_generating = bool(st.session_state.get("chat_generating", False))
                user_input = st.chat_input(
                    placeholder="Ask Dr. Mira about this report...",
                    disabled=chat_is_generating,
                    key="top_level_chat_input",
                )

                target_query = user_input or pending_query or followup_query
                if target_query:
                    handle_incoming_chat_turn(target_query)
                    st.rerun()

            elif active_section == "Diet and lifestyle":
                render_diet_and_precautions()
            elif active_section == "Doctor visit":
                render_doctor_questions()


if __name__ == "__main__":
    main()