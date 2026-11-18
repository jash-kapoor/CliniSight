import streamlit as st
try:
    from utils.api import delete_report_api, get_report_api, get_report_summary_api, upload_report_api
    from utils.compat import get_stretch_kwarg
except ImportError:
    from client.utils.api import delete_report_api, get_report_api, get_report_summary_api, upload_report_api
    from client.utils.compat import get_stretch_kwarg


def render_uploader_sidebar():
    """
    Renders the single lab report PDF uploader and active report management
    (including delete & privacy purge button) in the Streamlit sidebar.
    """
    st.sidebar.markdown("### Lab Report Upload")

    # File Uploader (PDF or multiple images)
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or page photos (JPG, PNG, HEIC)",
        type=["pdf", "jpg", "jpeg", "png", "heic"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload an official medical laboratory diagnostic report as PDF or multiple page photographs.",
    )

    from pathlib import Path
    sample_pdf = Path(__file__).resolve().parent.parent.parent / "Report.pdf"
    if not st.session_state.get("report_id") and sample_pdf.exists() and not uploaded_files:
        if st.sidebar.button(
            "Load Sample Report (Report.pdf)",
            icon=":material/description:",
            **get_stretch_kwarg(st.button),
        ):
            with st.sidebar:
                with st.spinner("Extracting parameters & indexing domain chunks..."):
                    with open(sample_pdf, "rb") as f:
                        file_bytes = f.read()
                    ok, res = upload_report_api(file_bytes, "Report.pdf")
                    if ok:
                        report_id = res["report_id"]
                        st.session_state["report_id"] = report_id
                        st.session_state["uploaded_filename"] = "Report.pdf"
                        st.session_state["uploaded_filenames"] = ["Report.pdf"]
                        st.session_state["upload_response"] = res

                        ok_rep, rep_data = get_report_api(report_id)
                        if ok_rep:
                            st.session_state["report_data"] = rep_data

                        is_verified = rep_data.get("verified_by_user", False) if ok_rep else False
                        st.session_state["verified_by_user"] = is_verified

                        if is_verified:
                            ok_sum, sum_data = get_report_summary_api(report_id)
                            if ok_sum:
                                st.session_state["insights"] = sum_data
                        st.rerun()
                    else:
                        st.sidebar.error(f"Analysis Failed: {res}")

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if st.session_state.get("uploaded_filenames") != current_names:
            btn_label = f"Analyze Lab Report ({len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''})"
            if st.sidebar.button(btn_label, icon=":material/analytics:", type="primary", **get_stretch_kwarg(st.button)):
                with st.sidebar:
                    with st.spinner("Extracting parameters & indexing domain chunks..."):
                        if len(uploaded_files) == 1:
                            primary_file = uploaded_files[0]
                            file_bytes = primary_file.getvalue()
                            ok, res = upload_report_api(file_bytes, primary_file.name)
                            uploaded_name_label = primary_file.name
                        else:
                            file_tuples = [(f.name, f.getvalue()) for f in uploaded_files]
                            ok, res = upload_report_api(file_tuples, uploaded_files[0].name)
                            uploaded_name_label = f"{len(uploaded_files)} Page Photos"

                        if ok:
                            report_id = res["report_id"]
                            st.session_state["report_id"] = report_id
                            st.session_state["uploaded_filename"] = uploaded_name_label
                            st.session_state["uploaded_filenames"] = current_names
                            st.session_state["upload_response"] = res

                            # Fetch full report data
                            ok_rep, rep_data = get_report_api(report_id)
                            if ok_rep:
                                st.session_state["report_data"] = rep_data

                            # Stage E: Check whether report has already been verified
                            is_verified = rep_data.get("verified_by_user", False) if ok_rep else False
                            st.session_state["verified_by_user"] = is_verified

                            if is_verified:
                                with st.spinner("Generating clinical insights & diet plan..."):
                                    ok_sum, sum_data = get_report_summary_api(report_id)
                                    if ok_sum:
                                        st.session_state["insights"] = sum_data
                                st.sidebar.success("Report loaded.")
                            else:
                                st.sidebar.info("Report extracted. Please verify your results below.")

                            st.rerun()
                        else:
                            st.sidebar.error(f"Analysis Failed: {res}")

    # Active Report Status & Delete Controls
    if st.session_state.get("report_id"):
        report_id = st.session_state["report_id"]
        filename = st.session_state.get("uploaded_filename", "Report.pdf")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Active Patient File")
        st.sidebar.markdown(f"**File:** `{filename}`")
        st.sidebar.markdown(f"**Report ID:** `{report_id}`")

        patient = st.session_state.get("upload_response", {}).get("patient") or {}
        p_name = patient.get("name") if patient else None
        if p_name:
            st.sidebar.markdown(f"**Patient:** {p_name}")

        if st.session_state.get("verified_by_user"):
            status_badge = '<span style="background-color: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">Confirmed</span>'
        else:
            status_badge = '<span style="background-color: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">Verification Pending</span>'
        st.sidebar.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Privacy Controls")
        if st.sidebar.button("Delete Report & Purge Data", icon=":material/delete:", type="secondary", **get_stretch_kwarg(st.button)):
            with st.sidebar:
                with st.spinner("Purging vectors, parsed data, and cached insights..."):
                    ok_del, del_res = delete_report_api(report_id)
                    if ok_del:
                        # Clear session state keys
                        for key in ["report_id", "uploaded_filename", "upload_response", "report_data", "insights", "messages", "verified_by_user"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.sidebar.success("All patient report data permanently purged.")
                        st.rerun()
                    else:
                        st.sidebar.error(f"Purge error: {del_res}")