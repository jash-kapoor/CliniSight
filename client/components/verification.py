"""Stage E: Human-In-The-Loop Interactive Verification Screen."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    from utils.api import confirm_report_api, get_report_api, get_report_page_image_url, get_report_summary_api, update_report_tests_api
    from utils.compat import get_stretch_kwarg
except ImportError:
    from client.utils.api import confirm_report_api, get_report_api, get_report_page_image_url, get_report_summary_api, update_report_tests_api
    from client.utils.compat import get_stretch_kwarg


def map_confidence_label(test: Dict[str, Any]) -> str:
    """Map internal confidence and review state to human-readable labels."""
    if test.get("needs_review"):
        return "Please check"
    conf = (test.get("confidence") or "high").lower()
    if conf == "high" and test.get("verified_by_ocr", True):
        return "Confirmed"
    if conf == "low":
        return "Could not read"
    return "Please check"


def unmap_confidence_label(label: str) -> Tuple[str, bool]:
    """Convert human-readable label back to (confidence, needs_review)."""
    lbl = (label or "Confirmed").strip()
    if lbl == "Confirmed":
        return "high", False
    elif lbl == "Could not read":
        return "low", True
    return "medium", True


def render_verification_screen():
    """
    Renders Stage E: Two-column Human-In-The-Loop verification view.
    Left: Rendered document page image (with optional bbox highlight).
    Right: Interactive editable table of tests with needs-review rows at the top.
    """
    report_id = st.session_state.get("report_id")
    report_data = st.session_state.get("report_data") or {}
    tests: List[Dict[str, Any]] = report_data.get("tests", [])

    if not report_id or not tests:
        st.warning("No report data available for verification.")
        return

    # If the report is already confirmed on backend, proceed to clinical dashboard immediately
    if report_data.get("status") == "confirmed" or report_data.get("verified_by_user"):
        st.session_state["verified_by_user"] = True
        if not st.session_state.get("insights"):
            ok_sum, sum_data = get_report_summary_api(report_id)
            if ok_sum:
                st.session_state["insights"] = sum_data
        st.rerun()

    # Header and Instructions
    import textwrap
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.6rem; font-weight: 700; color: inherit; margin-bottom: 4px;">
                Verify Your Lab Results
            </h2>
            <p style="font-size: 0.92rem; color: inherit; opacity: 0.85; margin: 0;">
                Please verify the extracted parameters against your original laboratory report before clinical reasoning and insights are generated.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # 1. Quality Warnings Alert & Degraded Mode Notice
    quality_warnings = report_data.get("quality_warnings", [])
    is_incomplete = (
        report_data.get("status") == "incomplete"
        or any("automatic reading was unavailable" in str(w).lower() for w in quality_warnings)
    )
    ack_incomplete = True
    if is_incomplete:
        st.warning(
            "Some results may be missing because automatic reading was unavailable. Every row requires your careful review before proceeding.",
            icon=":material/warning:",
        )
        ack_incomplete = st.checkbox(
            "I acknowledge that automatic reading was unavailable and I have verified the results.",
            key="ack_incomplete_reading",
            value=False,
        )

    if quality_warnings:
        with st.expander("Page Ingestion Notices", expanded=is_incomplete):
            for w in quality_warnings:
                st.markdown(f"- {w}")

    # 2. Coverage Summary Banner
    total_tests = len(tests)
    needs_review_items = [t for t in tests if t.get("needs_review") or t.get("confidence") in {"low", "medium"}]
    needs_review_count = len(needs_review_items)
    confirmed_count = total_tests - needs_review_count

    banner_color = "#fef3c7" if needs_review_count > 0 else "#ecfdf5"
    border_color = "#f59e0b" if needs_review_count > 0 else "#10b981"
    text_color = "#92400e" if needs_review_count > 0 else "#065f46"

    st.markdown(
        textwrap.dedent(f"""
        <div style="background-color: {banner_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 18px;">
            <span style="font-weight: 700; color: {text_color}; font-size: 0.95rem;">
                Coverage Summary: {confirmed_count} of {total_tests} values confirmed
                {f', <span style="text-decoration: underline;">{needs_review_count} need your check</span>' if needs_review_count > 0 else ' (100% verified)'}.
            </span>
            <div style="font-size: 0.85rem; color: {text_color}; margin-top: 4px; opacity: 0.9;">
                Review rows requiring attention (listed at the top), edit any values, or add missing rows using the table below.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Two-Column Layout
    col_img, col_editor = st.columns([1, 1.2], gap="medium")

    # LEFT COLUMN: Document Page Viewer
    page_count = report_data.get("page_count", 1)
    if not page_count or page_count < 1:
        pages_present = {t.get("page_number", 1) for t in tests}
        page_count = max(pages_present) if pages_present else 1

    with col_img:
        st.markdown("#### Original Report Document")
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_page = st.selectbox(
                "Page",
                options=list(range(1, page_count + 1)),
                format_func=lambda p: f"Page {p} of {page_count}",
                key="verify_page_select",
            )
        with c2:
            page_tests = [t for t in tests if t.get("page_number", 1) == selected_page]

            def doc_order_key(t: Dict[str, Any]) -> float:
                box = t.get("bbox")
                if box and len(box) == 4:
                    return float(box[0])
                return 999999.0

            page_tests_sorted = sorted(page_tests, key=doc_order_key)
            bbox_options = ["None"]
            label_to_test: Dict[str, Dict[str, Any]] = {}
            for t in page_tests_sorted:
                name = t.get("canonical_name") or t.get("test_name", "Test")
                if not t.get("bbox"):
                    label = f"{name} (location not found)"
                else:
                    label = name
                if label in label_to_test:
                    label = f"{label} (row {len(bbox_options)})"
                bbox_options.append(label)
                label_to_test[label] = t

            selected_bbox_test = st.selectbox(
                "Highlight Row",
                options=bbox_options,
                key="verify_bbox_select",
            )

        bbox_str = None
        if selected_bbox_test != "None":
            matched = label_to_test.get(selected_bbox_test)
            if matched and matched.get("bbox"):
                bbox_str = ",".join(str(x) for x in matched["bbox"])

        img_url = get_report_page_image_url(report_id, selected_page, bbox=bbox_str)
        st.image(img_url, caption=f"Document Page {selected_page}", **get_stretch_kwarg(st.image))

    # RIGHT COLUMN: Interactive Data Editor
    with col_editor:
        st.markdown("#### Extracted Parameters (Editable)")

        # Sort tests: rows that need review at the top!
        def sort_key(t: Dict[str, Any]) -> tuple:
            needs_check = 0 if (t.get("needs_review") or t.get("confidence") in {"low", "medium"}) else 1
            return (needs_check, t.get("page_number", 1), t.get("canonical_name", ""))

        sorted_tests = sorted(tests, key=sort_key)

        # Build DataFrame for st.data_editor
        table_rows = []
        for t in sorted_tests:
            val = t.get("result_value_numeric")
            val_display = str(val) if val is not None else (t.get("result_text") or "")
            ref_str = t.get("reference_text") or (
                f"{t.get('reference_low')} - {t.get('reference_high')}"
                if t.get("reference_low") is not None or t.get("reference_high") is not None
                else ""
            )
            table_rows.append(
                {
                    "Test Name": t.get("canonical_name") or t.get("test_name", ""),
                    "Value": val_display,
                    "Unit": t.get("unit") or "",
                    "Reference Range": ref_str,
                    "Status": t.get("flag", "normal"),
                    "Confidence": map_confidence_label(t),
                    "Review Reason": t.get("review_reason") or "",
                    "Page": int(t.get("page_number", 1)),
                }
            )

        df = pd.DataFrame(table_rows)

        column_config = {
            "Test Name": st.column_config.TextColumn("Test Name", required=True),
            "Value": st.column_config.TextColumn("Value", required=True),
            "Unit": st.column_config.TextColumn("Unit"),
            "Reference Range": st.column_config.TextColumn("Reference Range"),
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["normal", "low", "high", "critical", "unknown"],
                help="Computed clinical flag against reference bounds",
            ),
            "Confidence": st.column_config.SelectboxColumn(
                "Confidence",
                options=["Confirmed", "Please check", "Could not read"],
                help="Verification confidence level",
            ),
            "Review Reason": st.column_config.TextColumn(
                "Review Reason",
                help="Reason why this row requires verification or attention",
            ),
            "Page": st.column_config.NumberColumn("Page", min_value=1, step=1),
        }

        edited_df = st.data_editor(
            df,
            column_config=column_config,
            num_rows="dynamic",
            key="stage_e_editor",
            height=460,
            **get_stretch_kwarg(st.data_editor),
        )

        st.caption("Tip: You can edit values, change units or reference intervals, delete rows, or add missing tests at the bottom.")

        # Action Buttons
        btn_col1, btn_col2 = st.columns([1, 1.2])

        with btn_col1:
            if st.button("Recalculate & Save Changes", icon=":material/refresh:", **get_stretch_kwarg(st.button)):
                with st.spinner("Recomputing clinical flags and updating report..."):
                    updated_payload = _df_to_test_payload(edited_df)
                    ok_update, up_res = update_report_tests_api(report_id, updated_payload)
                    if ok_update:
                        ok_rep, rep_data = get_report_api(report_id)
                        if ok_rep:
                            st.session_state["report_data"] = rep_data
                        st.success("Changes saved and flags recomputed.")
                        st.rerun()
                    else:
                        st.error(f"Failed to update tests: {up_res}")

        with btn_col2:
            confirm_disabled = is_incomplete and not ack_incomplete
            if st.button("Confirm and Continue", icon=":material/check_circle:", type="primary", disabled=confirm_disabled, **get_stretch_kwarg(st.button)):
                if is_incomplete and not ack_incomplete:
                    st.error("Please acknowledge the notice above before confirming.")
                else:
                    with st.spinner("Finalizing verification, indexing vector store & generating insights..."):
                        updated_payload = _df_to_test_payload(edited_df)
                        ok_conf, conf_res = confirm_report_api(
                            report_id,
                            updated_payload,
                            acknowledged_incomplete=ack_incomplete,
                        )
                        if ok_conf:
                            ok_rep, rep_data = get_report_api(report_id)
                            if ok_rep:
                                st.session_state["report_data"] = rep_data
                            st.session_state["verified_by_user"] = True
                            st.session_state["insights"] = conf_res.get("insights")
                            if st.session_state.get("upload_response"):
                                st.session_state["upload_response"]["verified_by_user"] = True
                                if rep_data and "summary" in rep_data:
                                    st.session_state["upload_response"]["summary"] = rep_data["summary"]
                            st.success("Verification confirmed. Loading complete clinical analysis...")
                            st.rerun()
                        else:
                            # Resilience check: if confirm timed out or failed, check if report was actually confirmed on backend
                            ok_rep, rep_data = get_report_api(report_id)
                            if ok_rep and (rep_data.get("status") == "confirmed" or rep_data.get("verified_by_user")):
                                ok_sum, sum_data = get_report_summary_api(report_id)
                                st.session_state["report_data"] = rep_data
                                st.session_state["verified_by_user"] = True
                                if ok_sum:
                                    st.session_state["insights"] = sum_data
                                if st.session_state.get("upload_response"):
                                    st.session_state["upload_response"]["verified_by_user"] = True
                                    if rep_data and "summary" in rep_data:
                                        st.session_state["upload_response"]["summary"] = rep_data["summary"]
                                st.success("Verification confirmed. Loading complete clinical analysis...")
                                st.rerun()
                            else:
                                st.error(f"Failed to confirm report: {conf_res}")

            if is_incomplete and not ack_incomplete:
                st.caption("Please check the acknowledgement notice above to enable confirmation.")


def _df_to_test_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert edited DataFrame rows into TestItemUpdate payload dicts."""
    test_cat_map = {
        (t.get("canonical_name") or t.get("test_name", "")).strip().lower(): t.get("category")
        for t in st.session_state.get("report_data", {}).get("tests", [])
    }
    test_page_map = {
        (t.get("canonical_name") or t.get("test_name", "")).strip().lower(): int(t.get("page_number", 1))
        for t in st.session_state.get("report_data", {}).get("tests", [])
    }

    payload = []
    for _, row in df.iterrows():
        name = str(row.get("Test Name", "")).strip()
        if not name:
            continue

        raw_val = str(row.get("Value", "")).strip()
        unit = str(row.get("Unit", "")).strip() or None
        ref_text = str(row.get("Reference Range", "")).strip() or None
        status = str(row.get("Status", "normal")).strip().lower()
        conf_label = str(row.get("Confidence", "Confirmed")).strip()
        try:
            page_val = row.get("Page")
            page_num = int(page_val) if page_val is not None and not pd.isna(page_val) and int(page_val) > 0 else test_page_map.get(name.lower(), 1)
        except (ValueError, TypeError):
            page_num = test_page_map.get(name.lower(), 1)

        conf, needs_rev = unmap_confidence_label(conf_label)

        # Determine numeric vs qualitative
        num_val = None
        txt_val = None
        try:
            clean_num_str = raw_val.replace(",", "")
            num_val = float(clean_num_str)
        except ValueError:
            txt_val = raw_val if raw_val else None

        rev_reason = str(row.get("Review Reason", "")).strip() or None
        cat = test_cat_map.get(name.lower()) or "General"

        payload.append(
            {
                "test_name": name,
                "canonical_name": name,
                "category": cat,
                "result_value_numeric": num_val,
                "result_text": txt_val,
                "unit": unit,
                "reference_low": None,
                "reference_high": None,
                "reference_text": ref_text,
                "flag": status,
                "page_number": page_num,
                "confidence": conf,
                "needs_review": needs_rev,
                "review_reason": rev_reason,
            }
        )
    return payload
