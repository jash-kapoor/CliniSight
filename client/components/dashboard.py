import textwrap
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

# Add server modules to path for shared display helpers
_server_dir = Path(__file__).resolve().parent.parent.parent / "server"
if str(_server_dir) not in sys.path:
    sys.path.append(str(_server_dir))

from modules.display_helpers import format_test_display_reference, format_test_display_value
try:
    from utils.emoji import strip_emoji
except ImportError:
    from client.utils.emoji import strip_emoji


def get_flag_badge(flag: str) -> str:
    flag_lower = (flag or "").lower()
    if flag_lower == "normal":
        return '<span style="background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.03em;">NORMAL</span>'
    elif flag_lower == "high":
        return '<span style="background-color: #ffedd5; color: #9a3412; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.03em;">HIGH</span>'
    elif flag_lower == "low":
        return '<span style="background-color: #e0f2fe; color: #075985; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.03em;">LOW</span>'
    elif flag_lower == "critical":
        return '<span style="background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em;">CRITICAL</span>'
    else:
        return '<span style="background-color: rgba(148, 163, 184, 0.15); color: inherit; padding: 4px 10px; border-radius: 4px; font-weight: 500; font-size: 0.8rem;">UNKNOWN</span>'


def get_severity_badge_html(badge_text: str) -> str:
    clean = strip_emoji(badge_text)
    lower = clean.lower()
    if "high" in lower:
        return f'<span style="background-color: #fee2e2; color: #991b1b; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">{clean}</span>'
    elif "medium" in lower:
        return f'<span style="background-color: #ffedd5; color: #9a3412; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">{clean}</span>'
    elif "low" in lower:
        return f'<span style="background-color: #fef9c3; color: #854d0e; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">{clean}</span>'
    return f'<span style="background-color: rgba(148, 163, 184, 0.15); color: inherit; padding: 3px 9px; border-radius: 4px; font-weight: 600; font-size: 0.82rem;">{clean}</span>'




def render_dashboard():
    """Renders the comprehensive color-coded clinical dashboard with theme contrast."""
    report_data = st.session_state.get("report_data")
    insights = st.session_state.get("insights")
    upload_res = st.session_state.get("upload_response")

    if not report_data and not upload_res:
        st.info("Please upload a medical lab report PDF using the sidebar to view the clinical dashboard.")
        return

    # Extract patient and summary info
    patient = (report_data.get("patient") if report_data else None) or (upload_res.get("patient") if upload_res else {}) or {}
    summary = (report_data.get("summary") if report_data else None) or (upload_res.get("summary") if upload_res else None) or {}
    tests: List[Dict[str, Any]] = (report_data.get("tests") if report_data else None) or summary.get("abnormal_tests", [])

    # Recompute live summary from CURRENT saved tests (mutually exclusive counts)
    if tests:
        total_tests = len(tests)
        normal_count = sum(1 for t in tests if (t.get("flag") or "").lower() == "normal")
        critical_count = sum(1 for t in tests if (t.get("flag") or "").lower() == "critical")
        abnormal_count = sum(1 for t in tests if (t.get("flag") or "").lower() in ["high", "low"])
    else:
        total_tests = summary.get("total_tests", 0)
        normal_count = summary.get("normal_count", 0)
        critical_count = summary.get("critical_count", 0)
        abnormal_count = summary.get("abnormal_count", 0)

    # 1. Patient Demographics Banner
    st.markdown("### Patient Information & Specimen Details")
    p_name = patient.get("name") or "Not Specified"
    p_age = f"{patient.get('age'):.0f} yrs" if patient.get("age") is not None else "Not Specified"
    p_sex = (patient.get("sex") or "Not Specified").capitalize()
    p_date = patient.get("report_date") or "Not Specified"
    p_lab = patient.get("lab_name") or "Diagnostic Laboratory"

    # Referring doctor vs source: never invent "Self / OPD"
    ref_doc = patient.get("referring_doctor")
    ref_src = patient.get("referring_source")
    if ref_doc and str(ref_doc).strip() and str(ref_doc).strip().lower() not in {"self", "opd", "self / opd", "self/opd", "not specified"}:
        referral_text = f"**Referring Doctor:** {str(ref_doc).strip()}"
    elif ref_src and str(ref_src).strip() and str(ref_src).strip().lower() not in {"not specified", "none"}:
        referral_text = f"**Referred by:** {str(ref_src).strip()}"
    else:
        referral_text = "**Referred by:** Not stated"

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"**Patient Name:** {p_name}")
        st.markdown(f"**Age / Gender:** {p_age} | {p_sex}")
    with col_p2:
        st.markdown(f"**Report Date:** {p_date}")
        st.markdown(referral_text)
    with col_p3:
        st.markdown(f"**Diagnostic Lab:** {p_lab}")
        st.markdown(f"**Report ID:** `{st.session_state.get('report_id', 'N/A')}`")

    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.2);'>", unsafe_allow_html=True)

    # Urgent Critical Banner if any critical parameter exists
    if critical_count > 0:
        st.error(
            f"**CRITICAL ALERT: {critical_count} parameter(s) crossed explicit critical clinical thresholds. Immediate medical review is recommended.**",
            icon=":material/emergency:",
        )

    # 2. Key Metric Stat Cards
    st.markdown("### Test Statistics Overview")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            textwrap.dedent(f"""
            <div style="background-color: var(--secondary-background-color, rgba(148, 163, 184, 0.08)); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 10px; padding: 16px; text-align: center; color: inherit;">
                <div style="font-size: 0.85rem; color: inherit; opacity: 0.85; font-weight: 600; text-transform: uppercase;">Total Parameters</div>
                <div style="font-size: 2.1rem; font-weight: 700; color: inherit; margin: 4px 0;">{total_tests}</div>
                <div style="font-size: 0.8rem; color: inherit; opacity: 0.75;">Extracted & Validated</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            textwrap.dedent(f"""
            <div style="background-color: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 10px; padding: 16px; text-align: center;">
                <div style="font-size: 0.85rem; color: #22c55e; font-weight: 600; text-transform: uppercase;">Normal Tests</div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #22c55e; margin: 4px 0;">{normal_count}</div>
                <div style="font-size: 0.8rem; color: #22c55e; opacity: 0.9;">Within Physiological Range</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            textwrap.dedent(f"""
            <div style="background-color: rgba(249, 115, 22, 0.12); border: 1px solid rgba(249, 115, 22, 0.4); border-radius: 10px; padding: 16px; text-align: center;">
                <div style="font-size: 0.85rem; color: #f97316; font-weight: 600; text-transform: uppercase;">Abnormal Tests</div>
                <div style="font-size: 2.1rem; font-weight: 700; color: #f97316; margin: 4px 0;">{abnormal_count}</div>
                <div style="font-size: 0.8rem; color: #ea580c; opacity: 0.9;">Requiring Attention</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with m_col4:
        crit_bg = "rgba(239, 68, 68, 0.15)" if critical_count > 0 else "var(--secondary-background-color, rgba(148, 163, 184, 0.08))"
        crit_border = "rgba(239, 68, 68, 0.5)" if critical_count > 0 else "rgba(148, 163, 184, 0.25)"
        crit_color = "#ef4444" if critical_count > 0 else "inherit"
        st.markdown(
            textwrap.dedent(f"""
            <div style="background-color: {crit_bg}; border: 1px solid {crit_border}; border-radius: 10px; padding: 16px; text-align: center; color: inherit;">
                <div style="font-size: 0.85rem; color: {crit_color}; font-weight: 600; text-transform: uppercase;">Critical Alerts</div>
                <div style="font-size: 2.1rem; font-weight: 700; color: {crit_color}; margin: 4px 0;">{critical_count}</div>
                <div style="font-size: 0.8rem; color: {crit_color}; opacity: 0.9;">Immediate Clinical Review</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 20px 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.2);'>", unsafe_allow_html=True)

    # 3. Plain-Language Doctor Overview
    if insights and insights.get("overview"):
        st.markdown("### Physician's Clinical Overview")
        with st.container(border=True):
            st.markdown(strip_emoji(insights.get("overview")))
        st.markdown("")

    # 4. Priority Ranked Findings
    primary_cluster = insights.get("primary_cluster") if insights else None
    priority_findings = insights.get("priority_findings", []) if insights else []

    if primary_cluster or priority_findings:
        st.markdown("### Priority Clinical Findings")
        st.caption("Ranked by clinical importance (critical thresholds, diagnostic pattern significance, and deviations).")

        cluster_tests_keys = set()

        # 1. Primary Cluster Card at the top (if present)
        if primary_cluster:
            c_name = strip_emoji(primary_cluster.get("cluster_name", "Primary Diagnostic Pattern"))
            c_badge = primary_cluster.get("severity_badge", "Medium Concern")
            c_urgency = strip_emoji(primary_cluster.get("urgency", "Discuss with your physician"))
            c_summary = strip_emoji(primary_cluster.get("clinical_summary", ""))
            member_tests = primary_cluster.get("member_tests", [])

            for mt in member_tests:
                cluster_tests_keys.add((mt.get("canonical_name") or mt.get("test_name", "")).strip().lower())

            with st.container(border=True):
                head1, head2 = st.columns([3, 1])
                with head1:
                    st.markdown(f"### Main Diagnostic Finding: {c_name}")
                with head2:
                    st.markdown(get_severity_badge_html(c_badge), unsafe_allow_html=True)

                st.markdown(f"**Urgency:** **{c_urgency}**")
                if c_summary:
                    st.markdown(f"**Clinical Pattern Interpretation:** {c_summary}")

                st.markdown("##### Coordinated Pattern Parameters:")
                for mt in member_tests:
                    t_name = strip_emoji(mt.get("canonical_name") or mt.get("test_name"))
                    m_val_str = format_test_display_value(mt)
                    m_ref = format_test_display_reference(mt)
                    m_reason = strip_emoji(mt.get("clinical_reasoning") or "")
                    m_cite = strip_emoji(mt.get("evidence_citation") or "")

                    with st.expander(f"{t_name}: {m_val_str} (Reference: {m_ref})", expanded=True):
                        st.markdown(f"**Clinical Significance:** {m_reason}")
                        if m_cite:
                            st.caption(f"Evidence: `{m_cite}`")

        # 2. Remaining individual findings (not in primary cluster)
        remaining_findings = [
            f for f in priority_findings
            if (f.get("canonical_name") or f.get("test_name", "")).strip().lower() not in cluster_tests_keys
        ]
        if remaining_findings:
            if primary_cluster:
                st.markdown("##### Additional Key Findings")
            for item in remaining_findings:
                badge = item.get("severity_badge", "Low Concern")
                test_name = strip_emoji(item.get("canonical_name") or item.get("test_name"))
                val_str = format_test_display_value(item)
                ref = format_test_display_reference(item)
                urgency = strip_emoji(item.get("urgency") or "Review with physician")
                reasoning = strip_emoji(item.get("clinical_reasoning") or "")
                citation = strip_emoji(item.get("evidence_citation") or "")

                with st.container(border=True):
                    head_col1, head_col2 = st.columns([3, 1])
                    with head_col1:
                        st.markdown(f"#### {test_name}")
                    with head_col2:
                        st.markdown(get_severity_badge_html(badge), unsafe_allow_html=True)

                    st.markdown(
                        f"**Result:** `{val_str}` &nbsp;|&nbsp; **Standard Reference:** `{ref}` &nbsp;|&nbsp; **Urgency:** **{urgency}**"
                    )
                    if reasoning:
                        st.markdown(reasoning)
                    if citation:
                        st.caption(f"Evidence: `{citation}`")

        st.markdown("<br>", unsafe_allow_html=True)

    # 5. Interactive Complete Test Results (Grouped by Panel / Category)
    st.markdown("### Complete Laboratory Results")
    st.caption("Browse all extracted test parameters grouped by diagnostic panel with verified reference intervals.")

    filter_choice = st.radio(
        "Filter Parameters:",
        options=["All Parameters", "Abnormal Only", "Normal Only"],
        horizontal=True,
        key="dashboard_filter",
    )

    search_query = st.text_input("Search parameter name...", key="test_search").strip().lower()

    filtered_tests = []
    for t in tests:
        flag = (t.get("flag") or "").lower()
        if filter_choice == "Abnormal Only" and flag not in ["high", "low", "critical"]:
            continue
        if filter_choice == "Normal Only" and flag != "normal":
            continue
        if search_query:
            canonical = (t.get("canonical_name") or "").lower()
            name = (t.get("test_name") or "").lower()
            cat = (t.get("category") or "").lower()
            if search_query not in canonical and search_query not in name and search_query not in cat:
                continue
        filtered_tests.append(t)

    if not filtered_tests:
        st.info("No parameters match the selected criteria.")
    else:
        st.markdown(f"**Showing {len(filtered_tests)} parameter(s):**")

        # Group filtered tests by panel / category in clinical order
        PANEL_ORDER = [
            "CBC / Red cell indices",
            "White cell differential",
            "Absolute counts",
            "Platelets",
            "Kidney",
            "Electrolytes",
            "Liver",
            "Lipid",
            "Diabetes",
            "Thyroid",
            "Inflammation",
            "Vitamins",
            "Urine",
            "Infectious / Serology",
            "Other Tests",
        ]

        tests_by_category = defaultdict(list)
        for t in filtered_tests:
            cat = t.get("category") or "Other Tests"
            tests_by_category[cat].append(t)

        # Track displayed categories
        displayed_categories = set()

        for panel_name in PANEL_ORDER:
            panel_tests = tests_by_category.get(panel_name, [])
            if not panel_tests:
                continue
            displayed_categories.add(panel_name)

            st.markdown(f"#### {panel_name} ({len(panel_tests)})")
            for t in panel_tests:
                _render_single_test_row(t)

        # Render any remaining unmapped categories
        for cat_name, panel_tests in tests_by_category.items():
            if cat_name in displayed_categories or not panel_tests:
                continue
            st.markdown(f"#### {cat_name} ({len(panel_tests)})")
            for t in panel_tests:
                _render_single_test_row(t)


def _render_single_test_row(t: Dict[str, Any]):
    """Render a single table row for a test parameter."""
    t_name = strip_emoji(t.get("test_name") or "Unknown")
    c_name = strip_emoji(t.get("canonical_name") or t_name)
    cat = strip_emoji(t.get("category") or "General")
    val_display = format_test_display_value(t)
    ref = format_test_display_reference(t)
    flag = t.get("flag") or "unknown"
    page = t.get("page_number", 1)
    snippet = strip_emoji(t.get("source_snippet") or "")

    col_a, col_b, col_c, col_d, col_e = st.columns([3, 2, 2, 2, 2])
    with col_a:
        st.markdown(f"**{c_name}**")
        if c_name.lower() != t_name.lower():
            st.caption(f"Raw: {t_name}")
    with col_b:
        st.markdown(f"`{val_display}`")
    with col_c:
        st.markdown(f"<span style='color: inherit; opacity: 0.85;'>{ref}</span>", unsafe_allow_html=True)
    with col_d:
        st.markdown(get_flag_badge(flag), unsafe_allow_html=True)
    with col_e:
        st.caption(f"Page {page} &bull; {cat}")

    # Expandable evidence snippet
    if snippet:
        with st.expander(f"Grounding Evidence for {c_name} (Page {page})", expanded=False):
            st.code(snippet, language="text")
    st.markdown("<hr style='margin: 6px 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.15);'>", unsafe_allow_html=True)
