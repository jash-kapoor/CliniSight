"""Domain-specific lab report chunking, isolated indexing, and metadata-aware retrieval."""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document

from logger import logger
from modules.lab_parser import LabReportData, load_parsed_report
from modules.settings import PARSED_REPORTS_DIR, REPORT_PAGES_DIR, UPLOAD_DIR
from modules.vectorstore import (
    add_texts,
    delete_report_vectors,
    similarity_search,
)


GENERAL_QUERY_KEYWORDS = [
    "wrong",
    "abnormal",
    "summary",
    "summarize",
    "overall",
    "overview",
    "findings",
    "report",
    "problem",
    "problems",
    "issue",
    "issues",
    "concern",
    "concerns",
    "worry",
    "worrying",
    "explain all",
    "results",
    "test results",
    "how are my",
    "what does it say",
    "check my",
    "what do i have",
]


def build_lab_report_chunks(
    report_data: LabReportData,
) -> tuple[list[str], list[dict], list[str]]:
    """
    Construct clinically meaningful chunks for lab report data:
    a) One chunk for patient header info.
    b) One chunk per test result (with name, value, unit, range, flag, page, snippet).
    c) One chunk per category / medical panel.
    d) One chunk for doctor / pathologist remarks.
    """
    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    report_id = report_data.report_id

    # a) Patient & report header chunk
    p = report_data.patient
    patient_lines = [
        f"Patient & Report Header (Report ID: {report_id}):",
        f"- Patient Name: {p.name if p and p.name else 'Not specified'}",
        f"- Age: {p.age if p and p.age is not None else 'Not specified'}",
        f"- Sex: {p.sex if p and p.sex else 'Not specified'}",
        f"- Report Date: {p.report_date if p and p.report_date else 'Not specified'}",
        f"- Diagnostic Lab: {p.lab_name if p and p.lab_name else 'Not specified'}",
        f"- Referring Doctor: {p.referring_doctor if p and p.referring_doctor else 'Not specified'}",
        f"- Source File: {report_data.filename}",
    ]
    texts.append("\n".join(patient_lines))
    metadatas.append({
        "report_id": report_id,
        "page": 1,
        "category": "Patient Demographics",
        "test_name": "",
        "flag": "normal",
        "chunk_type": "patient_header",
    })
    ids.append(f"{report_id}-patient-header")

    # b) One chunk per test result
    for idx, test in enumerate(report_data.tests):
        ref_text = test.reference_text or (
            f"{test.reference_range_low} - {test.reference_range_high}"
            if (test.reference_range_low is not None and test.reference_range_high is not None)
            else "Not specified"
        )
        test_chunk = (
            f"Laboratory Test Result:\n"
            f"- Test: {test.canonical_name} ({test.test_name})\n"
            f"- Medical Category: {test.category}\n"
            f"- Observed Value: {test.value} {test.unit or ''}\n"
            f"- Biological Reference Range: {ref_text}\n"
            f"- Clinical Flag: {test.flag.upper()}\n"
            f"- Page in Report: {test.page_number}\n"
            f"- Report Source Snippet: {test.source_snippet}"
        )
        texts.append(test_chunk)
        metadatas.append({
            "report_id": report_id,
            "page": test.page_number,
            "category": test.category,
            "test_name": test.canonical_name,
            "flag": test.flag,
            "chunk_type": "test_result",
        })
        ids.append(f"{report_id}-test-{idx}")

    # c) One chunk per category / medical panel
    categories_map: dict[str, list] = {}
    for test in report_data.tests:
        cat = test.category or "General"
        categories_map.setdefault(cat, []).append(test)

    for cat_name, cat_tests in categories_map.items():
        panel_lines = [
            f"Category / Panel Summary: {cat_name.upper()} (Report ID: {report_id})",
            f"Total Tests in Panel: {len(cat_tests)}",
            "Tests and Observed Results:",
        ]
        has_abnormal = False
        min_page = min((t.page_number for t in cat_tests if t.page_number), default=1)
        for t in cat_tests:
            is_abn = t.flag.lower() in {"low", "high", "critical"}
            if is_abn:
                has_abnormal = True
            flag_str = f"[{t.flag.upper()}]" if is_abn else "[NORMAL]"
            ref_str = f"(Ref: {t.reference_text})" if t.reference_text else ""
            panel_lines.append(
                f"- {t.canonical_name}: {t.value} {t.unit or ''} {ref_str} {flag_str}"
            )

        clean_slug = re.sub(r"[^a-zA-Z0-9]+", "_", cat_name).strip("_").lower()
        texts.append("\n".join(panel_lines))
        metadatas.append({
            "report_id": report_id,
            "page": min_page,
            "category": cat_name,
            "test_name": "",
            "flag": "abnormal" if has_abnormal else "normal",
            "chunk_type": "category_summary",
        })
        ids.append(f"{report_id}-cat-{clean_slug}")

    # d) One chunk for doctor / lab remarks
    if report_data.remarks:
        remarks_joined = "\n".join(f"- {r}" for r in report_data.remarks)
        remarks_chunk = (
            f"Doctor & Pathologist Remarks / Clinical Interpretations (Report ID: {report_id}):\n"
            f"{remarks_joined}"
        )
        texts.append(remarks_chunk)
        metadatas.append({
            "report_id": report_id,
            "page": 1,
            "category": "Remarks",
            "test_name": "",
            "flag": "abnormal",
            "chunk_type": "doctor_remarks",
        })
        ids.append(f"{report_id}-doctor-remarks")

    return texts, metadatas, ids


def index_lab_report(report_data: LabReportData) -> int:
    """Index all structured chunks of a lab report into an isolated namespace."""
    texts, metadatas, ids = build_lab_report_chunks(report_data)
    logger.info(
        "Indexing %d clinical chunks for report %s", len(texts), report_data.report_id
    )
    add_texts(texts, metadatas, ids, namespace=report_data.report_id)
    return len(texts)


def is_general_overview_query(query: str) -> bool:
    """Detect if the patient is asking an open-ended, overview, or abnormality query."""
    q_lower = query.lower()
    return any(keyword in q_lower for keyword in GENERAL_QUERY_KEYWORDS)


def build_abnormal_synthesis_document(report_id: str) -> Optional[Document]:
    """Generate a high-priority synthetic document listing all abnormal findings."""
    report_data = load_parsed_report(report_id)
    if not report_data:
        return None

    abnormal_tests = [
        t for t in report_data.tests if t.flag.lower() in {"low", "high", "critical"}
    ]
    if not abnormal_tests:
        return Document(
            page_content=(
                f"[REPORT {report_id} CLINICAL STATUS]: All {len(report_data.tests)} "
                f"tested parameters fall within their physiological reference intervals."
            ),
            metadata={
                "report_id": report_id,
                "page": 1,
                "category": "Abnormal Summary",
                "test_name": "",
                "flag": "normal",
                "chunk_type": "abnormal_synthesis",
            },
        )

    lines = [
        f"[CRITICAL SYNTHESIS: ALL ABNORMAL FINDINGS IN REPORT {report_id}]",
        f"Total Findings Outside Reference Intervals: {len(abnormal_tests)} of {len(report_data.tests)} tests.",
        "List of Abnormal Tests:",
    ]
    for idx, t in enumerate(abnormal_tests, start=1):
        lines.append(
            f"{idx}. {t.canonical_name} ({t.category}): {t.value} {t.unit or ''} "
            f"[Ref: {t.reference_text or f'{t.reference_range_low} - {t.reference_range_high}'}] "
            f"-> Status: {t.flag.upper()} (Page {t.page_number})"
        )

    if report_data.remarks:
        lines.append("Doctor / Lab Remarks:")
        for r in report_data.remarks:
            lines.append(f"- {r}")

    return Document(
        page_content="\n".join(lines),
        metadata={
            "report_id": report_id,
            "page": 1,
            "category": "Abnormal Summary",
            "test_name": "",
            "flag": "abnormal",
            "chunk_type": "abnormal_synthesis",
        },
    )


def retrieve_report_context(
    report_id: str,
    query: str,
    k: int = 7,
) -> list[Document]:
    """
    Retrieve isolated top-k documents for this specific report.
    For general or overview queries, injects the complete list of abnormal tests
    from the parsed JSON to guarantee no abnormal finding is missed.
    """
    logger.info("Retrieving context for report %s: query='%s'", report_id, query)

    # Isolated similarity search within this report's namespace/collection
    docs = similarity_search(query=query, k=k, namespace=report_id)

    # Open-ended / overview / abnormality query injection
    if is_general_overview_query(query):
        logger.info(
            "General query detected for report %s; injecting abnormal tests synthesis",
            report_id,
        )
        synth_doc = build_abnormal_synthesis_document(report_id)
        if synth_doc:
            # Prepend synthesis document so it has highest context prominence
            docs = [synth_doc] + [
                d for d in docs if d.page_content != synth_doc.page_content
            ]

    return docs


def delete_report(report_id: str) -> dict:
    """
    Remove vectors, parsed JSON, and uploaded PDF for complete user privacy.
    """
    logger.info("Initiating deletion for report %s", report_id)

    # 1. Delete vectors
    deleted_vectors = delete_report_vectors(report_id)

    # 2. Delete parsed JSON and cached insights
    json_path = PARSED_REPORTS_DIR / f"{report_id}.json"
    deleted_json = False
    if json_path.exists():
        json_path.unlink()
        deleted_json = True
        logger.info("Deleted parsed JSON for report %s", report_id)

    insights_cache = PARSED_REPORTS_DIR / f"{report_id}_insights.json"
    if insights_cache.exists():
        insights_cache.unlink()
        logger.info("Deleted cached insights for report %s", report_id)

    # 3. Delete uploaded files and rendered page images
    import shutil
    pages_dir = REPORT_PAGES_DIR / report_id
    if pages_dir.exists():
        shutil.rmtree(pages_dir, ignore_errors=True)
        logger.info("Deleted rendered page images for report %s", report_id)

    deleted_pdf = False
    for p in UPLOAD_DIR.glob(f"{report_id}*"):
        try:
            if p.is_file():
                p.unlink()
                deleted_pdf = True
        except Exception:
            pass

    return {
        "report_id": report_id,
        "deleted_vectors": deleted_vectors,
        "deleted_json": deleted_json,
        "deleted_pdf": deleted_pdf,
        "message": f"Report {report_id} and all associated data deleted successfully.",
    }
