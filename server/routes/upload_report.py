"""Route for uploading, parsing, Stage E human verification, indexing, and deleting medical lab reports."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from logger import logger
from modules.insights import ReportInsightSummary, generate_report_insights
from modules.lab_indexer import delete_report, index_lab_report
from modules.lab_parser import (
    TestResult,
    UploadReportResponse,
    assign_test_category,
    build_report_summary,
    disambiguate_canonical_name,
    load_parsed_report,
    normalize_unit,
    parse_lab_report_inputs,
    parse_numeric_value,
    parse_reference_range,
    recompute_flag,
    run_plausibility_and_consistency_checks,
    save_parsed_report,
)
from modules.settings import MAX_UPLOAD_SIZE_MB, PARSED_REPORTS_DIR, UPLOAD_DIR, get_report_pages_dir

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".heic", ".heif"}


class TestItemUpdate(BaseModel):
    test_name: str
    canonical_name: Optional[str] = None
    category: Optional[str] = "General"
    result_value_numeric: Optional[float] = None
    result_text: Optional[str] = None
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    reference_text: Optional[str] = None
    method: Optional[str] = None
    printed_flag: Optional[str] = None
    flag: Optional[str] = None
    page_number: Optional[int] = None
    confidence: Optional[str] = "high"
    needs_review: bool = False
    review_reason: Optional[str] = None
    bbox: Optional[List[int]] = None


class UpdateTestsPayload(BaseModel):
    tests: List[TestItemUpdate]


class ConfirmReportPayload(BaseModel):
    tests: Optional[List[TestItemUpdate]] = None
    acknowledged_incomplete: Optional[bool] = False


@router.post(
    "/upload_report/",
    response_model=UploadReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload, parse, and prepare lab report for human verification (Stages A-D)",
)
async def upload_report(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts single or multi-page lab reports:
    - Clean digital PDFs, scanned PDFs, phone photos saved as PDF.
    - JPG, PNG, and HEIC photographs.
    - Multiple photographed pages combined in page order.
    Executes Stage A preprocessing and Stage B vision-first multimodal extraction.
    NOTE: Does NOT index into vector store or generate insights until Stage E verification is confirmed.
    """
    upload_list: List[UploadFile] = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)

    if not upload_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No report file provided. Upload a PDF, JPG, PNG, or HEIC file.",
        )

    if len(upload_list) > 1:
        upload_list.sort(key=lambda f: f.filename or "")

    report_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_inputs = []
    primary_filename = upload_list[0].filename or "report.pdf"

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for idx, ufile in enumerate(upload_list, start=1):
        filename = ufile.filename or f"page_{idx}.jpg"
        clean_name = Path(filename).name
        ext = Path(clean_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}' in '{clean_name}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        try:
            content = await ufile.read()
        except Exception as exc:
            logger.error("Failed to read uploaded file %s: %s", clean_name, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read uploaded content for '{clean_name}'.",
            )

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{clean_name}' exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB.",
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The uploaded file '{clean_name}' is empty.",
            )

        save_path = UPLOAD_DIR / f"{report_id}_p{idx}{ext}"
        with open(save_path, "wb") as handle:
            handle.write(content)

        saved_inputs.append((save_path, clean_name))

    logger.info(
        "Saved %d file(s) for report %s (primary: %s)",
        len(saved_inputs),
        report_id,
        primary_filename,
    )

    try:
        # 1. Parse report with Stage A, B, C & D Pipeline
        report_data, summary = parse_lab_report_inputs(
            inputs=saved_inputs,
            report_id=report_id,
            original_filename=primary_filename,
        )

        # Stage E: Human-In-The-Loop. Return parsed report without indexing yet.
        return UploadReportResponse(
            report_id=report_id,
            filename=primary_filename,
            patient=report_data.patient,
            summary=summary,
            verified_by_user=False,
            status=report_data.status,
            page_count=report_data.page_count,
            needs_review_count=report_data.needs_review_count,
            overall_coverage_pct=report_data.overall_coverage_pct,
            quality_warnings=report_data.quality_warnings,
            page_dimensions=report_data.page_dimensions,
        )
    except Exception as exc:
        logger.exception("Error processing lab report %s", report_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to parse lab report: {str(exc)}"},
        )


@router.get(
    "/report/{report_id}",
    summary="Get parsed lab report data by report ID",
)
async def get_report(report_id: str):
    """Retrieve full structured JSON data for a previously uploaded report."""
    report_data = load_parsed_report(report_id)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab report with ID '{report_id}' was not found.",
        )
    return report_data.model_dump()


@router.get(
    "/report/{report_id}/page/{page_number}/image",
    summary="Get rendered report page image with optional bbox highlight",
)
async def get_report_page_image(
    report_id: str,
    page_number: int,
    bbox: Optional[str] = None,
):
    """
    Return rendered page PNG for two-column Stage E verification view.
    If bbox=ymin,xmin,ymax,xmax is supplied, highlight that test row on the image
    with a clear colored rectangle plus a slight dim of the rest of the page.
    """
    pages_dir = get_report_pages_dir(report_id)
    img_path = pages_dir / f"page_{page_number}.png"
    if not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page image {page_number} for report {report_id} not found.",
        )

    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                from PIL import Image, ImageDraw
                img = Image.open(img_path).convert("RGBA")
                w, h = img.size
                ymin, xmin, ymax, xmax = parts
                if ymax <= 1000 and xmax <= 1000 and (ymax > 1 or xmax > 1):
                    ymin = (ymin / 1000.0) * h
                    ymax = (ymax / 1000.0) * h
                    xmin = (xmin / 1000.0) * w
                    xmax = (xmax / 1000.0) * w
                elif ymax <= 1.0 and xmax <= 1.0:
                    ymin *= h
                    ymax *= h
                    xmin *= w
                    xmax *= w

                ymin = max(0.0, min(float(h), ymin))
                ymax = max(0.0, min(float(h), ymax))
                xmin = max(0.0, min(float(w), xmin))
                xmax = max(0.0, min(float(w), xmax))

                # Slight dim of the rest of the page plus clear colored rectangle
                overlay = Image.new("RGBA", img.size, (15, 23, 42, 65))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle([xmin, ymin, xmax, ymax], fill=(255, 235, 59, 85), outline=(234, 88, 12, 255), width=4)
                highlighted = Image.alpha_composite(img, overlay).convert("RGB")
                buf = io.BytesIO()
                highlighted.save(buf, format="PNG")
                buf.seek(0)
                return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as exc:
            logger.warning("Bbox drawing failed for page %d: %s", page_number, exc)

    return FileResponse(img_path, media_type="image/png")


@router.put(
    "/report/{report_id}/tests",
    summary="Update and correct extracted lab test parameters with deterministic flag recomputation",
)
async def update_report_tests(
    report_id: str,
    payload: UpdateTestsPayload,
):
    """
    Stage E: User modifies values, units, or reference ranges, adds missing tests, or deletes rows.
    Deterministically recomputes clinical flags in Python after each change.
    """
    report_data = load_parsed_report(report_id)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab report with ID '{report_id}' was not found.",
        )

    existing_tests_map = {
        (t.canonical_name or t.test_name or "").strip().lower(): t
        for t in report_data.tests
    }
    updated_tests: List[TestResult] = []
    for item in payload.tests:
        canon_name = disambiguate_canonical_name(
            item.canonical_name or item.test_name or "Unknown Test",
            unit=item.unit,
            category=item.category,
        )
        norm_unit = normalize_unit(item.unit)
        num_val = parse_numeric_value(item.result_value_numeric)
        ref_low = item.reference_low
        ref_high = item.reference_high

        if ref_low is None and ref_high is None and item.reference_text:
            parsed_low, parsed_high = parse_reference_range(item.reference_text)
            ref_low = parsed_low
            ref_high = parsed_high

        computed_flag = recompute_flag(
            value=num_val,
            low=ref_low,
            high=ref_high,
            canonical_name=canon_name,
            unit=norm_unit,
            original_flag=item.printed_flag or item.flag or "unknown",
            result_text=item.result_text,
            ref_text=item.reference_text,
        )

        existing_t = existing_tests_map.get(canon_name.lower()) or existing_tests_map.get((item.test_name or "").strip().lower())
        final_page = item.page_number if (item.page_number is not None and item.page_number > 0) else (existing_t.page_number if existing_t else 1)

        test_res = TestResult(
            test_name=(item.test_name or canon_name).strip(),
            canonical_name=canon_name,
            category=assign_test_category(canon_name, test_name=item.test_name, fallback_category=item.category),
            result_value_numeric=num_val,
            result_text=item.result_text.strip() if item.result_text else None,
            unit=norm_unit,
            reference_low=ref_low,
            reference_high=ref_high,
            reference_text=item.reference_text.strip() if item.reference_text else None,
            method=item.method.strip() if item.method else None,
            printed_flag=item.printed_flag.strip() if item.printed_flag else None,
            flag=computed_flag,
            page_number=final_page,
            bbox=item.bbox if item.bbox is not None else (existing_t.bbox if existing_t else None),
            confidence=item.confidence or "high",
            needs_review=item.needs_review,
            review_reason=item.review_reason,
            verified_by_user=True,
        )
        updated_tests.append(test_res)

    run_plausibility_and_consistency_checks(updated_tests)
    report_data.tests = updated_tests
    report_data.needs_review_count = sum(1 for t in updated_tests if t.needs_review)
    summary = build_report_summary(report_data)
    report_data.summary = summary
    save_parsed_report(report_data)

    # Invalidate stale insights cache so insights recompute from current edits
    insights_cache_path = PARSED_REPORTS_DIR / f"{report_id}_insights.json"
    insights_cache_path.unlink(missing_ok=True)

    logger.info(
        "Updated %d tests for report %s. Recomputed abnormal count: %d, critical: %d",
        len(updated_tests),
        report_id,
        summary.abnormal_count,
        summary.critical_count,
    )

    return {
        "status": "updated",
        "report_id": report_id,
        "summary": summary.model_dump(),
        "tests": [t.model_dump() for t in updated_tests],
    }


@router.post(
    "/report/{report_id}/confirm",
    summary="Confirm verified lab report data, index vector store, and generate insights",
)
async def confirm_report_endpoint(
    report_id: str,
    payload: Optional[ConfirmReportPayload] = None,
):
    """
    Stage E: Save verified data (verified_by_user = true),
    then index clinical chunks into the vector store and generate insights.
    """
    report_data = load_parsed_report(report_id)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab report with ID '{report_id}' was not found.",
        )

    # Degraded-mode: block confirmation if automatic reading was incomplete and not acknowledged
    if report_data.status == "incomplete" and not (payload and payload.acknowledged_incomplete):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation blocked: Automatic reading was incomplete. Please review all extracted parameters and acknowledge before confirming.",
        )

    # If payload contains updated tests, apply them first
    if payload and payload.tests:
        existing_tests_map = {
            (t.canonical_name or t.test_name or "").strip().lower(): t
            for t in report_data.tests
        }
        updated_tests: List[TestResult] = []
        for item in payload.tests:
            canon_name = disambiguate_canonical_name(
                item.canonical_name or item.test_name or "Unknown Test",
                unit=item.unit,
                category=item.category,
            )
            norm_unit = normalize_unit(item.unit)
            num_val = parse_numeric_value(item.result_value_numeric)
            ref_low = item.reference_low
            ref_high = item.reference_high

            if ref_low is None and ref_high is None and item.reference_text:
                parsed_low, parsed_high = parse_reference_range(item.reference_text)
                ref_low = parsed_low
                ref_high = parsed_high

            computed_flag = recompute_flag(
                value=num_val,
                low=ref_low,
                high=ref_high,
                canonical_name=canon_name,
                unit=norm_unit,
                original_flag=item.printed_flag or item.flag or "unknown",
                result_text=item.result_text,
                ref_text=item.reference_text,
            )

            existing_t = existing_tests_map.get(canon_name.lower()) or existing_tests_map.get((item.test_name or "").strip().lower())
            final_page = item.page_number if (item.page_number is not None and item.page_number > 0) else (existing_t.page_number if existing_t else 1)

            test_res = TestResult(
                test_name=(item.test_name or canon_name).strip(),
                canonical_name=canon_name,
                category=assign_test_category(canon_name, test_name=item.test_name, fallback_category=item.category),
                result_value_numeric=num_val,
                result_text=item.result_text.strip() if item.result_text else None,
                unit=norm_unit,
                reference_low=ref_low,
                reference_high=ref_high,
                reference_text=item.reference_text.strip() if item.reference_text else None,
                method=item.method.strip() if item.method else None,
                printed_flag=item.printed_flag.strip() if item.printed_flag else None,
                flag=computed_flag,
                page_number=final_page,
                bbox=item.bbox if item.bbox is not None else (existing_t.bbox if existing_t else None),
                confidence=item.confidence or "high",
                needs_review=item.needs_review,
                review_reason=item.review_reason,
                verified_by_user=True,
            )
            updated_tests.append(test_res)

        run_plausibility_and_consistency_checks(updated_tests)
        report_data.tests = updated_tests

    was_already_confirmed = (report_data.status == "confirmed" and bool(report_data.verified_by_user))
    insights_cache_path = PARSED_REPORTS_DIR / f"{report_id}_insights.json"

    # Treat all remaining rows as user-verified unless truly unreadable / unresolved
    report_data.verified_by_user = True
    report_data.status = "confirmed"
    for t in report_data.tests:
        t.verified_by_user = True
        is_unresolved = (t.result_value_numeric is None and not t.result_text) or t.confidence in {"could_not_read", "unresolved"}
        if not is_unresolved:
            t.needs_review = False
            t.review_reason = None
            t.confidence = "verified"

    report_data.needs_review_count = sum(1 for t in report_data.tests if t.needs_review)
    summary = build_report_summary(report_data)
    report_data.summary = summary
    save_parsed_report(report_data)

    # 1. Index structured chunks into isolated per-report vector store
    indexed_count = index_lab_report(report_data)
    logger.info("Indexed %d clinical chunks after human confirmation for report %s", indexed_count, report_id)

    # 2. Regenerate clinical insights from verified data (reuse cache if already confirmed)
    force_refresh = not (was_already_confirmed and insights_cache_path.exists())
    insights = generate_report_insights(report_id, force_refresh=force_refresh)

    return {
        "status": "confirmed",
        "report_id": report_id,
        "summary": summary.model_dump() if hasattr(summary, "model_dump") else summary,
        "insights": insights.model_dump() if hasattr(insights, "model_dump") else insights,
        "report": report_data.model_dump(),
    }


@router.get(
    "/report/{report_id}/summary",
    response_model=ReportInsightSummary,
    summary="Get clinical summary, priority ranking, diet, and doctor questions",
)
async def get_report_summary_endpoint(report_id: str):
    """Generate or retrieve plain-language clinical insights for a parsed lab report."""
    try:
        insights = generate_report_insights(report_id)
        return insights
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab report with ID '{report_id}' was not found.",
        )
    except Exception as exc:
        logger.exception("Error generating insights for report %s", report_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate clinical insights: {str(exc)}",
        )


@router.delete(
    "/report/{report_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a report's vectors and parsed data for privacy",
)
async def delete_report_endpoint(report_id: str):
    """
    Remove all vector embeddings, parsed JSON data, and uploaded files
    associated with report_id for complete user privacy.
    """
    result = delete_report(report_id)
    return result
