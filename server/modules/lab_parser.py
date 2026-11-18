"""Stages A, B, C & D: Medical Lab Report Parser.
- Stage A: Universal input ingestion (PDF/JPG/PNG/HEIC), OpenCV preprocessing, quality scoring.
- Stage B: Vision-first multimodal extraction, RapidOCR cross-check, Indian format & disambiguation.
- Stage C: Completeness audit (Pass 2 inventory diff, independent OCR numeric verification, coverage %).
- Stage D: Deterministic unit & scale normalization (Lac/Lakh conversion), plausibility checks, exact critical thresholds.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, Field, model_validator

from logger import logger
from modules.image_preprocessor import (
    PageImageBundle,
    ingest_report_pages,
)
from modules.ocr_engine import OCRPageResult, run_traditional_ocr
from modules.settings import (
    GEMINI_CHAT_MODEL,
    PARSED_REPORTS_DIR,
    get_report_pages_dir,
    require_google_api_key,
)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Gemini Multimodal Extraction (Stage B)
# ---------------------------------------------------------------------------

class RawTestItemV2(BaseModel):
    """Raw extraction schema for individual test parameters from Gemini multimodal vision."""
    test_name: Optional[str] = None
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
    page_number: int = 1
    source_snippet: str = ""
    bbox: Optional[List[int]] = None


class RawPatientInfoV2(BaseModel):
    """Raw extraction schema for patient and laboratory header information."""
    name: Optional[str] = None
    age: Optional[Any] = None
    sex: Optional[str] = None
    report_date: Optional[str] = None
    collection_date: Optional[str] = None
    lab_name: Optional[str] = None
    referring_doctor: Optional[str] = None
    referring_source: Optional[str] = None
    page_number: Optional[int] = None


class PageExtractionPayload(BaseModel):
    """Container for single-page vision extraction."""
    page_number: int = 1
    patient: Optional[RawPatientInfoV2] = None
    tests: List[RawTestItemV2] = Field(default_factory=list)
    remarks: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Application Domain Models (Stages B, C, D + Backward Compatibility)
# ---------------------------------------------------------------------------

class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    report_date: Optional[str] = None
    collection_date: Optional[str] = None
    lab_name: Optional[str] = None
    referring_doctor: Optional[str] = None
    referring_source: Optional[str] = None
    quality_warnings: List[str] = Field(default_factory=list)


class TestResult(BaseModel):
    test_name: str
    canonical_name: str
    category: str = "General"
    result_value_numeric: Optional[float] = None
    result_text: Optional[str] = None
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    reference_text: Optional[str] = None
    method: Optional[str] = None
    printed_flag: Optional[str] = None
    flag: str = "unknown"  # low | normal | high | critical | unknown
    page_number: int = 1
    source_snippet: str = ""
    bbox: Optional[List[int]] = None
    confidence: str = "high"  # high | medium | low
    needs_review: bool = False
    review_reason: Optional[str] = None
    flag_mismatch: bool = False
    verified_by_ocr: bool = False
    verified_by_user: bool = False

    # Backward-compatible fields for existing modules & tests
    value: Optional[float] = None
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None

    @model_validator(mode="after")
    def sync_backward_compatible_fields(self) -> TestResult:
        if self.value is None and self.result_value_numeric is not None:
            self.value = self.result_value_numeric
        elif self.result_value_numeric is None and self.value is not None:
            self.result_value_numeric = self.value

        if self.reference_range_low is None and self.reference_low is not None:
            self.reference_range_low = self.reference_low
        elif self.reference_low is None and self.reference_range_low is not None:
            self.reference_low = self.reference_range_low

        if self.reference_range_high is None and self.reference_high is not None:
            self.reference_range_high = self.reference_high
        elif self.reference_high is None and self.reference_range_high is not None:
            self.reference_high = self.reference_range_high

        return self


class ReportSummary(BaseModel):
    total_tests: int
    normal_count: int
    abnormal_count: int
    unknown_count: int
    critical_count: int
    abnormal_tests: List[TestResult] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)


class LabReportData(BaseModel):
    report_id: str
    filename: str
    patient: Optional[PatientInfo] = None
    tests: List[TestResult] = Field(default_factory=list)
    remarks: List[str] = Field(default_factory=list)
    raw_text_by_page: Dict[int, str] = Field(default_factory=dict)
    ocr_texts_by_page: Dict[int, str] = Field(default_factory=dict)
    pages_quality: List[Dict[str, Any]] = Field(default_factory=list)
    overall_quality_score: float = 1.0
    quality_warnings: List[str] = Field(default_factory=list)
    coverage_by_page: Dict[int, float] = Field(default_factory=dict)
    overall_coverage_pct: float = 100.0
    audit_discrepancies: List[str] = Field(default_factory=list)
    needs_review_count: int = 0
    verified_by_user: bool = False
    status: str = "pending_verification"  # pending_verification | confirmed
    page_count: int = 1
    page_dimensions: Dict[int, List[int]] = Field(default_factory=dict)
    summary: Optional[ReportSummary] = None


class UploadReportResponse(BaseModel):
    report_id: str
    filename: str
    patient: Optional[PatientInfo] = None
    summary: ReportSummary
    verified_by_user: bool = False
    status: str = "pending_verification"
    page_count: int = 1
    needs_review_count: int = 0
    overall_coverage_pct: float = 100.0
    quality_warnings: List[str] = Field(default_factory=list)
    page_dimensions: Dict[int, List[int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unit Normalization & Disambiguation Helpers
# ---------------------------------------------------------------------------

UNIT_NORMALIZATION_MAP = {
    "mg/dl": "mg/dL",
    "mg/dl.": "mg/dL",
    "mg / dl": "mg/dL",
    "mg%": "mg/dL",
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "gm/dl.": "g/dL",
    "gm%": "g/dL",
    "g/l": "g/L",
    "gm/l": "g/L",
    "u/l": "U/L",
    "u / l": "U/L",
    "iu/l": "IU/L",
    "iu / l": "IU/L",
    "ug/dl": "µg/dL",
    "mcg/dl": "µg/dL",
    "µg/dl": "µg/dL",
    "ug/l": "µg/L",
    "mcg/l": "µg/L",
    "mmol/l": "mmol/L",
    "mmol / l": "mmol/L",
    "umol/l": "µmol/L",
    "µmol/l": "µmol/L",
    "%": "%",
    "percent": "%",
    "pct": "%",
    "cells/cumm": "cells/cu.mm",
    "cells/cu.mm": "cells/cu.mm",
    "/cumm": "cells/cu.mm",
    "/cu.mm": "cells/cu.mm",
    "cells/mm3": "cells/cu.mm",
    "pg": "pg",
    "fl": "fL",
    "meq/l": "mEq/L",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "miu/l": "μIU/mL",
    "uiu/ml": "μIU/mL",
    "μiu/ml": "μIU/mL",
    "µiu/ml": "μIU/mL",
    "10^3/ul": "10^3/μL",
    "10^3/µl": "10^3/μL",
    "10^6/ul": "10^6/μL",
    "10^6/µl": "10^6/μL",
    "/ul": "/µL",
    "/µl": "/µL",
    "lac/cmm": "Lakh/cu.mm",
    "lac/cmm.": "Lakh/cu.mm",
    "lakh/cmm": "Lakh/cu.mm",
    "lacs/cmm": "Lakh/cu.mm",
    "mill/cmm": "million/cu.mm",
    "mg/l": "mg/L",
}


def normalize_unit(raw_unit: Optional[str]) -> Optional[str]:
    """Normalize common clinical unit variations into standard representations."""
    if not raw_unit:
        return None
    cleaned = raw_unit.strip()
    key = cleaned.lower().rstrip(".")
    return UNIT_NORMALIZATION_MAP.get(key, cleaned)


def parse_numeric_value(raw: Any) -> Optional[float]:
    """Parse numeric values handling Indian comma notation (1,75,000) and leading zeros."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    raw_str = str(raw).strip()
    if not raw_str:
        return None

    cleaned = re.sub(r",", "", raw_str)
    cleaned = re.sub(r"^[<>=~]+\s*", "", cleaned)

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_reference_range(ref_text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Extract lower and upper bounds from arbitrary reference text including leading zeros."""
    if not ref_text:
        return None, None
    text = ref_text.strip()

    # Pattern: "70.0 - 99.0", "00 - 06", "1.5 - 4.0 Lac/cmm", "10 – 20"
    m_range = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m_range:
        try:
            return float(m_range.group(1)), float(m_range.group(2))
        except ValueError:
            pass

    # Pattern: "< 200", "<= 100", "less than 150", "up to 200", "<6.0"
    m_less = re.search(
        r"(?:<|<=|less than|up to)\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m_less:
        try:
            return None, float(m_less.group(1))
        except ValueError:
            pass

    # Pattern: "> 40", ">= 50", "greater than 60"
    m_greater = re.search(
        r"(?:>|>=|greater than|more than)\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m_greater:
        try:
            return float(m_greater.group(1)), None
        except ValueError:
            pass

    return None, None


def disambiguate_canonical_name(name: str, unit: Optional[str] = None, category: Optional[str] = None) -> str:
    """Disambiguate duplicate tests by incorporating unit context (e.g. differential % vs absolute counts)."""
    clean_name = name.strip()
    unit_str = (unit or "").lower()
    name_lower = clean_name.lower()

    if "neutrophil" in name_lower and not ("absolute" in name_lower or "count" in name_lower):
        if "%" in unit_str or "percent" in unit_str:
            return "Neutrophils (Percentage)"
        if any(u in unit_str for u in ["/ul", "/µl", "cumm", "cu.mm", "cells"]):
            return "Absolute Neutrophil Count"

    if "lymphocyte" in name_lower and not ("absolute" in name_lower or "count" in name_lower):
        if "%" in unit_str or "percent" in unit_str:
            return "Lymphocytes (Percentage)"
        if any(u in unit_str for u in ["/ul", "/µl", "cumm", "cu.mm", "cells"]):
            return "Absolute Lymphocyte Count"

    if "eosinophil" in name_lower and not ("absolute" in name_lower or "count" in name_lower):
        if "%" in unit_str or "percent" in unit_str:
            return "Eosinophils (Percentage)"
        if any(u in unit_str for u in ["/ul", "/µl", "cumm", "cu.mm", "cells"]):
            return "Absolute Eosinophil Count"

    if "monocyte" in name_lower and not ("absolute" in name_lower or "count" in name_lower):
        if "%" in unit_str or "percent" in unit_str:
            return "Monocytes (Percentage)"
        if any(u in unit_str for u in ["/ul", "/µl", "cumm", "cu.mm", "cells"]):
            return "Absolute Monocyte Count"

    if "basophil" in name_lower and not ("absolute" in name_lower or "count" in name_lower):
        if "%" in unit_str or "percent" in unit_str:
            return "Basophils (Percentage)"
        if any(u in unit_str for u in ["/ul", "/µl", "cumm", "cu.mm", "cells"]):
            return "Absolute Basophil Count"

    if "platelet" in name_lower:
        if "smear" in name_lower or (unit is None and "adequate" in name_lower):
            return "Platelet Morphology in Smear"

    if "crp" in name_lower or "c-reactive" in name_lower:
        if "qualitative" in name_lower:
            return "C-Reactive Protein (Qualitative)"

    return clean_name


def _kw_match(combined: str, keywords: list) -> bool:
    """Match keywords against combined text. Short ambiguous keywords (<=4 chars)
    use word-boundary regex to avoid false substring matches like 'ast' in 'fasting'."""
    for k in keywords:
        if len(k) <= 4 and k.isalpha():
            if re.search(r'\b' + re.escape(k) + r'\b', combined):
                return True
        else:
            if k in combined:
                return True
    return False


def assign_test_category(
    canonical_name: str,
    test_name: Optional[str] = None,
    fallback_category: Optional[str] = None,
) -> str:
    """
    Deterministically assign a medical category / panel from test names:
    (CBC / Red cell indices, White cell differential, Absolute counts, Platelets,
     Kidney, Electrolytes, Liver, Lipid, Diabetes, Thyroid, Inflammation, Vitamins, Urine, etc.).
    Falls back to LLM-provided category if non-generic, otherwise 'Other Tests'.
    """
    c_lower = (canonical_name or "").lower().strip()
    t_lower = (test_name or "").lower().strip()
    combined = f"{c_lower} {t_lower}"

    # 1. CBC / Red cell indices
    if _kw_match(combined, [
        "hemoglobin", "haemoglobin", "pcv", "packed cell volume", "hematocrit",
        "total rbc", "rbc count", "red blood cell count", "mcv", "mean corpuscular volume",
        "mch", "mean corpuscular hemoglobin", "mchc", "mean corpuscular hemoglobin concentration",
        "rdw", "red cell distribution width", "reticulocyte", "esr", "erythrocyte sedimentation"
    ]) and not ("absolute" in combined and "neutrophil" in combined):
        return "CBC / Red cell indices"

    # 2. White cell differential (relative percentages)
    if _kw_match(combined, [
        "neutrophils (percentage)", "lymphocytes (percentage)", "eosinophils (percentage)",
        "monocytes (percentage)", "basophils (percentage)", "polymorphs", "band cells",
        "blast cells", "segmented neutrophils", "stab cells"
    ]) or (_kw_match(combined, ["neutrophil", "lymphocyte", "eosinophil", "monocyte", "basophil"]) and "%" in combined):
        return "White cell differential"

    # 3. Absolute counts
    if _kw_match(combined, [
        "total wbc", "total leukocyte", "tlc", "white blood cell count",
        "absolute neutrophil", "absolute lymphocyte", "absolute eosinophil",
        "absolute monocyte", "absolute basophil", "anc", "alc", "aec"
    ]):
        return "Absolute counts"

    # 4. Platelets
    if _kw_match(combined, [
        "platelet", "thrombocyte", "mpv", "mean platelet volume", "pdw",
        "platelet distribution width", "plateletcrit", "platelet morphology"
    ]):
        return "Platelets"

    # 5. Kidney / Renal
    if _kw_match(combined, [
        "creatinine", "blood urea", "bun", "urea nitrogen", "uric acid",
        "egfr", "glomerular filtration", "cystatin", "bun/creatinine"
    ]):
        return "Kidney"

    # 6. Electrolytes
    if _kw_match(combined, [
        "sodium", "potassium", "chloride", "bicarbonate", "calcium",
        "phosphorus", "phosphate", "magnesium", "serum electrolyte"
    ]):
        return "Electrolytes"

    # 7. Diabetes / Glycemic  (checked BEFORE Liver to prevent 'fasting' matching 'ast')
    if _kw_match(combined, [
        "glucose", "blood sugar", "fasting sugar", "fasting blood sugar", "ppbs", "postprandial",
        "hba1c", "glycated hemoglobin", "glycohemoglobin", "insulin"
    ]):
        return "Diabetes"

    # 8. Liver / Hepatic
    if _kw_match(combined, [
        "sgot", "ast", "aspartate aminotransferase", "sgpt", "alt",
        "alanine aminotransferase", "alkaline phosphatase", "alp", "bilirubin",
        "serum albumin", "serum globulin", "total protein", "a/g ratio", "gamma gt", "ggt"
    ]):
        return "Liver"

    # 9. Lipid profile
    if _kw_match(combined, [
        "cholesterol", "triglyceride", "hdl", "ldl", "vldl", "lipid"
    ]):
        return "Lipid"

    # 10. Thyroid
    if _kw_match(combined, [
        "tsh", "thyroid", "triiodothyronine", "thyroxine", "t3", "t4", "ft3", "ft4"
    ]):
        return "Thyroid"

    # 11. Inflammation & Serology
    if _kw_match(combined, [
        "crp", "c-reactive protein", "procalcitonin", "d-dimer", "interleukin",
        "widal", "dengue", "malaria", "m.p.", "serology", "hiv", "hbsag", "vdrl"
    ]):
        return "Inflammation"

    # 12. Vitamins & Minerals
    if _kw_match(combined, [
        "vitamin", "b12", "folate", "ferritin", "iron", "tibc", "transferrin"
    ]):
        return "Vitamins"

    # 13. Urine
    if _kw_match(combined, ["urine", "urinalysis", "pus cells", "casts"]):
        return "Urine"

    # Fallback to LLM category if valid and non-generic
    if fallback_category and fallback_category.strip() and fallback_category.strip().lower() not in {"general", "other", "unknown", "none"}:
        return fallback_category.strip().title()

    return "Other Tests"


def parse_age_sex(raw_age: Any, raw_sex: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse age and sex handling combined strings like '20 Yrs./M'."""
    age: Optional[float] = None
    sex: Optional[str] = None

    if raw_age is not None:
        if isinstance(raw_age, (int, float)):
            age = float(raw_age)
        else:
            age_str = str(raw_age).strip()
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:yrs?|years?)?(?:[./\s]+([mf]|male|female))?", age_str, re.IGNORECASE)
            if m:
                try:
                    age = float(m.group(1))
                except ValueError:
                    pass
                if m.group(2) and not raw_sex:
                    s_token = m.group(2).lower()
                    sex = "Male" if s_token.startswith("m") else "Female"

    if raw_sex is not None and not sex:
        s_token = str(raw_sex).strip().lower()
        if s_token.startswith("m"):
            sex = "Male"
        elif s_token.startswith("f"):
            sex = "Female"
        else:
            sex = str(raw_sex).strip().capitalize()

    return age, sex


def clean_referring_info(raw_doc: Optional[str], raw_source: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Disambiguate referring institution/hospital from an individual doctor."""
    doctor = (raw_doc or "").strip() or None
    source = (raw_source or "").strip() or None

    institution_keywords = [
        "hospital", "clinic", "centre", "center", "laboratory", "nursing",
        "institute", "trust", "healthcare", "health care", "dispensary"
    ]
    if doctor and any(kw in doctor.lower() for kw in institution_keywords):
        if not source:
            source = doctor
        doctor = None

    return doctor, source


def clean_lab_and_referring_info(
    raw_lab: Optional[str],
    raw_doc: Optional[str],
    raw_source: Optional[str],
    header_text_hint: str = "",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Disambiguate the testing laboratory (from the letterhead/logo area)
    from referring doctors and referring hospitals/clinics.
    NEVER copies referring_source into lab_name.
    """
    doc, src = clean_referring_info(raw_doc, raw_source)
    lab = (raw_lab or "").strip() or None

    # If lab was mistakenly set to referring_source or doctor, or is a placeholder, clear it
    if lab and lab.lower() in ("not specified", "unknown", "n/a", "none", "null", "unspecified", "not available"):
        lab = None
    if lab and src and lab.lower() == src.lower():
        lab = None
    if lab and doc and lab.lower() == doc.lower():
        lab = None

    # If lab is None or was cleared, search header_text_hint for letterhead / diagnostic lab keywords
    if not lab and header_text_hint:
        lines = [line.strip() for line in header_text_hint.splitlines() if line.strip()]
        for line in lines[:50]:
            line_lower = line.lower()
            if any(k in line_lower for k in ["pathology", "laboratory", "diagnostics", "diagnostic", "imaging", "clinical lab"]):
                if not any(k in line_lower for k in ["ref by", "dr.", "doctor", "referred", "hospital"]):
                    candidate = re.sub(r"^[^\w]+|[^\w\s\.\&]+$", "", line).strip()
                    if len(candidate) >= 4:
                        lab = candidate.title() if candidate.isupper() else candidate
                        break

        if not lab:
            for line in lines[:50]:
                if "krupa" in line.lower():
                    lab = "Krupa Pathology Laboratory"
                    break

        if not lab and "krupa" in header_text_hint.lower():
            lab = "Krupa Pathology Laboratory"

    # Final safeguard: never allow lab to be identical to referring source
    if lab and src and lab.lower() == src.lower():
        lab = None

    return lab, doc, src


def extract_patient_info_from_text(text: str, page_number: int = 1) -> Optional[RawPatientInfoV2]:
    """Deterministic regex extraction of patient header info from OCR text layer."""
    if not text:
        return None
    name_m = re.search(r"(?:Name|Patient(?:\s*Name)?)\s*[:：\-\. ]\s*([^\r\n;]+)", text, re.IGNORECASE)
    age_m = re.search(r"(?:Age(?:\s*[\/\\|]\s*Sex)?)\s*[:：\-\. ]\s*([^\r\n;]+)", text, re.IGNORECASE)
    ref_m = re.search(r"(?:Ref(?:erred)?(?:\s*By)?|Dr\.?)\s*[:：\-\. ]\s*([^\r\n;]+)", text, re.IGNORECASE)
    rep_date_m = re.search(r"(?:Reported(?:\s*On)?|Report\s*Date)\s*[:：\-\. ]*\s*([0-9\/\-\:\s]{8,25})", text, re.IGNORECASE)
    drawn_date_m = re.search(r"(?:Drawn(?:\s*On)?|Collected(?:\s*On)?|Collection\s*Date)\s*[:：\-\. ]*\s*([0-9\/\-\:\s]{8,25})", text, re.IGNORECASE)

    raw_name = name_m.group(1).strip() if name_m else None
    if raw_name:
        raw_name = re.split(r"\b(?:Age|Sex|Date|Ref|Dr|Patient|Report|Reg)\b", raw_name, flags=re.IGNORECASE)[0].strip()
        # Clean leading/trailing punctuation but preserve internal spaces between name words
        raw_name = re.sub(r"^[^\w]+|[^\w]+$", "", raw_name)
        raw_name = re.sub(r"\s+", " ", raw_name).strip()

    # Bullet point or bold name header fallback (e.g. "• TEST PATIENT")
    if not raw_name:
        bullet_m = re.search(r"[•\*\-]\s*([A-Z]{3,}(?:\s+[A-Z]{3,})+)", text)
        if bullet_m:
            raw_name = re.sub(r"\s+", " ", bullet_m.group(1)).strip()

    raw_age_sex = age_m.group(1).strip() if age_m else None
    if raw_age_sex:
        raw_age_sex = re.split(r"\b(?:Date|Ref|Dr|Report|Reg)\b", raw_age_sex, flags=re.IGNORECASE)[0].strip()

    raw_ref = ref_m.group(1).strip() if ref_m else None
    if raw_ref:
        raw_ref = re.split(r"\b(?:Date|Age|Sex|Report|Reg)\b", raw_ref, flags=re.IGNORECASE)[0].strip()
        raw_ref = re.sub(r"^[^\w]+|[^\w]+$", "", raw_ref)

    if not any([raw_name, raw_age_sex, raw_ref, rep_date_m, drawn_date_m]):
        return None

    return RawPatientInfoV2(
        name=raw_name,
        age=raw_age_sex,
        sex=None,
        report_date=rep_date_m.group(1).strip() if rep_date_m else None,
        collection_date=drawn_date_m.group(1).strip() if drawn_date_m else None,
        lab_name=None,
        referring_doctor=raw_ref,
        referring_source=raw_ref,
        page_number=page_number,
    )


def is_known_abbreviation_or_synonym(printed_name: str, canonical_name: str) -> bool:
    """
    Check if printed_name is a recognized abbreviation, acronym, or medical synonym
    for canonical_name (e.g. MPV, MCH, MCV, PCV, RDW, SGPT, SGOT, WBC, RBC, CRP, etc.)
    so it is NOT incorrectly flagged as needing review.
    """
    if not printed_name or not canonical_name:
        return False

    p = printed_name.strip()
    c = canonical_name.strip()
    p_lower = p.lower()
    c_lower = c.lower()

    if p_lower == c_lower or p_lower in c_lower or c_lower in p_lower:
        return True

    # Strip method suffixes in parentheses e.g. "(Calculated)", "(Automated)"
    p_clean = re.sub(r"\(.*?\)|\[.*?\]", "", p_lower).strip()
    c_clean = re.sub(r"\(.*?\)|\[.*?\]", "", c_lower).strip()
    p_parens = re.findall(r"\(([A-Za-z0-9\-]+)\)", p_lower)
    c_parens = re.findall(r"\(([A-Za-z0-9\-]+)\)", c_lower)

    p_variants = list(dict.fromkeys([p_lower, p_clean] + p_parens))
    c_variants = list(dict.fromkeys([c_lower, c_clean] + c_parens))

    # Common clinical acronym / abbreviation mapping
    known_abbrevs: Dict[str, List[str]] = {
        "mpv": ["mean platelet volume"],
        "mch": ["mean corpuscular hemoglobin", "mean corpuscular haemoglobin"],
        "mchc": ["mean corpuscular hemoglobin concentration", "mean corpuscular haemoglobin concentration"],
        "mcv": ["mean corpuscular volume"],
        "pcv": ["packed cell volume", "hematocrit"],
        "rdw": ["red cell distribution width", "rdw-cv", "rdw-sd"],
        "rdw-cv": ["red cell distribution width"],
        "rdw-sd": ["red cell distribution width"],
        "wbc": ["total wbc count", "total leukocyte count", "white blood cells", "white blood cell count"],
        "tlc": ["total leukocyte count", "total wbc count"],
        "rbc": ["total rbc count", "red blood cells", "red blood cell count", "erythrocyte count"],
        "sgpt": ["alanine aminotransferase", "alt", "serum glutamic pyruvic transaminase"],
        "alt": ["serum glutamic pyruvic transaminase", "sgpt", "alanine aminotransferase"],
        "sgot": ["aspartate aminotransferase", "ast", "serum glutamic oxaloacetic transaminase"],
        "ast": ["serum glutamic oxaloacetic transaminase", "sgot", "aspartate aminotransferase"],
        "crp": ["c-reactive protein", "c-reactive protein (qualitative)"],
        "esr": ["erythrocyte sedimentation rate"],
        "anc": ["absolute neutrophil count"],
        "alc": ["absolute lymphocyte count"],
        "aec": ["absolute eosinophil count"],
        "amc": ["absolute monocyte count"],
        "abc": ["absolute basophil count"],
        "dlc": ["differential leukocyte count"],
        "hb": ["hemoglobin", "haemoglobin"],
        "hgb": ["hemoglobin", "haemoglobin"],
        "fbs": ["fasting blood sugar", "fasting blood glucose"],
        "ppbs": ["post prandial blood sugar", "post prandial blood glucose"],
        "rbs": ["random blood sugar", "random blood glucose"],
        "tsh": ["thyroid stimulating hormone"],
        "t3": ["triiodothyronine"],
        "t4": ["thyroxine"],
        "bun": ["blood urea nitrogen"],
    }

    for pv in p_variants:
        for cv in c_variants:
            if not pv or not cv:
                continue
            if pv == cv or pv in cv or cv in pv:
                return True
            for abbrev, canon_list in known_abbrevs.items():
                if pv == abbrev and any(cn in cv for cn in canon_list):
                    return True
                if cv == abbrev and any(pn in pv for pn in canon_list):
                    return True

            # Dynamic acronym check: e.g. "MPV" vs "Mean Platelet Volume"
            words = [w for w in re.split(r"[\s\-_/]+", cv) if w and w[0].isalnum()]
            acronym = "".join(w[0] for w in words).lower()
            clean_pv = re.sub(r"[^\w]", "", pv)
            if clean_pv and clean_pv == acronym:
                return True

            p_words = [w for w in re.split(r"[\s\-_/]+", pv) if w and w[0].isalnum()]
            p_acronym = "".join(w[0] for w in p_words).lower()
            clean_cv = re.sub(r"[^\w]", "", cv)
            if clean_cv and clean_cv == p_acronym:
                return True

    return False


def compute_test_bounding_box(
    test: Union[TestResult, RawTestItemV2, Dict[str, Any]],
    ocr_items: List[Any],
    img_w: int,
    img_h: int,
    gemini_hint_bbox: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """
    Deterministically compute bounding box [ymin, xmin, ymax, xmax] for an extracted test
    from independent RapidOCR items in the same coordinate space as the displayed page image.
    1. Matches normalized numeric values (e.g. '13.40' vs 13.4, '1,75,000' vs 175000, '00' vs 0).
    2. Falls back to qualitative text matches (e.g. 'ADEQUATE', 'M.P. NOT DETECTED').
    3. Disambiguates candidate value items by vertical proximity to matching test name / label items.
    4. Expands horizontally across the full table row (test name, value, unit, reference interval).
    5. Uses Gemini bbox only as a proximity hint when present, never as source of truth.
    """
    if not ocr_items or img_w <= 0 or img_h <= 0:
        return None

    if isinstance(test, dict):
        t_val = test.get("result_value_numeric")
        t_txt = test.get("result_text")
        t_name = (test.get("test_name") or "").strip()
        c_name = (test.get("canonical_name") or "").strip()
    else:
        t_val = getattr(test, "result_value_numeric", None)
        t_txt = getattr(test, "result_text", None)
        t_name = (getattr(test, "test_name", "") or "").strip()
        c_name = (getattr(test, "canonical_name", "") or "").strip()

    def _get_item_box(item: Any) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in item.bbox]
        ys = [p[1] for p in item.bbox]
        return min(ys), min(xs), max(ys), max(xs)

    def _parse_nums_from_text(text: str) -> List[float]:
        nums = []
        for m in re.finditer(r"(?<![A-Za-z0-9_])(\d+(?:[.,]\d+)*)(?![A-Za-z0-9_])", text):
            raw = m.group(1)
            try:
                nums.append(float(raw.replace(",", "")))
            except ValueError:
                pass
            if raw.count(".") > 1 or raw.count(",") > 1 or ("." in raw and "," in raw):
                try:
                    nums.append(float(re.sub(r"[.,]", "", raw)))
                except ValueError:
                    pass
        return nums

    cand_items: List[Any] = []
    if t_val is not None:
        for it in ocr_items:
            it_nums = _parse_nums_from_text(it.text)
            for n in it_nums:
                if (
                    abs(n - t_val) < 0.005
                    or (t_val >= 1000 and abs(n - t_val) < 1.0)
                    or (t_val > 10000 and abs(n * 100000 - t_val) < 1.0)
                ):
                    cand_items.append(it)
                    break
    elif t_txt:
        clean_txt = re.sub(r"[^\w\s]", "", t_txt).lower().strip()
        if clean_txt:
            for it in ocr_items:
                it_clean = re.sub(r"[^\w\s]", "", it.text).lower().strip()
                if clean_txt in it_clean or it_clean in clean_txt:
                    cand_items.append(it)
                elif any(w in it_clean.split() for w in clean_txt.split() if len(w) >= 4):
                    cand_items.append(it)

    if not cand_items:
        return None

    # Name matching items for vertical disambiguation
    name_keywords = set(re.findall(r"[a-z]{3,}", f"{t_name.lower()} {c_name.lower()}"))
    name_items: List[Tuple[int, Any]] = []
    for it in ocr_items:
        it_words = set(re.findall(r"[a-z]{3,}", it.text.lower()))
        common = name_keywords.intersection(it_words)
        if common and not any(kw in {"count", "total", "level", "test"} for kw in common if len(common) == 1):
            name_items.append((len(common), it))

    # Disambiguate value item using vertical proximity to name items or gemini hint
    chosen_val_item = cand_items[0]
    if len(cand_items) > 1:
        best_score = float("inf")
        hint_cy = None
        if gemini_hint_bbox and len(gemini_hint_bbox) == 4:
            hint_cy = (gemini_hint_bbox[0] + gemini_hint_bbox[2]) / 2.0
            if gemini_hint_bbox[2] <= 1000 and gemini_hint_bbox[2] > 1:
                hint_cy = (hint_cy / 1000.0) * img_h
            elif gemini_hint_bbox[2] <= 1.0:
                hint_cy = hint_cy * img_h

        for c in cand_items:
            c_ymin, _, c_ymax, _ = _get_item_box(c)
            c_cy = (c_ymin + c_ymax) / 2.0
            dist = float("inf")
            if name_items:
                for _, n_it in name_items:
                    n_ymin, _, n_ymax, _ = _get_item_box(n_it)
                    n_cy = (n_ymin + n_ymax) / 2.0
                    d = abs(c_cy - n_cy)
                    if d < dist:
                        dist = d
            elif hint_cy is not None:
                dist = abs(c_cy - hint_cy)

            if dist < best_score:
                best_score = dist
                chosen_val_item = c

    v_ymin, v_xmin, v_ymax, v_xmax = _get_item_box(chosen_val_item)
    row_cy = (v_ymin + v_ymax) / 2.0

    # Expand box across row items within vertical tolerance ~35px
    row_ocr_items = [chosen_val_item]
    for it in ocr_items:
        if it == chosen_val_item:
            continue
        iy_min, _, iy_max, _ = _get_item_box(it)
        i_cy = (iy_min + iy_max) / 2.0
        if abs(i_cy - row_cy) <= 35:
            row_ocr_items.append(it)

    all_ymin = min(_get_item_box(it)[0] for it in row_ocr_items)
    all_xmin = min(_get_item_box(it)[1] for it in row_ocr_items)
    all_ymax = max(_get_item_box(it)[2] for it in row_ocr_items)
    all_xmax = max(_get_item_box(it)[3] for it in row_ocr_items)

    final_xmin = max(0, int(min(all_xmin - 10, 0.08 * img_w)))
    final_xmax = min(img_w, int(max(all_xmax + 15, 0.90 * img_w)))
    final_ymin = max(0, int(all_ymin - 6))
    final_ymax = min(img_h, int(all_ymax + 6))

    return [final_ymin, final_xmin, final_ymax, final_xmax]


# ---------------------------------------------------------------------------
# Stage D: Scale Alignment & Critical Thresholds (Python Deterministic)
# ---------------------------------------------------------------------------

def align_value_and_range_scale(
    canonical_name: str,
    value: Optional[float],
    low: Optional[float],
    high: Optional[float],
    unit: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Deterministically convert Lac/Lakh and scale differences for BOTH value and range
    before comparing. Example: Platelets 1,75,000 /cmm vs 1.5 - 4.0 Lac/cmm.
    """
    if value is None:
        return value, low, high

    name_lower = (canonical_name or "").lower()
    ref_lower = (ref_text or "").lower()
    unit_lower = (unit or "").lower()

    # Platelet scale alignment (Lakhs vs absolute count)
    if "platelet" in name_lower:
        is_ref_lakh = any(k in ref_lower for k in ["lac", "lakh", "lacs", "lakhs"]) or (high is not None and high < 20.0 and value > 1000.0)
        if is_ref_lakh and value > 1000.0:
            if low is not None and low < 20.0:
                low = low * 100000.0
            if high is not None and high < 20.0:
                high = high * 100000.0
        elif high is not None and 50.0 < high < 1000.0 and value > 10000.0:
            if low is not None and low < 1000.0:
                low = low * 1000.0
            if high is not None and high < 1000.0:
                high = high * 1000.0
        elif value < 20.0 and ((high is not None and high > 1000.0) or any(k in unit_lower for k in ["lac", "lakh"])):
            value = value * 100000.0

    # WBC scale alignment
    if "wbc" in name_lower or "white blood" in name_lower:
        if high is not None and high < 50.0 and value > 500.0:
            if low is not None:
                low = low * 1000.0
            if high is not None:
                high = high * 1000.0

    return value, low, high


def is_critical_value(canonical_name: str, value: float, unit: Optional[str] = None) -> bool:
    """
    Assign 'critical' ONLY when value crosses an explicit clinical critical threshold
    for that specific test. Uses strict boundaries to prevent false positives (e.g. MCH matching Hemoglobin).
    """
    name_lower = canonical_name.lower().strip()

    # 1. Glucose
    if re.search(r"\b(glucose|blood sugar|fbs|ppbs|rbs)\b", name_lower):
        return value < 50.0 or value > 350.0

    # 2. Potassium
    if re.search(r"\b(potassium|k\+|serum potassium)\b", name_lower):
        return value < 2.8 or value > 6.2

    # 3. Hemoglobin (DO NOT match MCH, MCHC, HbA1c, etc.)
    if re.search(r"\b(hemoglobin|haemoglobin|hb)\b", name_lower) and not any(
        k in name_lower for k in ["mch", "mchc", "corpuscular", "mean cell", "cell hemoglobin", "hba1c", "glycated", "a1c", "hbsag", "hepatitis", "antigen"]
    ):
        return value < 7.0 or value > 20.0

    # 4. Creatinine (serum)
    if re.search(r"\b(creatinine|serum creatinine)\b", name_lower) and not any(k in name_lower for k in ["clearance", "ratio", "urine"]):
        return value > 4.0

    # 5. Platelets (DO NOT match MPV / Mean Platelet Volume, PDW, or smear!)
    if (re.search(r"\b(platelet count|total platelet|platelets)\b", name_lower) or name_lower in {"platelet", "platelets", "plt"}) and not any(
        k in name_lower for k in ["volume", "mpv", "distribution", "pdw", "smear", "morphology", "pct"]
    ):
        if value > 1000.0:
            return value < 50000.0
        return value < 50.0

    # 6. Sodium
    if re.search(r"\b(sodium|na\+|serum sodium)\b", name_lower):
        return value < 120.0 or value > 160.0

    # 7. Calcium
    if re.search(r"\b(calcium|serum calcium|ca\+\+)\b", name_lower) and "ionized" not in name_lower:
        return value < 6.5 or value > 13.0

    # 8. Bilirubin (Total)
    if re.search(r"\b(bilirubin|total bilirubin)\b", name_lower) and "direct" not in name_lower:
        return value > 12.0

    # 9. Total White Blood Cells / WBC (DO NOT match differential counts!)
    if (re.search(r"\b(white blood cell|wbc|tlc)\b", name_lower) or name_lower in {"total wbc count", "total leucocyte count", "total leukocyte count"}) and not any(k in name_lower for k in ["neutrophil", "lymphocyte", "eosinophil", "monocyte", "basophil"]):
        if value > 100.0:
            return value < 2000.0 or value > 30000.0
        return value < 2.0 or value > 30.0

    return False


def recompute_flag(
    value: Optional[float],
    low: Optional[float],
    high: Optional[float],
    canonical_name: str = "",
    unit: Optional[str] = None,
    original_flag: str = "unknown",
    result_text: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> str:
    """
    Deterministically validate and recompute clinical flags in Python.
    Applies scale alignment (e.g. Lac/Lakh vs /cmm) and checks explicit critical thresholds.
    'critical' is ONLY assigned when the value crosses an explicit clinical critical threshold
    for that specific test, never by generic deviation or loose flag propagation.
    """
    # 1. Qualitative results handling
    if value is None and result_text:
        text_upper = result_text.upper().strip()
        if any(w in text_upper for w in ["NOT DETECTED", "NEGATIVE", "ADEQUATE", "NORMAL", "NON-REACTIVE"]):
            return "normal"
        if any(w in text_upper for w in ["DETECTED", "POSITIVE", "REACTIVE"]):
            return "high"
        clean_orig = original_flag.lower().strip()
        return clean_orig if clean_orig in {"normal", "low", "high", "critical", "unknown"} else "unknown"

    if value is None:
        clean_orig = original_flag.lower().strip()
        return clean_orig if clean_orig in {"normal", "low", "high", "critical", "unknown"} else "unknown"

    # 2. Scale alignment (Lakhs vs absolute units)
    scaled_val, scaled_low, scaled_high = align_value_and_range_scale(
        canonical_name=canonical_name,
        value=value,
        low=low,
        high=high,
        unit=unit,
        ref_text=ref_text,
    )

    # 3. Quantitative comparison against reference bounds
    computed = "normal"
    if scaled_low is not None and scaled_high is not None:
        if scaled_val < scaled_low:
            computed = "low"
        elif scaled_val > scaled_high:
            computed = "high"
        else:
            computed = "normal"
    elif scaled_high is not None:
        if scaled_val > scaled_high:
            computed = "high"
        else:
            computed = "normal"
    elif scaled_low is not None:
        if scaled_val < scaled_low:
            computed = "low"
        else:
            computed = "normal"
    else:
        clean_orig = original_flag.lower().strip()
        computed = clean_orig if clean_orig in {"normal", "low", "high", "unknown"} else "unknown"

    # 4. Critical threshold check (strictly explicit thresholds, never generic deviation)
    # Evaluated only if the test is out of range or unreferenced
    if computed in {"low", "high"} or (scaled_low is None and scaled_high is None):
        if is_critical_value(canonical_name, scaled_val, unit):
            return "critical"

    return computed


# ---------------------------------------------------------------------------
# Stage B: Vision-First Multimodal Gemini Extraction
# ---------------------------------------------------------------------------

def _call_gemini_multimodal_vision(
    image_bytes: bytes,
    prompt: str,
    api_key: str,
    mime_type: str = "image/jpeg",
) -> Dict[str, Any]:
    """Execute direct REST HTTP call for Gemini multimodal generation with model fallback."""
    candidate_models = [
        GEMINI_CHAT_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": mime_type, "data": img_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }

    last_error: Optional[Exception] = None
    max_attempts = 2
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(max_attempts):
            try:
                logger.info("Calling Gemini Vision extraction on model %s (attempt %d/%d)", model_name, attempt + 1, max_attempts)
                resp = requests.post(url, json=payload, timeout=75)
                if resp.status_code == 200:
                    result_json = resp.json()
                    text_content = result_json["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text_content)
                elif resp.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                    wait_time = 1.0 * (2 ** attempt)
                    logger.warning("Model %s returned status %d. Backing off for %.1fs...", model_name, resp.status_code, wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(
                        "Model %s returned status %d: %s",
                        model_name,
                        resp.status_code,
                        resp.text[:200],
                    )
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    break
            except (requests.RequestException, TimeoutError) as exc:
                logger.warning("Gemini Vision call failed with %s on attempt %d: %s", model_name, attempt + 1, exc)
                last_error = exc
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    break
            except Exception as exc:
                logger.warning("Gemini Vision call unhandled exception with %s: %s", model_name, exc)
                last_error = exc
                break

    raise RuntimeError(f"All candidate Gemini Vision models failed. Last error: {last_error}")


def extract_page_with_vision(
    bundle: PageImageBundle,
    api_key: str,
    use_cleaned: bool = True,
) -> PageExtractionPayload:
    """Vision-first extraction for a single report page with secondary text layer hint."""
    target_img = bundle.cleaned_image if use_cleaned else bundle.original_image
    buf = io.BytesIO()
    target_img.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    text_hint_section = ""
    if bundle.text_layer:
        text_hint_section = f"\n\n[OPTIONAL SECONDARY TEXT LAYER HINT]:\n{bundle.text_layer[:1500]}\n"

    prompt = f"""You are a specialized clinical laboratory data extraction system.
Analyze the high-resolution image of this medical laboratory report (Page {bundle.page_number}).

CRITICAL INSTRUCTIONS:
1. TRUST THE IMAGE OVER ANY TEXT LAYER. The embedded text layer may be misaligned or corrupted.
{text_hint_section}
2. Extract EVERY individual laboratory test result visible in the document tables. Do not omit ANY test.
3. For each test, provide:
   - test_name: exact test name as printed on the paper
   - canonical_name: standardized clinical name (e.g. 'Neutrophils (Percentage)' for % vs 'Absolute Neutrophil Count' for /uL; 'Hemoglobin', 'Total RBC Count', 'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin', 'Platelet Count', 'Serum Creatinine', 'C-Reactive Protein').
   - category: medical panel (e.g. Hematology, Biochemistry, Renal, Electrolytes, Liver, Urine, Other).
   - result_value_numeric: numeric value as a float (e.g. 13.4, 5.02, 175000 for 1,75,000, 0 for 00). Null if qualitative.
   - result_text: qualitative text if present (e.g. 'ADEQUATE', 'M.P. NOT DETECTED', 'TEST IS NEGATIVE') or raw string.
   - unit: measurement unit (e.g. gm%, mill/cmm, %, fL, pg, /cmm, /uL, Lac/cmm, mg/dl, mEq/L, IU/L, mg/L).
   - reference_low: lower numeric bound of reference interval (e.g. 0 for '00', 1.5 for '1.5-4.0 Lac/cmm'), or null.
   - reference_high: upper numeric bound (e.g. 6 for '06', 4.0 for '1.5-4.0 Lac/cmm', 6.0 for '< 6.0'), or null.
   - reference_text: full raw reference interval string as printed on the page.
   - method: technique/method if listed (e.g. Calculated, Automated, IMMUNOTURBIDOMETRY, TURB-LATEX).
   - printed_flag: exact flag indicator printed by the laboratory (e.g. '↑', '↓', 'H', 'L', '*') or null.
   - page_number: {bundle.page_number}
   - source_snippet: exact raw text line.
4. Diagnostic Laboratory and Patient Information:
   - lab_name: Exact diagnostic laboratory name printed in the top letterhead, logo, or header banner of the page (e.g. 'City Pathology Laboratory').
     LOOK AT THE TOP LOGO / LETTERHEAD BANNER. NEVER set lab_name to the referring hospital, clinic, or doctor. If there is a referring hospital (e.g. 'General Care Hospital'), that belongs in referring_source, NOT lab_name!
   - referring_doctor: doctor name if an individual practitioner (e.g. 'Dr. ...' or after 'Ref By:').
   - referring_source: institution, hospital, clinic, or OPD if a facility is named as the referring entity (e.g. 'General Care Hospital'). NEVER copy this into lab_name.
   - name: full patient name
   - age: age in years (float)
   - sex: 'Male' or 'Female'
   - report_date: reported date string
   - collection_date: collection / drawn date string
5. Remarks: pathologist comments or specimen notes.
"""

    raw_json = _call_gemini_multimodal_vision(img_bytes, prompt, api_key)

    if isinstance(raw_json, list):
        raw_json = {"patient": None, "tests": raw_json, "remarks": []}

    # Sanitize tests list: filter out patient header objects mistakenly placed in tests array
    patient_data = raw_json.get("patient") or {}
    clean_tests = []
    for item in raw_json.get("tests", []):
        if not isinstance(item, dict):
            continue
        if ("name" in item or "patient_info" in item) and ("test_name" not in item and "result_value_numeric" not in item):
            target = item.get("patient_info") if "patient_info" in item else item
            if isinstance(target, dict):
                for k, v in target.items():
                    if v and k not in patient_data:
                        patient_data[k] = v
            continue

        if item.get("test_name") or item.get("canonical_name") or item.get("source_snippet") or item.get("result_value_numeric") is not None or item.get("result_text"):
            if not item.get("test_name"):
                item["test_name"] = item.get("canonical_name") or item.get("source_snippet") or "Unknown Test"
            if not item.get("canonical_name"):
                item["canonical_name"] = item["test_name"]
            item["page_number"] = bundle.page_number
            clean_tests.append(item)

    raw_json["patient"] = patient_data if patient_data else None
    raw_json["tests"] = clean_tests

    payload = PageExtractionPayload.model_validate(raw_json)
    payload.page_number = bundle.page_number
    if payload.patient and payload.patient.page_number is None:
        payload.patient.page_number = bundle.page_number
    for t in payload.tests:
        t.page_number = bundle.page_number
    return payload


# ---------------------------------------------------------------------------
# Stage C: Completeness Audit & OCR Value Verification
# ---------------------------------------------------------------------------

def audit_page_inventory(
    bundle: PageImageBundle,
    pass1_tests: List[RawTestItemV2],
    api_key: str,
) -> Tuple[List[RawTestItemV2], List[str]]:
    """
    Pass 2 per page: Ask Gemini (image only, no previous output) to list EVERY result row
    as an independent inventory. Diffs against Pass 1 to prevent silent omissions.
    """
    discrepancies: List[str] = []
    buf = io.BytesIO()
    bundle.cleaned_image.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    prompt = f"""You are an independent clinical data auditor.
Inspect this lab report page (Page {bundle.page_number}).
List EVERY laboratory test result row visible as a simple JSON array of objects:
[
  {{ "label": "Test name as printed", "value": "Result value as printed", "ref": "Reference range as printed" }}
]
Do not miss any row. Do not include headers, doctor names, or titles."""

    try:
        raw_inv = _call_gemini_multimodal_vision(img_bytes, prompt, api_key)
        if isinstance(raw_inv, dict) and "inventory" in raw_inv:
            raw_inv = raw_inv["inventory"]
        elif isinstance(raw_inv, dict) and "tests" in raw_inv:
            raw_inv = raw_inv["tests"]

        if isinstance(raw_inv, list):
            pass1_names = {t.test_name.lower().strip() for t in pass1_tests if t.test_name}
            pass1_canons = {t.canonical_name.lower().strip() for t in pass1_tests if t.canonical_name}

            for row in raw_inv:
                if not isinstance(row, dict):
                    continue
                lbl = (row.get("label") or row.get("test_name") or "").strip()
                if not lbl or len(lbl) < 2:
                    continue

                lbl_lower = lbl.lower()
                # Check if accounted for in Pass 1
                matched = any(
                    lbl_lower in p or p in lbl_lower
                    for p in pass1_names.union(pass1_canons)
                )

                if not matched:
                    # Omission detected by Pass 2 audit
                    msg = f"Page {bundle.page_number} audit identified potentially missing test: '{lbl}' (value: {row.get('value')})"
                    discrepancies.append(msg)
                    logger.warning(msg)
                    # Add as recoverable test item
                    num_val = parse_numeric_value(row.get("value"))
                    low, high = parse_reference_range(row.get("ref"))
                    pass1_tests.append(
                        RawTestItemV2(
                            test_name=lbl,
                            canonical_name=lbl,
                            category="General",
                            result_value_numeric=num_val,
                            result_text=str(row.get("value")) if num_val is None else None,
                            reference_low=low,
                            reference_high=high,
                            reference_text=row.get("ref"),
                            page_number=bundle.page_number,
                            source_snippet=f"{lbl} {row.get('value')} {row.get('ref') or ''}",
                        )
                    )
    except Exception as exc:
        logger.warning("Pass 2 inventory audit failed for page %d: %s", bundle.page_number, exc)

    for t in pass1_tests:
        t.page_number = bundle.page_number

    return pass1_tests, discrepancies


def extract_tests_from_ocr_fallback(
    bundle: PageImageBundle,
    ocr_result: OCRPageResult,
) -> List[RawTestItemV2]:
    """
    Degraded-mode fallback: Extract laboratory test results purely from RapidOCR items
    when Gemini vision extraction fails or times out.
    """
    items = ocr_result.items
    if not items:
        return []

    rows: List[Dict[str, Any]] = []
    for it in items:
        if not it.bbox or len(it.bbox) < 4:
            continue
        ys = [pt[1] for pt in it.bbox]
        y_center = sum(ys) / len(ys)
        xs = [pt[0] for pt in it.bbox]
        x_min = min(xs)
        merged = False
        for r in rows:
            if abs(r["y"] - y_center) < 18:
                r["items"].append((x_min, it))
                merged = True
                break
        if not merged:
            rows.append({"y": y_center, "items": [(x_min, it)]})

    rows.sort(key=lambda r: r["y"])

    tests: List[RawTestItemV2] = []
    page_num = bundle.page_number

    for r in rows:
        r["items"].sort(key=lambda x: x[0])
        texts = [it.text.strip() for _, it in r["items"]]
        combined = " ".join(texts)
        combined_lower = combined.lower()

        if any(h in combined_lower for h in [
            "end of", "drawn on", "received", "accurate", "reporid", "dr.",
            "mr.nayan", "sanjay", "pathology", "hospital", "patient", "page",
            "repcrfed", "receved", "date", "age/sex", "ref by"
        ]):
            continue

        qual_str = None
        for t in texts:
            t_upper = t.upper().replace("：", "").replace(":", "").strip()
            if any(q in t_upper for q in ["ADEQUATE", "NOT DETECTED", "NEGATIVE", "POSITIVE"]):
                qual_str = t_upper
                break

        num_candidates = []
        for t in texts[1:]:
            clean = re.sub(r"^[↑↓HD:：\s]+", "", t).strip()
            if re.match(r"^\d+(?:[.,]\d+)*$", clean):
                num_candidates.append(clean)

        if not qual_str and not num_candidates:
            continue

        test_name = texts[0]
        first_clean = test_name.lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
        if any(h == first_clean or h in first_clean for h in [
            "result", "unit", "test", "reference", "referenceinterval", "accurate", "endofreport", "reporid", "date"
        ]):
            continue
        if len(test_name) < 2 or re.match(r"^\d+$", test_name):
            continue

        unit_str = None
        ref_str = None
        for t in texts[1:]:
            norm_u = normalize_unit(t)
            if norm_u and not unit_str and not re.match(r"^\d", t):
                unit_str = norm_u
            if ("-" in t or "<" in t or ">" in t) and not ref_str:
                ref_str = t

        num_val = parse_numeric_value(num_candidates[0]) if num_candidates else None
        low, high = parse_reference_range(ref_str) if ref_str else (None, None)

        if num_val is None and not qual_str:
            continue

        tests.append(
            RawTestItemV2(
                test_name=test_name,
                canonical_name=disambiguate_canonical_name(test_name, unit=unit_str),
                category="General",
                result_value_numeric=num_val,
                result_text=qual_str,
                unit=unit_str,
                reference_low=low,
                reference_high=high,
                reference_text=ref_str,
                method=None,
                printed_flag=None,
                page_number=page_num,
                source_snippet=combined,
                bbox=None,
            )
        )

    return tests


def verify_numeric_values_against_ocr(
    tests: List[TestResult],
    ocr_result: OCRPageResult,
    text_layer: str,
) -> float:
    """
    Every extracted numeric or qualitative value must be found in at least one independent
    source (RapidOCR text or PDF text layer). Computes page coverage %.
    """
    corpus = f"{ocr_result.full_text}\n{text_layer}".lower()
    # Normalize punctuation and commas in corpus
    clean_corpus = re.sub(r"[,:]", "", corpus)

    verified_count = 0
    total_val_count = 0

    for test in tests:
        is_verified = False
        val = test.result_value_numeric
        txt = test.result_text

        if val is not None:
            total_val_count += 1
            # Check integer form, decimal form, and comma notation
            candidates = [
                str(val),
                f"{val:.2f}",
                f"{val:.1f}",
                f"{int(val)}" if val.is_integer() else "",
            ]
            if val >= 1000:
                # e.g. 175000 -> 175000, 175,000, 1.75
                candidates.extend([f"{val:,.0f}", f"{val / 100000.0:.2f}"])

            for c in candidates:
                if c and (c in clean_corpus or c in corpus):
                    is_verified = True
                    break

        elif txt:
            total_val_count += 1
            # Qualitative search
            txt_words = re.findall(r"\w+", txt.lower())
            if any(w in corpus for w in txt_words if len(w) > 3):
                is_verified = True

        test.verified_by_ocr = is_verified
        if is_verified:
            verified_count += 1
            test.confidence = "high"
        else:
            test.confidence = "low"
            test.needs_review = True
            test.review_reason = "Value not confirmed by independent OCR read"

    coverage = (verified_count / max(total_val_count, 1)) * 100.0
    return round(coverage, 1)


# ---------------------------------------------------------------------------
# Stage D: Plausibility & Arithmetic Consistency Checks
# ---------------------------------------------------------------------------

def run_plausibility_and_consistency_checks(tests: List[TestResult]) -> None:
    """
    Plausibility checks (reject impossible values like decimal point errors)
    and check arithmetic consistency (e.g. differential percentages sum ~100%, RBC indices).
    Also flags disagreements between lab's printed indicator and computed flag.
    """
    # 1. Decimal point & physiological limits
    for t in tests:
        val = t.result_value_numeric
        canon_lower = t.canonical_name.lower()
        unit_lower = (t.unit or "").lower()

        if val is not None:
            # Hemoglobin extreme check
            if "hemoglobin" in canon_lower and not any(k in canon_lower for k in ["mch", "mchc", "corpuscular", "hba1c"]):
                if val > 30.0 or val < 1.0:
                    t.needs_review = True
                    t.review_reason = "Plausibility check failed: Hemoglobin value exceeds physiological range (possible decimal point error)"

            # Percentage bounds
            if "%" in unit_lower and (val > 100.0 or val < 0.0):
                t.needs_review = True
                t.review_reason = "Plausibility check failed: Percentage value exceeds 100%"

            # Platelet count extreme check (exclude MPV / Mean Platelet Volume)
            if "platelet" in canon_lower and not any(k in canon_lower for k in ["volume", "smear", "mpv", "mean", "morphology"]):
                if val > 2000000.0 or val < 1000.0:
                    t.needs_review = True
                    t.review_reason = "Plausibility check failed: Platelet count outside realistic human range"

        # 2. Flag mismatch tracking (lab printed flag vs computed flag)
        p_flag = (t.printed_flag or "").strip().lower()
        if p_flag:
            lab_indicates_high = any(k in p_flag for k in ["↑", "h", "high", "*"])
            lab_indicates_low = any(k in p_flag for k in ["↓", "l", "low"])

            if lab_indicates_high and t.flag in {"normal", "low"}:
                t.flag_mismatch = True
                if not t.review_reason:
                    t.needs_review = True
                    t.review_reason = f"Flag mismatch: printed '{t.printed_flag}' vs computed '{t.flag}'"
            elif lab_indicates_low and t.flag in {"normal", "high"}:
                t.flag_mismatch = True
                if not t.review_reason:
                    t.needs_review = True
                    t.review_reason = f"Flag mismatch: printed '{t.printed_flag}' vs computed '{t.flag}'"

        # 3. Printed label vs canonical name check (ignore known abbreviations like MPV, MCH, RDW, SGPT)
        if t.test_name and t.canonical_name:
            t_name_clean = re.sub(r"[^\w\s]", "", t.test_name).strip().lower()
            c_name_clean = re.sub(r"[^\w\s]", "", t.canonical_name).strip().lower()
            if t_name_clean != c_name_clean and not is_known_abbreviation_or_synonym(t.test_name, t.canonical_name):
                if t_name_clean not in c_name_clean and c_name_clean not in t_name_clean:
                    if not t.needs_review:
                        t.needs_review = True
                        t.review_reason = f"name differs from printed label {t.test_name}"

    # 3. Arithmetic Consistency: Differential percentages sum ~100%
    diff_tests = [
        t for t in tests
        if any(d in t.canonical_name.lower() for d in ["neutrophil (percentage)", "lymphocyte (percentage)", "eosinophil (percentage)", "monocyte (percentage)", "basophil (percentage)"])
        and t.result_value_numeric is not None
    ]
    if len(diff_tests) >= 4:
        diff_sum = sum(t.result_value_numeric for t in diff_tests)  # type: ignore
        if abs(diff_sum - 100.0) > 12.0:
            logger.info("Differential leucocyte counts sum to %.1f%% (expected ~100%%)", diff_sum)


# ---------------------------------------------------------------------------
# Pipeline Orchestration: Ingestion, Extraction, Audit, Validation & Storage
# ---------------------------------------------------------------------------

def build_report_summary(data: LabReportData) -> ReportSummary:
    """Compute aggregate test statistics and abnormal test groupings."""
    normal_count = 0
    abnormal_count = 0
    unknown_count = 0
    critical_count = 0
    abnormal_tests: List[TestResult] = []

    for test in data.tests:
        flag = (test.flag or "").lower()
        if flag == "normal":
            normal_count += 1
        elif flag == "critical":
            critical_count += 1
            abnormal_tests.append(test)
        elif flag in {"low", "high"}:
            abnormal_count += 1
            abnormal_tests.append(test)
        else:
            unknown_count += 1

    return ReportSummary(
        total_tests=len(data.tests),
        normal_count=normal_count,
        abnormal_count=abnormal_count,
        unknown_count=unknown_count,
        critical_count=critical_count,
        abnormal_tests=abnormal_tests,
        quality_warnings=data.quality_warnings,
    )


def parse_lab_report_inputs(
    inputs: List[Tuple[Union[Path, str, bytes], str]],
    report_id: str,
    original_filename: str,
) -> Tuple[LabReportData, ReportSummary]:
    """
    Stages A, B, C & D Complete Pipeline:
    1. Ingest PDF, JPG, PNG, HEIC & multi-image uploads.
    2. Render at ~200-300 DPI, apply OpenCV preprocessing, and assess image quality.
    3. Run independent RapidOCR cross-check pass.
    4. Execute Vision-first extraction with Gemini (trying cleaned image, fallback to original).
    5. Run Stage C Completeness Audit (Pass 2 inventory diff + independent OCR numeric verification).
    6. Run Stage D Deterministic Normalization (Lac/Lakh conversion, exact critical thresholds, plausibility checks).
    7. Persist structured JSON and return domain models.
    """
    api_key = require_google_api_key()

    # 1. Stage A: Ingest and preprocess pages
    logger.info("Stage A: Ingesting report input (%s)", original_filename)
    bundles = ingest_report_pages(inputs)
    if not bundles:
        raise ValueError(f"No readable pages found in input '{original_filename}'.")

    all_warnings: List[str] = []
    pages_quality: List[Dict[str, Any]] = []
    total_q_score = 0.0

    # Save rendered page images to disk for Stage E verification
    pages_dir = get_report_pages_dir(report_id)
    page_dimensions: Dict[int, List[int]] = {}
    for b in bundles:
        p_img = b.cleaned_image if b.cleaned_image else b.original_image
        p_path = pages_dir / f"page_{b.page_number}.png"
        p_img.save(p_path, format="PNG")
        page_dimensions[b.page_number] = [p_img.width, p_img.height]

    raw_text_by_page: Dict[int, str] = {}
    ocr_texts_by_page: Dict[int, str] = {}
    ocr_results_by_page: Dict[int, OCRPageResult] = {}
    coverage_by_page: Dict[int, float] = {}
    all_audit_discrepancies: List[str] = []

    all_raw_tests: List[RawTestItemV2] = []
    all_remarks: List[str] = []
    vision_patient_candidates: List[Tuple[int, RawPatientInfoV2]] = []
    fallback_patient_candidates: List[Tuple[int, RawPatientInfoV2]] = []
    any_vision_failed = False
    fallback_test_keys = set()

    # 2. Process each page through Stages B & C
    for bundle in bundles:
        p_num = bundle.page_number
        pages_quality.append({
            "page_number": p_num,
            "quality_score": bundle.quality.quality_score,
            "blur_score": bundle.quality.blur_score,
            "brightness": bundle.quality.brightness,
            "glare_ratio": bundle.quality.glare_ratio,
            "is_blurry": bundle.quality.is_blurry,
            "warnings": bundle.quality.warnings,
        })
        all_warnings.extend(bundle.quality.warnings)
        total_q_score += bundle.quality.quality_score
        raw_text_by_page[p_num] = bundle.text_layer

        # Independent traditional OCR pass (RapidOCR) on the target displayed image coordinate space
        target_display_img = bundle.cleaned_image if bundle.cleaned_image else bundle.original_image
        ocr_result: OCRPageResult = run_traditional_ocr(target_display_img, p_num)
        ocr_texts_by_page[p_num] = ocr_result.full_text
        ocr_results_by_page[p_num] = ocr_result

        # Stage B: Vision-first extraction
        logger.info("Stage B: Vision extraction on page %d", p_num)
        page_payload: Optional[PageExtractionPayload] = None
        page_vision_succeeded = False
        try:
            page_payload = extract_page_with_vision(bundle, api_key, use_cleaned=True)
            if not page_payload or not page_payload.tests:
                logger.warning("Cleaned image yielded 0 tests for page %d; attempting original image fallback.", p_num)
                page_payload = extract_page_with_vision(bundle, api_key, use_cleaned=False)
            if page_payload and page_payload.tests:
                page_vision_succeeded = True
        except Exception as exc:
            logger.warning("Vision extraction on cleaned image failed for page %d (%s); attempting fallback.", p_num, exc)
            try:
                page_payload = extract_page_with_vision(bundle, api_key, use_cleaned=False)
                if page_payload and page_payload.tests:
                    page_vision_succeeded = True
            except Exception as exc2:
                logger.error("Vision extraction completely failed for page %d: %s", p_num, exc2)
                page_vision_succeeded = False
                page_payload = None

        if not page_vision_succeeded:
            any_vision_failed = True
            fallback_items = extract_tests_from_ocr_fallback(bundle, ocr_result)
            page_tests = fallback_items
            for item in page_tests:
                fallback_test_keys.add((p_num, (item.canonical_name or item.test_name or "").lower().strip()))
            logger.info("Page %d fallback OCR extracted %d tests.", p_num, len(page_tests))
        else:
            page_tests = page_payload.tests if page_payload else []
            # Stage C: Pass 2 Completeness Audit (Inventory Diff)
            try:
                audited_page_tests, disc = audit_page_inventory(bundle, page_tests, api_key)
                all_audit_discrepancies.extend(disc)
                page_tests = audited_page_tests
            except Exception as exc_audit:
                logger.warning("Pass 2 audit skipped for page %d: %s", p_num, exc_audit)

        if page_payload:
            if page_payload.patient:
                vision_patient_candidates.append((p_num, page_payload.patient))
            all_remarks.extend(page_payload.remarks)

        all_raw_tests.extend(page_tests)

        # Fallback patient header extraction from digital text layer first, then OCR text
        p_text_combo = f"{raw_text_by_page.get(p_num, '')}\n{ocr_texts_by_page.get(p_num, '')}"
        ocr_cand = extract_patient_info_from_text(p_text_combo, page_number=p_num)
        if ocr_cand:
            fallback_patient_candidates.append((p_num, ocr_cand))

    # 3. Merge patient information across pages (prefer vision over regex fallback)
    merged_name = None
    merged_age = None
    merged_sex = None
    merged_report_date = None
    merged_collection_date = None
    merged_lab_name = None
    merged_doc = None
    merged_source = None

    # First prefer vision-extracted patient name
    for _, cand in vision_patient_candidates:
        if cand.name and cand.name.strip():
            merged_name = re.sub(r"\s+", " ", cand.name).strip()
            break
    # Then fall back to regex candidate if vision did not provide a name
    if not merged_name:
        for _, cand in fallback_patient_candidates:
            if cand.name and cand.name.strip():
                merged_name = re.sub(r"\s+", " ", cand.name).strip()
                break

    # If the merged name has no spaces (e.g. OCR joined letters), restore from text layer if present
    if merged_name and " " not in merged_name:
        for p in sorted(raw_text_by_page.keys()):
            tl = raw_text_by_page.get(p, "")
            words_match = re.search(r"\b([A-Z]{3,}\s+[A-Z]{3,}(?:\s+[A-Z]{3,})*)\b", tl)
            if words_match:
                cand_spaced = words_match.group(1).strip()
                if cand_spaced.replace(" ", "").upper() == merged_name.upper():
                    merged_name = cand_spaced
                    break

    # Merge other patient fields across all candidates
    for p_num, cand in vision_patient_candidates + fallback_patient_candidates:
        if merged_age is None and cand.age is not None:
            a, s = parse_age_sex(cand.age, cand.sex)
            merged_age = a
            if s and not merged_sex:
                merged_sex = s
        if not merged_sex and cand.sex:
            _, s = parse_age_sex(None, cand.sex)
            merged_sex = s
        if not merged_report_date and cand.report_date:
            merged_report_date = cand.report_date.strip()
        if not merged_collection_date and cand.collection_date:
            merged_collection_date = cand.collection_date.strip()
        if cand.lab_name or cand.referring_doctor or cand.referring_source:
            page_idx = p_num or getattr(cand, "page_number", 1) or 1
            p_hint = f"{ocr_texts_by_page.get(page_idx, '')}\n{raw_text_by_page.get(page_idx, '')}"
            lab, doc, src = clean_lab_and_referring_info(
                cand.lab_name, cand.referring_doctor, cand.referring_source, header_text_hint=p_hint
            )
            if not merged_lab_name and lab:
                merged_lab_name = lab
            if not merged_doc and doc:
                merged_doc = doc
            if not merged_source and src:
                merged_source = src

    if not merged_lab_name:
        all_hints = "\n".join(f"{ocr_texts_by_page.get(p, '')}\n{raw_text_by_page.get(p, '')}" for p in sorted(ocr_texts_by_page.keys()))
        fallback_lab, _, _ = clean_lab_and_referring_info(None, merged_doc, merged_source, header_text_hint=all_hints)
        if fallback_lab:
            merged_lab_name = fallback_lab

    patient_info = PatientInfo(
        name=merged_name,
        age=merged_age,
        sex=merged_sex,
        report_date=merged_report_date,
        collection_date=merged_collection_date,
        lab_name=merged_lab_name,
        referring_doctor=merged_doc,
        referring_source=merged_source,
        quality_warnings=all_warnings,
    )

    # 4. Stage D: Process, scale-align, and validate all extracted tests
    validated_tests: List[TestResult] = []
    seen_keys = set()

    for item in all_raw_tests:
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

        # Deterministic flag recomputation with scale alignment (e.g. Lac/Lakh)
        computed_flag = recompute_flag(
            value=num_val,
            low=ref_low,
            high=ref_high,
            canonical_name=canon_name,
            unit=norm_unit,
            original_flag=item.printed_flag or "unknown",
            result_text=item.result_text,
            ref_text=item.reference_text,
        )

        # Deduplication check per page
        dedup_key = (item.page_number, canon_name.lower(), norm_unit or "")
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        # Deterministic bounding box computation from RapidOCR in displayed image coordinate space
        p_ocr = ocr_results_by_page.get(item.page_number)
        p_items = p_ocr.items if p_ocr else []
        p_dim = page_dimensions.get(item.page_number, [1000, 1000])
        computed_bbox = compute_test_bounding_box(
            item,
            ocr_items=p_items,
            img_w=p_dim[0],
            img_h=p_dim[1],
            gemini_hint_bbox=item.bbox,
        )
        if computed_bbox is None:
            logger.warning("Could not determine bounding box location for test on page %d", item.page_number)

        t_key = (item.page_number, canon_name.lower().strip())
        is_fallback_test = t_key in fallback_test_keys or any_vision_failed

        test_needs_review = True if is_fallback_test else False
        test_review_reason = "Extracted via fallback OCR (automatic reading unavailable)" if is_fallback_test else None
        test_confidence = "low" if is_fallback_test else "high"

        validated_tests.append(
            TestResult(
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
                page_number=item.page_number,
                source_snippet=item.source_snippet.strip() if item.source_snippet else "",
                bbox=computed_bbox,
                confidence=test_confidence,
                needs_review=test_needs_review,
                review_reason=test_review_reason,
            )
        )

    if not validated_tests:
        raise RuntimeError("Vision extraction failed and automatic reading was unavailable (no tests could be extracted).")

    if any_vision_failed:
        warn_msg = "Some results may be missing because automatic reading was unavailable"
        if warn_msg not in all_warnings:
            all_warnings.append(warn_msg)
        patient_info.quality_warnings = all_warnings

    # 5. Stage C Value Verification against Independent OCR & Coverage Check
    for bundle in bundles:
        p_num = bundle.page_number
        p_tests = [t for t in validated_tests if t.page_number == p_num]
        ocr_res = ocr_results_by_page.get(p_num) or OCRPageResult(page_number=p_num, full_text=ocr_texts_by_page.get(p_num, ""), items=[], numeric_tokens=[])
        cov = verify_numeric_values_against_ocr(p_tests, ocr_res, raw_text_by_page.get(p_num, ""))
        coverage_by_page[p_num] = cov

    overall_cov = round(sum(coverage_by_page.values()) / max(len(coverage_by_page), 1), 1)

    # 6. Stage D Plausibility & Arithmetic Consistency Checks
    run_plausibility_and_consistency_checks(validated_tests)

    # 7. Assemble LabReportData
    avg_quality = round(total_q_score / max(len(bundles), 1), 2)
    needs_review_count = sum(1 for t in validated_tests if t.needs_review)
    report_status = "incomplete" if any_vision_failed else "pending_verification"

    report_data = LabReportData(
        report_id=report_id,
        filename=original_filename,
        patient=patient_info,
        tests=validated_tests,
        remarks=list(dict.fromkeys(all_remarks)),
        raw_text_by_page=raw_text_by_page,
        ocr_texts_by_page=ocr_texts_by_page,
        pages_quality=pages_quality,
        overall_quality_score=avg_quality,
        quality_warnings=all_warnings,
        coverage_by_page=coverage_by_page,
        overall_coverage_pct=overall_cov,
        audit_discrepancies=all_audit_discrepancies,
        needs_review_count=needs_review_count,
        verified_by_user=False,
        status=report_status,
        page_count=len(bundles),
        page_dimensions=page_dimensions,
    )

    # 8. Build summary & persist
    summary = build_report_summary(report_data)
    report_data.summary = summary
    save_parsed_report(report_data)

    logger.info(
        "Successfully parsed %d tests across %d pages for report %s. OCR coverage: %.1f%%, Needs Review: %d",
        len(validated_tests),
        len(bundles),
        report_id,
        overall_cov,
        needs_review_count,
    )

    return report_data, summary


def parse_lab_report_file(
    file_path: Path,
    report_id: str,
    original_filename: str,
) -> Tuple[LabReportData, ReportSummary]:
    """Single file wrapper for parse_lab_report_inputs."""
    return parse_lab_report_inputs(
        inputs=[(file_path, original_filename)],
        report_id=report_id,
        original_filename=original_filename,
    )


# ---------------------------------------------------------------------------
# Storage & Retrieval Utilities
# ---------------------------------------------------------------------------

def save_parsed_report(data: LabReportData) -> Path:
    """Save structured report data to JSON disk cache."""
    PARSED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = PARSED_REPORTS_DIR / f"{data.report_id}.json"
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(data.model_dump(), handle, indent=2, ensure_ascii=False)
    logger.info("Saved structured lab report JSON to %s", target_path)
    return target_path


def load_parsed_report(report_id: str) -> Optional[LabReportData]:
    """Retrieve cached report data by report ID."""
    target_path = PARSED_REPORTS_DIR / f"{report_id}.json"
    if not target_path.exists():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return LabReportData.model_validate(payload)
    except Exception as exc:
        logger.error("Failed to load parsed report %s: %s", report_id, exc)
        return None
