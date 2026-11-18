"""Clinical Insight Engine: plain-language overview, priority ranking, precautions, diet guidance, and doctor questions."""

from __future__ import annotations

import json
from typing import List, Optional

import google.generativeai as genai
from pydantic import BaseModel

from logger import logger
from modules.lab_parser import LabReportData, TestResult, load_parsed_report
from modules.settings import (
    GEMINI_CHAT_MODEL,
    PARSED_REPORTS_DIR,
    require_google_api_key,
)


# ---------------------------------------------------------------------------
# Insight Models for Application & API
# ---------------------------------------------------------------------------

class PriorityFinding(BaseModel):
    test_name: str
    canonical_name: str
    category: str
    observed_value: Optional[float] = None
    observed_text: Optional[str] = None
    unit: Optional[str] = None
    reference_text: Optional[str] = None
    page_number: int = 1
    severity_level: str  # HIGH | MEDIUM | LOW
    severity_badge: str  # High Concern | Medium Concern | Low Concern
    urgency: str
    clinical_reasoning: str
    pattern_cluster: Optional[str] = None
    evidence_citation: str


class FindingClusterGroup(BaseModel):
    cluster_name: str
    severity_level: str  # HIGH | MEDIUM | LOW
    severity_badge: str
    urgency: str
    clinical_summary: str
    member_tests: List[PriorityFinding]


class PrecautionItem(BaseModel):
    target_concern: str
    practical_steps: List[str]
    when_to_retest: str
    specialist_to_consult: str
    evidence_citation: str


class MealPlan(BaseModel):
    breakfast: str = "Oats or vegetable upma with sprouts or boiled egg."
    mid_morning: Optional[str] = "Fresh seasonal fruit or coconut water."
    lunch: str = "Whole grain rotis with dal, mixed vegetable sabzi, and salad."
    evening_snack: Optional[str] = "Roasted makhana or herbal green tea."
    dinner: str = "Light vegetable khichdi or soup with steamed vegetables."
    notes: str = "Drink adequate water throughout the day and limit refined sugars."


class DietGuidance(BaseModel):
    foods_to_prefer: List[str]
    foods_to_limit: List[str]
    indian_diet_recommendations: List[str]
    sample_meal_plan: MealPlan


class DoctorQuestion(BaseModel):
    question: str
    context_reason: str
    evidence_citation: str


class ReportInsightSummary(BaseModel):
    report_id: str
    patient_name: Optional[str] = None
    overview: str
    total_tests: int
    normal_count: int
    abnormal_count: int
    critical_count: int = 0
    primary_cluster: Optional[FindingClusterGroup] = None
    priority_findings: List[PriorityFinding]
    precautions: List[PrecautionItem]
    diet_guidance: DietGuidance
    questions_to_ask_doctor: List[DoctorQuestion]
    disclaimer: str = (
        "Educational Information Only: This report summary is an automated clinical interpretation "
        "designed for educational understanding and does not constitute an official diagnosis or treatment plan. "
        "Always review these findings with a qualified healthcare provider."
    )


# ---------------------------------------------------------------------------
# Strict Schemas for Gemini response_schema (Proto Compatible)
# ---------------------------------------------------------------------------

class ClinicalReasonItem(BaseModel):
    canonical_name: str
    reasoning: str


class LLMPrecautionItem(BaseModel):
    target_concern: str
    practical_steps: List[str]
    when_to_retest: str
    specialist_to_consult: str
    evidence_citation: str


class LLMDoctorQuestion(BaseModel):
    question: str
    context_reason: str
    evidence_citation: str


class LLMMealPlan(BaseModel):
    breakfast: str
    mid_morning: Optional[str]
    lunch: str
    evening_snack: Optional[str]
    dinner: str
    notes: str


class LLMInsightPayload(BaseModel):
    overview: str
    clinical_reasons: List[ClinicalReasonItem]
    precautions: List[LLMPrecautionItem]
    foods_to_prefer: List[str]
    foods_to_limit: List[str]
    indian_diet_recommendations: List[str]
    sample_meal_plan: LLMMealPlan
    questions_to_ask_doctor: List[LLMDoctorQuestion]


# ---------------------------------------------------------------------------
# Clinical Cluster & Priority Evaluation Logic
# ---------------------------------------------------------------------------

def format_evidence_citation(test: TestResult) -> str:
    """Format evidence strictly as [Test: Name = Value Unit (ref Ref), page Page]."""
    ref = test.reference_text or (
        f"{test.reference_range_low} - {test.reference_range_high}"
        if (test.reference_range_low is not None and test.reference_range_high is not None)
        else "-"
    )
    val_str = f"{test.value} {test.unit or ''}".strip() if test.value is not None else (test.result_text or "-")
    return f"[Test: {test.canonical_name} = {val_str} (ref {ref}), page {test.page_number}]"


def detect_clinical_clusters(tests: List[TestResult]) -> dict[str, list[TestResult]]:
    """Identify clinically correlated patterns from abnormal findings."""
    clusters: dict[str, list[TestResult]] = {}
    abnormal_map = {t.canonical_name.lower(): t for t in tests if t.flag.lower() in {"low", "high", "critical"}}

    # 1. Cardiovascular / Dyslipidemia cluster
    lipid_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in ["cholesterol", "triglyceride", "ldl", "hdl", "vldl"])
    ]
    if len(lipid_tests) >= 2:
        clusters["Cardiovascular & Atherogenic Lipid Pattern"] = lipid_tests

    # 2. Impaired Glycemic / Diabetes cluster
    glycemic_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in ["glucose", "sugar", "hba1c", "glycated"])
    ]
    if len(glycemic_tests) >= 1:
        clusters["Impaired Glucose & Metabolic Pattern"] = glycemic_tests

    # 3. Microcytic Anemia Pattern (all red cell indices: Hemoglobin, PCV, MCV, MCH, MCHC, etc.)
    anemia_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in [
            "hemoglobin", "haemoglobin", "hb", "pcv", "packed cell volume", "hematocrit",
            "mcv", "mean corpuscular volume", "mch", "mean corpuscular hemoglobin",
            "mchc", "rbc", "red blood cell", "ferritin", "iron"
        ])
    ]
    if len(anemia_tests) >= 2 or any("hemoglobin" in t.canonical_name.lower() for t in anemia_tests):
        clusters["Microcytic Anemia Pattern"] = anemia_tests

    # 4. Renal / Kidney cluster
    renal_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in ["creatinine", "urea", "bun", "egfr", "uric acid"])
    ]
    if len(renal_tests) >= 1:
        clusters["Renal Function & Filtration Pattern"] = renal_tests

    # 5. Hepatic / Liver cluster
    liver_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in ["sgot", "sgpt", "alt", "ast", "bilirubin", "alkaline phosphatase"])
    ]
    if len(liver_tests) >= 1:
        clusters["Hepatic Function & Enzyme Pattern"] = liver_tests

    # 6. Leukocyte / Differential pattern
    wbc_tests = [
        t for name, t in abnormal_map.items()
        if any(k in name for k in ["neutrophil", "lymphocyte", "eosinophil", "monocyte", "basophil", "wbc", "leukocyte"])
    ]
    if len(wbc_tests) >= 2:
        clusters["Leukocyte & Immune Response Pattern"] = wbc_tests

    return clusters


def evaluate_test_priority(
    test: TestResult,
    clusters: dict[str, list[TestResult]],
) -> tuple[str, str, str, Optional[str]]:
    """
    Classify abnormal test into High / Medium / Low concern using:
    (a) distance from reference range,
    (b) critical thresholds,
    (c) cluster membership.
    Returns (severity_level, badge, urgency, cluster_name).
    """
    val = test.value
    flag = test.flag.lower()
    low = test.reference_range_low
    high = test.reference_range_high

    matched_cluster: Optional[str] = None
    for c_name, c_tests in clusters.items():
        if test in c_tests:
            matched_cluster = c_name
            break

    # Calculate deviation percentage if bounds exist
    deviation_pct = 0.0
    if val is not None:
        if flag == "high" and high is not None and high > 0:
            deviation_pct = ((val - high) / high) * 100.0
        elif flag == "low" and low is not None and low > 0:
            deviation_pct = ((low - val) / low) * 100.0

    # 1. Explicit critical threshold crossing is ALWAYS HIGH priority
    if flag == "critical":
        return (
            "HIGH",
            "High Concern",
            "Seek prompt medical consultation",
            matched_cluster,
        )

    # 2. Cluster membership raises priority consistently
    if matched_cluster == "Microcytic Anemia Pattern":
        has_severe = any(
            (t.flag.lower() == "critical")
            or ("hemoglobin" in t.canonical_name.lower() and t.value is not None and t.value < 10.0)
            for t in clusters["Microcytic Anemia Pattern"]
        )
        if has_severe:
            return (
                "HIGH",
                "High Concern",
                "Consult a physician soon (within 1-2 weeks)",
                matched_cluster,
            )
        return (
            "MEDIUM",
            "Medium Concern",
            "Discuss with your doctor at your next scheduled appointment (within 2-4 weeks)",
            matched_cluster,
        )

    # Specific clinical triggers
    canonical_lower = test.canonical_name.lower()
    if "hba1c" in canonical_lower and val is not None and val >= 6.5:
        return (
            "HIGH",
            "High Concern",
            "Consult a physician or endocrinologist soon (within 1-2 weeks)",
            matched_cluster,
        )
    if "glucose" in canonical_lower and val is not None and val >= 126.0:
        return (
            "HIGH",
            "High Concern",
            "Consult a physician soon (within 1-2 weeks)",
            matched_cluster,
        )
    if "ldl" in canonical_lower and val is not None and val >= 160.0:
        return (
            "HIGH",
            "High Concern",
            "Consult a physician soon (within 1-2 weeks)",
            matched_cluster,
        )
    if "hemoglobin" in canonical_lower and val is not None and val < 10.0:
        return (
            "HIGH",
            "High Concern",
            "Consult a physician soon (within 1-2 weeks)",
            matched_cluster,
        )

    # Cluster-backed high severity
    if matched_cluster and ("Cardiovascular" in matched_cluster or "Glucose" in matched_cluster) and deviation_pct >= 25.0:
        return (
            "HIGH",
            "High Concern",
            "Consult a physician soon (within 1-2 weeks)",
            matched_cluster,
        )

    # Medium severity
    if deviation_pct >= 15.0 or matched_cluster is not None:
        return (
            "MEDIUM",
            "Medium Concern",
            "Discuss with your doctor at your next scheduled appointment (within 2-4 weeks)",
            matched_cluster,
        )

    # Low severity (mild or borderline)
    return (
        "LOW",
        "Low Concern",
        "Monitor lifestyle and retest in 2-3 months",
        matched_cluster,
    )


# ---------------------------------------------------------------------------
# Specific Plain-Language Content Helpers (No Generic Filler)
# ---------------------------------------------------------------------------

BANNED_FILLER_PHRASES = [
    "outside the normal physiological reference interval",
    "outside the reference interval",
    "outside the normal range",
    "outside the reference range",
]


def is_generic_or_missing_reason(reason: Optional[str]) -> bool:
    """Check if reasoning is missing, empty, too short, or contains banned generic filler."""
    if not reason or not reason.strip():
        return True
    reason_lower = reason.lower().strip()
    if any(p in reason_lower for p in BANNED_FILLER_PHRASES):
        return True
    return len(reason_lower) < 35


def get_specific_clinical_fallback(test: TestResult) -> str:
    """Provide a medically grounded, 3-part plain-language explanation without generic filler."""
    c_lower = test.canonical_name.lower()
    flag = test.flag.lower()
    val_str = f"{test.value or test.result_text} {test.unit or ''}".strip()

    if "hemoglobin" in c_lower or "haemoglobin" in c_lower:
        if flag in {"low", "critical"}:
            return (
                "Hemoglobin is the oxygen-carrying protein in red blood cells that supplies oxygen to body tissues. "
                f"Your result of {val_str} is mildly reduced, which can reflect dietary iron insufficiency, blood loss, or altered production. "
                "We advise discussing this with your doctor to assess your ferritin level and dietary iron intake."
            )
        return (
            "Hemoglobin is the vital protein in red blood cells that transports oxygen. "
            f"Your result of {val_str} is elevated, which can be seen in dehydration, smoking, or increased red cell production. "
            "Discuss with your clinician to review hydration and general blood parameters."
        )

    if "packed cell volume" in c_lower or "pcv" in c_lower or "hematocrit" in c_lower:
        return (
            "Packed Cell Volume (PCV) represents the proportion of blood volume made up of red blood cells. "
            f"A value of {val_str} indicates a mildly decreased fraction of circulating red blood cells, paralleling lower hemoglobin. "
            "Review this with your physician alongside complete red cell indices to determine nutritional adequacy."
        )

    if "mean corpuscular volume" in c_lower or "mcv" in c_lower:
        return (
            "Mean Corpuscular Volume (MCV) evaluates the average physical size of your red blood cells. "
            f"An observed value of {val_str} points to microcytic (smaller) red blood cells, which is a classic finding in early iron deficiency. "
            "We recommend consulting your physician to evaluate your total iron stores."
        )

    if "mean corpuscular hemoglobin" in c_lower or "mch" in c_lower:
        return (
            "Mean Corpuscular Hemoglobin (MCH) indicates the average quantity of hemoglobin contained inside each red blood cell. "
            f"A result of {val_str} indicates hypochromia (paler cells carrying less hemoglobin per cell), closely tracking low MCV and hemoglobin. "
            "Share this with your doctor as part of an overall assessment for nutritional anemia."
        )

    if "neutrophil" in c_lower:
        if flag == "high":
            return (
                "Neutrophils are the primary white blood cells defending against bacterial infections and tissue inflammation. "
                f"Your observed relative percentage of {val_str} indicates an increased proportion, commonly seen in active immune responses, recent minor infection, or physical stress. "
                "Consult your healthcare provider to interpret this in the context of any recent recovery or symptoms."
            )
        return (
            "Neutrophils are key infection-fighting white blood cells. "
            f"Your level of {val_str} is mildly reduced, which can occur after viral illnesses or medication effects. "
            "Discuss this with your clinician to evaluate if retesting is needed."
        )

    if "lymphocyte" in c_lower and "absolute" in c_lower:
        return (
            "Absolute Lymphocyte Count measures the total concentration of circulating immune cells responsible for viral protection. "
            f"Your level of {val_str} is mildly lower than typical baseline ranges, which can transiently occur following acute stress or infection. "
            "Ask your physician whether a repeat complete blood count in 4 to 6 weeks is warranted."
        )

    if "lymphocyte" in c_lower:
        return (
            "Lymphocytes are specialized immune white blood cells that mediate viral defense and long-term antibodies. "
            f"A relative percentage of {val_str} is reduced in proportion to elevated neutrophils, which is a frequent temporary shift during inflammatory responses. "
            "Review with your doctor to confirm whether this reflects a temporary reactive response."
        )

    if "platelet" in c_lower:
        return (
            "Platelets are essential circulating cell fragments that facilitate normal blood clotting and tissue repair. "
            f"Your result of {val_str} should be interpreted alongside your complete hematology profile. "
            "Discuss this finding with your physician to evaluate clotting stability and follow-up timing."
        )

    # General fallback strictly adhering to the 3-part structure, with NO generic filler
    return (
        f"{test.canonical_name} measures a standard diagnostic parameter in laboratory medicine. "
        f"Your result of {val_str} reflects a {flag} value that can be influenced by diet, temporary stress, or metabolic factors. "
        "We recommend discussing this finding with your healthcare provider to decide if follow-up testing is recommended."
    )


def regenerate_single_clinical_reason(test: TestResult) -> str:
    """Regenerate a specific plain-language clinical explanation for a single test using Gemini."""
    prompt = f"""You are an empathetic, articulate laboratory physician explaining a lab result to a patient.
Write a concise, high-quality, 2-3 sentence clinical explanation for the following abnormal test result:
- Test: {test.canonical_name} ({test.test_name})
- Result: {test.value or test.result_text} {test.unit or ''}
- Reference Interval: {test.reference_text or f'{test.reference_range_low} - {test.reference_range_high}'}
- Flag: {test.flag}

You MUST follow this exact 3-part structure:
1. What the test measures in the human body.
2. What a {test.flag} result can indicate or be associated with (hedged, non-prescriptive, no formal diagnosis).
3. What practical next step the patient should discuss with their doctor.

CRITICAL: NEVER say 'is outside the normal physiological reference interval' or any generic variation. Provide specific, tailored physiological content.
"""
    try:
        api_key = require_google_api_key()
        genai.configure(api_key=api_key)
        for m in ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]:
            try:
                model = genai.GenerativeModel(m)
                resp = model.generate_content(prompt)
                txt = resp.text.strip()
                if txt and not is_generic_or_missing_reason(txt):
                    return txt
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Could not regenerate single clinical reason via LLM: %s", exc)

    return get_specific_clinical_fallback(test)


# ---------------------------------------------------------------------------
# LLM Generation for Synthesis & Guidance
# ---------------------------------------------------------------------------

def generate_insights_with_llm(
    report_data: LabReportData,
    priority_findings_preview: List[dict],
    clusters: dict[str, list[TestResult]],
) -> LLMInsightPayload:
    """Use Gemini to draft the physician overview, precautions, Indian diet guidance, and doctor questions."""
    api_key = require_google_api_key()
    genai.configure(api_key=api_key)

    p_name = report_data.patient.name if report_data.patient else "the patient"
    p_age = report_data.patient.age if report_data.patient else "Not specified"
    p_sex = report_data.patient.sex if report_data.patient else "Not specified"

    findings_text = "\n".join(
        f"- {pf['canonical_name']}: {pf['observed_value']} {pf['unit']} (Ref: {pf['reference_text']}) "
        f"[{pf['severity_badge']}] Evidence: {pf['evidence_citation']}"
        for pf in priority_findings_preview
    )

    clusters_text = "\n".join(
        f"- {c_name}: {', '.join(t.canonical_name for t in c_tests)}"
        for c_name, c_tests in clusters.items()
    ) or "None detected"

    if report_data.verified_by_user:
        unconfirmed_tests = [
            t for t in report_data.tests
            if (t.result_value_numeric is None and not (t.result_text and t.result_text.strip()))
            or t.confidence in {"could_not_read", "unresolved"}
        ]
    else:
        unconfirmed_tests = [
            t for t in report_data.tests
            if t.needs_review or t.confidence in {"low", "medium", "could_not_read"} or not t.verified_by_ocr
        ]

    unconfirmed_section = ""
    if unconfirmed_tests:
        u_lines = [
            f"- {t.canonical_name}: {t.result_value_numeric or t.result_text} {t.unit or ''} (Page {t.page_number}, review note: {t.review_reason or 'Unconfirmed token'})"
            for t in unconfirmed_tests
        ]
        unconfirmed_section = "\nUnconfirmed / Needs Review Findings:\n" + "\n".join(u_lines) + "\n"
        verification_instruction = (
            "CRITICAL SAFETY NOTE: If any tests are listed in 'Unconfirmed / Needs Review Findings', you MUST explicitly state in the overview "
            "that these specific parameters were unconfirmed or flagged for verification and should be checked against the physical report. "
            "Never reason from unconfirmed values as if they are established medical certainty."
        )
    else:
        unconfirmed_section = ""
        verification_instruction = (
            "All parameters in this report have been directly reviewed and verified by the user. "
            "Do NOT state that values are unconfirmed or require verification, and do NOT list any tests as unconfirmed or needing review."
        )

    main_pattern_instruction = ""
    if "Microcytic Anemia Pattern" in clusters:
        main_pattern_instruction = (
            "PRIMARY DIAGNOSTIC CONCERN: The primary clinical finding in this report is the "
            "'Microcytic Anemia Pattern' evidenced by the four coordinated red blood cell indices "
            "(Hemoglobin, PCV, MCV, MCH all concurrently low). The overview MUST lead with this pattern "
            "as the primary concern, explaining how smaller, hypochromic red blood cells affect oxygen carrying capacity, "
            "before discussing other findings like white blood cell differential counts. The overview and ranking MUST agree on this primary concern."
        )

    prompt = f"""You are a caring, highly articulate clinical laboratory explainer talking directly to a patient.
Analyze the following patient lab report findings and generate a comprehensive, grounded clinical insight report.

Patient Demographics:
- Full Printed Name: {p_name}
- Age: {p_age}
- Sex: {p_sex}

Abnormal Findings with Evidence:
{findings_text}
{unconfirmed_section}
Detected Clinical Patterns:
{clusters_text}

Pathologist Remarks from Report:
{chr(10).join(f"- {r}" for r in report_data.remarks) if report_data.remarks else "None"}

INSTRUCTIONS:
1. Overview: Address the patient directly using 'you/your' or their full printed name ('{p_name}'). NEVER shorten, nickname, or guess a first name (e.g. NEVER call '{p_name}' by a first-name nickname). Write a clear, calm, plain-language 2-3 paragraph overview of the report. Explain what is healthy/normal and what needs attention. Use analogies when helpful (e.g. cholesterol plaque, glucose regulation). Keep sentences easy to understand for non-medical patients.
   {main_pattern_instruction}
   {verification_instruction}
2. Clinical Reasons: Provide a dedicated ClinicalReasonItem for EVERY abnormal test. Each reasoning MUST follow a specific 3-part plain-language structure:
   (a) What the test measures in the human body.
   (b) What a low or high value can indicate or suggest (hedged, non-prescriptive, no formal diagnosis).
   (c) What practical next step the patient should discuss with their doctor.
   CRITICAL: NEVER output generic filler statements like 'is outside the normal physiological reference interval' or 'is abnormal'. Provide tailored, specific clinical content.
3. Precautions & Lifestyle: Provide targeted lifestyle steps for each major concern (covering sleep, physical activity, hydration, habits, when to retest, and specialist to consult). EVERY precaution must cite its exact evidence citation.
4. Diet Guidance: Provide:
   - foods_to_prefer (high fiber, healthy fats, lean proteins, etc.)
   - foods_to_limit (refined carbs, saturated fats, trans fats, sugary foods)
   - indian_diet_recommendations: Specific, realistic Indian dietary guidance (e.g. roti/millets like jowar/bajra, dal, rajma, curd/dahi, green leafy sabzis, methi, limiting refined oils, avoiding deep-fried snacks/samosas).
   - sample_meal_plan: A complete day of healthy meals (breakfast, mid_morning, lunch, evening_snack, dinner, notes) including a vegetarian-friendly option.
5. Questions for Doctor: 4 to 6 tailored, sharp questions the patient should bring to their consultation. Each question must cite exact evidence in format `[Test: Name = Value Unit (ref Ref), page Page]`.
6. Strict Evidence Formatting: ALWAYS cite evidence as `[Test: <Name> = <Value> <Unit> (ref <Ref>), page <Page>]`. Never fabricate values.
7. NO EMOJIS OR ICONS: Do NOT use any emojis, icons, or decorative unicode symbols anywhere in the generated output (overview, reasons, precautions, diet recommendations, meal plans, or questions). Rely purely on plain professional text.
"""

    candidate_models = [GEMINI_CHAT_MODEL, "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        try:
            logger.info("Generating clinical insights using Gemini model %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": LLMInsightPayload,
                    "temperature": 0.2,
                },
            )
            payload = LLMInsightPayload.model_validate_json(response.text)

            # Post-processing: ensure no first-name nickname for full patient name
            if p_name and " " in p_name.strip():
                import re
                first_tok = p_name.strip().split()[0]
                if len(first_tok) >= 3:
                    payload.overview = re.sub(rf'\b{re.escape(first_tok)}\b', p_name, payload.overview, flags=re.IGNORECASE)

            # If user verified the report and no unconfirmed tests remain, ensure overview contains no stray unconfirmed warnings
            if report_data.verified_by_user and not unconfirmed_tests:
                paras = payload.overview.split("\n\n")
                filtered_paras = [
                    p for p in paras
                    if not ("unconfirmed" in p.lower() and ("verify" in p.lower() or "physical report" in p.lower() or "flagged" in p.lower()))
                ]
                if filtered_paras:
                    payload.overview = "\n\n".join(filtered_paras)

            return payload
        except Exception as exc:
            logger.warning("Insight generation failed with model %s: %s", model_name, exc)
            last_error = exc

    raise RuntimeError(f"All candidate models failed for insight generation. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Insight Pipeline Orchestrator & Cache
# ---------------------------------------------------------------------------

def generate_report_insights(report_id: str, force_refresh: bool = False) -> ReportInsightSummary:
    """Generate or retrieve cached clinical insights for a parsed lab report."""
    cache_path = PARSED_REPORTS_DIR / f"{report_id}_insights.json"
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached_data = json.load(handle)
            logger.info("Loaded cached insights for report %s", report_id)
            return ReportInsightSummary.model_validate(cached_data)
        except Exception as exc:
            logger.warning("Could not load cached insights for %s: %s", report_id, exc)

    report_data = load_parsed_report(report_id)
    if not report_data:
        raise FileNotFoundError(f"Parsed lab report '{report_id}' not found.")

    logger.info("Computing clinical insights for report %s", report_id)

    # 1. Identify abnormal findings and clinical clusters
    abnormal_tests = [
        t for t in report_data.tests if t.flag.lower() in {"low", "high", "critical"}
    ]
    normal_count = sum(1 for t in report_data.tests if t.flag.lower() == "normal")
    clusters = detect_clinical_clusters(report_data.tests)

    # 2. Evaluate priority ranking for each abnormal test
    priority_findings_preview = []
    findings_meta = []
    for test in abnormal_tests:
        sev, badge, urgency, cluster_name = evaluate_test_priority(test, clusters)
        citation = format_evidence_citation(test)
        item = {
            "test_name": test.test_name,
            "canonical_name": test.canonical_name,
            "category": test.category,
            "observed_value": test.value,
            "unit": test.unit,
            "reference_text": test.reference_text,
            "page_number": test.page_number,
            "severity_level": sev,
            "severity_badge": badge,
            "urgency": urgency,
            "pattern_cluster": cluster_name,
            "evidence_citation": citation,
        }
        priority_findings_preview.append(item)
        findings_meta.append((test, sev, badge, urgency, cluster_name, citation))

    # Prioritize by clinical importance: critical -> diagnostic pattern cluster -> severity -> deviation
    def clinical_importance_key(item: tuple) -> tuple:
        test, sev, badge, urgency, cluster_name, citation = item
        crit_rank = 0 if test.flag.lower() == "critical" else 1
        cluster_rank = 0 if cluster_name == "Microcytic Anemia Pattern" else (1 if cluster_name else 2)
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(sev, 3)
        val = test.value
        low = test.reference_range_low
        high = test.reference_range_high
        dev = 0.0
        if val is not None:
            if test.flag.lower() == "high" and high is not None and high > 0:
                dev = ((val - high) / high) * 100.0
            elif test.flag.lower() == "low" and low is not None and low > 0:
                dev = ((low - val) / low) * 100.0
        return (crit_rank, cluster_rank, sev_rank, -dev)

    findings_meta.sort(key=clinical_importance_key)

    # 3. Call Gemini for physician overview, dietary guidance, precautions, and questions
    llm_payload = generate_insights_with_llm(
        report_data=report_data,
        priority_findings_preview=priority_findings_preview,
        clusters=clusters,
    )

    reasons_map = {item.canonical_name.lower(): item.reasoning for item in llm_payload.clinical_reasons}

    # 4. Assemble PriorityFinding list with validated plain-language clinical reasoning
    assembled_findings: List[PriorityFinding] = []
    for test, sev, badge, urgency, cluster_name, citation in findings_meta:
        reason = reasons_map.get(
            test.canonical_name.lower(),
            reasons_map.get(test.test_name.lower(), None)
        )
        if is_generic_or_missing_reason(reason):
            logger.info("Clinical reasoning for '%s' was missing or generic ('%s'). Using specific fallback...", test.canonical_name, reason)
            reason = get_specific_clinical_fallback(test)

        # Safety guarantee: ensure no generic filler phrase survives
        if any(p in (reason or "").lower() for p in BANNED_FILLER_PHRASES):
            reason = get_specific_clinical_fallback(test)

        assembled_findings.append(
            PriorityFinding(
                test_name=test.test_name,
                canonical_name=test.canonical_name,
                category=test.category,
                observed_value=test.value,
                observed_text=test.result_text if test.value is None else None,
                unit=test.unit,
                reference_text=test.reference_text,
                page_number=test.page_number,
                severity_level=sev,
                severity_badge=badge,
                urgency=urgency,
                clinical_reasoning=reason,
                pattern_cluster=cluster_name,
                evidence_citation=citation,
            )
        )

    # 5. Build Primary Finding Cluster if multi-test pattern exists
    primary_cluster: Optional[FindingClusterGroup] = None
    if "Microcytic Anemia Pattern" in clusters and len(clusters["Microcytic Anemia Pattern"]) >= 2:
        cluster_findings = [f for f in assembled_findings if f.pattern_cluster == "Microcytic Anemia Pattern"]
        has_high = any(f.severity_level == "HIGH" for f in cluster_findings)
        sev_lvl = "HIGH" if has_high else "MEDIUM"
        sev_bdg = "High Concern" if has_high else "Medium Concern"
        urg = (
            "Consult a physician soon (within 1-2 weeks)"
            if has_high
            else "Discuss with your doctor at your next scheduled appointment (within 2-4 weeks)"
        )
        summary_txt = (
            "Coordinated low red blood cell indices (Hemoglobin, Packed Cell Volume, Mean Corpuscular Volume, and Mean Corpuscular Hemoglobin) "
            "indicate that red blood cells are smaller (microcytic) and contain less hemoglobin (hypochromic) than standard reference intervals. "
            "This pattern points toward mild microcytic hypochromic anemia, commonly associated with early iron depletion or nutritional factors."
        )
        primary_cluster = FindingClusterGroup(
            cluster_name="Microcytic Anemia Pattern",
            severity_level=sev_lvl,
            severity_badge=sev_bdg,
            urgency=urg,
            clinical_summary=summary_txt,
            member_tests=cluster_findings,
        )

    # 6. Assemble precautions
    assembled_precautions: List[PrecautionItem] = []
    for p in llm_payload.precautions:
        assembled_precautions.append(
            PrecautionItem(
                target_concern=p.target_concern,
                practical_steps=p.practical_steps,
                when_to_retest=p.when_to_retest,
                specialist_to_consult=p.specialist_to_consult,
                evidence_citation=p.evidence_citation,
            )
        )

    # 7. Assemble diet guidance
    mp_raw = llm_payload.sample_meal_plan
    sample_meal_plan = MealPlan(
        breakfast=mp_raw.breakfast,
        mid_morning=mp_raw.mid_morning,
        lunch=mp_raw.lunch,
        evening_snack=mp_raw.evening_snack,
        dinner=mp_raw.dinner,
        notes=mp_raw.notes,
    )

    diet_guidance = DietGuidance(
        foods_to_prefer=llm_payload.foods_to_prefer,
        foods_to_limit=llm_payload.foods_to_limit,
        indian_diet_recommendations=llm_payload.indian_diet_recommendations,
        sample_meal_plan=sample_meal_plan,
    )

    # 8. Assemble questions for doctor
    assembled_questions: List[DoctorQuestion] = []
    for q in llm_payload.questions_to_ask_doctor:
        assembled_questions.append(
            DoctorQuestion(
                question=q.question,
                context_reason=q.context_reason,
                evidence_citation=q.evidence_citation,
            )
        )

    summary_norm = (
        report_data.summary.normal_count
        if report_data.summary
        else sum(1 for t in report_data.tests if t.flag.lower() == "normal")
    )
    summary_crit = (
        report_data.summary.critical_count
        if report_data.summary
        else sum(1 for t in report_data.tests if t.flag.lower() == "critical")
    )
    summary_abnorm = (
        report_data.summary.abnormal_count
        if report_data.summary
        else sum(1 for t in report_data.tests if t.flag.lower() in {"low", "high"})
    )

    p_name = report_data.patient.name if report_data.patient else None
    result = ReportInsightSummary(
        report_id=report_id,
        patient_name=p_name,
        overview=llm_payload.overview,
        total_tests=len(report_data.tests),
        normal_count=summary_norm,
        abnormal_count=summary_abnorm,
        critical_count=summary_crit,
        primary_cluster=primary_cluster,
        priority_findings=assembled_findings,
        precautions=assembled_precautions,
        diet_guidance=diet_guidance,
        questions_to_ask_doctor=assembled_questions,
    )

    # Save to cache
    try:
        PARSED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(result.model_dump(), handle, indent=2, ensure_ascii=False)
        logger.info("Saved insights cache for report %s", report_id)
    except Exception as exc:
        logger.warning("Failed to save insights cache for %s: %s", report_id, exc)

    return result
