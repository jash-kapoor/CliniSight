"""Unit tests for Stage A (Input handling & preprocessing) and Stage B (Vision-first extraction & normalization)."""

import sys
import unittest
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(server_dir))

from modules.image_preprocessor import (
    PageQualityInfo,
    assess_image_quality,
    ingest_report_pages,
)
from modules.lab_parser import (
    clean_lab_and_referring_info,
    clean_referring_info,
    disambiguate_canonical_name,
    normalize_unit,
    parse_age_sex,
    parse_numeric_value,
    parse_reference_range,
    recompute_flag,
)
from PIL import Image


class TestStageAandB(unittest.TestCase):

    def test_indian_number_parsing(self):
        self.assertEqual(parse_numeric_value("1,75,000"), 175000.0)
        self.assertEqual(parse_numeric_value("1,75,000.50"), 175000.5)
        self.assertEqual(parse_numeric_value("00"), 0.0)
        self.assertEqual(parse_numeric_value("08"), 8.0)
        self.assertEqual(parse_numeric_value("< 6.0"), 6.0)
        self.assertEqual(parse_numeric_value("> 50"), 50.0)
        self.assertIsNone(parse_numeric_value(None))
        self.assertIsNone(parse_numeric_value("ADEQUATE"))

    def test_referring_source_vs_doctor(self):
        # Hospital should be detected as referring_source, NOT referring_doctor
        doc, src = clean_referring_info(raw_doc="TEST REFERRAL HOSPITAL", raw_source=None)
        self.assertIsNone(doc)
        self.assertEqual(src, "TEST REFERRAL HOSPITAL")

        # Real physician should be kept as referring_doctor
        doc2, src2 = clean_referring_info(raw_doc="Dr. Rajesh Patel", raw_source="Apex Clinic")
        self.assertEqual(doc2, "Dr. Rajesh Patel")
        self.assertEqual(src2, "Apex Clinic")

    def test_lab_name_never_copies_referring_source(self):
        # When lab_name was erroneously set to referring_source, it must NEVER be copied into lab_name
        lab, doc, src = clean_lab_and_referring_info(
            raw_lab="TEST REFERRAL HOSPITAL",
            raw_doc=None,
            raw_source="TEST REFERRAL HOSPITAL",
            header_text_hint="",
        )
        self.assertIsNone(lab)
        self.assertEqual(src, "TEST REFERRAL HOSPITAL")

        # When header text contains letterhead / logo info for Test Pathology Laboratory, extract actual lab
        hint = "TEST PATHOLOGY LABORATORY\nAccurate | Caring | Instant\nRef By: TEST REFERRAL HOSPITAL"
        lab2, doc2, src2 = clean_lab_and_referring_info(
            raw_lab="TEST REFERRAL HOSPITAL",
            raw_doc=None,
            raw_source="TEST REFERRAL HOSPITAL",
            header_text_hint=hint,
        )
        self.assertEqual(lab2, "Test Pathology Laboratory")
        self.assertEqual(src2, "TEST REFERRAL HOSPITAL")

        # When lab is a genuine separate lab name, keep it
        lab3, doc3, src3 = clean_lab_and_referring_info(
            raw_lab="Metropolis Diagnostics",
            raw_doc="Dr. Anil Sharma",
            raw_source="Apollo Hospital",
            header_text_hint="",
        )
        self.assertEqual(lab3, "Metropolis Diagnostics")
        self.assertEqual(doc3, "Dr. Anil Sharma")
        self.assertEqual(src3, "Apollo Hospital")

    def test_age_sex_parsing(self):
        age, sex = parse_age_sex("20 Yrs./M", None)
        self.assertEqual(age, 20.0)
        self.assertEqual(sex, "Male")

        age2, sex2 = parse_age_sex(45, "F")
        self.assertEqual(age2, 45.0)
        self.assertEqual(sex2, "Female")

    def test_disambiguate_duplicate_tests(self):
        # Differential % vs Absolute count
        name_pct = disambiguate_canonical_name("Neutrophils", unit="%")
        self.assertEqual(name_pct, "Neutrophils (Percentage)")

        name_abs = disambiguate_canonical_name("Neutrophils", unit="/uL")
        self.assertEqual(name_abs, "Absolute Neutrophil Count")

        name_lymph_pct = disambiguate_canonical_name("Lymphocytes", unit="%")
        self.assertEqual(name_lymph_pct, "Lymphocytes (Percentage)")

        name_lymph_abs = disambiguate_canonical_name("Lymphocytes", unit="/µL")
        self.assertEqual(name_lymph_abs, "Absolute Lymphocyte Count")

    def test_qualitative_flag_recomputation(self):
        # Normal qualitative results
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="ADEQUATE"), "normal")
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="M.P. NOT DETECTED"), "normal")
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="TEST IS NEGATIVE"), "normal")
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="NON-REACTIVE"), "normal")

        # Abnormal qualitative results
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="POSITIVE"), "high")
        self.assertEqual(recompute_flag(value=None, low=None, high=None, result_text="REACTIVE"), "high")

    def test_reference_range_with_leading_zeros_and_lakhs(self):
        low, high = parse_reference_range("00 - 06 %")
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 6.0)

        low2, high2 = parse_reference_range("1.5 - 4.0 Lac/cmm.")
        self.assertEqual(low2, 1.5)
        self.assertEqual(high2, 4.0)

        low3, high3 = parse_reference_range("< 6.0 mg/L")
        self.assertIsNone(low3)
        self.assertEqual(high3, 6.0)

    def test_explicit_critical_thresholds_and_marginal_deviation(self):
        # MCH 26.69 pg (ref 27 - 31) is marginally low, NOT critical (only ~1% below lower bound)
        flag_mch = recompute_flag(26.69, low=27.0, high=31.0, canonical_name="Mean Corpuscular Hemoglobin", unit="pg")
        self.assertEqual(flag_mch, "low")

        # Hemoglobin 13.4 (ref 13.5 - 17.5) -> low, NOT critical
        flag_hb_low = recompute_flag(13.4, low=13.5, high=17.5, canonical_name="Hemoglobin", unit="g/dL")
        self.assertEqual(flag_hb_low, "low")

        # Hemoglobin 6.5 (ref 13.5 - 17.5) -> critical (< 7.0 g/dL is explicit clinical panic value)
        flag_hb_crit = recompute_flag(6.5, low=13.5, high=17.5, canonical_name="Hemoglobin", unit="g/dL")
        self.assertEqual(flag_hb_crit, "critical")

        # Potassium 6.5 (ref 3.5 - 5.5) -> critical (> 6.2 mEq/L is explicit clinical panic value)
        flag_k_crit = recompute_flag(6.5, low=3.5, high=5.5, canonical_name="Potassium", unit="mEq/L")
        self.assertEqual(flag_k_crit, "critical")

        # Potassium 3.81 (ref 3.5 - 5.5) -> normal
        flag_k_norm = recompute_flag(3.81, low=3.5, high=5.5, canonical_name="Potassium", unit="mEq/L")
        self.assertEqual(flag_k_norm, "normal")

    def test_platelet_lac_conversion_scale_alignment(self):
        # Platelets 1,75,000 /cmm vs 1.5 - 4.0 Lac/cmm must be NORMAL
        flag_plt = recompute_flag(
            value=175000.0,
            low=1.5,
            high=4.0,
            canonical_name="Platelet Count",
            unit="/cmm",
            ref_text="1.5 - 4.0 Lac/cmm.",
        )
        self.assertEqual(flag_plt, "normal")

        # Platelets 35,000 /cmm vs 1.5 - 4.0 Lac/cmm must be CRITICAL (< 50,000)
        flag_plt_crit = recompute_flag(
            value=35000.0,
            low=1.5,
            high=4.0,
            canonical_name="Platelet Count",
            unit="/cmm",
            ref_text="1.5 - 4.0 Lac/cmm.",
        )
        self.assertEqual(flag_plt_crit, "critical")

    def test_stage_a_ingest_report_pdf(self):
        report_pdf = Path(__file__).resolve().parent.parent / "Report.pdf"
        if report_pdf.exists():
            bundles = ingest_report_pages([(report_pdf, "Report.pdf")])
            self.assertEqual(len(bundles), 2)
            for b in bundles:
                self.assertIsNotNone(b.original_image)
                self.assertIsNotNone(b.cleaned_image)
                self.assertGreaterEqual(b.quality.quality_score, 0.0)
                self.assertLessEqual(b.quality.quality_score, 1.0)
                self.assertGreater(b.quality.blur_score, 0.0)


if __name__ == "__main__":
    unittest.main()
