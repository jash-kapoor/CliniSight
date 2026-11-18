"""Unit tests for Phase 2: lab report parser, unit normalization, and flag recomputation."""

import unittest
from pathlib import Path
import sys

# Add server directory to path
server_dir = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(server_dir))

from modules.lab_parser import (
    TestResult,
    build_report_summary,
    is_critical_value,
    normalize_unit,
    parse_reference_range,
    recompute_flag,
    LabReportData,
)


class TestLabParserValidation(unittest.TestCase):

    def test_unit_normalization(self):
        self.assertEqual(normalize_unit("mg/dl"), "mg/dL")
        self.assertEqual(normalize_unit("MG/DL."), "mg/dL")
        self.assertEqual(normalize_unit("gm/dl"), "g/dL")
        self.assertEqual(normalize_unit("u/l"), "U/L")
        self.assertEqual(normalize_unit("iu/l"), "IU/L")
        self.assertEqual(normalize_unit("mcg/dl"), "µg/dL")
        self.assertEqual(normalize_unit("cells/cumm"), "cells/cu.mm")
        self.assertEqual(normalize_unit(""), None)
        self.assertEqual(normalize_unit(None), None)

    def test_parse_reference_range(self):
        self.assertEqual(parse_reference_range("70 - 99 mg/dL"), (70.0, 99.0))
        self.assertEqual(parse_reference_range("12.0 – 15.5"), (12.0, 15.5))
        self.assertEqual(parse_reference_range("13.5 to 17.5"), (13.5, 17.5))
        self.assertEqual(parse_reference_range("< 200"), (None, 200.0))
        self.assertEqual(parse_reference_range("<= 100"), (None, 100.0))
        self.assertEqual(parse_reference_range("> 50"), (50.0, None))
        self.assertEqual(parse_reference_range(">= 40"), (40.0, None))
        self.assertEqual(parse_reference_range("Negative"), (None, None))
        self.assertEqual(parse_reference_range(None), (None, None))

    def test_recompute_flag_normal_and_abnormal(self):
        # Within range
        flag = recompute_flag(85.0, low=70.0, high=99.0, canonical_name="Fasting Blood Sugar")
        self.assertEqual(flag, "normal")

        # Above high bound
        flag = recompute_flag(128.0, low=70.0, high=99.0, canonical_name="Fasting Blood Sugar")
        self.assertEqual(flag, "high")

        # Below low bound
        flag = recompute_flag(10.5, low=12.0, high=15.5, canonical_name="Hemoglobin")
        self.assertEqual(flag, "low")

        # Upper bound only (< 200)
        flag = recompute_flag(242.0, low=None, high=200.0, canonical_name="Total Cholesterol")
        self.assertEqual(flag, "high")
        flag = recompute_flag(175.0, low=None, high=200.0, canonical_name="Total Cholesterol")
        self.assertEqual(flag, "normal")

        # Lower bound only (> 50)
        flag = recompute_flag(38.0, low=50.0, high=None, canonical_name="HDL Cholesterol")
        self.assertEqual(flag, "low")
        flag = recompute_flag(58.0, low=50.0, high=None, canonical_name="HDL Cholesterol")
        self.assertEqual(flag, "normal")

    def test_recompute_flag_critical_escalation(self):
        # Critical high glucose
        flag = recompute_flag(450.0, low=70.0, high=99.0, canonical_name="Fasting Glucose")
        self.assertEqual(flag, "critical")

        # Critical low potassium
        flag = recompute_flag(2.3, low=3.5, high=5.0, canonical_name="Serum Potassium")
        self.assertEqual(flag, "critical")

        # Critical low hemoglobin
        flag = recompute_flag(5.8, low=12.0, high=15.5, canonical_name="Hemoglobin")
        self.assertEqual(flag, "critical")

    def test_build_report_summary(self):
        data = LabReportData(
            report_id="test_001",
            filename="test.pdf",
            tests=[
                TestResult(test_name="Glucose", canonical_name="Glucose", value=90.0, flag="normal"),
                TestResult(test_name="Cholesterol", canonical_name="Total Cholesterol", value=240.0, flag="high"),
                TestResult(test_name="Hb", canonical_name="Hemoglobin", value=6.0, flag="critical"),
            ]
        )
        summary = build_report_summary(data)
        self.assertEqual(summary.total_tests, 3)
        self.assertEqual(summary.normal_count, 1)
        self.assertEqual(summary.abnormal_count, 1)
        self.assertEqual(summary.critical_count, 1)
    def test_build_report_chunks(self):
        from modules.lab_indexer import build_lab_report_chunks, is_general_overview_query
        from modules.lab_parser import PatientInfo

        data = LabReportData(
            report_id="rep_999",
            filename="patient_report.pdf",
            patient=PatientInfo(name="John Doe", age=50.0, sex="Male"),
            tests=[
                TestResult(test_name="S. Chol", canonical_name="Total Cholesterol", category="Lipid", value=220.0, unit="mg/dL", flag="high"),
                TestResult(test_name="HDL", canonical_name="HDL Cholesterol", category="Lipid", value=45.0, unit="mg/dL", flag="normal"),
            ],
            remarks=["Follow up lipid profile in 3 months."]
        )
        texts, metadatas, ids = build_lab_report_chunks(data)

        # Expected: 1 header + 2 tests + 1 category summary + 1 remarks = 5 chunks
        self.assertEqual(len(texts), 5)
        self.assertEqual(len(metadatas), 5)
        self.assertEqual(len(ids), 5)

        chunk_types = [m["chunk_type"] for m in metadatas]
        self.assertIn("patient_header", chunk_types)
        self.assertIn("test_result", chunk_types)
        self.assertIn("category_summary", chunk_types)
        self.assertIn("doctor_remarks", chunk_types)

        # General query detection
        self.assertTrue(is_general_overview_query("What is wrong with my report?"))
        self.assertTrue(is_general_overview_query("Summarize my findings"))
        self.assertFalse(is_general_overview_query("What is my sodium level?"))

    def test_clinical_clusters_and_priorities(self):
        from modules.insights import (
            detect_clinical_clusters,
            evaluate_test_priority,
            format_evidence_citation,
        )

        test_ldl = TestResult(
            test_name="LDL-C",
            canonical_name="LDL Cholesterol",
            category="Lipid Profile",
            value=165.0,
            unit="mg/dL",
            reference_text="< 100",
            reference_range_high=100.0,
            page_number=2,
            flag="high",
        )
        test_chol = TestResult(
            test_name="Cholesterol",
            canonical_name="Total Cholesterol",
            category="Lipid Profile",
            value=240.0,
            unit="mg/dL",
            reference_text="< 200",
            reference_range_high=200.0,
            page_number=2,
            flag="high",
        )
        test_hba1c = TestResult(
            test_name="HbA1c",
            canonical_name="HbA1c",
            category="Glycemic",
            value=6.8,
            unit="%",
            reference_text="< 5.7",
            reference_range_high=5.7,
            page_number=1,
            flag="high",
        )

        # 1. Evidence citation formatting
        citation = format_evidence_citation(test_ldl)
        self.assertEqual(citation, "[Test: LDL Cholesterol = 165.0 mg/dL (ref < 100), page 2]")

        # 2. Cluster detection
        clusters = detect_clinical_clusters([test_ldl, test_chol, test_hba1c])
        self.assertIn("Cardiovascular & Atherogenic Lipid Pattern", clusters)
        self.assertIn("Impaired Glucose & Metabolic Pattern", clusters)

        # 3. Priority evaluation
        sev_ldl, badge_ldl, urgency_ldl, cluster_ldl = evaluate_test_priority(test_ldl, clusters)
        self.assertEqual(sev_ldl, "HIGH")
        self.assertEqual(badge_ldl, "High Concern")
        self.assertEqual(cluster_ldl, "Cardiovascular & Atherogenic Lipid Pattern")

        sev_a1c, badge_a1c, urgency_a1c, cluster_a1c = evaluate_test_priority(test_hba1c, clusters)
        self.assertEqual(sev_a1c, "HIGH")
        self.assertEqual(badge_a1c, "High Concern")


if __name__ == "__main__":
    unittest.main()

