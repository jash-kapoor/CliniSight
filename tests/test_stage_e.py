"""Unit tests for Stage E: Human-In-The-Loop Verification API and letterhead lab_name resolution."""

import json
import sys
import unittest
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(server_dir))

client_dir = Path(__file__).resolve().parent.parent / "client"
sys.path.insert(0, str(client_dir))

from fastapi.testclient import TestClient
from main import app
from modules.lab_parser import (
    LabReportData,
    PatientInfo,
    ReportSummary,
    TestResult,
    assign_test_category,
    clean_lab_and_referring_info,
    save_parsed_report,
)
from modules.settings import PARSED_REPORTS_DIR, UPLOAD_DIR
from PIL import Image


class TestStageE(unittest.TestCase):
    created_report_ids = set()

    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch

        class DummyEmbeddings:
            def embed_documents(self, texts):
                return [[0.01] * 3072 for _ in texts]
            def embed_query(self, text):
                return [0.01] * 3072

        cls.embed_patcher = patch("modules.vectorstore.get_embeddings", return_value=DummyEmbeddings())
        cls.embed_patcher.start()

        class MockGeminiResponse:
            text = json.dumps({
                "overview": "Your report shows high Hemoglobin requiring physician attention.",
                "clinical_reasons": [],
                "precautions": [],
                "foods_to_prefer": ["Green leafy vegetables"],
                "foods_to_limit": ["Refined sugar"],
                "indian_diet_recommendations": ["Roti with dal"],
                "sample_meal_plan": {
                    "breakfast": "Oats or vegetable upma",
                    "mid_morning": "Fresh fruits",
                    "lunch": "Roti and dal with green sabzi",
                    "evening_snack": "Roasted chana",
                    "dinner": "Light khichdi",
                    "notes": "Drink adequate water",
                },
                "questions_to_ask_doctor": [],
            })

        cls.gemini_patcher = patch("google.generativeai.GenerativeModel.generate_content", return_value=MockGeminiResponse())
        cls.gemini_patcher.start()

        cls.client = TestClient(app)
        cls.test_report_id = "test_stage_e_001"
        cls.created_report_ids.add(cls.test_report_id)

        # Create dummy rendered page images
        cls.pages_dir = UPLOAD_DIR / "report_pages" / cls.test_report_id
        cls.pages_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        img.save(cls.pages_dir / "page_1.png", format="PNG")

        # Create initial parsed report in pending_verification state
        initial_data = LabReportData(
            report_id=cls.test_report_id,
            filename="sample_test.pdf",
            patient=PatientInfo(
                name="Test Patient",
                age=20.0,
                sex="Male",
                lab_name="Test Pathology Laboratory",
                referring_source="TEST REFERRAL HOSPITAL",
            ),
            tests=[
                TestResult(
                    test_name="Hemoglobin",
                    canonical_name="Hemoglobin",
                    result_value_numeric=13.4,
                    unit="g/dL",
                    reference_low=13.5,
                    reference_high=17.5,
                    flag="low",
                    confidence="high",
                    needs_review=False,
                    page_number=1,
                    bbox=[100, 100, 150, 400],
                ),
                TestResult(
                    test_name="Platelet Count",
                    canonical_name="Platelet Count",
                    result_value_numeric=175000.0,
                    unit="/cmm",
                    reference_text="1.5 - 4.0 Lac/cmm.",
                    reference_low=1.5,
                    reference_high=4.0,
                    flag="normal",
                    confidence="high",
                    needs_review=False,
                    page_number=1,
                ),
                TestResult(
                    test_name="Potassium",
                    canonical_name="Potassium",
                    result_value_numeric=6.5,
                    unit="mEq/L",
                    reference_low=3.5,
                    reference_high=5.5,
                    flag="critical",
                    confidence="low",
                    needs_review=True,
                    review_reason="Needs human check",
                    page_number=1,
                ),
            ],
            needs_review_count=1,
            verified_by_user=False,
            status="pending_verification",
            page_count=1,
        )
        save_parsed_report(initial_data)

    def test_lab_name_extraction_from_letterhead_vs_referring_source(self):
        # 1. When lab_name was erroneously set to referring_source, verify it is cleared
        lab, doc, src = clean_lab_and_referring_info(
            raw_lab="TEST REFERRAL HOSPITAL",
            raw_doc=None,
            raw_source="TEST REFERRAL HOSPITAL",
            header_text_hint="",
        )
        self.assertIsNone(lab)
        self.assertEqual(src, "TEST REFERRAL HOSPITAL")

        # 2. Extract Test Pathology Laboratory from letterhead
        hint = "TEST PATHOLOGY LABORATORY\nAccurate | Caring | Instant\nPatient: Test Patient\nRef By : TEST REFERRAL HOSPITAL"
        lab2, doc2, src2 = clean_lab_and_referring_info(
            raw_lab="TEST REFERRAL HOSPITAL",
            raw_doc=None,
            raw_source="TEST REFERRAL HOSPITAL",
            header_text_hint=hint,
        )
        self.assertEqual(lab2, "Test Pathology Laboratory")
        self.assertEqual(src2, "TEST REFERRAL HOSPITAL")

    def test_put_report_tests_recomputes_flags(self):
        # User edits Potassium from 6.5 (critical) to 4.2 (normal)
        payload = {
            "tests": [
                {
                    "test_name": "Hemoglobin",
                    "canonical_name": "Hemoglobin",
                    "result_value_numeric": 13.4,
                    "unit": "g/dL",
                    "reference_low": 13.5,
                    "reference_high": 17.5,
                    "page_number": 1,
                    "confidence": "high",
                    "needs_review": False,
                },
                {
                    "test_name": "Platelet Count",
                    "canonical_name": "Platelet Count",
                    "result_value_numeric": 175000.0,
                    "unit": "/cmm",
                    "reference_text": "1.5 - 4.0 Lac/cmm.",
                    "reference_low": 1.5,
                    "reference_high": 4.0,
                    "page_number": 1,
                    "confidence": "high",
                    "needs_review": False,
                },
                {
                    "test_name": "Potassium",
                    "canonical_name": "Potassium",
                    "result_value_numeric": 4.2,
                    "unit": "mEq/L",
                    "reference_low": 3.5,
                    "reference_high": 5.5,
                    "page_number": 1,
                    "confidence": "high",
                    "needs_review": False,
                },
            ]
        }

        resp = self.client.put(f"/report/{self.test_report_id}/tests", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "updated")
        self.assertEqual(data["summary"]["abnormal_count"], 1)  # Only Hemoglobin is low
        self.assertEqual(data["summary"]["critical_count"], 0)  # Potassium is now normal

        # Verify Potassium is normal in response
        k_test = next(t for t in data["tests"] if t["canonical_name"] == "Potassium")
        self.assertEqual(k_test["flag"], "normal")
        self.assertEqual(k_test["verified_by_user"], True)

    def test_get_page_image_with_optional_bbox(self):
        # 1. Plain image
        resp = self.client.get(f"/report/{self.test_report_id}/page/1/image")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")

        # 2. Image with bbox highlight
        resp_bbox = self.client.get(f"/report/{self.test_report_id}/page/1/image?bbox=100,100,200,400")
        self.assertEqual(resp_bbox.status_code, 200)
        self.assertEqual(resp_bbox.headers["content-type"], "image/png")

        # 3. Nonexistent page returns 404
        resp_404 = self.client.get(f"/report/{self.test_report_id}/page/99/image")
        self.assertEqual(resp_404.status_code, 404)

    def test_post_confirm_report(self):
        # Confirming report marks verified_by_user = True and indexes into vector store
        resp = self.client.post(f"/report/{self.test_report_id}/confirm", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "confirmed")
        self.assertIn("summary", data)
        self.assertIn("insights", data)

        # Verify persisted report has status confirmed
        get_resp = self.client.get(f"/report/{self.test_report_id}")
        self.assertEqual(get_resp.status_code, 200)
        rep_json = get_resp.json()
        self.assertEqual(rep_json["verified_by_user"], True)
        self.assertEqual(rep_json["status"], "confirmed")

    def test_upload_report_pdf_endpoint_pending_verification(self):
        """Upload Report.pdf through the endpoint with recorded fixtures and expect 200 with exactly 28 tests and 7 abnormal."""
        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        # Load golden fixtures to mock Gemini vision calls deterministically offline
        fixture_dir = Path(__file__).resolve().parent / "fixtures"
        fixtures = []
        for idx in range(4):
            fix_path = fixture_dir / f"vision_call_{idx}.json"
            if fix_path.exists():
                with open(fix_path, "r") as f:
                    fixtures.append(json.load(f))

        if len(fixtures) < 4:
            self.skipTest("Golden fixtures for vision calls not found in tests/fixtures/")

        call_idx = {"count": 0}
        def fixture_vision(image_bytes, prompt, api_key, mime_type="image/jpeg"):
            res = fixtures[call_idx["count"] % len(fixtures)]
            call_idx["count"] += 1
            return res

        with open(report_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        from unittest.mock import patch
        with patch("modules.lab_parser._call_gemini_multimodal_vision", side_effect=fixture_vision):
            resp = self.client.post(
                "/upload_report/",
                files={"file": ("Report.pdf", pdf_bytes, "application/pdf")},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.created_report_ids.add(data["report_id"])

        self.assertEqual(data["status"], "pending_verification")
        self.assertEqual(data["verified_by_user"], False)
        self.assertIn("report_id", data)
        # Strict assertions: exactly 28 tests and 7 abnormal from Report.pdf
        self.assertEqual(data["summary"]["total_tests"], 28)
        self.assertEqual(data["summary"]["abnormal_count"], 7)
        self.assertEqual(data["patient"]["lab_name"], "Krupa Pathology Laboratory")

        # Verify page images were created on disk
        from modules.settings import REPORT_PAGES_DIR
        page_1_img = REPORT_PAGES_DIR / data["report_id"] / "page_1.png"
        self.assertTrue(page_1_img.exists(), f"Expected {page_1_img} to exist")

    def test_upload_report_pdf_endpoint_live_network(self):
        """Optional live network integration test: upload Report.pdf without mocks (skipped without network/API key)."""
        import os
        from modules.settings import GOOGLE_API_KEY
        api_key = GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GOOGLE_API_KEY / GEMINI_API_KEY not configured for live test")

        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        # Check network connectivity to Gemini API host
        try:
            import socket
            with socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=3):
                pass
        except OSError:
            self.skipTest("Network connection to Google Gemini API unreachable")

        with open(report_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        resp = self.client.post(
            "/upload_report/",
            files={"file": ("Report.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.created_report_ids.add(data["report_id"])
        self.assertIn(data["status"], {"pending_verification", "incomplete"})
        self.assertGreaterEqual(data["summary"]["total_tests"], 22)

    def test_degraded_mode_ocr_fallback_incomplete_and_blocked_confirm(self):
        """
        Degraded-mode Path 2: When Gemini vision fails, pipeline must NOT silently return a partial result:
        1. Fallback to OCR-only extraction
        2. Set status to 'incomplete'
        3. Visible warning: 'Some results may be missing because automatic reading was unavailable'
        4. Every extracted row from fallback has needs_review=True
        5. Block confirmation until user acknowledges
        6. Confirm succeeds when acknowledged
        """
        from unittest.mock import patch
        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        with open(report_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        with patch("modules.lab_parser._call_gemini_multimodal_vision", side_effect=RuntimeError("Quota exceeded 429")):
            resp = self.client.post(
                "/upload_report/",
                files={"file": ("Report.pdf", pdf_bytes, "application/pdf")},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        rep_id = data["report_id"]
        self.created_report_ids.add(rep_id)

        # 1. Assert status is incomplete
        self.assertEqual(data["status"], "incomplete")
        self.assertEqual(data["verified_by_user"], False)

        # 2. Assert visible quality warning
        has_warning = any("automatic reading was unavailable" in w.lower() for w in data["quality_warnings"])
        self.assertTrue(has_warning, f"Expected warning about automatic reading in {data['quality_warnings']}")

        # 3. Retrieve report and assert every test has needs_review = True
        get_resp = self.client.get(f"/report/{rep_id}")
        self.assertEqual(get_resp.status_code, 200)
        rep_json = get_resp.json()
        self.assertGreater(len(rep_json["tests"]), 0)
        for t in rep_json["tests"]:
            self.assertTrue(t["needs_review"], f"Expected test {t['test_name']} to have needs_review=True")

        # 4. Attempt confirm WITHOUT acknowledgement -> should be blocked (HTTP 400)
        block_resp = self.client.post(
            f"/report/{rep_id}/confirm",
            json={"acknowledged_incomplete": False},
        )
        self.assertEqual(block_resp.status_code, 400)
        self.assertIn("Confirmation blocked", block_resp.json()["detail"])

        # 5. Confirm WITH acknowledgement -> should succeed (HTTP 200)
        with patch("routes.upload_report.index_lab_report", return_value=10):
            with patch("routes.upload_report.generate_report_insights", return_value={"overview": "Degraded mode verified"}):
                ok_resp = self.client.post(
                    f"/report/{rep_id}/confirm",
                    json={"acknowledged_incomplete": True},
                )
        self.assertEqual(ok_resp.status_code, 200)
        ok_json = ok_resp.json()
        self.assertEqual(ok_json["report"]["status"], "confirmed")
        self.assertEqual(ok_json["report"]["verified_by_user"], True)

    def test_degraded_mode_clear_error_when_all_extraction_fails(self):
        """
        Degraded-mode Path 1: When both Gemini vision and fallback extraction fail:
        Return a clear error to the user (HTTP 500) rather than a silent failure or partial empty result.
        """
        from unittest.mock import patch
        from modules.ocr_engine import OCRPageResult

        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        with open(report_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        with patch("modules.lab_parser._call_gemini_multimodal_vision", side_effect=RuntimeError("Connection timed out")):
            with patch("modules.lab_parser.run_traditional_ocr", return_value=OCRPageResult(page_number=1, full_text="", items=[], numeric_tokens=[])):
                resp = self.client.post(
                    "/upload_report/",
                    files={"file": ("Report.pdf", pdf_bytes, "application/pdf")},
                )

        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertIn("error", data)
        self.assertIn("automatic reading was unavailable", data["error"].lower())

    def test_patient_name_preserves_spaces(self):
        """Ensure spaces between patient name words are preserved and vision is preferred."""
        from modules.lab_parser import extract_patient_info_from_text, parse_lab_report_file

        # 1. Direct regex extraction test with space
        cand = extract_patient_info_from_text("Name : FIRSTNAME LASTNAME\nAge/Sex : 20 Yrs./M\nRef By : TEST REFERRAL HOSPITAL")
        self.assertIsNotNone(cand)
        self.assertEqual(cand.name, "FIRSTNAME LASTNAME")

        # 2. End-to-end parsed Report.pdf patient name must have space
        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if report_pdf_path.exists():
            report_data, _ = parse_lab_report_file(
                file_path=report_pdf_path,
                report_id="test_name_unit_run",
                original_filename="Report.pdf",
            )
            self.assertIsNotNone(report_data.patient)
            self.assertTrue(len(report_data.patient.name.split()) >= 2)

    def test_report_pdf_bbox_coverage_and_hemoglobin_location(self):
        """Ensure >= 90% of tests in Report.pdf get a bbox and Hemoglobin bbox lies near 13.40 on page 1."""
        from modules.lab_parser import parse_lab_report_file

        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        fixture_dir = Path(__file__).resolve().parent / "fixtures"
        fixtures = []
        for idx in range(4):
            fix_path = fixture_dir / f"vision_call_{idx}.json"
            if fix_path.exists():
                with open(fix_path, "r") as f:
                    fixtures.append(json.load(f))
        call_idx = {"count": 0}
        def fixture_vision(image_bytes, prompt, api_key, mime_type="image/jpeg"):
            res = fixtures[call_idx["count"] % len(fixtures)]
            call_idx["count"] += 1
            return res

        from unittest.mock import patch
        with patch("modules.lab_parser._call_gemini_multimodal_vision", side_effect=fixture_vision):
            report_data, _ = parse_lab_report_file(
                file_path=report_pdf_path,
                report_id="test_bbox_unit_run",
                original_filename="Report.pdf",
            )

        tests = report_data.tests
        self.assertGreaterEqual(len(tests), 25)

        tests_with_bbox = [t for t in tests if t.bbox and len(t.bbox) == 4]
        coverage_pct = (len(tests_with_bbox) / len(tests)) * 100.0
        self.assertGreaterEqual(coverage_pct, 90.0, f"Expected >= 90% bbox coverage, got {coverage_pct:.1f}%")

        # Hemoglobin on page 1 must lie near the 13.40 token (y ~ 458-495)
        hb_test = next((t for t in tests if t.canonical_name == "Hemoglobin" and t.page_number == 1), None)
        self.assertIsNotNone(hb_test, "Hemoglobin test not found on page 1")
        self.assertIsNotNone(hb_test.bbox, "Hemoglobin bbox is None")

        ymin, xmin, ymax, xmax = hb_test.bbox
        # Assert vertical bounds enclose the 13.40 token (457 to 495)
        self.assertLessEqual(ymin, 460, f"Hemoglobin ymin {ymin} should be <= 460")
        self.assertGreaterEqual(ymax, 490, f"Hemoglobin ymax {ymax} should be >= 490")
        # Assert horizontal bounds enclose value column and span row
        self.assertLessEqual(xmin, 520, f"Hemoglobin xmin {xmin} should span from test name/value")
        self.assertGreaterEqual(xmax, 650, f"Hemoglobin xmax {xmax} should cover value/range")

    def test_abbreviations_do_not_flag_review(self):
        """Ensure clinical abbreviations (MPV, MCH, RDW, SGPT) are not flagged for review."""
        from modules.lab_parser import (
            TestResult,
            is_known_abbreviation_or_synonym,
            run_plausibility_and_consistency_checks,
        )

        # Check helper
        self.assertTrue(is_known_abbreviation_or_synonym("MPV", "Mean Platelet Volume"))
        self.assertTrue(is_known_abbreviation_or_synonym("MCH", "Mean Corpuscular Hemoglobin"))
        self.assertTrue(is_known_abbreviation_or_synonym("MCH (Calculated)", "Mean Corpuscular Hemoglobin"))
        self.assertTrue(is_known_abbreviation_or_synonym("RDW", "Red Cell Distribution Width"))
        self.assertTrue(is_known_abbreviation_or_synonym("SGPT", "Alanine Aminotransferase"))

        # Verify tests are not flagged
        t_mpv = TestResult(test_name="MPV", canonical_name="Mean Platelet Volume", result_value_numeric=9.4, unit="fL")
        t_mch = TestResult(test_name="MCH (Calculated)", canonical_name="Mean Corpuscular Hemoglobin", result_value_numeric=26.69, unit="pg")
        t_rdw = TestResult(test_name="RDW", canonical_name="Red Cell Distribution Width", result_value_numeric=10.8, unit="%")
        t_sgpt = TestResult(test_name="SGPT", canonical_name="Alanine Aminotransferase", result_value_numeric=18.8, unit="U/L")

        test_list = [t_mpv, t_mch, t_rdw, t_sgpt]
        run_plausibility_and_consistency_checks(test_list)

        for t in test_list:
            self.assertFalse(t.needs_review, f"Test {t.test_name} was unexpectedly flagged: {t.review_reason}")

        # Non-abbreviation different name must be flagged
        t_mismatch = TestResult(test_name="Completely Different", canonical_name="Serum Creatinine", result_value_numeric=0.94, unit="mg/dL")
        run_plausibility_and_consistency_checks([t_mismatch])
        self.assertTrue(t_mismatch.needs_review)
        self.assertIn("name differs from printed label", t_mismatch.review_reason)


    def test_edit_to_critical_and_confirm(self):
        """Edit Hemoglobin to 6.5 (critical), confirm, and verify summary counts, priority finding, and insights."""
        report_id = "test_critical_hb_run"
        report_pdf_path = Path(__file__).resolve().parent.parent / "Report.pdf"
        if not report_pdf_path.exists():
            self.skipTest("Report.pdf not found in workspace root")

        # Parse or load Report.pdf
        from modules.lab_parser import build_report_summary, load_parsed_report, parse_lab_report_file
        report_data = load_parsed_report(report_id)
        if not report_data:
            report_data, _ = parse_lab_report_file(
                file_path=report_pdf_path,
                report_id=report_id,
                original_filename="Report.pdf",
            )
        # Ensure initial state has Hemoglobin at 13.4 (low, not critical)
        for t in report_data.tests:
            if t.canonical_name == "Hemoglobin":
                t.result_value_numeric = 13.4
                t.flag = "low"
        report_data.summary = build_report_summary(report_data)
        save_parsed_report(report_data)

        # Before edit: 7 abnormal, 0 critical
        self.assertEqual(report_data.summary.abnormal_count, 7)
        self.assertEqual(report_data.summary.critical_count, 0)

        # Convert tests to payload, editing Hemoglobin to 6.5
        payload_tests = []
        for t in report_data.tests:
            t_dict = t.model_dump()
            if t.canonical_name == "Hemoglobin":
                t_dict["result_value_numeric"] = 6.5
                t_dict["flag"] = "critical"
            payload_tests.append(t_dict)

        # 1. Update tests via PUT /report/{report_id}/tests
        put_resp = self.client.put(f"/report/{report_id}/tests", json={"tests": payload_tests})
        self.assertEqual(put_resp.status_code, 200)
        put_data = put_resp.json()
        self.assertEqual(put_data["summary"]["critical_count"], 1)
        self.assertEqual(put_data["summary"]["abnormal_count"], 6)

        # 2. Confirm report via POST /report/{report_id}/confirm
        conf_resp = self.client.post(f"/report/{report_id}/confirm", json={"tests": payload_tests})
        self.assertEqual(conf_resp.status_code, 200)
        conf_data = conf_resp.json()
        self.assertEqual(conf_data["status"], "confirmed")
        self.assertEqual(conf_data["summary"]["critical_count"], 1)
        self.assertEqual(conf_data["summary"]["abnormal_count"], 6)

        insights = conf_data.get("insights", {})
        self.assertEqual(insights.get("critical_count"), 1)
        self.assertEqual(insights.get("abnormal_count"), 6)

        # Check priority findings: Hemoglobin must be marked HIGH concern
        hb_finding = next((pf for pf in insights.get("priority_findings", []) if pf["canonical_name"] == "Hemoglobin"), None)
        self.assertIsNotNone(hb_finding)
        self.assertEqual(hb_finding["severity_level"], "HIGH")
        self.assertIn("High Concern", hb_finding["severity_badge"])

        # Check patient full name in overview: must not contain standalone first name nickname
        import re
        overview = insights.get("overview", "")
        self.assertFalse(re.search(r'\bTest\b', overview), f"Overview contains shortened nickname: {overview}")

        # Check no bogus unconfirmed warnings remain
        self.assertNotIn("unconfirmed", overview.lower())
        self.assertNotIn("require verification", overview.lower())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "gemini_patcher"):
            cls.gemini_patcher.stop()
        if hasattr(cls, "embed_patcher"):
            cls.embed_patcher.stop()
        from modules.lab_indexer import delete_report
        for rep_id in list(cls.created_report_ids):
            try:
                delete_report(rep_id)
            except Exception:
                pass

    def test_qualitative_display_and_reference(self):
        """Ensure qualitative results display actual printed text, and missing ref intervals show '-' not 'N/A'."""
        from modules.display_helpers import format_test_display_reference, format_test_display_value

        # Platelet Morphology in Smear -> "ADEQUATE", ref -> "-"
        t_morph = {
            "canonical_name": "Platelet Morphology in Smear",
            "result_value_numeric": None,
            "result_text": "ADEQUATE",
            "unit": None,
            "reference_text": None,
        }
        self.assertEqual(format_test_display_value(t_morph), "ADEQUATE")
        self.assertNotEqual(format_test_display_value(t_morph), "Not detected")
        self.assertEqual(format_test_display_reference(t_morph), "-")
        self.assertNotEqual(format_test_display_reference(t_morph), "N/A")

        # Malaria Parasite -> "M.P. NOT DETECTED"
        t_mp = {
            "canonical_name": "Malaria Parasite (M.P.)",
            "result_value_numeric": None,
            "result_text": "M.P. NOT DETECTED",
            "reference_text": "",
        }
        self.assertEqual(format_test_display_value(t_mp), "M.P. NOT DETECTED")
        self.assertEqual(format_test_display_reference(t_mp), "-")

        # Qualitative with printed expected text
        t_qual_exp = {
            "canonical_name": "HIV Antibodies",
            "result_value_numeric": None,
            "result_text": "TEST IS NEGATIVE",
            "reference_text": "NON-REACTIVE",
        }
        self.assertEqual(format_test_display_value(t_qual_exp), "TEST IS NEGATIVE")
        self.assertEqual(format_test_display_reference(t_qual_exp), "NON-REACTIVE")

    def test_referring_doctor_and_source_resolution(self):
        """Never invent 'Self / OPD'. Show referring_doctor only if named person; otherwise referring_source or Not stated."""
        # Scenario A: Only referring hospital/facility exists
        p_facility = {"referring_doctor": None, "referring_source": "TEST REFERRAL HOSPITAL"}
        ref_doc = p_facility.get("referring_doctor")
        ref_src = p_facility.get("referring_source")
        if ref_doc and str(ref_doc).strip() and str(ref_doc).strip().lower() not in {"self", "opd", "self / opd", "self/opd", "not specified"}:
            text_a = f"Referring Doctor: {str(ref_doc).strip()}"
        elif ref_src and str(ref_src).strip():
            text_a = f"Referred by: {str(ref_src).strip()}"
        else:
            text_a = "Referred by: Not stated"
        self.assertEqual(text_a, "Referred by: TEST REFERRAL HOSPITAL")
        self.assertNotIn("Self / OPD", text_a)

        # Scenario B: Named physician
        p_doc = {"referring_doctor": "Dr. A. Sharma", "referring_source": "TEST REFERRAL HOSPITAL"}
        ref_doc = p_doc.get("referring_doctor")
        ref_src = p_doc.get("referring_source")
        if ref_doc and str(ref_doc).strip() and str(ref_doc).strip().lower() not in {"self", "opd", "self / opd", "self/opd", "not specified"}:
            text_b = f"Referring Doctor: {str(ref_doc).strip()}"
        elif ref_src and str(ref_src).strip():
            text_b = f"Referred by: {str(ref_src).strip()}"
        else:
            text_b = "Referred by: Not stated"
        self.assertEqual(text_b, "Referring Doctor: Dr. A. Sharma")

        # Scenario C: Neither provided
        p_none = {"referring_doctor": None, "referring_source": None}
        ref_doc = p_none.get("referring_doctor")
        ref_src = p_none.get("referring_source")
        if ref_doc and str(ref_doc).strip() and str(ref_doc).strip().lower() not in {"self", "opd", "self / opd", "self/opd", "not specified"}:
            text_c = f"Referring Doctor: {str(ref_doc).strip()}"
        elif ref_src and str(ref_src).strip():
            text_c = f"Referred by: {str(ref_src).strip()}"
        else:
            text_c = "Referred by: Not stated"
        self.assertEqual(text_c, "Referred by: Not stated")

    def test_deterministic_test_categories(self):
        """Ensure tests are deterministically mapped to categories and not left as 'General'."""
        self.assertEqual(assign_test_category("Hemoglobin"), "CBC / Red cell indices")
        self.assertEqual(assign_test_category("Packed Cell Volume"), "CBC / Red cell indices")
        self.assertEqual(assign_test_category("Mean Corpuscular Volume"), "CBC / Red cell indices")
        self.assertEqual(assign_test_category("Mean Corpuscular Hemoglobin"), "CBC / Red cell indices")
        self.assertEqual(assign_test_category("Neutrophils (Percentage)"), "White cell differential")
        self.assertEqual(assign_test_category("Lymphocytes (Percentage)"), "White cell differential")
        self.assertEqual(assign_test_category("Absolute Lymphocyte Count"), "Absolute counts")
        self.assertEqual(assign_test_category("Total Leukocyte Count"), "Absolute counts")
        self.assertEqual(assign_test_category("Platelet Count"), "Platelets")
        self.assertEqual(assign_test_category("Platelet Morphology in Smear"), "Platelets")
        self.assertEqual(assign_test_category("Serum Creatinine"), "Kidney")
        self.assertEqual(assign_test_category("Potassium"), "Electrolytes")
        self.assertEqual(assign_test_category("SGPT"), "Liver")
        self.assertEqual(assign_test_category("Total Cholesterol"), "Lipid")
        self.assertEqual(assign_test_category("Fasting Blood Sugar"), "Diabetes")
        self.assertEqual(assign_test_category("TSH"), "Thyroid")
        self.assertEqual(assign_test_category("C-Reactive Protein"), "Inflammation")
        self.assertEqual(assign_test_category("Vitamin D"), "Vitamins")

    def test_microcytic_anemia_cluster_and_priority_consistency(self):
        """Verify Hemoglobin, PCV, MCV, MCH are in Microcytic Anemia Pattern cluster and have elevated priority."""
        from modules.insights import detect_clinical_clusters, evaluate_test_priority

        tests = [
            TestResult(test_name="Hemoglobin", canonical_name="Hemoglobin", result_value_numeric=13.4, flag="low", reference_low=13.5, reference_high=17.0),
            TestResult(test_name="Packed Cell Volume", canonical_name="Packed Cell Volume", result_value_numeric=38.5, flag="low", reference_low=40.0, reference_high=54.0),
            TestResult(test_name="Mean Corpuscular Volume", canonical_name="Mean Corpuscular Volume", result_value_numeric=76.69, flag="low", reference_low=80.0, reference_high=96.0),
            TestResult(test_name="Mean Corpuscular Hemoglobin", canonical_name="Mean Corpuscular Hemoglobin", result_value_numeric=26.69, flag="low", reference_low=27.0, reference_high=31.0),
        ]

        clusters = detect_clinical_clusters(tests)
        self.assertIn("Microcytic Anemia Pattern", clusters)
        cluster_names = [t.canonical_name for t in clusters["Microcytic Anemia Pattern"]]
        self.assertIn("Hemoglobin", cluster_names)
        self.assertIn("Packed Cell Volume", cluster_names)
        self.assertIn("Mean Corpuscular Volume", cluster_names)
        self.assertIn("Mean Corpuscular Hemoglobin", cluster_names)

        # All 4 tests must have their priority raised consistently (MEDIUM or HIGH, never LOW)
        for t in tests:
            sev, badge, urgency, matched_cluster = evaluate_test_priority(t, clusters)
            self.assertEqual(matched_cluster, "Microcytic Anemia Pattern")
            self.assertIn(sev, {"MEDIUM", "HIGH"}, f"Test {t.canonical_name} severity {sev} was not raised consistently")

    def test_no_generic_filler_phrase_in_priority_findings(self):
        """Fail if any priority finding contains the generic template filler phrase."""
        from modules.insights import BANNED_FILLER_PHRASES, generate_report_insights

        insights = generate_report_insights(self.test_report_id, force_refresh=True)
        self.assertTrue(len(insights.priority_findings) > 0)

        for finding in insights.priority_findings:
            reason = finding.clinical_reasoning
            for banned in BANNED_FILLER_PHRASES:
                self.assertNotIn(
                    banned,
                    reason.lower(),
                    f"Priority finding for '{finding.canonical_name}' contains banned filler: '{reason}'",
                )
            self.assertGreater(len(reason.strip()), 40, f"Reason too short for '{finding.canonical_name}': {reason}")


if __name__ == "__main__":
    unittest.main()


