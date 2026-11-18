"""Unit tests for Phase 6: Streamlit client utilities and dashboard formatting."""

import unittest
from client.components.dashboard import get_flag_badge
from client.utils.api import DEFAULT_TIMEOUT


class TestClientPhase6(unittest.TestCase):
    def test_flag_badge_normal(self):
        badge = get_flag_badge("normal")
        self.assertIn("NORMAL", badge)
        self.assertIn("#dcfce7", badge)

    def test_flag_badge_high(self):
        badge = get_flag_badge("high")
        self.assertIn("HIGH", badge)
        self.assertIn("#ffedd5", badge)

    def test_flag_badge_low(self):
        badge = get_flag_badge("low")
        self.assertIn("LOW", badge)
        self.assertIn("#e0f2fe", badge)

    def test_flag_badge_critical(self):
        badge = get_flag_badge("critical")
        self.assertIn("CRITICAL", badge)
        self.assertIn("#fee2e2", badge)

    def test_flag_badge_unknown(self):
        badge = get_flag_badge("random_string")
        self.assertIn("UNKNOWN", badge)

    def test_default_timeout(self):
        self.assertGreaterEqual(DEFAULT_TIMEOUT, 60)

    def test_stage_e_confidence_mapping(self):
        from client.components.verification import map_confidence_label, unmap_confidence_label

        # Confirmed
        self.assertEqual(map_confidence_label({"confidence": "high", "verified_by_ocr": True, "needs_review": False}), "Confirmed")
        # Needs check
        self.assertEqual(map_confidence_label({"confidence": "high", "needs_review": True}), "Please check")
        self.assertEqual(map_confidence_label({"confidence": "medium"}), "Please check")
        # Could not read
        self.assertEqual(map_confidence_label({"confidence": "low"}), "Could not read")

        # Unmapping
        conf1, rev1 = unmap_confidence_label("Confirmed")
        self.assertEqual(conf1, "high")
        self.assertFalse(rev1)

        conf2, rev2 = unmap_confidence_label("Please check")
        self.assertEqual(conf2, "medium")
        self.assertTrue(rev2)

        conf3, rev3 = unmap_confidence_label("Could not read")
        self.assertEqual(conf3, "low")
        self.assertTrue(rev3)

    def test_theme_css(self):
        from client.theme import get_theme_css, COLORS
        css = get_theme_css()
        self.assertIn("chat-layout-wrapper", css)
        self.assertIn("max-width: 800px", css)
        self.assertIn("report-chip", css)
        self.assertIn("chat-input-caption", css)
        self.assertIn("primary", COLORS)

    def test_report_chip_formatting(self):
        import streamlit as st
        from client.components.chatUI import get_report_chip_text
        st.session_state["report_data"] = {"patient": {"name": "Jane Doe", "report_date": "2026-05-10"}}
        chip = get_report_chip_text()
        self.assertEqual(chip, "Report: Jane Doe | 2026-05-10")


if __name__ == "__main__":
    unittest.main()
