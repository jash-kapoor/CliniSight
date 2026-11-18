import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add server directory to path
server_dir = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(server_dir))

from langchain.docstore.document import Document
from modules.chat_memory import ChatMemoryManager
from modules.lab_parser import LabReportData, load_parsed_report, save_parsed_report
from modules.lab_indexer import build_lab_report_chunks, delete_report, index_lab_report
from modules.llm import DOCTOR_SYSTEM_PROMPT, MANDATORY_DISCLAIMER, build_clinical_chat_prompt, clean_chat_answer
from modules.query_handlers import detect_safety_escalation, extract_sources_and_flags, handle_chat_query
from modules.vectorstore import similarity_search


class TestChatRAGPhase5(unittest.TestCase):

    def test_chat_memory_manager_turns_and_trimming(self):
        memory = ChatMemoryManager(max_turns=6)
        session = "test_user_123"

        # Add 4 turns (2 user, 2 assistant)
        memory.add_turn(session, "What is my cholesterol?", "Your total cholesterol is 242 mg/dL.")
        memory.add_turn(session, "Why is that high?", "It exceeds the normal reference limit of 200 mg/dL.")

        history = memory.get_history(session)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

        # Add 2 more turns -> total 6 turns
        memory.add_turn(session, "What about my HbA1c?", "Your HbA1c is 6.8%.")
        history = memory.get_history(session)
        self.assertEqual(len(history), 6)

        # Add 1 more pair (2 turns) -> should trim to the most recent 6
        memory.add_turn(session, "Should I be worried?", "Let us discuss precautions.")
        history = memory.get_history(session)
        self.assertEqual(len(history), 6)
        # Oldest turns should have shifted out
        self.assertEqual(history[-2]["content"], "Should I be worried?")
        self.assertEqual(history[-1]["content"], "Let us discuss precautions.")

        # Test prompt formatting
        formatted = memory.format_history_for_prompt(session)
        self.assertIn("Patient: Should I be worried?", formatted)
        self.assertIn("Doctor / Assistant: Let us discuss precautions.", formatted)

        # Test clearing
        memory.clear_history(session)
        self.assertEqual(len(memory.get_history(session)), 0)

    def test_safety_escalation_detection(self):
        normal_doc = Document(
            page_content="Test: Total Cholesterol\n- Observed Value: 210 mg/dL\n- Clinical Flag: HIGH",
            metadata={"test_name": "Total Cholesterol", "flag": "high"},
        )
        critical_doc = Document(
            page_content="Test: Fasting Blood Sugar\n- Observed Value: 450 mg/dL\n- Clinical Flag: CRITICAL",
            metadata={"test_name": "Fasting Blood Sugar", "flag": "critical"},
        )

        # 1. Non-emergency query
        is_em, warn = detect_safety_escalation("What is my cholesterol level?", [normal_doc])
        self.assertFalse(is_em)
        self.assertIsNone(warn)

        # 2. Emergency symptom query: chest pain
        is_em_pain, warn_pain = detect_safety_escalation("I am having severe chest pain and shortness of breath", [normal_doc])
        self.assertTrue(is_em_pain)
        self.assertIsNotNone(warn_pain)
        self.assertIn("chest pain", warn_pain.lower())
        self.assertIn("emergency", warn_pain.lower())

        # 3. Emergency symptom query: fainting
        is_em_faint, warn_faint = detect_safety_escalation("I fainted this morning", [normal_doc])
        self.assertTrue(is_em_faint)
        self.assertIn("fainted", warn_faint.lower())

        # 4. Critical lab value flag escalation
        is_em_crit, warn_crit = detect_safety_escalation("Can you explain my blood sugar?", [critical_doc])
        self.assertTrue(is_em_crit)
        self.assertIn("critical", warn_crit.lower())

    def test_extract_sources_and_flags(self):
        doc1 = Document(
            page_content=(
                "Laboratory Test Result:\n"
                "- Test: LDL Cholesterol (LDL-C)\n"
                "- Medical Category: Lipid\n"
                "- Observed Value: 161.0 mg/dL\n"
                "- Biological Reference Range: < 100\n"
                "- Clinical Flag: HIGH\n"
                "- Page in Report: 1\n"
                "- Report Source Snippet: LDL Cholesterol 161.0 mg/dL"
            ),
            metadata={"test_name": "LDL Cholesterol", "flag": "high", "page": 1, "chunk_type": "test_result"},
        )
        doc2 = Document(
            page_content=(
                "Laboratory Test Result:\n"
                "- Test: Potassium (Serum Potassium)\n"
                "- Medical Category: Electrolytes\n"
                "- Observed Value: 2.3 mmol/L\n"
                "- Biological Reference Range: 3.5 - 5.0\n"
                "- Clinical Flag: CRITICAL\n"
                "- Page in Report: 2\n"
                "- Report Source Snippet: Serum Potassium 2.3 mmol/L"
            ),
            metadata={"test_name": "Potassium", "flag": "critical", "page": 2, "chunk_type": "test_result"},
        )

        sources, flags = extract_sources_and_flags([doc1, doc2])

        # Verify sources
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["test"], "LDL Cholesterol")
        self.assertEqual(sources[0]["value"], "161.0 mg/dL")
        self.assertEqual(sources[0]["range"], "< 100")
        self.assertEqual(sources[0]["page"], 1)

        # Verify priority flags
        self.assertEqual(len(flags), 2)
        self.assertEqual(flags[0]["test"], "LDL Cholesterol")
        self.assertEqual(flags[0]["flag"], "high")
        self.assertEqual(flags[0]["urgency"], "🟠 Medium Concern")

        self.assertEqual(flags[1]["test"], "Potassium")
        self.assertEqual(flags[1]["flag"], "critical")
        self.assertEqual(flags[1]["urgency"], "🔴 High Concern")

    def test_prompt_construction_and_disclaimer(self):
        prompt = build_clinical_chat_prompt(
            context_text="LDL: 161 mg/dL",
            history_text="Patient: Hi\nDoctor: Hello",
            question="Why is my LDL high?",
            emergency_advisory="Seek immediate care for chest pain",
        )
        self.assertIn("Dr. Mira", prompt)
        self.assertIn("CRITICAL SAFETY ALERT", prompt)
        self.assertIn("Seek immediate care for chest pain", prompt)
        self.assertIn("Why is my LDL high?", prompt)
        self.assertIn("LDL: 161 mg/dL", prompt)
        self.assertIn(MANDATORY_DISCLAIMER, DOCTOR_SYSTEM_PROMPT)


    def test_serum_creatinine_page_two_saved_indexed_and_ask_source(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "report_pdf_parsed.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["report_id"] = "test_serum_creat_p2"
        rep_obj = LabReportData.model_validate(data)
        save_parsed_report(rep_obj)

        try:
            # 1. Assert Serum Creatinine = page 2 in the saved data
            saved = load_parsed_report("test_serum_creat_p2")
            self.assertIsNotNone(saved)
            creat_saved = [t for t in saved.tests if t.canonical_name == "Serum Creatinine"]
            self.assertTrue(len(creat_saved) > 0)
            self.assertEqual(creat_saved[0].page_number, 2)

            # 2. Assert Serum Creatinine = page 2 in the indexed chunk metadata
            texts, metadatas, ids = build_lab_report_chunks(saved)
            creat_chunks = [m for m in metadatas if m.get("test_name") == "Serum Creatinine"]
            self.assertTrue(len(creat_chunks) > 0)
            self.assertEqual(creat_chunks[0]["page"], 2)

            # Index into chroma and check retrieved document metadata
            class DummyEmbeddings:
                def embed_documents(self, texts):
                    vectors = []
                    for t in texts:
                        vec = [0.0] * 3072
                        if "Serum Creatinine" in t:
                            vec[0] = 1.0
                        vectors.append(vec)
                    return vectors
                def embed_query(self, text):
                    vec = [0.0] * 3072
                    if "Serum Creatinine" in text:
                        vec[0] = 1.0
                    return vec

            with patch("modules.vectorstore.get_embeddings", return_value=DummyEmbeddings()):
                index_lab_report(saved)
                indexed_docs = similarity_search("Serum Creatinine", k=5, namespace="test_serum_creat_p2")
                doc_matches = [d for d in indexed_docs if (d.metadata or {}).get("test_name") == "Serum Creatinine"]
                self.assertTrue(len(doc_matches) > 0)
                self.assertEqual(doc_matches[0].metadata["page"], 2)

                # 3. Assert Serum Creatinine = page 2 in the chat source returned by /ask/
                with patch("modules.query_handlers.generate_chat_response") as mock_gen:
                    mock_gen.return_value = "Your Serum Creatinine is 0.94 mg/dL which is within normal limits."
                    res = handle_chat_query(
                    question="What is my Serum Creatinine level?",
                    report_id="test_serum_creat_p2",
                    session_id="test_sess_creat_p2",
                )
                self.assertIn("sources", res)
                source_creat = [s for s in res["sources"] if s.get("test") == "Serum Creatinine"]
                self.assertTrue(len(source_creat) > 0)
                self.assertEqual(source_creat[0]["page"], 2)
                self.assertEqual(source_creat[0]["value"], "0.94 mg/dL")
                self.assertEqual(source_creat[0]["unit"], "mg/dL")
        finally:
            delete_report("test_serum_creat_p2")

    def test_emergency_response_single_advisory_no_evidence_short_reply(self):
        doc1 = Document(
            page_content="Test: Hemoglobin\n- Observed Value: 13.4 gm%\n- Page in Report: 1",
            metadata={"test_name": "Hemoglobin", "page": 1, "flag": "normal", "chunk_type": "test_result"},
        )
        with patch("modules.query_handlers.similarity_search", return_value=[doc1]):
            with patch("modules.query_handlers.generate_chat_response") as mock_gen:
                mock_gen.return_value = (
                    "I hear that you are having severe chest pain and difficulty breathing, which is very concerning. "
                    "Please call emergency services like 102 or 911 immediately or go directly to the nearest hospital emergency room. "
                    "This routine laboratory report cannot evaluate acute heart conditions or emergency symptoms."
                )
                res = handle_chat_query(
                    question="I have severe chest pain and breathlessness. What should I do?",
                    session_id="emergency_sess_1",
                )
                # 1. Structured emergency_warning present for UI banner
                self.assertIsNotNone(res["emergency_warning"])
                self.assertIn("chest pain", res["emergency_warning"].lower())

                # 2. Model's answer must NOT repeat the advisory text banner
                self.assertNotIn("URGENT MEDICAL ADVISORY", res["answer"])
                self.assertNotIn("EMERGENCY SAFETY ADVISORY", res["answer"])

                # 3. Model reply is 3 to 5 sentences
                sentences = [s.strip() for s in res["answer"].replace(MANDATORY_DISCLAIMER, "").split(".") if s.strip()]
                self.assertTrue(3 <= len(sentences) <= 5)

                # 4. No evidence list, no lab walkthrough, no diet advice
                self.assertEqual(res["sources"], [])
                self.assertEqual(res["priority_flags"], [])
                self.assertNotIn("[Test:", res["answer"])
                self.assertNotIn("diet", res["answer"].lower())
                self.assertNotIn("hemoglobin", res["answer"].lower())

    def test_evidence_formatting_no_inline_brackets_and_relevance_filtering(self):
        # 1. clean_chat_answer strips inline brackets and Evidence: headers
        raw = (
            "Your blood tests look mostly stable.\n\n"
            "**Evidence:**\n"
            "[Test: Serum Creatinine = 0.94 mg/dL (ref 0.6 - 1.4 mg/dL), page 2] "
            "[Test: Hemoglobin = 13.4 gm% (ref 13.5 - 17.0 gm%), page 1]"
        )
        cleaned = clean_chat_answer(raw)
        self.assertNotIn("[Test:", cleaned)
        self.assertNotIn("Evidence:", cleaned)
        self.assertIn("Your blood tests look mostly stable.", cleaned)

        # 2. Relevance filtering: question about Creatinine should not include Hemoglobin in sources
        doc_creat = Document(
            page_content="Laboratory Test Result:\n- Test: Serum Creatinine (Creatinine)\n- Observed Value: 0.94 mg/dL\n- Biological Reference Range: 0.6 - 1.4 mg/dL\n- Clinical Flag: NORMAL\n- Page in Report: 2\n- Report Source Snippet: Serum Creatinine 0.94 mg/dL",
            metadata={"test_name": "Serum Creatinine", "page": 2, "flag": "normal", "chunk_type": "test_result"},
        )
        doc_hb = Document(
            page_content="Laboratory Test Result:\n- Test: Hemoglobin (Hb)\n- Observed Value: 13.4 gm%\n- Biological Reference Range: 13.5 - 17.0 gm%\n- Clinical Flag: LOW\n- Page in Report: 1\n- Report Source Snippet: Hemoglobin 13.4 gm%",
            metadata={"test_name": "Hemoglobin", "page": 1, "flag": "low", "chunk_type": "test_result"},
        )
        sources, _ = extract_sources_and_flags(
            [doc_creat, doc_hb],
            question="What is my Serum Creatinine level?",
            answer="Your Serum Creatinine is 0.94 mg/dL, which is within the normal range.",
        )
        # Only Serum Creatinine should be included
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["test"], "Serum Creatinine")
        self.assertEqual(sources[0]["value"], "0.94 mg/dL")
        self.assertEqual(sources[0]["unit"], "mg/dL")
        self.assertEqual(sources[0]["page"], 2)

    def test_greeting_dr_mira_once_per_session(self):
        # Turn 1: Empty history, Dr. Mira can introduce herself
        raw_t1 = "Hello! I am Dr. Mira, your clinical explainer. Your hemoglobin is slightly low."
        cleaned_t1 = clean_chat_answer(raw_t1, is_ongoing_turn=False)
        self.assertIn("Dr. Mira", cleaned_t1)

        # Turn 2: Ongoing turn (history exists), introduction must be stripped
        raw_t2 = "Hello Test Patient! I am Dr. Mira, your clinical lab explainer. Your platelet count is normal."
        cleaned_t2 = clean_chat_answer(raw_t2, is_ongoing_turn=True)
        self.assertNotIn("Dr. Mira", cleaned_t2)
        self.assertIn("Your platelet count is normal.", cleaned_t2)


if __name__ == "__main__":
    unittest.main()
