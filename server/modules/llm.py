"""Gemini chat LLM and Clinical Doctor Explainer prompt engine."""

from __future__ import annotations

import re
from typing import Generator, List, Optional
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from logger import logger
from modules.settings import GEMINI_CHAT_MODEL, require_google_api_key

DOCTOR_SYSTEM_PROMPT = """You are Dr. Mira, a friendly, compassionate clinical lab explainer talking directly to a patient about their laboratory test report.

YOUR PERSONA & COMMUNICATION STYLE:
- Talk like a patient-centered doctor explaining test results in a clinic room: warm, clear, empathetic, and reassuring.
- Use short, straightforward sentences.
- Avoid unexplained medical jargon. If you must use a medical term, immediately explain it simply in parentheses (e.g. "dyslipidemia (unhealthy blood fat levels)").
- Use everyday analogies where helpful (e.g. comparing arteries to water pipes, hemoglobin to oxygen delivery vans, or insulin to a key opening cells for energy).

GREETING & INTRODUCTION RULES:
- Introduce yourself as Dr. Mira ONLY on the very first answer of the session (when there is no previous conversation history).
- For all subsequent answers in the session, NEVER introduce yourself again. Do NOT say "I am Dr. Mira" or "Hello, I'm Dr. Mira". Acknowledge the patient warmly and jump straight into answering.

GROUNDING & TRUTHFULNESS:
1. REPORT EVIDENCE ONLY: For any factual claim about the patient's test values, reference intervals, dates, or results, rely EXCLUSIVELY on the provided REPORT CONTEXT.
2. ZERO FABRICATION: If the patient asks about a test, condition, or number NOT in the report, clearly state that it is not in their report. Never invent or guess values.
3. CLEAR SEPARATION: You may draw upon general medical knowledge for lifestyle, diet, and physiological causes, but you must explicitly label it as "General Guidance", clearly separated from report facts.

STRUCTURED CLINICAL EXPLANATION:
When answering clinical questions about specific test results or findings, organize your response with these clear sections where relevant using bold run-in labels rather than markdown headings (# or ##):
- **What it means:** Explain what the test measures and what their observed level means in plain words.
- **Why it matters:** Explain how this affects their health or body systems.
- **What you can do:** Practical, actionable lifestyle steps, precautions, and dietary habits (e.g. high-fiber foods, hydration, exercise).
- **Priority and urgency:** State the urgency (e.g. routine lifestyle monitoring vs. discuss with your physician at your next visit vs. prompt consultation).
Do not use markdown headings (#, ##, ###); use bold run-in labels ("**What it means:**") directly in paragraphs.

EVIDENCE FORMATTING RULES:
- Do NOT include inline bracket citations like `[Test: ...]` or an inline Evidence block in your answer text.
- Grounding evidence is displayed separately in a dedicated UI panel. Keep your answer text clean, fluent, and conversational.

EMERGENCY SYMPTOMS PROTOCOL:
- When the patient reports acute or emergency symptoms (such as chest pain, chest pressure, shortness of breath, fainting, severe dizziness, stroke symptoms, etc.):
  1. The UI already displays a dedicated emergency advisory banner. DO NOT repeat or duplicate the advisory banner text.
  2. Your reply MUST be short (strictly 3 to 5 sentences total):
     - Empathetically acknowledge their symptoms.
     - Strongly urge them to seek immediate emergency medical care without delay (call emergency services like 102/108/911 or go to the nearest emergency room).
     - State clearly in one line that this routine laboratory report cannot evaluate heart conditions or acute medical emergencies.
  3. ABSOLUTELY NO evidence list, NO lab walkthrough, and NO diet or lifestyle advice.

SAFETY & ETHICAL BOUNDARIES:
- Do NOT provide a definitive diagnosis (use "possible reasons" or "findings suggest", never "you definitely have condition X").
- Do NOT prescribe medications, suggest pharmaceutical doses, or advise stopping prescribed medication.
- If any test in the evidence/context is marked as [STATUS: UNCONFIRMED / NEEDS HUMAN REVIEW] or has low confidence, you must explicitly alert the user that this parameter could not be confirmed with certainty from the document image and must be verified against their physical report. Never reason from unconfirmed values as established certainty.

NO EMOJIS OR ICONS:
- Do NOT use any emojis, icons, or decorative unicode symbols anywhere in your response. Rely entirely on clear, professional plain text.

MANDATORY DISCLAIMER:
Always end your response with this exact one-line statement at the very end:
"This information is for educational understanding and is not a medical diagnosis or treatment plan. Always consult your doctor or healthcare professional."
"""

MANDATORY_DISCLAIMER = (
    "This information is for educational understanding and is not a medical diagnosis "
    "or treatment plan. Always consult your doctor or healthcare professional."
)


def clean_chat_answer(
    text: str,
    is_ongoing_turn: bool = False,
    is_emergency_symptom: bool = False,
    emergency_advisory: Optional[str] = None,
) -> str:
    """Clean the raw LLM output according to evidence, emergency, and greeting formatting constraints."""
    cleaned = text

    # 1. Remove inline [Test: ...] brackets
    cleaned = re.sub(r"\[Test:[^\]]*\]", "", cleaned)

    # 2. Remove orphaned Evidence section headers
    cleaned = re.sub(r"\n+\s*(?:\*{1,2})?Evidence:?(?:\*{1,2})?\s*\n*", "\n\n", cleaned, flags=re.IGNORECASE)

    # 3. For emergency queries, strip repeated advisory banners/quotes if the model generated them
    if is_emergency_symptom or emergency_advisory:
        cleaned = re.sub(r"(?:[\U0001F6A8\U000026A0\U0000FE0F])?\s*(?:URGENT MEDICAL ADVISORY|EMERGENCY SAFETY ADVISORY|CRITICAL SAFETY ALERT):?[^\n]*\n*", "", cleaned, flags=re.IGNORECASE)
        if emergency_advisory:
            cleaned = cleaned.replace(emergency_advisory, "")

    # 4. If ongoing turn (not first turn), strip repeated introduction
    if is_ongoing_turn:
        cleaned = re.sub(
            r"^(?:[^\n.!?]*\b(?:hello|hi|greetings)[^\n.!?]*[.!?]\s*)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:i am|i'm|this is)\s+dr\.\s*mira[^\n.!?]*[.!?]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # 5. Remove any accidental emojis or decorative symbols from output
    cleaned = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F]", "", cleaned)

    # 6. Collapse excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Ensure mandatory disclaimer is present
    if MANDATORY_DISCLAIMER not in cleaned:
        cleaned = f"{cleaned}\n\n{MANDATORY_DISCLAIMER}"

    return cleaned


def get_chat_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model configured from the environment or specified name."""
    api_key = require_google_api_key()
    model = model_name or GEMINI_CHAT_MODEL
    logger.info("Using Gemini chat model %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.2,
    )


def build_clinical_chat_prompt(
    context_text: str,
    history_text: str,
    question: str,
    emergency_advisory: Optional[str] = None,
    is_emergency_symptom: bool = False,
    is_ongoing_turn: bool = False,
) -> str:
    """Compose the doctor-style prompt with grounding context, memory, and safety escalation."""
    emergency_instructions = ""
    if is_emergency_symptom or emergency_advisory:
        advisory_line = f"- System Advisory: {emergency_advisory}\n" if emergency_advisory else ""
        emergency_instructions = (
            f"\nCRITICAL SAFETY ALERT & EMERGENCY PROTOCOL ACTIVE:\n"
            f"{advisory_line}"
            "- A dedicated emergency advisory banner is already displayed to the patient in the UI. DO NOT repeat the advisory text or banner.\n"
            "- Your reply MUST be short (strictly 3 to 5 sentences total):\n"
            "  1. Empathetically acknowledge their symptoms.\n"
            "  2. Urge immediate emergency care (call 102/108/911 or visit the emergency room without delay).\n"
            "  3. State clearly in one line that this routine lab report cannot evaluate heart conditions or acute medical emergencies.\n"
            "- ABSOLUTELY NO evidence list, NO lab walkthrough, and NO diet or lifestyle advice.\n\n"
        )

    greeting_instructions = ""
    has_history = is_ongoing_turn or bool(history_text and history_text.strip() and history_text.strip() != "No previous conversation.")
    if has_history:
        greeting_instructions = (
            "\nNOTE: This is an ongoing conversation. DO NOT introduce yourself or say 'I am Dr. Mira'. "
            "Proceed directly to addressing the user's question with warmth.\n\n"
        )

    return f"""{DOCTOR_SYSTEM_PROMPT}
{emergency_instructions}{greeting_instructions}
---
RELEVANT LAB REPORT CONTEXT:
{context_text or "No specific report context found for this query."}

---
PREVIOUS CONVERSATION:
{history_text or "No previous conversation."}

---
PATIENT QUESTION:
{question}

DOCTOR'S RESPONSE:"""


def get_candidate_models() -> List[str]:
    """Return ordered list of models for automatic fallback."""
    candidates = [GEMINI_CHAT_MODEL, "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]
    seen = set()
    return [m for m in candidates if not (m in seen or seen.add(m))]


def generate_chat_response(
    context_text: str,
    history_text: str,
    question: str,
    emergency_advisory: Optional[str] = None,
    is_emergency_symptom: bool = False,
    is_ongoing_turn: bool = False,
) -> str:
    """Call Gemini to generate a doctor-style response with automatic model fallback."""
    api_key = require_google_api_key()
    genai.configure(api_key=api_key)
    prompt = build_clinical_chat_prompt(
        context_text=context_text,
        history_text=history_text,
        question=question,
        emergency_advisory=emergency_advisory,
        is_emergency_symptom=is_emergency_symptom,
        is_ongoing_turn=is_ongoing_turn,
    )

    last_error = None
    for model_name in get_candidate_models():
        try:
            logger.info("Generating chat response using model %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
            )
            raw_text = (response.text or "").strip()
            cleaned_text = clean_chat_answer(
                raw_text,
                is_ongoing_turn=is_ongoing_turn,
                is_emergency_symptom=is_emergency_symptom,
                emergency_advisory=emergency_advisory,
            )
            return cleaned_text
        except Exception as exc:
            logger.warning("Chat generation failed on %s: %s", model_name, exc)
            last_error = exc

    raise RuntimeError(f"All candidate models failed for chat response. Last error: {last_error}")


def stream_chat_response(
    context_text: str,
    history_text: str,
    question: str,
    emergency_advisory: Optional[str] = None,
    is_emergency_symptom: bool = False,
    is_ongoing_turn: bool = False,
) -> Generator[str, None, None]:
    """Stream chat response chunks using Gemini API."""
    api_key = require_google_api_key()
    genai.configure(api_key=api_key)
    prompt = build_clinical_chat_prompt(
        context_text=context_text,
        history_text=history_text,
        question=question,
        emergency_advisory=emergency_advisory,
        is_emergency_symptom=is_emergency_symptom,
        is_ongoing_turn=is_ongoing_turn,
    )

    last_error = None
    for model_name in get_candidate_models():
        try:
            logger.info("Streaming chat response using model %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            logger.warning("Streaming failed on %s: %s", model_name, exc)
            last_error = exc

    raise RuntimeError(f"All candidate models failed for chat streaming. Last error: {last_error}")


def get_llm_chain(retriever, model_name: str | None = None):
    """Backwards-compatible RetrievalQA chain."""
    llm = get_chat_llm(model_name=model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{DOCTOR_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}\n\nAnswer:",
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
