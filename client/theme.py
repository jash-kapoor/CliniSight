"""Central design system and theme styles for the clinical assistant client.

Provides colors, typography, spacing tokens, and CSS supporting both light
and dark modes with WCAG AA compliance and accessible focus states.
"""

from __future__ import annotations

# Theme color tokens
COLORS = {
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "primary_light": "#dbeafe",
    "text_primary_light": "#0f172a",
    "text_secondary_light": "#475569",
    "text_primary_dark": "#f8fafc",
    "text_secondary_dark": "#94a3b8",
    "border_light": "#e2e8f0",
    "border_dark": "#334155",
    "bg_subtle_light": "#f8fafc",
    "bg_subtle_dark": "#1e293b",
    "user_msg_bg_light": "#eff6ff",
    "user_msg_bg_dark": "rgba(37, 99, 235, 0.15)",
    "user_msg_border_light": "#bfdbfe",
    "user_msg_border_dark": "rgba(59, 130, 246, 0.3)",
    "emergency_bg_light": "#fef2f2",
    "emergency_bg_dark": "rgba(239, 68, 68, 0.12)",
    "emergency_border": "#ef4444",
    "emergency_text_light": "#991b1b",
    "emergency_text_dark": "#fca5a5",
}


def get_theme_css() -> str:
    """Return custom CSS rules for application layout, navigation, and chat styling."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main container bounds */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    /* Visible focus states for keyboard accessibility (WCAG AA) */
    button:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible,
    [tabindex]:focus-visible {
        outline: 2px solid #2563eb !important;
        outline-offset: 2px !important;
    }

    /* Main Title Styling */
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @media (prefers-color-scheme: dark) {
        .main-title {
            background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .main-subtitle {
        font-size: 0.92rem;
        color: var(--text-color, #475569);
        opacity: 0.85;
        margin-top: 0;
        margin-bottom: 12px;
    }

    /* Navigation control custom styling */
    [data-testid="stSegmentedControl"] {
        margin-bottom: 1rem;
    }

    /* Centered Chat Layout Column (max-width ~800px; full width on mobile) */
    .chat-layout-wrapper {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 4px;
    }

    @media (max-width: 768px) {
        .chat-layout-wrapper {
            max-width: 100% !important;
            padding: 0 2px !important;
        }
    }

    /* Chat header chip: Report metadata */
    .report-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        background-color: var(--secondary-background-color, rgba(148, 163, 184, 0.12));
        color: var(--text-color, #334155);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-top: 4px;
        margin-bottom: 12px;
    }

    /* Chat messages styling */
    /* User message: light tinted background, compact */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
    [data-testid="stChatMessage"]:has([data-testid="stIconMaterial-person"]) {
        background-color: var(--user-msg-bg, rgba(37, 99, 235, 0.06)) !important;
        border: 1px solid rgba(37, 99, 235, 0.18) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
    }

    /* Assistant message: no heavy background, normal body text, max ~75 characters line length */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]),
    [data-testid="stChatMessage"]:has([data-testid="stIconMaterial-medical_services"]) {
        background-color: transparent !important;
        border: none !important;
        padding: 6px 4px !important;
        margin-bottom: 12px !important;
        max-width: 75ch;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        max-width: 75ch;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Headings inside chat answers: small, semibold (1rem to 1.1rem) */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
        margin-bottom: 4px !important;
        line-height: 1.4 !important;
    }

    /* Calm but prominent emergency banner */
    .chat-emergency-banner {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #f87171;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.88rem;
        line-height: 1.5;
        font-weight: 500;
        margin-bottom: 10px;
    }

    @media (prefers-color-scheme: dark) {
        .chat-emergency-banner {
            background-color: rgba(239, 68, 68, 0.12);
            color: #fca5a5;
            border-color: rgba(248, 113, 113, 0.3);
            border-left: 4px solid #ef4444;
        }
    }

    /* Small disclaimer caption above/near input */
    .chat-input-caption {
        font-size: 0.76rem;
        color: var(--text-color, #64748b);
        opacity: 0.8;
        text-align: center;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    /* Sources expander title */
    .sources-expander-summary {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color, #475569);
    }

    /* Outlined suggested question buttons */
    .suggestion-btn button {
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        background-color: var(--secondary-background-color, transparent) !important;
        color: var(--text-color, inherit) !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Evidence chips container & individual chips */
    .evidence-chips-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
        margin-bottom: 6px;
    }

    .evidence-chip {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.78rem;
        font-weight: 500;
        border-radius: 6px;
        background-color: var(--secondary-background-color, rgba(148, 163, 184, 0.12));
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: var(--text-color, #334155);
    }

    /* Chat follow-up pills container above pinned input */
    .chat-followup-container {
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .chat-followup-container button {
        font-size: 0.82rem !important;
        padding: 4px 8px !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(37, 99, 235, 0.3) !important;
        background-color: var(--secondary-background-color, rgba(37, 99, 235, 0.04)) !important;
        color: var(--text-color, inherit) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .chat-followup-container button:hover {
        border-color: #2563eb !important;
        background-color: rgba(37, 99, 235, 0.12) !important;
        color: #2563eb !important;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """
