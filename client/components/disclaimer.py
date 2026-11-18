import textwrap
import streamlit as st


def render_disclaimer_banner():
    """Renders prominent educational medical disclaimer banner with theme contrast."""
    banner_html = textwrap.dedent("""
    <div style="background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.35); border-left: 4px solid #ef4444; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 0.90rem; color: inherit; display: flex; align-items: flex-start; gap: 12px;">
        <span style="background-color: rgba(239, 68, 68, 0.2); color: #b91c1c; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 2px;">Notice</span>
        <div style="line-height: 1.5;">
            <strong style="color: #ef4444;">Medical & Clinical Disclaimer:</strong>
            This application is an AI-powered clinical laboratory explainer intended for informational and educational purposes only.
            It does <em>not</em> provide formal medical diagnoses, clinical determinations, or prescriptive treatment plans.
            Always consult your licensed physician or healthcare specialist to interpret your diagnostic results and before initiating any dietary or medical changes.
        </div>
    </div>
    """).strip()
    st.markdown(banner_html, unsafe_allow_html=True)
