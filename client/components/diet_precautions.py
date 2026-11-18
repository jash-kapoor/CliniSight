import streamlit as st
try:
    from utils.emoji import strip_emoji
except ImportError:
    from client.utils.emoji import strip_emoji


def render_diet_and_precautions():
    """Renders clinical precautions, Indian diet guidance, and 1-day meal plan with theme contrast."""
    insights = st.session_state.get("insights")

    if not insights:
        st.info("Please upload and analyze a lab report first to view personalized diet & lifestyle precautions.")
        return

    precautions = insights.get("precautions", [])
    diet = insights.get("diet_guidance", {})
    meal_plan = diet.get("sample_meal_plan", {}) if diet else {}

    # Section 1: Clinical Precautions & Action Steps
    st.markdown("### Clinical Precautions & Action Plan")
    st.caption("Targeted behavioral and lifestyle interventions derived from your diagnostic parameters.")

    if not precautions:
        st.write("No specific lifestyle precautions flagged for this report.")
    else:
        for idx, p in enumerate(precautions, 1):
            concern = strip_emoji(p.get("target_concern", "Health Observation"))
            steps = p.get("practical_steps", [])
            retest = strip_emoji(p.get("when_to_retest", "3 months"))
            specialist = strip_emoji(p.get("specialist_to_consult", "General Physician"))
            citation = strip_emoji(p.get("evidence_citation", ""))

            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(f"#### Concern #{idx}: {concern}")
                with head_col2:
                    st.markdown(f"`Specialist: {specialist}`")

                if steps:
                    st.markdown("**Practical Lifestyle Steps:**")
                    for step in steps:
                        st.markdown(f"- {strip_emoji(step)}")

                st.markdown(f"**When to Retest:** {retest}")
                if citation:
                    st.caption(f"Evidence: `{citation}`")

    st.markdown("<hr style='margin: 25px 0; border: 0; border-top: 1px solid rgba(148, 163, 184, 0.2);'>", unsafe_allow_html=True)

    # Section 2: Dietary Guidelines
    st.markdown("### Dietary Guidance & Food Choices")
    st.caption("Nutritional advice tailored to your biochemical markers.")

    col_prefer, col_limit = st.columns(2)

    with col_prefer:
        with st.container(border=True):
            st.markdown("#### Foods to Prefer")
            prefer_items = diet.get("foods_to_prefer", [])
            if prefer_items:
                for item in prefer_items:
                    st.markdown(f"- {strip_emoji(item)}")
            else:
                st.caption("No specific items listed.")

    with col_limit:
        with st.container(border=True):
            st.markdown("#### Foods to Limit or Avoid")
            limit_items = diet.get("foods_to_limit", [])
            if limit_items:
                for item in limit_items:
                    st.markdown(f"- {strip_emoji(item)}")
            else:
                st.caption("No specific items listed.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Indian Diet Recommendations
    indian_recs = diet.get("indian_diet_recommendations", [])
    if indian_recs:
        st.markdown("### Indian Diet Recommendations")
        st.caption("Culturally adapted dietary tips utilizing Indian staples, pulses, and traditional preparations.")
        
        with st.container(border=True):
            for r in indian_recs:
                st.markdown(f"- **{strip_emoji(r)}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 4: Sample 1-Day Indian Meal Plan
    if meal_plan:
        st.markdown("### Sample 1-Day Indian Meal Plan")
        st.caption("Balanced sample meal schedule incorporating high-fiber, low-glycemic, and cardioprotective options.")

        mp_cols = st.columns(5)
        with mp_cols[0]:
            st.markdown("**Breakfast**")
            st.info(strip_emoji(meal_plan.get("breakfast", "Oats / Vegetable Poha with sprouts")))
        with mp_cols[1]:
            st.markdown("**Mid-Morning**")
            st.info(strip_emoji(meal_plan.get("mid_morning") or "Green tea & handful of roasted walnuts/almonds"))
        with mp_cols[2]:
            st.markdown("**Lunch**")
            st.info(strip_emoji(meal_plan.get("lunch", "Multigrain / Jowar roti, yellow dal, green leafy sabzi, and fresh salad")))
        with mp_cols[3]:
            st.markdown("**Evening Snack**")
            st.info(strip_emoji(meal_plan.get("evening_snack") or "Roasted makhana / Sprouted moong chaat"))
        with mp_cols[4]:
            st.markdown("**Dinner**")
            st.info(strip_emoji(meal_plan.get("dinner", "Light khichdi with vegetables or paneer/tofu salad")))

        notes = meal_plan.get("notes")
        if notes:
            st.caption(f"**Clinical Dietitian's Note:** {strip_emoji(notes)}")
