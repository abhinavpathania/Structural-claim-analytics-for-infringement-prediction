"""Streamlit frontend for Patent Infringement Risk Prediction."""
from __future__ import annotations

import os
os.environ.setdefault("HF_TOKEN", "hf_suppress")

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Import from project
from src.feature_builder import FeatureBuilder
from src.explain import explain_single

# Page config
st.set_page_config(
    page_title="Patent Infringement Risk Analyzer",
    page_icon="⚖️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[0]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

# Risk thresholds
RISK_THRESHOLDS = {
    "Minimal Risk": (0, 20),
    "Low Risk": (20, 50),
    "Moderate Risk": (50, 75),
    "High Risk": (75, 90),
    "Critical Risk": (90, 100),
}

RISK_COLORS = {
    "Minimal Risk": "🟢",
    "Low Risk": "🟢",
    "Moderate Risk": "🟡",
    "High Risk": "🟠",
    "Critical Risk": "🔴",
}

def get_risk_category(score: float) -> str:
    """Categorize risk score into descriptive labels."""
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return category
    if score >= 100:
        return "Critical Risk"
    return "Minimal Risk"

def load_model():
    """Load the trained model and feature columns."""
    pack = joblib.load(MODEL_PATH)
    return pack["model"], pack.get("calibrator"), pack["feature_columns"]

@st.cache_resource
def get_feature_builder():
    """Get or create the FeatureBuilder (cached)."""
    return FeatureBuilder()

def predict_risk(claim: str, product: str, show_explanation: bool = False):
    """Run the risk prediction."""
    model, calibrator, cols = load_model()
    fb = get_feature_builder()

    feats = fb.build_feature_vector(claim, product)
    X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0)
    score = float(model.predict(X)[0])
    score = np.clip(score, 0, 100)

    if calibrator is not None:
        score = float(calibrator.predict([score])[0])

    category = get_risk_category(score)
    semantic_sim = feats["semantic_similarity"]

    result = {
        "score": score,
        "category": category,
        "semantic_similarity": semantic_sim * 100,
        "features": feats
    }

    if show_explanation:
        result["explanation"] = explain_single(claim, product, topk=8)

    return result

def render_gauge(score: float) -> None:
    """Render a visual gauge for the risk score."""
    # Calculate width percentages for color bands
    minimal = min(score / 20, 1.0) * 20 if score <= 20 else 20
    low = min(max(score - 20, 0) / 30, 1.0) * 30 if score > 20 else 0
    moderate = min(max(score - 50, 0) / 25, 1.0) * 25 if score > 50 else 0
    high = min(max(score - 75, 0) / 15, 1.0) * 15 if score > 75 else 0
    critical = min(max(score - 90, 0) / 10, 1.0) * 10 if score > 90 else 0

    if score <= 20:
        low = moderate = high = critical = 0
    elif score <= 50:
        moderate = high = critical = 0
    elif score <= 75:
        high = critical = 0
    elif score <= 90:
        critical = 0

    html = f"""
    <div style="width: 100%; max-width: 500px; margin: 0 auto;">
        <div style="display: flex; height: 30px; border-radius: 4px; overflow: hidden; background: linear-gradient(to right, #22c55e 0%, #22c55e 20%, #84cc16 20%, #84cc16 50%, #eab308 50%, #eab308 75%, #f97316 75%, #f97316 90%, #ef4444 90%, #ef4444 100%);">
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
            <span style="color: #22c55e;">0</span>
            <span style="color: #84cc16;">20</span>
            <span style="color: #eab308;">50</span>
            <span style="color: #f97316;">75</span>
            <span style="color: #ef4444;">90</span>
            <span style="color: #ef4444;">100</span>
        </div>
        <div style="text-align: center; font-size: 48px; font-weight: bold; margin-top: 20px; color: {'#ef4444' if score >= 75 else '#eab308' if score >= 50 else '#22c55e'};">
            {score:.1f}%
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_feature_breakdown(features: dict) -> None:
    """Render feature breakdown."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Claim Features")
        claim_feats = {
            "Word Count": features.get("claim_word_count", 0),
            "Token Count": features.get("claim_token_count", 0),
            "Clause Markers": features.get("claim_clause_markers", 0),
            "Dependency Depth (Max)": features.get("claim_dep_depth_max", 0),
            "Functional Verb Density": features.get("claim_func_verb_density", 0),
            "Structural Noun Density": features.get("claim_struct_noun_density", 0),
            "Abstraction Score": features.get("claim_abstraction_score", 0),
            "Abstractness": features.get("claim_abstractness", 0),
        }
        for key, value in claim_feats.items():
            if isinstance(value, float):
                st.metric(key, f"{value:.3f}")
            else:
                st.metric(key, int(value))

    with col2:
        st.subheader("🏭 Product Features")
        prod_feats = {
            "Token Count": features.get("prod_token_count", 0),
            "Component Terms": features.get("prod_component_terms", 0),
            "Interaction Terms": features.get("prod_interaction_terms", 0),
            "Functional Terms": features.get("prod_functional_terms", 0),
            "Functional Density": features.get("prod_functional_density", 0),
        }
        for key, value in prod_feats.items():
            if isinstance(value, float):
                st.metric(key, f"{value:.3f}")
            else:
                st.metric(key, int(value))

    st.divider()

    col3, col4, col5 = st.columns(3)

    with col3:
        st.subheader("🔗 Overlap Metrics")
        overlap = {
            "Semantic Similarity": f"{features.get('semantic_similarity', 0) * 100:.1f}%",
            "Noun Jaccard": f"{features.get('noun_jaccard', 0):.3f}",
            "Bigram Jaccard": f"{features.get('bigram_jaccard', 0):.3f}",
            "Trigram Jaccard": f"{features.get('trigram_jaccard', 0):.3f}",
        }
        for key, value in overlap.items():
            st.metric(key, value)

    with col4:
        st.subheader("🎯 Domain Alignment")
        domain = {
            "Same Domain": "Yes" if features.get("same_domain", 0) else "No",
            "Domain Overlap": int(features.get("domain_overlap_count", 0)),
            "Domain Mismatch": "Yes" if features.get("domain_mismatch_penalty", 0) else "No",
            "Abstraction Alignment": f"{features.get('abs_alignment', 0):.3f}",
        }
        for key, value in domain.items():
            st.metric(key, value)

    with col5:
        st.subheader("📏 Length Metrics")
        length = {
            "Length Ratio": f"{features.get('len_ratio', 0):.3f}",
            "Length Diff": f"{features.get('len_diff_norm', 0):.3f}",
            "Complexity Balance": f"{features.get('complexity_balance', 0):.3f}",
        }
        for key, value in length.items():
            st.metric(key, value)

def render_explanation(explanation: dict) -> None:
    """Render SHAP explanation."""
    if not explanation:
        return

    st.subheader("🔬 SHAP Feature Explanations")

    if "shap_values" in explanation:
        importance = explanation["shap_values"]
        if importance:
            # Convert dict to DataFrame for chart
            items = list(importance.items())
            df = pd.DataFrame(items, columns=["Feature", "Contribution"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Base Score", f"{explanation.get('base_value', 0):.2f}")
            with col2:
                st.metric("Final Score", f"{explanation.get('prediction', 0):.2f}")

            st.markdown("**Top Contributing Features:**")
            st.bar_chart(df.set_index("Feature"))

            # Show grouped contribution
            if "grouped_shap" in explanation:
                st.markdown("**Feature Groups:**")
                grouped = explanation["grouped_shap"]
                group_df = pd.DataFrame(list(grouped.items()), columns=["Group", "Contribution"])
                st.dataframe(group_df, use_container_width=True)

            with st.expander("View raw SHAP values"):
                for feat, val in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
                    direction = "↑" if val > 0 else "↓"
                    st.write(f"{direction} {feat}: {val:+.5f}")

# Main UI
def main():
    st.title("⚖️ Patent Infringement Risk Analyzer")
    st.markdown("""
    Analyze patent claims against product descriptions to assess infringement risk.
    The model evaluates structural, semantic, and domain-level features to predict risk.
    """)

    # Input section
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📜 Patent Claim")
            claim = st.text_area(
                "Enter patent claim text",
                height=200,
                placeholder="Example: A system for synchronizing data between a server and a client device, comprising: a processor, a memory, and a network interface; wherein said processor is configured to transmit update signals..."
            )

        with col2:
            st.subheader("🏭 Product Description")
            product = st.text_area(
                "Enter product description",
                height=200,
                placeholder="Example: A smartphone app that syncs calendar events with cloud storage using real-time push notifications..."
            )

    # Options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_details = st.checkbox("Show Feature Details", value=True)
    with col2:
        show_explanation = st.checkbox("Show SHAP Explanation")

    # Predict button
    if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
        if not claim.strip() or not product.strip():
            st.warning("Please enter both claim and product text.")
            return

        with st.spinner("Analyzing..."):
            try:
                result = predict_risk(claim, product, show_explanation)

                st.divider()

                # Results header
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Risk Score", f"{result['score']:.2f}%")

                with col2:
                    category = result['category']
                    emoji = RISK_COLORS.get(category, "⚪")
                    st.markdown(f"### {emoji} {category}")

                with col3:
                    st.metric("Semantic Similarity", f"{result['semantic_similarity']:.1f}%")

                # Gauge visualization
                render_gauge(result['score'])

                # Feature breakdown
                if show_details:
                    st.divider()
                    render_feature_breakdown(result['features'])

                # SHAP explanation
                if show_explanation and "explanation" in result:
                    st.divider()
                    render_explanation(result['explanation'])

                # Disclaimer
                st.caption("""
                ⚠️ **Disclaimer**: This tool is for demonstration purposes only.
                Actual patent infringement determination requires legal expertise
                and comprehensive analysis by qualified professionals.
                """)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please ensure the model has been trained before using the predictor.")

    # Example cases section
    with st.expander("📋 Example Test Cases"):
        st.markdown("""
        **Test different risk levels:**

        | Risk Level | Claim | Product |
        |------------|-------|---------|
        | **Low** | A system for displaying images on a screen | A smartphone with a color LCD display |
        | **Moderate** | A method of processing digital signals using Fourier transforms to filter frequency components | An audio equalizer app that processes sound waves using FFT analysis |
        | **High/Critical** | A neural network apparatus comprising: an input layer, a plurality of hidden layers, and an output layer; wherein said hidden layers use gradient descent optimization for training | A deep learning framework implementing multi-layer perceptrons trained via backpropagation with gradient descent |
        """)

if __name__ == "__main__":
    main()
