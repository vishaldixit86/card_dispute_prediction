# ---------------------------
# HARD FIX for Windows/Streamlit import path
# ---------------------------
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------
# Standard imports
# ---------------------------
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Package imports (now always resolvable)
from src.config import TEST_PATH
from src.agent1 import agent1_node
from src.agent2_runtime import load_agent2_artifacts, run_agent2, compute_shap_for_instance
from src.agent3_explainer import generate_explanation, generate_followup_explanation


# ---------------------------
# Session state
# ---------------------------

def init_session_state():
    if "stage" not in st.session_state:
        # 0=select, 1=agent1, 2=agent2, 3=agent3, 4=done
        st.session_state.stage = 0

    if "raw_input" not in st.session_state:
        st.session_state.raw_input: Dict[str, Any] = {}

    if "enriched_features" not in st.session_state:
        st.session_state.enriched_features: Dict[str, Any] = {}
    if "feature_explanations" not in st.session_state:
        st.session_state.feature_explanations: Dict[str, Any] = {}

    if "agent2_predictions" not in st.session_state:
        st.session_state.agent2_predictions: Dict[str, Any] = {}
    if "agent2_shap_top_features" not in st.session_state:
        st.session_state.agent2_shap_top_features = []

    if "nl_explanation" not in st.session_state:
        st.session_state.nl_explanation = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ""

    if "human_feedback" not in st.session_state:
        st.session_state.human_feedback = {
            "approved_enrichment": None,
            "approved_explanation": None,
        }

    if "agent2_loaded" not in st.session_state:
        st.session_state.agent2_loaded = False


def reset_downstream():
    st.session_state.enriched_features = {}
    st.session_state.feature_explanations = {}
    st.session_state.agent2_predictions = {}
    st.session_state.agent2_shap_top_features = []
    st.session_state.nl_explanation = ""
    st.session_state.conversation_history = ""
    st.session_state.human_feedback = {
        "approved_enrichment": None,
        "approved_explanation": None,
    }


# ---------------------------
# Data loading
# ---------------------------

def load_test_data() -> pd.DataFrame:
    if not os.path.exists(TEST_PATH):
        st.error(f"Missing file: {TEST_PATH}")
        st.info("Fix: ensure your test data exists at project_root/data/disputes_test.csv")
        st.stop()
    return pd.read_csv(TEST_PATH)


# ---------------------------
# Stage header / navigation
# ---------------------------

def render_stage_header():
    stage = st.session_state.stage
    cols = st.columns(5)
    labels = ["0 Select", "1 Agent1", "2 Agent2", "3 Agent3", "Done"]
    for i, c in enumerate(cols):
        with c:
            if i == stage:
                st.button(labels[i], disabled=True, use_container_width=True)
            else:
                if st.button(labels[i], use_container_width=True):
                    st.session_state.stage = i


# ---------------------------
# Stage 0: Select dispute
# ---------------------------

def stage_select_dispute(df: pd.DataFrame):
    st.header("Step 0 — Select dispute")

    id_col = "dispute_id" if "dispute_id" in df.columns else None

    left, right = st.columns([2, 1])

    with left:
        if id_col:
            dispute_id = st.selectbox("Select dispute_id", df[id_col].tolist())
            row = df[df[id_col] == dispute_id].iloc[0]
        else:
            idx = st.number_input("Row index", 0, len(df) - 1, 0, 1)
            row = df.iloc[int(idx)]
        selected_row = row.to_dict()

    # New selection => reset pipeline
    if (not st.session_state.raw_input) or (st.session_state.raw_input.get("dispute_id") != selected_row.get("dispute_id")):
        st.session_state.raw_input = selected_row
        reset_downstream()

    with right:
        st.caption("Actions")
        if st.button("🔄 Reload row", use_container_width=True):
            st.session_state.raw_input = selected_row
            reset_downstream()
            st.success("Row reloaded & pipeline reset.")
        if st.button("➡️ Go to Agent 1", use_container_width=True):
            st.session_state.stage = 1

    with st.expander("Raw dispute — tabular (Excel-like)", expanded=True):
        st.dataframe(pd.DataFrame([st.session_state.raw_input]), use_container_width=True, height=200)

    with st.expander("Raw dispute — JSON", expanded=False):
        st.json(st.session_state.raw_input)


# ---------------------------
# Stage 1: Agent 1
# ---------------------------

def stage_agent1():
    st.header("Step 1 — Agent 1: Enrichment + Human review")

    if not st.session_state.raw_input:
        st.warning("No dispute selected. Go to Step 0.")
        if st.button("⬅️ Back to Step 0", use_container_width=True):
            st.session_state.stage = 0
        return

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("⚙️ Run Enrichment", use_container_width=True):
            state = {"raw_input": st.session_state.raw_input}
            state = agent1_node(state)
            st.session_state.enriched_features = state["enriched_features"]
            st.session_state.feature_explanations = state["feature_explanations"]
            st.session_state.human_feedback["approved_enrichment"] = None
            st.success("Agent 1 complete.")
    with colB:
        if st.button("⬅️ Back to Step 0", use_container_width=True):
            st.session_state.stage = 0
    with colC:
        if st.button("➡️ Go to Agent 2", use_container_width=True):
            st.session_state.stage = 2

    with st.expander("Raw dispute (read-only)", expanded=False):
        st.dataframe(pd.DataFrame([st.session_state.raw_input]), use_container_width=True, height=180)

    if not st.session_state.enriched_features:
        st.info("Run enrichment to generate features.")
        return

    st.subheader("Human-in-loop overrides (optional)")
    st.caption("Edit key fields if needed, then click “Apply + Recompute”.")

    edit = st.session_state.raw_input.copy()
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        edit["mcc"] = st.number_input("mcc", value=int(edit.get("mcc", 0) or 0), step=1)
    with e2:
        edit["merchant_risk_tier"] = st.number_input("merchant_risk_tier", value=int(edit.get("merchant_risk_tier", 1) or 1), min_value=1, max_value=4, step=1)
    with e3:
        edit["txn_amount"] = st.number_input("txn_amount", value=float(edit.get("txn_amount", 0.0) or 0.0), step=1.0)
    with e4:
        edit["time_to_raise_days"] = st.number_input(
            "time_to_raise_days",
            value=float(edit.get("time_to_raise_days", edit.get("time_to_raise", 0.0)) or 0.0),
            step=1.0,
        )

    colX, colY, colZ = st.columns([1, 1, 1])
    with colX:
        if st.button("📝 Apply + Recompute", use_container_width=True):
            st.session_state.raw_input = edit
            reset_downstream()
            state = {"raw_input": st.session_state.raw_input}
            state = agent1_node(state)
            st.session_state.enriched_features = state["enriched_features"]
            st.session_state.feature_explanations = state["feature_explanations"]
            st.success("Recomputed Agent 1 with overrides.")
    with colY:
        if st.button("✅ Approve Enrichment → Agent 2", use_container_width=True):
            st.session_state.human_feedback["approved_enrichment"] = True
            st.session_state.stage = 2
    with colZ:
        if st.button("❌ Not approved (stay here)", use_container_width=True):
            st.session_state.human_feedback["approved_enrichment"] = False
            st.warning("Not approved. Adjust overrides and recompute.")

    with st.expander("Agent 1 output — enriched features", expanded=True):
        df_feats = pd.DataFrame(list(st.session_state.enriched_features.items()), columns=["feature", "value"])
        st.dataframe(df_feats, use_container_width=True, height=320)

    with st.expander("Agent 1 output — explanations", expanded=False):
        for k, v in st.session_state.feature_explanations.items():
            with st.expander(k, expanded=False):
                st.write(v)


# ---------------------------
# Stage 2: Agent 2
# ---------------------------

def stage_agent2():
    st.header("Step 2 — Agent 2: Prediction + SHAP")

    if not st.session_state.enriched_features:
        st.warning("Missing enriched features. Go back to Agent 1.")
        if st.button("⬅️ Back to Agent 1", use_container_width=True):
            st.session_state.stage = 1
        return

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("📊 Run Prediction", use_container_width=True):
            preds, shap_top = run_agent2(st.session_state.enriched_features)
            st.session_state.agent2_predictions = preds
            st.session_state.agent2_shap_top_features = shap_top
            st.success("Agent 2 complete.")
    with colB:
        if st.button("⬅️ Back to Agent 1", use_container_width=True):
            st.session_state.stage = 1
    with colC:
        if st.button("➡️ Go to Agent 3", use_container_width=True):
            st.session_state.stage = 3

    with st.expander("Input row (raw)", expanded=False):
        st.dataframe(pd.DataFrame([st.session_state.raw_input]), use_container_width=True, height=180)

    with st.expander("Input row (enriched features)", expanded=False):
        df_feats = pd.DataFrame(list(st.session_state.enriched_features.items()), columns=["feature", "value"])
        st.dataframe(df_feats, use_container_width=True, height=260)

    if not st.session_state.agent2_predictions:
        st.info("Run prediction to see outputs.")
        return

    preds = st.session_state.agent2_predictions

    with st.expander("Agent 2 output — summary", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted category", preds.get("predicted_category", "N/A"))
        with c2:
            st.metric("Customer-favor prob", f"{preds.get('predicted_customer_favor_prob', 0.0):.2%}")
        with c3:
            st.write("Category probs")
            st.write(preds.get("category_proba", {}))

    with st.expander("Agent 2 output — SHAP (table)", expanded=True):
        st.dataframe(pd.DataFrame(st.session_state.agent2_shap_top_features), use_container_width=True, height=260)

    with st.expander("Agent 2 output — SHAP (plot)", expanded=True):
        shap_vec, feat_cols = compute_shap_for_instance(st.session_state.enriched_features)
        abs_vals = np.abs(shap_vec)
        top_idx = np.argsort(-abs_vals)[:10]

        top_features = [feat_cols[i] for i in top_idx]
        top_vals = [float(shap_vec[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_features[::-1], top_vals[::-1])
        ax.set_xlabel("SHAP value (impact)")
        ax.set_title("Top contributions (single dispute)")
        plt.tight_layout()
        st.pyplot(fig)


# ---------------------------
# Stage 3: Agent 3
# ---------------------------

def stage_agent3():
    st.header("Step 3 — Agent 3: Explanation + Q&A")

    if not st.session_state.agent2_predictions:
        st.warning("Missing predictions. Go back to Agent 2.")
        if st.button("⬅️ Back to Agent 2", use_container_width=True):
            st.session_state.stage = 2
        return

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("🧠 Generate explanation", use_container_width=True):
            state = {
                "raw_input": st.session_state.raw_input,
                "enriched_features": st.session_state.enriched_features,
                "agent2_predictions": st.session_state.agent2_predictions,
                "agent2_shap_top_features": st.session_state.agent2_shap_top_features,
            }
            expl = generate_explanation(state)
            st.session_state.nl_explanation = expl
            st.session_state.conversation_history = expl
            st.session_state.human_feedback["approved_explanation"] = None
            st.success("Explanation generated.")
    with colB:
        if st.button("⬅️ Back to Agent 2", use_container_width=True):
            st.session_state.stage = 2
    with colC:
        if st.button("✅ Approve & Close", use_container_width=True):
            st.session_state.human_feedback["approved_explanation"] = True
            st.session_state.stage = 4

    if not st.session_state.nl_explanation:
        st.info("Generate explanation to proceed.")
        return

    with st.expander("Agent 3 output — explanation", expanded=True):
        st.text_area("Explanation", value=st.session_state.nl_explanation, height=230, disabled=True)

    st.subheader("Follow-up questions (human-in-loop)")
    q = st.text_input("Ask a follow-up question")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💬 Ask", use_container_width=True):
            if not q.strip():
                st.warning("Type a question first.")
            else:
                state = {
                    "raw_input": st.session_state.raw_input,
                    "enriched_features": st.session_state.enriched_features,
                    "agent2_predictions": st.session_state.agent2_predictions,
                    "agent2_shap_top_features": st.session_state.agent2_shap_top_features,
                }
                ans = generate_followup_explanation(
                    state=state,
                    previous_explanation=st.session_state.conversation_history,
                    user_question=q,
                )
                st.session_state.conversation_history += f"\n\n[Q] {q}\n[A] {ans}"
                st.session_state.nl_explanation += f"\n\nFollow-up:\n{ans}"
                st.success("Added follow-up.")
    with col2:
        if st.button("🧹 Reset Q&A history", use_container_width=True):
            st.session_state.conversation_history = st.session_state.nl_explanation
            st.success("History reset.")

    with st.expander("Conversation history", expanded=False):
        st.text_area("History", value=st.session_state.conversation_history, height=280, disabled=True)


# ---------------------------
# Stage 4: Done
# ---------------------------

def stage_done():
    st.header("✅ Case closed")

    preds = st.session_state.agent2_predictions or {}

    with st.expander("Final summary", expanded=True):
        st.write("**dispute_id:**", st.session_state.raw_input.get("dispute_id"))
        st.write("**predicted_category:**", preds.get("predicted_category"))
        st.write("**customer_favor_prob:**", f"{preds.get('predicted_customer_favor_prob', 0.0):.2%}")

    with st.expander("Final explanation", expanded=True):
        st.text_area("Explanation", value=st.session_state.nl_explanation, height=260, disabled=True)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🆕 Start new dispute", use_container_width=True):
            st.session_state.stage = 0
            st.session_state.raw_input = {}
            reset_downstream()
    with colB:
        if st.button("🔍 Review Agent 2", use_container_width=True):
            st.session_state.stage = 2


# ---------------------------
# Main
# ---------------------------

def main():
    st.set_page_config(page_title="Dispute Wizard", layout="wide")
    st.title("🧠 Dispute Prediction & Explainability Wizard")

    init_session_state()

    # Load models once (early)
    if not st.session_state.agent2_loaded:
        load_agent2_artifacts()
        st.session_state.agent2_loaded = True

    render_stage_header()
    st.markdown("---")

    df = load_test_data()

    stage = st.session_state.stage
    if stage == 0:
        stage_select_dispute(df)
    elif stage == 1:
        stage_agent1()
    elif stage == 2:
        stage_agent2()
    elif stage == 3:
        stage_agent3()
    else:
        stage_done()


if __name__ == "__main__":
    main()
