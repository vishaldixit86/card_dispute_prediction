# interactive_pipeline.py

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import os as _os
import pandas as pd

from agent1 import agent1_node
from agent2_runtime import load_agent2_artifacts, agent2_node
from agent3_explainer import agent3_node, generate_followup_explanation

DATA_DIR = "data"


def human_review_enrichment(state: dict) -> dict:
    """
    Level 1 human-in-loop:
    - Show raw_input & enriched features
    - Allow user to edit raw_input fields
    - When user APPROVES -> approved_enrichment = True (go forward)
    - When user REJECTS -> approved_enrichment = False (graph will loop back to agent1)
    """
    while True:
        raw = state["raw_input"]
        enriched = state["enriched_features"]
        expl = state["feature_explanations"]

        print("\n===== HUMAN REVIEW: ENRICHMENT (Agent 1) =====")
        print("Raw dispute (subset):")
        for k in ["txn_amount", "mcc", "mcc_group", "merchant_risk_tier",
                  "geo_distance_km", "time_to_raise_days", "dispute_text"]:
            if k in raw:
                print(f"  {k}: {raw[k]}")

        print("\nKey enriched features:")
        for k in ["transaction_risk_behavior_score", "policy_rules_count",
                  "policy_severity_score", "keyword_risk_score", "sentiment_score"]:
            if k in enriched:
                print(f"  {k}: {enriched[k]}")

        print("\nExplanations (summary):")
        for feat_name, info in expl.items():
            print(f"- {feat_name}: {info.get('explanation', '')}")

        print("\nOptions:")
        print("  1) Edit a raw field (e.g., mcc, merchant_risk_tier, limit_violation)")
        print("  2) Approve enrichment and continue to prediction")
        print("  3) Reject enrichment and recompute from Agent 1 (loop back in graph)")
        choice = input("Choose 1, 2, or 3 (default=2): ").strip() or "2"

        if choice == "2":
            state.setdefault("human_feedback", {})
            state["human_feedback"]["approved_enrichment"] = True
            return state

        if choice == "3":
            state.setdefault("human_feedback", {})
            state["human_feedback"]["approved_enrichment"] = False
            return state

        field_name = input("Enter raw_input field name to edit (or 'cancel' to go back): ").strip()
        if not field_name or field_name.lower() == "cancel":
            continue

        old_val = raw.get(field_name, None)
        print(f"Current value of raw_input['{field_name}'] = {old_val}")
        new_val = input("Enter new value: ").strip()

        if new_val == "":
            print("No change.")
        else:
            if isinstance(old_val, (int, float)):
                try:
                    cast_val = float(new_val)
                except ValueError:
                    cast_val = new_val
                raw[field_name] = cast_val
            else:
                raw[field_name] = new_val

            state["raw_input"] = raw
            print("Re-running Agent1 enrichment with updated raw_input...")
            state = {"raw_input": state["raw_input"]}
            state = agent1_node(state)


def human_review_explanation(state: dict) -> dict:
    """
    Level 2 human-in-loop:
    - Show initial explanation
    - Allow analyst to ask follow-up questions
    - 'approve' -> approved_explanation = True and finish
    - 'no'/Enter -> approved_explanation = False and finish
    """
    base_explanation = state.get("nl_explanation", "")
    if not base_explanation:
        print("No initial explanation found in state['nl_explanation'].")
        return state

    print("\n===== INITIAL MODEL EXPLANATION (Agent 3) =====\n")
    print(base_explanation)

    conversation_history = base_explanation

    while True:
        print("\nYou can:")
        print("  - Ask a follow-up question (type your question)")
        print("  - Type 'approve' to accept explanation and close")
        print("  - Press Enter or type 'no' to stop without explicit approval")
        follow_q = input("Your input: ").strip()

        if follow_q.lower() == "approve":
            state.setdefault("human_feedback", {})
            state["human_feedback"]["approved_explanation"] = True
            print("Explanation approved. Ending explanation loop.")
            break

        if not follow_q or follow_q.lower() in ("no", "n"):
            state.setdefault("human_feedback", {})
            state["human_feedback"]["approved_explanation"] = False
            print("No more questions. Ending explanation loop.")
            break

        follow_ans = generate_followup_explanation(
            state=state,
            previous_explanation=conversation_history,
            user_question=follow_q,
        )
        print("\n--- Follow-up answer ---\n")
        print(follow_ans)
        conversation_history += "\n\n" + follow_ans

    return state


def main():
    load_agent2_artifacts()

    test_path = _os.path.join(DATA_DIR, "disputes_test.csv")
    if not _os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found. Run train_agent2.py first."
        )

    df_test = pd.read_csv(test_path)
    sample_row = df_test.iloc[0].to_dict()

    state = {"raw_input": sample_row}
    state = agent1_node(state)

    state = human_review_enrichment(state)

    state = agent2_node(state)

    print("\n===== Agent2 Predictions =====")
    print(state["agent2_predictions"])

    print("\n===== Agent2 Top SHAP features =====")
    for item in state["agent2_shap_top_features"]:
        print(item)

    state = agent3_node(state)

    state = human_review_explanation(state)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
