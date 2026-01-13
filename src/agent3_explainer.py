# agent3_explainer.py

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from typing import Dict, Any
from openai import OpenAI

client = OpenAI()


def build_explainer_prompt(state: Dict[str, Any]) -> str:
    raw = state.get("raw_input", {})
    feats = state.get("enriched_features", {})
    preds = state.get("agent2_predictions", {})
    shap_top = state.get("agent2_shap_top_features", [])

    predicted_category = preds.get("predicted_category", "N/A")
    category_proba = preds.get("category_proba", {})
    favor_prob = preds.get("predicted_customer_favor_prob", 0.0)

    prompt = f"""
You are a credit card dispute risk analyst assistant.

You are given:

1) Raw dispute record (transaction + customer + merchant + text):
{raw}

2) Enriched risk features (numeric signals used by the model):
{feats}

3) Model predictions:
- Predicted dispute category: {predicted_category}
- Category probabilities: {category_proba}
- Probability of resolving in customer favor: {favor_prob:.2f}

4) Top SHAP feature contributions for the predicted category:
{shap_top}

TASK:
Explain the model output to a human analyst in a **very concise** way.

OUTPUT FORMAT (IMPORTANT):
- Return **5–7 bullet points only**.
- Each bullet should be **one short sentence**, max 2 lines.
- No long paragraphs, no repetition, no generic filler.

CONTENT TO COVER:
- Why the dispute is classified in this category (True Dispute, First Party Fraud, or Third Party Fraud).
- The top 3–5 features that pushed the score **towards** this category.
- Any key feature(s) that **reduced** the risk or pulled away from this category.
- How confident the model is (high / medium / low, based on probabilities).
- A short statement on likelihood of customer-favor outcome and why.

Be precise but brief. Avoid overclaiming; use wording like "likely", "strong signal", "weaker signal".
"""
    return prompt


def generate_explanation(state: Dict[str, Any]) -> str:
    prompt = build_explainer_prompt(state)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise risk analyst. "
                    "Always answer in short bullet points, no long paragraphs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    return response.choices[0].message.content


def generate_followup_explanation(
    state: Dict[str, Any],
    previous_explanation: str,
    user_question: str,
) -> str:
    raw = state.get("raw_input", {})
    preds = state.get("agent2_predictions", {})
    shap_top = state.get("agent2_shap_top_features", [])

    prompt = f"""
You previously explained the dispute prediction as follows:

[PREVIOUS EXPLANATION]
{previous_explanation}

The analyst now asks a follow-up question:
[QUESTION]
{user_question}

CONSTRAINTS:
- Do NOT change or contradict the original model prediction or probabilities.
- Use the raw dispute, SHAP features, and previous explanation as context.
- Answer **very concisely** in bullet points only.

OUTPUT FORMAT:
- Return **3–5 bullet points**.
- Each bullet: one short sentence.
- Focus on clarifying the reasoning, feature impact, or what-if logic.

Context:
- Raw dispute: {raw}
- Model predictions: {preds}
- Top SHAP features: {shap_top}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a detailed but concise risk model explainer. "
                    "Use 3–5 very short bullet points; no long paragraphs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content


def agent3_node(state: Dict[str, Any]) -> Dict[str, Any]:
    explanation = generate_explanation(state)
    state["nl_explanation"] = explanation
    return state


if __name__ == "__main__":
    # lightweight smoke test (requires OPENAI_API_KEY)
    demo_state = {
        "raw_input": {
            "txn_amount": 123.45,
            "dispute_text": "This charge is unauthorized.",
            "dispute_id": "DEMO_001",
        },
        "enriched_features": {
            "txn_amount": 123.45,
            "transaction_risk_behavior_score": 0.8,
            "keyword_risk_score": 0.9,
        },
        "agent2_predictions": {
            "predicted_category": "third_party_fraud",
            "predicted_customer_favor_prob": 0.78,
            "category_proba": {
                "first_party_fraud": 0.05,
                "third_party_fraud": 0.78,
                "true_dispute": 0.17,
            },
        },
        "agent2_shap_top_features": [
            {
                "feature": "transaction_risk_behavior_score",
                "value": 0.8,
                "shap_value": 0.25,
                "direction": "pushes_towards_class",
            }
        ],
    }

    first = generate_explanation(demo_state)
    print("=== First explanation ===\n", first)

    follow = generate_followup_explanation(
        demo_state,
        previous_explanation=first,
        user_question="Why exactly did it classify as third-party fraud instead of true dispute?",
    )
    print("\n=== Follow-up explanation ===\n", follow)
