import os
import json
from typing import Dict, Any, Tuple

import numpy as np

from config import FEATURE_COLS_PATH


# --- Your keyword map can be expanded anytime ---
KEYWORD_WEIGHTS = {
    "unauthorized": 0.35,
    "fraud": 0.35,
    "scam": 0.25,
    "never happened": 0.25,
    "stolen": 0.25,
    "chargeback": 0.20,
    "duplicate": 0.15,
    "refund": 0.10,
    "cancel": 0.10,
}

# Very lightweight sentiment proxy (keeps dependencies minimal)
NEGATIVE_WORDS = {"angry", "frustrated", "upset", "terrible", "worst", "hate"}
POSITIVE_WORDS = {"please", "thanks", "thank you"}


def load_feature_cols():
    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            f"{FEATURE_COLS_PATH} not found. Run train_agent2.py first."
        )
    with open(FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _text_sentiment_score(text: str) -> float:
    """
    Cheap sentiment proxy in [-1, 1].
    Negative words push down; polite words push up slightly.
    """
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 0.15
    for w in POSITIVE_WORDS:
        if w in t:
            score += 0.05
    return float(np.clip(score, -1.0, 1.0))


def _keyword_risk(text: str) -> Tuple[float, int, Dict[str, float], list]:
    if not text:
        return 0.0, 0, {}, []
    t = text.lower()
    matched = []
    weights = {}
    score = 0.0
    for k, w in KEYWORD_WEIGHTS.items():
        if k in t:
            matched.append(k)
            weights[k] = w
            score += w
    score = float(min(score, 1.0))
    flag = 1 if score >= 0.25 else 0
    return score, flag, weights, matched


def enrich_dispute(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Produces enriched_features + feature_explanations.
    Also filters output to only include feature columns used by Agent2.
    """
    feature_cols = load_feature_cols()

    txn_amount = _safe_float(raw.get("txn_amount"), 0.0)
    avg_monthly_spend = _safe_float(raw.get("avg_monthly_spend"), 0.0)
    time_to_raise_days = _safe_float(raw.get("time_to_raise_days"), raw.get("time_to_raise", 0.0))
    merchant_risk_tier = int(_safe_float(raw.get("merchant_risk_tier"), 1))
    geo_distance_km = _safe_float(raw.get("geo_distance_km"), 0.0)

    dispute_text = (
        raw.get("dispute_reason_text")
        or raw.get("dispute_text")
        or raw.get("customer_dispute_text")
        or ""
    )

    # --- Example policy rule signals ---
    limit_violation = 1 if (avg_monthly_spend > 0 and txn_amount > 0.8 * avg_monthly_spend) else 0
    policy_rules_count = int(limit_violation)  # you can add more rules here
    policy_severity_score = float(limit_violation * 2.0)

    # --- amount flags ---
    medium_amount_flag = 1 if txn_amount >= 200 else 0

    # --- dispute timing flags ---
    very_fast_dispute_flag = 1 if time_to_raise_days <= 1 else 0
    late_dispute_flag = 1 if time_to_raise_days >= 30 else 0

    # --- text features ---
    sentiment_score = _text_sentiment_score(dispute_text)
    keyword_risk_score, keyword_risk_flag, kw_weights, matched_keywords = _keyword_risk(dispute_text)

    # --- composite score (simple) ---
    # This is NOT the model; just a helpful “risk summary”
    tier_score = {1: 0.10, 2: 0.30, 3: 0.60, 4: 0.90}.get(merchant_risk_tier, 0.30)
    geo_boost = 0.2 if geo_distance_km > 200 else 0.05 if geo_distance_km > 20 else 0.0
    transaction_risk_behavior_score = float(
        np.clip(0.4 * tier_score + 0.3 * (policy_severity_score / 10.0) + 0.3 * (keyword_risk_score + geo_boost), 0, 1)
    )

    # --- assemble ---
    enriched = {
        "txn_amount": txn_amount,
        "avg_monthly_spend": avg_monthly_spend,
        "time_to_raise_days": time_to_raise_days,
        "merchant_risk_tier": merchant_risk_tier,
        "geo_distance_km": geo_distance_km,

        "limit_violation": limit_violation,
        "policy_rules_count": policy_rules_count,
        "policy_severity_score": policy_severity_score,

        "medium_amount_flag": medium_amount_flag,
        "very_fast_dispute_flag": very_fast_dispute_flag,
        "late_dispute_flag": late_dispute_flag,

        "keyword_risk_flag": keyword_risk_flag,
        "keyword_risk_score": keyword_risk_score,
        "sentiment_score": sentiment_score,

        "transaction_risk_behavior_score": transaction_risk_behavior_score,
    }

    explanations = {
        "keyword_risk_score": {
            "value": keyword_risk_score,
            "keyword_risk_flag": keyword_risk_flag,
            "matched_keywords": matched_keywords,
            "weights": kw_weights,
            "raw_text": dispute_text,
            "explanation": "Risk score = sum(keyword weights) capped at 1.0. Flag if score >= 0.25.",
        },
        "policy_rules": {
            "limit_violation": limit_violation,
            "policy_rules_count": policy_rules_count,
            "policy_severity_score": policy_severity_score,
            "explanation": "Simple POC rule: txn_amount > 0.8 * avg_monthly_spend triggers limit_violation.",
        },
        "timing": {
            "time_to_raise_days": time_to_raise_days,
            "very_fast_dispute_flag": very_fast_dispute_flag,
            "late_dispute_flag": late_dispute_flag,
            "explanation": "Very fast disputes (<=1 day) can correlate with fraud; late disputes (>=30 days) can correlate with buyer’s remorse.",
        },
        "transaction_risk_behavior_score": {
            "value": transaction_risk_behavior_score,
            "explanation": "POC composite: 0.4*tier + 0.3*policy + 0.3*(keyword + geo boost).",
        },
    }

    # Keep only columns Agent2 was trained on
    filtered_enriched = {c: float(enriched.get(c, 0.0)) for c in feature_cols}

    return filtered_enriched, explanations


def agent1_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw = state.get("raw_input", {})
    enriched, expl = enrich_dispute(raw)
    state["enriched_features"] = enriched
    state["feature_explanations"] = expl
    return state
