import os
import json
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import shap

from config import (
    MODELS_DIR,
    FEATURE_COLS_PATH,
    CATEGORY_MODEL_PATH,
    FAVOR_MODEL_PATH,
    LABEL_ENCODER_PATH,
)

AGENT2_OBJECTS: Dict[str, Any] = {}


def load_agent2_artifacts() -> None:
    """Load trained model artifacts and build SHAP explainer."""
    global AGENT2_OBJECTS

    missing = [p for p in [CATEGORY_MODEL_PATH, FAVOR_MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_COLS_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts:\n" + "\n".join(missing) +
            "\nRun train_agent2.py to generate these in /models."
        )

    with open(CATEGORY_MODEL_PATH, "rb") as f:
        category_model = pickle.load(f)

    with open(FAVOR_MODEL_PATH, "rb") as f:
        favor_model = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    explainer = shap.TreeExplainer(category_model)

    AGENT2_OBJECTS = {
        "category_model": category_model,
        "favor_model": favor_model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "explainer": explainer,
        "models_dir": MODELS_DIR,
    }


def ensure_loaded() -> None:
    if not AGENT2_OBJECTS:
        raise RuntimeError("Agent2 artifacts not loaded. Call load_agent2_artifacts() first.")


def _vectorize(enriched_features: Dict[str, Any], feature_cols: List[str]) -> np.ndarray:
    return np.array([[float(enriched_features.get(col, 0.0)) for col in feature_cols]], dtype=float)


def _get_predicted_class_index(cat_model, x: np.ndarray) -> Tuple[int, np.ndarray]:
    proba = cat_model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    return pred_idx, proba


def _extract_shap_vector_for_class(shap_values, pred_idx: int) -> np.ndarray:
    """
    Robust across SHAP return shapes/versions:
      - list of arrays per class
      - ndarray (1, n_features, n_classes) or (n_classes, 1, n_features)
      - ndarray (1, n_features)
    """
    if isinstance(shap_values, list):
        return np.array(shap_values[pred_idx][0], dtype=float)

    arr = np.array(shap_values)
    if arr.ndim == 2:
        return np.array(arr[0], dtype=float)

    if arr.ndim == 3:
        # (1, n_features, n_classes)
        if arr.shape[0] == 1 and arr.shape[2] > 1:
            return np.array(arr[0, :, pred_idx], dtype=float)
        # (n_classes, 1, n_features)
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            return np.array(arr[pred_idx, 0, :], dtype=float)

    # fallback
    try:
        return np.array(arr[pred_idx][0], dtype=float)
    except Exception:
        return np.array(arr).reshape(-1).astype(float)


def run_agent2(enriched_features: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Predict category + customer-favor probability + top SHAP features for predicted category.
    Returns: (predictions_dict, shap_top_list)
    """
    ensure_loaded()

    cat_model = AGENT2_OBJECTS["category_model"]
    fav_model = AGENT2_OBJECTS["favor_model"]
    le = AGENT2_OBJECTS["label_encoder"]
    feature_cols = AGENT2_OBJECTS["feature_cols"]
    explainer = AGENT2_OBJECTS["explainer"]

    x = _vectorize(enriched_features, feature_cols)
    pred_idx, proba = _get_predicted_class_index(cat_model, x)

    pred_label = le.inverse_transform([pred_idx])[0]
    favor_proba = float(fav_model.predict_proba(x)[0, 1])

    shap_values = explainer.shap_values(x)
    shap_vec = _extract_shap_vector_for_class(shap_values, pred_idx)

    abs_contrib = np.abs(shap_vec)
    top_idx = np.argsort(-abs_contrib)[:10]

    shap_top = []
    for i in top_idx:
        sv = float(shap_vec[i])
        shap_top.append({
            "feature": feature_cols[i],
            "value": float(x[0, i]),
            "shap_value": sv,
            "direction": "pushes_towards_class" if sv > 0 else "pushes_away_from_class",
        })

    predictions = {
        "predicted_category": pred_label,
        "predicted_customer_favor_prob": favor_proba,
        "category_proba": {cls: float(proba[i]) for i, cls in enumerate(le.classes_)},
    }

    return predictions, shap_top


def compute_shap_for_instance(enriched_features: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    For one row: returns (shap_vector_for_predicted_class, feature_cols)
    Used for plotting.
    """
    ensure_loaded()

    cat_model = AGENT2_OBJECTS["category_model"]
    feature_cols = AGENT2_OBJECTS["feature_cols"]
    explainer = AGENT2_OBJECTS["explainer"]

    x = _vectorize(enriched_features, feature_cols)
    pred_idx, _ = _get_predicted_class_index(cat_model, x)

    shap_values = explainer.shap_values(x)
    shap_vec = _extract_shap_vector_for_class(shap_values, pred_idx)

    return shap_vec, feature_cols


def agent2_node(state: Dict[str, Any]) -> Dict[str, Any]:
    preds, shap_top = run_agent2(state["enriched_features"])
    state["agent2_predictions"] = preds
    state["agent2_shap_top_features"] = shap_top
    return state


if __name__ == "__main__":
    load_agent2_artifacts()
    print("Loaded Agent2 artifacts OK.")
