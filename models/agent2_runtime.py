import os
import json
import pickle
import numpy as np
import shap
import pandas as pd  # used in __main__ test

MODELS_DIR = "models"
AGENT2_OBJECTS = {}


def load_agent2_artifacts():
    global AGENT2_OBJECTS

    cat_path = os.path.join(MODELS_DIR, "category_model.pkl")
    fav_path = os.path.join(MODELS_DIR, "favor_model.pkl")
    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    feat_path = os.path.join(MODELS_DIR, "feature_cols.json")

    if not (os.path.exists(cat_path) and os.path.exists(fav_path)
            and os.path.exists(le_path) and os.path.exists(feat_path)):
        raise FileNotFoundError(
            "Model artifacts missing in 'models/'. Run train_agent2.py first."
        )

    with open(cat_path, "rb") as f:
        category_model = pickle.load(f)
    with open(fav_path, "rb") as f:
        favor_model = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    explainer = shap.TreeExplainer(category_model)

    AGENT2_OBJECTS = {
        "category_model": category_model,
        "favor_model": favor_model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "explainer": explainer,
    }

    print("[Agent2] Loaded models and SHAP explainer.")
    print("[Agent2] Features:", feature_cols)


def run_agent2(enriched_features: dict):
    if not AGENT2_OBJECTS:
        raise RuntimeError("Call load_agent2_artifacts() first.")

    cat_model = AGENT2_OBJECTS["category_model"]
    fav_model = AGENT2_OBJECTS["favor_model"]
    le = AGENT2_OBJECTS["label_encoder"]
    feature_cols = AGENT2_OBJECTS["feature_cols"]
    explainer = AGENT2_OBJECTS["explainer"]

    x = np.array([[float(enriched_features.get(col, 0.0)) for col in feature_cols]])

    proba = cat_model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]

    favor_proba = float(fav_model.predict_proba(x)[0, 1])

    shap_values = explainer.shap_values(x)
    contrib = shap_values[pred_idx][0]
    abs_contrib = np.abs(contrib)
    top_idx = np.argsort(-abs_contrib)[:10]

    shap_top = []
    for i in top_idx:
        shap_top.append({
            "feature": feature_cols[i],
            "value": float(x[0, i]),
            "shap_value": float(contrib[i]),
            "direction": "pushes_towards_class" if contrib[i] > 0 else "pushes_away_from_class",
        })

    predictions = {
        "predicted_category": pred_label,
        "predicted_customer_favor_prob": favor_proba,
        "category_proba": {
            cls: float(proba[i]) for i, cls in enumerate(le.classes_)
        },
    }

    return predictions, shap_top


def agent2_node(state: dict) -> dict:
    preds, shap_top = run_agent2(state["enriched_features"])
    state["agent2_predictions"] = preds
    state["agent2_shap_top_features"] = shap_top
    return state


if __name__ == "__main__":
    DATA_DIR = "data"
    test_path = os.path.join(DATA_DIR, "disputes_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found. Run train_agent2.py first."
        )

    load_agent2_artifacts()
    df_test = pd.read_csv(test_path)
    feature_cols = AGENT2_OBJECTS["feature_cols"]

    row = df_test.iloc[0]
    enriched_features = {col: float(row[col]) for col in feature_cols}

    preds, shap_top = run_agent2(enriched_features)

    print("\n=== Agent2 predictions ===")
    print(preds)

    print("\n=== Top SHAP features ===")
    for item in shap_top:
        print(item)
