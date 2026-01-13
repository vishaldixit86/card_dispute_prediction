import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
MODELS_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FULL_PATH = os.path.join(DATA_DIR, "disputes_full_10k.csv")


def main():
    if not os.path.exists(FULL_PATH):
        raise FileNotFoundError(
            f"{FULL_PATH} not found. Run generator.py first to create 10k records."
        )

    print("[TRAIN] Loading dataset from:", FULL_PATH)
    df = pd.read_csv(FULL_PATH, parse_dates=["txn_datetime", "dispute_datetime"])
    print("[TRAIN] Data shape:", df.shape)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["dispute_category"],
    )

    train_path = os.path.join(DATA_DIR, "disputes_train.csv")
    test_path = os.path.join(DATA_DIR, "disputes_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("[TRAIN] Saved train to:", train_path)
    print("[TRAIN] Saved test  to:", test_path)

    feature_cols = [
        "txn_amount",
        "geo_distance_km",
        "customer_tenure_months",
        "credit_risk_score",
        "avg_monthly_spend",
        "historical_dispute_count",
        "payment_behavior_score",
        "merchant_risk_tier",
        "merchant_dispute_rate",
        "txn_count_10min",
        "txn_count_1hr",
        "txn_count_24hr",
        "amount_velocity_10min",
        "amount_velocity_1hr",
        "amount_velocity_24hr",
        "policy_rules_count",
        "policy_severity_score",
        "supporting_docs_count",
        "evidence_quality_score",
        "evidence_to_claim_ratio",
        "sentiment_score",
        "keyword_risk_score",
        "transaction_risk_behavior_score",
        "time_to_raise_days",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing}")

    X_train = train_df[feature_cols].fillna(0.0).values
    X_test = test_df[feature_cols].fillna(0.0).values

    le = LabelEncoder()
    y_cat_train = le.fit_transform(train_df["dispute_category"])
    y_cat_test = le.transform(test_df["dispute_category"])

    y_fav_train = train_df["customer_favor"].values
    y_fav_test = test_df["customer_favor"].values

    print("[TRAIN] Training category_model...")
    category_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    category_model.fit(X_train, y_cat_train)
    cat_acc = category_model.score(X_test, y_cat_test)
    print(f"[TRAIN] Category model accuracy: {cat_acc:.3f}")

    print("[TRAIN] Training favor_model...")
    favor_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    favor_model.fit(X_train, y_fav_train)
    fav_acc = favor_model.score(X_test, y_fav_test)
    print(f"[TRAIN] Customer-favor model accuracy: {fav_acc:.3f}")

    cat_path = os.path.join(MODELS_DIR, "category_model.pkl")
    fav_path = os.path.join(MODELS_DIR, "favor_model.pkl")
    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    feat_path = os.path.join(MODELS_DIR, "feature_cols.json")

    with open(cat_path, "wb") as f:
        pickle.dump(category_model, f)
    with open(fav_path, "wb") as f:
        pickle.dump(favor_model, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    print("[TRAIN] Saved category_model to:", cat_path)
    print("[TRAIN] Saved favor_model    to:", fav_path)
    print("[TRAIN] Saved label_encoder  to:", le_path)
    print("[TRAIN] Saved feature_cols   to:", feat_path)
    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()
