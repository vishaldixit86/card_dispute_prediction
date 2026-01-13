import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

# ---------------------------
# 1. Helper lookups & config
# ---------------------------

AGE_GROUPS = ["18-25", "26-35", "36-50", "51-65", "65+"]

COUNTRIES = ["US", "UK", "CA", "IN", "DE", "AU"]

CHANNELS = ["POS", "Online", "ATM", "App"]
DISPUTE_CHANNELS = ["App", "Web", "Call", "Branch"]

DISPUTE_REASON_CODES = [
    "FRAUD", "NOT_RECEIVED", "DUPLICATE",
    "NOT_RECOGNIZED", "QUALITY", "OTHER"
]

# MCC groups and tiers – from your mapping
MCC_TIER_MAP = {
    # Tier 1: Low risk
    4900: ("Utilities", 1),
    9399: ("Government Services", 1),
    6300: ("Insurance", 1),
    8211: ("Education", 1),
    4111: ("Local Transport", 1),
    6011: ("ATM Withdrawals", 1),
    8062: ("Hospitals", 1),
    8220: ("Colleges/Universities", 1),
    # Tier 2: Medium risk
    5311: ("Department Stores", 2),
    5812: ("Restaurants", 2),
    5541: ("Service Stations", 2),
    4722: ("Travel Agencies", 2),
    5691: ("Apparel Stores", 2),
    7011: ("Hotels", 2),
    5814: ("Fast Food", 2),
    4112: ("Passenger Railways", 2),
    # Tier 3: High risk
    7995: ("Gambling", 3),
    5966: ("Telemarketing", 3),
    4829: ("Money Transfer", 3),
    5816: ("Digital Goods - Games", 3),
    5817: ("Digital Goods - Apps", 3),
    5818: ("Digital Goods - Multi", 3),
    5999: ("Misc Retail", 3),
    5734: ("Software Stores", 3),
    7299: ("Personal Services", 3),
    # Tier 4: Extreme risk
    5968: ("Adult Entertainment", 4),
    6051: ("Crypto Exchanges", 4),
    5967: ("Subscription Services", 4),
    7994: ("Video Game Arcades", 4),
    7841: ("Video Rental Stores", 4),
    7993: ("Bowling Alleys", 4),
}

MCC_LIST = list(MCC_TIER_MAP.keys())

# Risk score per tier
TIER_TO_SCORE = {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.9}

TODAY = datetime(2025, 1, 1)  # fixed for reproducibility

# ---------------------------
# 2. Generate customers
# ---------------------------

def generate_customers(n_customers=2000):
    customers = []
    for i in range(n_customers):
        customer_id = f"CUST_{i:05d}"
        age_group = np.random.choice(AGE_GROUPS, p=[0.2, 0.3, 0.25, 0.15, 0.1])
        tenure_months = np.random.randint(1, 180)  # 1–15 years
        credit_risk_score = np.clip(np.random.normal(0.6, 0.15), 0, 1)
        avg_monthly_spend = float(np.exp(np.random.normal(7.5, 0.5)))  # log-normal-ish
        # Dispute count: skewed
        hist_dispute = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            p=[0.35, 0.25, 0.15, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02]
        )
        payment_behavior = np.clip(np.random.normal(0.7, 0.2), 0, 1)
        customer_country = np.random.choice(COUNTRIES, p=[0.5, 0.1, 0.1, 0.15, 0.05, 0.1])

        customers.append({
            "customer_id": customer_id,
            "age_group": age_group,
            "customer_tenure_months": tenure_months,
            "credit_risk_score": credit_risk_score,
            "avg_monthly_spend": avg_monthly_spend,
            "historical_dispute_count": hist_dispute,
            "payment_behavior_score": payment_behavior,
            "customer_country": customer_country,
        })
    return pd.DataFrame(customers)

# ---------------------------
# 3. Generate merchants
# ---------------------------

def generate_merchants(n_merchants=1000):
    merchants = []
    for i in range(n_merchants):
        merchant_id = f"MER_{i:04d}"
        mcc = int(np.random.choice(MCC_LIST))
        mcc_group, tier = MCC_TIER_MAP[mcc]
        # Dispute rate based on tier
        if tier == 1:
            dispute_rate = np.random.uniform(0.01, 0.05)
        elif tier == 2:
            dispute_rate = np.random.uniform(0.05, 0.15)
        elif tier == 3:
            dispute_rate = np.random.uniform(0.15, 0.35)
        else:
            dispute_rate = np.random.uniform(0.30, 0.50)
        merchant_country = np.random.choice(COUNTRIES, p=[0.4, 0.15, 0.1, 0.15, 0.1, 0.1])

        merchants.append({
            "merchant_id": merchant_id,
            "mcc": mcc,
            "mcc_group": mcc_group,
            "merchant_risk_tier": tier,
            "merchant_dispute_rate": dispute_rate,
            "merchant_country": merchant_country,
        })
    return pd.DataFrame(merchants)

# ---------------------------
# 4. Dispute text templates
# ---------------------------

FRAUD_TEXTS = [
    "This transaction is unauthorized, I never made this purchase.",
    "My card was stolen and this charge is fraud.",
    "I did not recognize this charge, it looks like a scam.",
]

MERCHANT_ERROR_TEXTS = [
    "I was charged twice for the same purchase.",
    "I never received the product I paid for.",
    "The service was not provided but I was billed.",
]

FRIENDLY_FRAUD_TEXTS = [
    "I don't remember making this purchase.",
    "This amount is too high, I want to dispute it.",
    "I thought this subscription was cancelled.",
]

def sample_dispute_text(label):
    if label == "third_party_fraud":
        return np.random.choice(FRAUD_TEXTS)
    elif label == "true_dispute":
        return np.random.choice(MERCHANT_ERROR_TEXTS)
    else:  # first_party_fraud
        return np.random.choice(FRIENDLY_FRAUD_TEXTS)

# ---------------------------
# 5. One dispute row generator
# ---------------------------

def generate_one_dispute(customers_df, merchants_df, idx):
    # randomly pick customer & merchant
    cust = customers_df.sample(1).iloc[0]
    merch = merchants_df.sample(1).iloc[0]

    dispute_id = f"DISP_{idx:06d}"
    transaction_id = f"TXN_{idx:06d}"
    customer_id = cust["customer_id"]
    merchant_id = merch["merchant_id"]

    # transaction datetime in last 365 days
    days_ago = np.random.randint(1, 365)
    txn_datetime = TODAY - timedelta(
        days=int(days_ago),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )

    # txn amount based on merchant & customer
    base = cust["avg_monthly_spend"] / np.random.uniform(10, 60)
    risk_multiplier = 1 + (merch["merchant_risk_tier"] - 1) * np.random.uniform(0.1, 0.8)
    txn_amount = float(np.clip(
        np.random.normal(base * risk_multiplier, base * 0.3),
        5,
        10000
    ))

    # channel
    if (merch["mcc_group"].startswith("Digital Goods")
        or merch["mcc_group"] in ["Gambling", "Crypto Exchanges", "Subscription Services"]):
        channel = np.random.choice(["Online", "App"], p=[0.7, 0.3])
        card_present = 0
    elif merch["mcc_group"] in ["Restaurants", "Fast Food", "Service Stations", "Bowling Alleys"]:
        channel = "POS"
        card_present = 1
    else:
        channel = np.random.choice(CHANNELS, p=[0.4, 0.4, 0.05, 0.15])
        card_present = 1 if channel == "POS" else 0

    # currency type
    currency_type = np.random.choice(["Domestic", "Foreign"], p=[0.9, 0.1])

    # geo distance: mostly local, sometimes far
    if np.random.rand() < 0.8:
        geo_distance_km = float(np.random.uniform(0, 30))
    else:
        geo_distance_km = float(np.random.uniform(200, 2000))

    txn_country = merch["merchant_country"]

    # fraudiness signal
    fraud_signal = (
        (merch["merchant_risk_tier"] >= 3)
        + (currency_type == "Foreign")
        + (geo_distance_km > 200)
    )

    # decide proto label and time-to-raise
    if fraud_signal >= 2 and np.random.rand() < 0.7:
        proto_label = "third_party_fraud"
        time_to_raise_days = np.random.randint(0, 4)
    elif np.random.rand() < 0.5:
        proto_label = "true_dispute"
        time_to_raise_days = np.random.randint(3, 31)
    else:
        proto_label = "first_party_fraud"
        time_to_raise_days = np.random.randint(20, 61)

    dispute_datetime = txn_datetime + timedelta(
        days=int(time_to_raise_days),
        hours=np.random.randint(0, 5)
    )

    # dispute channel & reason
    dispute_channel = np.random.choice(DISPUTE_CHANNELS, p=[0.5, 0.25, 0.2, 0.05])
    if proto_label == "third_party_fraud":
        dispute_reason_code = "FRAUD"
    elif proto_label == "true_dispute":
        dispute_reason_code = np.random.choice(["NOT_RECEIVED", "DUPLICATE", "QUALITY"])
    else:
        dispute_reason_code = np.random.choice(DISPUTE_REASON_CODES)

    dispute_amount = txn_amount  # POC: same as txn_amount

    # ============================
    # Velocity & behavior
    # ============================
    if merch["merchant_risk_tier"] >= 3 and proto_label == "third_party_fraud":
        txn_count_10min = np.random.randint(3, 8)
        txn_count_1hr = txn_count_10min + np.random.randint(1, 5)
        txn_count_24hr = txn_count_1hr + np.random.randint(1, 10)
        distinct_geo_10min = np.random.randint(1, 4)
        time_gap_prev_txn_min = float(np.random.uniform(0, 2))
        device_switch_count_24hr = np.random.randint(1, 4)
        velocity_pattern_label = "high_velocity"
    elif proto_label == "first_party_fraud" and np.random.rand() < 0.5:
        # low velocity but high amount
        txn_count_10min = np.random.randint(1, 3)
        txn_count_1hr = txn_count_10min + np.random.randint(0, 2)
        txn_count_24hr = txn_count_1hr + np.random.randint(0, 5)
        distinct_geo_10min = 1
        time_gap_prev_txn_min = float(np.random.uniform(60, 1440))
        device_switch_count_24hr = np.random.randint(0, 2)
        velocity_pattern_label = "low_velocity_high_amount"
    else:
        # normal pattern
        txn_count_10min = np.random.randint(1, 4)
        txn_count_1hr = txn_count_10min + np.random.randint(0, 3)
        txn_count_24hr = txn_count_1hr + np.random.randint(0, 5)
        distinct_geo_10min = np.random.randint(1, 2)
        time_gap_prev_txn_min = float(np.random.uniform(5, 180))
        device_switch_count_24hr = np.random.randint(0, 2)
        velocity_pattern_label = "normal"

    amount_velocity_10min = float(txn_amount * txn_count_10min / 10)
    amount_velocity_1hr = float(txn_amount * txn_count_1hr / 60)
    amount_velocity_24hr = float(txn_amount * txn_count_24hr / (24 * 60))

    # ============================
    # Policy violations
    # ============================
    limit_violation = int(
        (txn_amount > cust["avg_monthly_spend"] * 0.8) and np.random.rand() < 0.6
    )
    high_risk_country_violation = int(
        (currency_type == "Foreign" or txn_country != cust["customer_country"])
        and np.random.rand() < 0.5
    )
    behavior_pattern_violation = int(
        fraud_signal >= 2 or velocity_pattern_label == "high_velocity"
    )
    aml_kyc_violation = int(np.random.rand() < 0.05)
    fraud_pattern_violation = int(velocity_pattern_label == "high_velocity")

    policy_rules_count = (
        limit_violation
        + high_risk_country_violation
        + behavior_pattern_violation
        + aml_kyc_violation
        + fraud_pattern_violation
    )
    policy_severity_score = (
        limit_violation * 2
        + high_risk_country_violation * 2
        + behavior_pattern_violation * 3
        + aml_kyc_violation * 3
        + fraud_pattern_violation * 3
    )

    # ============================
    # Evidence
    # ============================
    if proto_label == "true_dispute":
        supporting_docs_count = np.random.randint(1, 5)
        evidence_quality_score = float(np.clip(np.random.normal(0.8, 0.1), 0, 1))
    elif proto_label == "third_party_fraud":
        supporting_docs_count = np.random.randint(1, 4)
        evidence_quality_score = float(np.clip(np.random.normal(0.7, 0.15), 0, 1))
    else:  # first_party_fraud
        supporting_docs_count = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        evidence_quality_score = float(np.clip(np.random.normal(0.4, 0.2), 0, 1))

    evidence_to_claim_ratio = supporting_docs_count / max(dispute_amount, 1.0)

    # ============================
    # Sentiment & keyword risk
    # ============================
    if proto_label == "third_party_fraud":
        sentiment_score = float(np.clip(np.random.normal(-0.8, 0.2), -1, 1))
        keyword_risk_flag = 1
        keyword_risk_score = float(np.clip(np.random.normal(0.9, 0.1), 0, 1))
    elif proto_label == "true_dispute":
        sentiment_score = float(np.clip(np.random.normal(-0.4, 0.3), -1, 1))
        keyword_risk_flag = np.random.choice([0, 1], p=[0.6, 0.4])
        keyword_risk_score = float(np.clip(np.random.normal(0.6, 0.2), 0, 1))
    else:
        sentiment_score = float(np.clip(np.random.normal(-0.2, 0.3), -1, 1))
        keyword_risk_flag = np.random.choice([0, 1], p=[0.7, 0.3])
        keyword_risk_score = float(np.clip(np.random.normal(0.4, 0.2), 0, 1))

    # ============================
    # Composite risk
    # ============================
    mcc_risk_score = TIER_TO_SCORE[merch["merchant_risk_tier"]]
    composite_raw = (
        0.4 * mcc_risk_score
        + 0.3 * (policy_severity_score / 10.0)
        + 0.3 * (
            1.0 if velocity_pattern_label == "high_velocity"
            else 0.3 if velocity_pattern_label == "normal"
            else 0.6
        )
    )
    transaction_risk_behavior_score = float(np.clip(composite_raw, 0, 1))

    # ============================
    # Outcome probabilities & labels
    # ============================
    if proto_label == "third_party_fraud":
        p_customer_favor = 0.7 + 0.2 * evidence_quality_score
        p_merchant_arbitration = 0.1 + 0.2 * (1 - evidence_quality_score)
    elif proto_label == "true_dispute":
        p_customer_favor = 0.6 + 0.3 * evidence_quality_score
        p_merchant_arbitration = 0.15
    else:  # first_party_fraud
        p_customer_favor = 0.3 + 0.3 * evidence_quality_score
        p_merchant_arbitration = 0.2 + 0.2 * (1 - evidence_quality_score)

    p_customer_favor = float(np.clip(p_customer_favor, 0, 1))
    p_merchant_arbitration = float(np.clip(p_merchant_arbitration, 0, 1))

    base_assert = 0.2 + 0.3 * abs(sentiment_score) + 0.2 * (txn_amount > cust["avg_monthly_spend"] * 0.7)
    p_customer_assertion = float(np.clip(base_assert, 0, 1))

    customer_favor = int(np.random.rand() < p_customer_favor)
    merchant_arbitration = int(np.random.rand() < p_merchant_arbitration)
    customer_assertion = int(np.random.rand() < p_customer_assertion)

    # small chance to flip category for realism
    flip_prob = 0.1
    if np.random.rand() < flip_prob:
        dispute_category = np.random.choice(
            ["true_dispute", "first_party_fraud", "third_party_fraud"]
        )
    else:
        dispute_category = proto_label

    dispute_text = sample_dispute_text(dispute_category)

    row = {
        # IDs
        "dispute_id": dispute_id,
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "merchant_id": merchant_id,

        # Transaction
        "txn_amount": txn_amount,
        "txn_datetime": txn_datetime,
        "channel": channel,
        "card_present": card_present,
        "currency_type": currency_type,
        "geo_distance_km": geo_distance_km,
        "txn_country": txn_country,
        "time_to_raise_days": time_to_raise_days,
        "mcc": merch["mcc"],
        "mcc_group": merch["mcc_group"],

        # Customer
        "customer_tenure_months": cust["customer_tenure_months"],
        "age_group": cust["age_group"],
        "credit_risk_score": cust["credit_risk_score"],
        "avg_monthly_spend": cust["avg_monthly_spend"],
        "historical_dispute_count": cust["historical_dispute_count"],
        "payment_behavior_score": cust["payment_behavior_score"],
        "customer_country": cust["customer_country"],

        # Merchant
        "merchant_risk_tier": merch["merchant_risk_tier"],
        "merchant_dispute_rate": merch["merchant_dispute_rate"],
        "merchant_country": merch["merchant_country"],

        # Velocity
        "txn_count_10min": txn_count_10min,
        "txn_count_1hr": txn_count_1hr,
        "txn_count_24hr": txn_count_24hr,
        "distinct_geo_10min": distinct_geo_10min,
        "time_gap_prev_txn_min": time_gap_prev_txn_min,
        "device_switch_count_24hr": device_switch_count_24hr,
        "velocity_pattern_label": velocity_pattern_label,
        "amount_velocity_10min": amount_velocity_10min,
        "amount_velocity_1hr": amount_velocity_1hr,
        "amount_velocity_24hr": amount_velocity_24hr,

        # Policy
        "limit_violation": limit_violation,
        "high_risk_country_violation": high_risk_country_violation,
        "behavior_pattern_violation": behavior_pattern_violation,
        "aml_kyc_violation": aml_kyc_violation,
        "fraud_pattern_violation": fraud_pattern_violation,
        "policy_rules_count": policy_rules_count,
        "policy_severity_score": policy_severity_score,

        # Evidence
        "supporting_docs_count": supporting_docs_count,
        "evidence_quality_score": evidence_quality_score,
        "evidence_to_claim_ratio": evidence_to_claim_ratio,

        # Text / sentiment
        "sentiment_score": sentiment_score,
        "keyword_risk_flag": keyword_risk_flag,
        "keyword_risk_score": keyword_risk_score,
        "dispute_text": dispute_text,

        # Composite risk
        "transaction_risk_behavior_score": transaction_risk_behavior_score,

        # Outcome probabilities
        "p_customer_favor": p_customer_favor,
        "p_merchant_arbitration": p_merchant_arbitration,
        "p_customer_assertion": p_customer_assertion,

        # Final labels
        "customer_favor": customer_favor,
        "merchant_arbitration": merchant_arbitration,
        "customer_assertion": customer_assertion,
        "dispute_category": dispute_category,
        "dispute_reason_code": dispute_reason_code,
        "dispute_channel": dispute_channel,
        "dispute_amount": dispute_amount,
        "dispute_datetime": dispute_datetime,
    }

    return row

# ---------------------------
# 6. Generate full dataset
# ---------------------------

def generate_disputes_dataset(n_records=10000):
    customers_df = generate_customers()
    merchants_df = generate_merchants()
    rows = []
    for i in range(n_records):
        rows.append(generate_one_dispute(customers_df, merchants_df, i))
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# 7. Main: generate 10k and save to data/
# ---------------------------

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating 10k synthetic disputes...")
    df = generate_disputes_dataset(10000)
    print("Shape:", df.shape)

    out_path = os.path.join("data", "disputes_full_10k.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(df.head())
