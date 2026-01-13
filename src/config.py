import os

# Project root = parent of src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "disputes_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "disputes_test.csv")

FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_cols.json")
CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_model.pkl")
FAVOR_MODEL_PATH = os.path.join(MODELS_DIR, "favor_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
