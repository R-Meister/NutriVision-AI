# config.py — Central configuration for Food Calorie Estimator
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
REAL_DATA_DIR   = os.path.join(DATA_DIR, "food_images")
SAVED_MODELS    = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

CNN_MODEL_PATH  = os.path.join(SAVED_MODELS, "cnn_model.keras")
TL_MODEL_PATH   = os.path.join(SAVED_MODELS, "mobilenet_model.keras")
ANN_MODEL_PATH  = os.path.join(SAVED_MODELS, "ann_model.keras")
NUTRITION_CSV   = os.path.join(DATA_DIR, "nutrition.csv")
DB_PATH         = os.path.join(BASE_DIR, "food_tracker.db")

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)
IMG_CHANNELS    = 3
BATCH_SIZE      = 32
RANDOM_SEED     = 42

# ── Food classes (12 recognisable categories) ─────────────────────────────────
FOOD_CLASSES = [
    "pizza", "burger", "french_fries", "samosa", "dosa",
    "idli", "biryani", "fried_rice", "apple", "banana",
    "sandwich", "omelette"
]
NUM_CLASSES = len(FOOD_CLASSES)

# ── Training hyper-parameters ─────────────────────────────────────────────────
CNN_EPOCHS          = 30
TL_EPOCHS           = 20
FINE_TUNE_EPOCHS    = 10
ANN_EPOCHS          = 100
LEARNING_RATE       = 1e-3
FINE_TUNE_LR        = 1e-5
DROPOUT_RATE        = 0.5
VALIDATION_SPLIT    = 0.2
TEST_SPLIT          = 0.1
PATIENCE            = 5          # Early stopping patience

# ── Default portion size (grams) ─────────────────────────────────────────────
DEFAULT_PORTION     = 150        # grams
MIN_PORTION         = 50
MAX_PORTION         = 500
CONFIDENCE_THRESHOLD = 0.40     # Below this → show "uncertain" warning

# ── MobileNetV2 fine-tune layer count ────────────────────────────────────────
FINE_TUNE_AT        = 100       # Unfreeze layers from this index onwards

# ── Data augmentation ────────────────────────────────────────────────────────
AUG_ROTATION        = 20
AUG_ZOOM            = 0.2
AUG_BRIGHTNESS      = [0.8, 1.2]

# Ensure required directories exist at import time
for _d in [DATA_DIR, SAVED_MODELS, RESULTS_DIR, ASSETS_DIR]:
    os.makedirs(_d, exist_ok=True)
