"""
ann_model.py
Artificial Neural Network for nutritional value regression / estimation.

Role
----
The CNN / MobileNetV2 classifies the food IMAGE → food class.
This ANN takes NUMERICAL FEATURES (food class encoding + portion weight)
as input and predicts { calories, protein, carbs, fat }.

This demonstrates that ANNs can be used for structured / tabular data
prediction independently of image-based CNNs.

Architecture
------------
Input: [food_class_one_hot(12), portion_grams(1)] → 13 features
  → Dense(64) + ReLU + BatchNorm + Dropout(0.3)
  → Dense(128) + ReLU + BatchNorm + Dropout(0.3)
  → Dense(64)  + ReLU
  → Dense(4)   [calories, protein, carbs, fat] – linear output

Loss      : Mean Absolute Error (MAE)
Optimiser : Adam
Metrics   : MAE, MSE
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


N_FOOD_CLASSES = config.NUM_CLASSES  # 12
N_INPUTS       = N_FOOD_CLASSES + 1  # one-hot(12) + portion_grams
N_OUTPUTS      = 4                   # calories, protein, carbs, fat


def build_ann(n_inputs: int = N_INPUTS,
              n_outputs: int = N_OUTPUTS,
              dropout_rate: float = 0.3) -> Model:
    """
    Build and return the nutritional ANN.

    Parameters
    ----------
    n_inputs     : int – number of input features (default 13)
    n_outputs    : int – number of output values   (default  4)
    dropout_rate : float

    Returns
    -------
    tf.keras.Model (not compiled)
    """
    tf.random.set_seed(config.RANDOM_SEED)

    inputs = layers.Input(shape=(n_inputs,), name="nutrition_features")

    x = layers.Dense(64, name="fc1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(128, name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    x = layers.Dense(64, name="fc3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    outputs = layers.Dense(n_outputs, activation="linear",
                           name="nutrition_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="NutritionANN")
    return model


def compile_ann(model: Model,
                learning_rate: float = config.LEARNING_RATE) -> Model:
    """Compile with Adam + MAE loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=["mae", "mse"],
    )
    return model


def build_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for the ANN from the nutrition database.

    For each food class and a range of portion sizes, compute the ground-truth
    nutritional values.  Add Gaussian noise to simulate real-world variation.

    Returns
    -------
    X : np.ndarray shape (N, N_INPUTS)
    y : np.ndarray shape (N, N_OUTPUTS)
    """
    from utils.nutrition_database import NUTRITION_DATA

    rng = np.random.default_rng(config.RANDOM_SEED)
    portions = np.arange(50, 501, 10, dtype=float)  # 50 g – 500 g in 10 g steps

    X_rows, y_rows = [], []
    food_list = list(NUTRITION_DATA.keys())

    for class_idx, food in enumerate(food_list):
        per100 = NUTRITION_DATA[food]
        one_hot = np.zeros(N_FOOD_CLASSES)
        one_hot[class_idx] = 1.0

        for p in portions:
            factor = p / 100.0
            # Ground-truth nutrition
            gt = np.array([
                per100["calories"] * factor,
                per100["protein"]  * factor,
                per100["carbs"]    * factor,
                per100["fat"]      * factor,
            ])
            # Add realistic noise (±5 %)
            noise = rng.normal(0, 0.05, size=gt.shape) * gt
            y_noisy = np.clip(gt + noise, 0, None)

            feat = np.concatenate([one_hot, [p / 500.0]])   # normalise portion
            X_rows.append(feat)
            y_rows.append(y_noisy)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def get_ann_callbacks() -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.ANN_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]


def predict_nutrition(model: Model,
                      food_class_index: int,
                      portion_grams: float) -> dict:
    """
    Use the trained ANN to predict nutritional values.

    Parameters
    ----------
    model            : trained ANN model
    food_class_index : int – index in FOOD_CLASSES list
    portion_grams    : float – portion in grams

    Returns
    -------
    dict: { calories, protein, carbs, fat }
    """
    one_hot = np.zeros(N_FOOD_CLASSES, dtype=np.float32)
    one_hot[food_class_index] = 1.0
    portion_norm = np.array([portion_grams / 500.0], dtype=np.float32)
    feat = np.concatenate([one_hot, portion_norm])[np.newaxis, :]  # (1, 13)

    pred = model.predict(feat, verbose=0)[0]  # shape (4,)
    pred = np.clip(pred, 0, None)

    return {
        "calories": round(float(pred[0]), 1),
        "protein":  round(float(pred[1]), 1),
        "carbs":    round(float(pred[2]), 1),
        "fat":      round(float(pred[3]), 1),
    }


if __name__ == "__main__":
    m = build_ann()
    compile_ann(m)
    m.summary()
    X, y = build_training_data()
    print(f"Training data: X={X.shape}, y={y.shape}")
