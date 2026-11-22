"""
cnn_model.py
Custom CNN architecture for food image classification.

Architecture (documented for college evaluation)
------------------------------------------------
Input (224×224×3)
  → Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomBrightness)
  → Conv2D(32, 3×3)  + BatchNorm + ReLU + MaxPool(2×2)
  → Conv2D(64, 3×3)  + BatchNorm + ReLU + MaxPool(2×2)
  → Conv2D(128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
  → Conv2D(256, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
  → Flatten
  → Dense(512) + ReLU + Dropout(0.5)
  → Dense(256) + ReLU + Dropout(0.3)
  → Dense(NUM_CLASSES) + Softmax

Deep-learning concepts implemented
-----------------------------------
  Forward propagation  : standard Keras call
  Backpropagation      : automatic differentiation via model.fit / Adam
  Activation functions : ReLU (hidden), Softmax (output)
  Loss                 : Categorical cross-entropy
  Optimiser            : Adam
  Regularisation       : Batch normalisation + Dropout
  Pooling              : Max pooling (2×2 stride)
  Flattening           : after last conv block
  Data augmentation    : applied inside the model (training only)
  Early stopping       : via EarlyStopping callback
  Checkpointing        : via ModelCheckpoint callback
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn(num_classes: int = config.NUM_CLASSES,
              input_shape: tuple = (*config.IMG_SIZE, config.IMG_CHANNELS),
              dropout_rate: float = config.DROPOUT_RATE) -> Model:
    """
    Build and return the custom CNN model.

    Parameters
    ----------
    num_classes  : int   – number of food categories
    input_shape  : tuple – (height, width, channels), default (224,224,3)
    dropout_rate : float – dropout probability for FC layers

    Returns
    -------
    tf.keras.Model (not yet compiled)
    """
    tf.random.set_seed(config.RANDOM_SEED)

    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Data augmentation (active only during training) ───────────────────
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(config.AUG_ROTATION / 360.0, name="aug_rot")(x)
    x = layers.RandomZoom(config.AUG_ZOOM, name="aug_zoom")(x)

    # ── Conv Block 1: 32 filters ──────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Conv Block 2: 64 filters ──────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Conv Block 3: 128 filters ─────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Conv Block 4: 256 filters ─────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Flatten + Fully-Connected Head ────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, name="fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(256, name="fc2")(x)
    x = layers.Activation("relu", name="relu_fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    # ── Output: Softmax multiclass ────────────────────────────────────────
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FoodCNN")
    return model


def compile_cnn(model: Model,
                learning_rate: float = config.LEARNING_RATE) -> Model:
    """Compile the CNN with Adam + categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: str = config.CNN_MODEL_PATH) -> list:
    """Return standard training callbacks: EarlyStopping + ModelCheckpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


if __name__ == "__main__":
    model = build_cnn()
    compile_cnn(model)
    model.summary()
