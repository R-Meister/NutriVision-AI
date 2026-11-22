"""
transfer_learning.py
MobileNetV2-based transfer-learning model for food classification.

Architecture
------------
Input (224×224×3)
  → MobileNetV2 (ImageNet weights, frozen base)
  → GlobalAveragePooling2D
  → Dense(256) + ReLU + BatchNorm + Dropout(0.5)
  → Dense(128) + ReLU + Dropout(0.3)
  → Dense(NUM_CLASSES) + Softmax

Transfer-learning workflow
--------------------------
Phase 1 – Feature extraction
  Freeze all MobileNetV2 layers, train the head only.

Phase 2 – Fine-tuning (optional)
  Unfreeze layers from FINE_TUNE_AT onwards and train with a low LR.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet(num_classes: int = config.NUM_CLASSES,
                    input_shape: tuple = (*config.IMG_SIZE, config.IMG_CHANNELS),
                    freeze_base: bool = True) -> Model:
    """
    Build MobileNetV2 transfer-learning model.

    Parameters
    ----------
    num_classes  : number of output food categories
    input_shape  : (H, W, C) – must be (224,224,3) for MobileNetV2
    freeze_base  : freeze the entire MobileNetV2 backbone initially

    Returns
    -------
    tf.keras.Model (not compiled)
    """
    tf.random.set_seed(config.RANDOM_SEED)

    # ── Pretrained backbone (ImageNet weights) ─────────────────────────────
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    # ── Build the full model ───────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects pixels in [-1, 1]; scale from [0,1]
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale")(inputs)

    # Data augmentation (applied during training only)
    x = layers.RandomFlip("horizontal", name="aug_flip")(x)
    x = layers.RandomRotation(config.AUG_ROTATION / 360.0, name="aug_rot")(x)
    x = layers.RandomZoom(config.AUG_ZOOM, name="aug_zoom")(x)

    # Backbone
    x = base_model(x, training=not freeze_base)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)

    x = layers.Dense(128, name="fc2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MobileNetV2_FoodClassifier")
    model._base_model_ref = base_model   # keep reference for fine-tuning
    return model


def enable_fine_tuning(model: Model,
                       fine_tune_at: int = config.FINE_TUNE_AT,
                       learning_rate: float = config.FINE_TUNE_LR) -> Model:
    """
    Unfreeze the top layers of the backbone for fine-tuning.

    Parameters
    ----------
    model        : the already-trained (phase-1) model
    fine_tune_at : unfreeze backbone layers from this index onwards
    learning_rate: smaller LR for fine-tuning (default 1e-5)
    """
    # Find the MobileNetV2 sub-model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("⚠️  Could not locate MobileNetV2 sub-model; skipping fine-tune.")
        return model

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Fine-tuning: {n_trainable} trainable layers in backbone.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compile_mobilenet(model: Model,
                      learning_rate: float = config.LEARNING_RATE) -> Model:
    """Compile with Adam + categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: str = config.TL_MODEL_PATH) -> list:
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
            min_lr=1e-7,
            verbose=1,
        ),
    ]


if __name__ == "__main__":
    m = build_mobilenet()
    compile_mobilenet(m)
    m.summary()
