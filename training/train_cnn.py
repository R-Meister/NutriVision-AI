"""
train_cnn.py
Train the custom CNN and MobileNetV2 transfer-learning model.

Usage
-----
  python training/train_cnn.py --model cnn      # train custom CNN only
  python training/train_cnn.py --model mobilenet # train MobileNetV2 only
  python training/train_cnn.py --model all       # train both (default)
  python training/train_cnn.py --dataset synthetic --n_per_class 20  # quick test
"""
import os, sys, json, argparse, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from training.dataset_loader import load_dataset
from models.cnn_model import build_cnn, compile_cnn, get_callbacks as cnn_callbacks
from models.transfer_learning import (build_mobilenet, compile_mobilenet,
                                       get_callbacks as tl_callbacks,
                                       enable_fine_tuning)


def _save_history(hist: dict, name: str):
    path = os.path.join(config.RESULTS_DIR, f"{name}_history.json")
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.items()}, f, indent=2)
    print(f"History → {path}")


def train_cnn_model(X_train, y_train, X_val, y_val,
                    epochs: int = config.CNN_EPOCHS,
                    batch_size: int = config.BATCH_SIZE) -> dict:
    print("\n" + "=" * 60)
    print("  Training Custom CNN")
    print("=" * 60)
    model = build_cnn()
    compile_cnn(model)
    model.summary()

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cnn_callbacks(),
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n✅ CNN trained in {elapsed:.0f}s. "
          f"Best val_acc = {max(history.history['val_accuracy']):.4f}")
    _save_history(history.history, "cnn")
    return history.history


def train_mobilenet_model(X_train, y_train, X_val, y_val,
                           epochs: int = config.TL_EPOCHS,
                           fine_tune: bool = False,
                           batch_size: int = config.BATCH_SIZE) -> dict:
    print("\n" + "=" * 60)
    print("  Training MobileNetV2 Transfer Learning")
    print("=" * 60)
    model = build_mobilenet(freeze_base=True)
    compile_mobilenet(model)
    model.summary()

    # Phase 1: Feature extraction
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=tl_callbacks(),
        verbose=1,
    )
    print(f"\n✅ Phase-1 done. Best val_acc = {max(history.history['val_accuracy']):.4f}")

    # Phase 2: Fine-tuning (optional)
    if fine_tune and epochs > 5:
        print("\n── Fine-tuning phase ──")
        model = enable_fine_tuning(model)
        ft_history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.FINE_TUNE_EPOCHS,
            batch_size=batch_size // 2,
            verbose=1,
        )
        # Merge histories
        for k in history.history:
            if k in ft_history.history:
                history.history[k].extend(ft_history.history[k])

    elapsed = time.time() - t0
    print(f"\n✅ MobileNetV2 total time: {elapsed:.0f}s. "
          f"Best val_acc = {max(history.history['val_accuracy']):.4f}")
    _save_history(history.history, "mobilenet")
    return history.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="all",
                        choices=["cnn", "mobilenet", "all"])
    parser.add_argument("--dataset",     default="auto",
                        choices=["auto", "food101", "synthetic"])
    parser.add_argument("--n_per_class", type=int, default=100)
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=config.BATCH_SIZE)
    parser.add_argument("--fine_tune",   action="store_true")
    args = parser.parse_args()

    # Load data
    print(f"[train_cnn] Loading dataset: mode={args.dataset}, "
          f"n_per_class={args.n_per_class}")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(
        args.dataset, args.n_per_class
    )
    print(f"Train={X_tr.shape[0]} | Val={X_val.shape[0]} | Test={X_te.shape[0]}")

    epochs = args.epochs

    if args.model in ("cnn", "all"):
        ep = epochs or config.CNN_EPOCHS
        train_cnn_model(X_tr, y_tr, X_val, y_val, ep, args.batch_size)

    if args.model in ("mobilenet", "all"):
        ep = epochs or config.TL_EPOCHS
        train_mobilenet_model(X_tr, y_tr, X_val, y_val, ep,
                               args.fine_tune, args.batch_size)

    print("\n🎉 Training complete! Saved models are in:", config.SAVED_MODELS)


if __name__ == "__main__":
    main()
