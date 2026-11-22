"""
evaluate.py
Model evaluation script — generates metrics and plots saved to results/.

Metrics computed
----------------
- Training / Validation accuracy & loss curves
- Test accuracy, Precision, Recall, F1-score
- Confusion matrix (heatmap)
- Classification report
- Per-class accuracy bar chart

Usage
-----
  python training/evaluate.py --model mobilenet   # evaluate MobileNetV2 (default)
  python training/evaluate.py --model cnn         # evaluate custom CNN
  python training/evaluate.py --model ann         # evaluate ANN
"""
import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support, accuracy_score)


def load_history(name: str) -> dict | None:
    path = os.path.join(config.RESULTS_DIR, f"{name}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(history: dict, name: str):
    """Save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name.upper()} — Learning Curves", fontsize=14)

    # Accuracy
    ax = axes[0]
    ax.plot(history.get("accuracy", []),     label="Train Acc",  color="#2196F3")
    ax.plot(history.get("val_accuracy", []), label="Val Acc",    color="#FF9800")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(history.get("loss", []),     label="Train Loss", color="#F44336")
    ax.plot(history.get("val_loss", []), label="Val Loss",   color="#9C27B0")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    path = os.path.join(config.RESULTS_DIR, f"{name}_learning_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")
    return path


def plot_confusion_matrix(cm: np.ndarray, classes: list, name: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes],
                        rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title(f"{name.upper()} — Confusion Matrix")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    path = os.path.join(config.RESULTS_DIR, f"{name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")
    return path


def evaluate_classifier(model_path: str, name: str, dataset_mode: str = "real",
                         n_per_class: int | None = None):
    """Evaluate a classification model (CNN or MobileNetV2)."""
    import tensorflow as tf
    from training.dataset_loader import load_dataset, load_test_dataset

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return {}

    model = tf.keras.models.load_model(model_path)
    if dataset_mode == "real":
        X_test, y_test = load_test_dataset(n_per_class)
    else:
        _, _, _, _, X_test, y_test = load_dataset(dataset_mode, n_per_class)

    if len(X_test) == 0:
        print("❌ Empty test set.")
        return {}

    print(f"Evaluating {name} on {len(X_test)} test samples…")
    preds   = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(preds, axis=1)
    y_true  = np.argmax(y_test, axis=1)

    labels = list(range(len(config.FOOD_CLASSES)))
    acc    = accuracy_score(y_true, y_pred)
    top3   = np.argsort(preds, axis=1)[:, -3:]
    top3_acc = float(np.any(top3 == y_true[:, None], axis=1).mean())
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                   labels=labels,
                                                   average="macro",
                                                   zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                      labels=labels,
                                                      average="weighted",
                                                      zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred,
                                   labels=labels,
                                   target_names=config.FOOD_CLASSES,
                                   zero_division=0)

    print(f"\n{name.upper()} Test Results")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Top-3 Accuracy  : {top3_acc:.4f}")
    print(f"  Precision (macro): {p:.4f}")
    print(f"  Recall    (macro): {r:.4f}")
    print(f"  F1-score  (macro): {f1:.4f}")
    print(f"\n{report}")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "precision": p, "recall": r, "f1": f1,
        "precision_weighted": wp, "recall_weighted": wr, "f1_weighted": wf1,
        "test_samples": int(len(y_true)),
    }
    mpath = os.path.join(config.RESULTS_DIR, f"{name}_metrics.json")
    with open(mpath, "w") as mf:
        json.dump({k: float(v) for k, v in metrics.items()}, mf, indent=2)

    rpath = os.path.join(config.RESULTS_DIR, f"{name}_classification_report.txt")
    with open(rpath, "w") as rf:
        rf.write(report)

    # Plots
    history = load_history(name)
    if history:
        plot_learning_curves(history, name)
    plot_confusion_matrix(cm, config.FOOD_CLASSES, name)

    return metrics


def evaluate_ann(dataset_mode: str = "synthetic"):
    """Evaluate the ANN on held-out nutrition data."""
    import tensorflow as tf
    from models.ann_model import build_training_data

    ann_path = config.ANN_MODEL_PATH
    if not os.path.exists(ann_path):
        print(f"❌ ANN model not found: {ann_path}")
        return {}

    model = tf.keras.models.load_model(ann_path)
    X, y = build_training_data()
    n    = len(X)
    test_X, test_y = X[int(0.8 * n):], y[int(0.8 * n):]

    results = model.evaluate(test_X, test_y, verbose=0)
    print(f"\nANN Evaluation: loss={results[0]:.4f}, mae={results[1]:.4f}")

    # Plot ANN learning curves
    history = load_history("ann")
    if history:
        # ANN uses loss/mae/mse keys instead of accuracy
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.get("loss", []),     label="Train MAE", color="#2196F3")
        ax.plot(history.get("val_loss", []), label="Val MAE",   color="#FF9800")
        ax.set_title("ANN — Training Curves (MAE)"); ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE"); ax.legend(); ax.grid(True, alpha=0.3)
        path = os.path.join(config.RESULTS_DIR, "ann_learning_curves.png")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f"Saved → {path}")

    metrics = {"test_mae": float(results[1])}
    mpath   = os.path.join(config.RESULTS_DIR, "ann_metrics.json")
    with open(mpath, "w") as mf:
        json.dump(metrics, mf, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet",
                        choices=["cnn", "mobilenet", "ann", "all"])
    parser.add_argument("--dataset", default="real",
                        choices=["real", "synthetic", "food101"])
    parser.add_argument("--n_per_class", type=int, default=0,
                        help="Cap test images per class (0 = all)")
    args = parser.parse_args()

    n_per_class = args.n_per_class or None
    if args.dataset == "synthetic":
        print("⚠️  WARNING: synthetic images are random noise; metrics are "
              "meaningless. Use --dataset real for reported results.")

    if args.model in ("mobilenet", "all"):
        evaluate_classifier(config.TL_MODEL_PATH, "mobilenet",
                            args.dataset, n_per_class)
    if args.model in ("cnn", "all"):
        evaluate_classifier(config.CNN_MODEL_PATH, "cnn",
                            args.dataset, n_per_class)
    if args.model in ("ann", "all"):
        evaluate_ann(args.dataset)

    print("\n✅ Evaluation complete. Results saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
