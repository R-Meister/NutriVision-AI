"""Model performance page: evaluation metrics for each trained model."""
import json
import os

import pandas as pd
import streamlit as st

import config
from ui.theme import page_header, section_title

CNN_MODEL_NAME = "Custom CNN"


def tab_performance():
    page_header(
        "Model",
        "Held-out test results for the food classifier and the nutrition regressor.",
    )

    test_size = _test_sample_count()
    if test_size:
        st.caption(
            f"Test set: {test_size} real-photo images the models never saw during training."
        )

    tabs = st.tabs(["MobileNetV2", "Custom CNN", "ANN", "Comparison"])

    with tabs[0]:
        section_title("MobileNetV2")
        st.caption(
            "ImageNet backbone kept frozen, features pooled with global average "
            "pooling, and a linear probe trained on top."
        )
        _render_model_results("mobilenet", "MobileNetV2")

    with tabs[1]:
        section_title("Custom 4-Block CNN")
        st.caption(
            "From-scratch ConvNet with four convolution blocks, trained only on the "
            "food dataset."
        )
        _render_model_results("cnn", CNN_MODEL_NAME)

    with tabs[2]:
        section_title("Nutritional ANN")
        _render_ann_results()

    with tabs[3]:
        section_title("Architecture comparison")
        _render_comparison_table()


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _mobile_metrics() -> dict:
    """Merge the classifier metrics with the linear-probe run details."""
    metrics = dict(_read_json(os.path.join(config.RESULTS_DIR, "mobilenet_metrics.json")))
    probe = _read_json(os.path.join(config.RESULTS_DIR, "mobilenet_linear_metrics.json"))

    # The linear-probe file is the authoritative source for the production run,
    # so it overrides the classifier metrics when both are present.
    if "test_accuracy" in probe:
        metrics["accuracy"] = probe["test_accuracy"]
    if "top3_accuracy" in probe:
        metrics["top3_accuracy"] = probe["top3_accuracy"]
    if "test_samples" in probe and "test_samples" not in metrics:
        metrics["test_samples"] = probe["test_samples"]
    return metrics


def _test_sample_count():
    for metrics in (
        _mobile_metrics(),
        _read_json(os.path.join(config.RESULTS_DIR, "cnn_metrics.json")),
    ):
        samples = metrics.get("test_samples")
        if samples:
            return int(samples)
    return None


def _render_model_results(model_prefix: str, model_title: str):
    if model_prefix == "mobilenet":
        metrics = _mobile_metrics()
    else:
        metrics = _read_json(os.path.join(config.RESULTS_DIR, f"{model_prefix}_metrics.json"))

    if not metrics:
        if model_prefix == "mobilenet":
            st.info("No evaluation metrics found for MobileNetV2. Run the evaluation script first.")
        else:
            st.info(
                f"The optional {model_title} has not been trained and evaluated yet. "
                "MobileNetV2 is the model used in the app, so this section stays empty "
                "unless you train the custom network."
            )
        return

    cm_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_confusion_matrix.png")
    lc_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_learning_curves.png")
    rep_path = os.path.join(config.RESULTS_DIR, f"{model_prefix}_classification_report.txt")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test top-1 accuracy", _pct(metrics.get("accuracy"), 2))
        c2.metric("Test top-3 accuracy", _pct(metrics.get("top3_accuracy"), 2))
        c3.metric("Macro F1", _pct(metrics.get("f1"), 2))
        c4.metric("Held-out test images", f"{int(metrics.get('test_samples', 0))}")

        st.caption(
            f"Macro precision {_pct(metrics.get('precision'))} · "
            f"macro recall {_pct(metrics.get('recall'))}"
        )

    st.write("")

    has_curve = os.path.exists(lc_path)
    has_cm = os.path.exists(cm_path)
    if has_curve and has_cm:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(
                lc_path,
                caption=f"{model_title} training and validation curves",
                width="stretch",
            )
        with col_b:
            st.image(cm_path, caption=f"{model_title} confusion matrix", width="stretch")
    elif has_cm:
        st.image(cm_path, caption=f"{model_title} confusion matrix", width="stretch")
    elif has_curve:
        st.image(
            lc_path,
            caption=f"{model_title} training and validation curves",
            width="stretch",
        )

    if os.path.exists(rep_path):
        with open(rep_path) as f:
            report = f.read()
        with st.expander("Classification report"):
            st.code(report, language="text")


def _render_ann_results():
    metrics = _read_json(os.path.join(config.RESULTS_DIR, "ann_metrics.json"))
    lc_path = os.path.join(config.RESULTS_DIR, "ann_learning_curves.png")

    if not metrics:
        st.info("No evaluation metrics found for the nutritional ANN. Run the evaluation script first.")
    else:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("Test MAE (calories)", f"{float(metrics.get('test_mae', 0)):.2f} kcal")
            c2.metric("Outputs", "Calories, protein, carbs, fat")

        st.caption(
            "Average absolute error when predicting calories for a given food and portion."
        )

    if os.path.exists(lc_path):
        st.write("")
        st.image(lc_path, caption="ANN training and validation MAE", width="stretch")

    st.write("")
    st.markdown(
        "The CNN reads the food photo and outputs a class. The ANN works on a "
        "different kind of input: 13 numbers (a one-hot food class plus the "
        "normalised portion weight) map to four nutrition values in kcal or grams. "
        "It is a compact dense network, so nutrition estimates cost almost no "
        "compute once the food class and portion are known."
    )


def _render_comparison_table():
    comparison = pd.DataFrame(
        [
            {
                "Model": "MobileNetV2",
                "Type": "Frozen CNN + linear probe",
                "Input": "224 × 224 × 3 image",
                "Parameters": "~2.3M backbone + ~15K head",
                "Advantage": "91.1% top-1 / 98.6% top-3 on held-out photos; fast on CPU",
            },
            {
                "Model": "Custom 4-Block CNN",
                "Type": "From-scratch ConvNet",
                "Input": "224 × 224 × 3 image",
                "Parameters": "~26M",
                "Advantage": "Optional baseline; no pretrained weights",
            },
            {
                "Model": "Nutritional ANN",
                "Type": "Dense multi-layer perceptron",
                "Input": "13 numerical features",
                "Parameters": "~18.5K",
                "Advantage": "Near-instant nutrition regression, no image processing",
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        "MobileNetV2 is the model serving predictions. The custom CNN is optional "
        "and only appears when it has been trained and evaluated."
    )
