"""About / architecture page."""
import streamlit as st

from ui.theme import page_header

_TRAINING_DATASETS = "Food-101, Indian Foods, Indian Food Images, and Fruits-360"


def tab_about():
    page_header(
        "About",
        "What NutriVision AI recognizes, how it works, and where it stops.",
    )

    st.markdown(
        "NutriVision AI identifies food from a single photo and estimates its "
        "nutrition. It classifies the image into one of twelve supported "
        "dishes, estimates the portion weight, looks up calories and macros "
        "from a local nutrition database, and lets you log the result to a "
        "persistent daily food diary."
    )

    st.divider()

    # ── How it works ─────────────────────────────────────────────────────────
    st.subheader("How it works")
    st.markdown(
        "1. **Upload a photo** — a meal photo, or one of the bundled samples.\n"
        "2. **Detect regions** — OpenCV finds separate food regions on the plate, "
        "so a meal with several items is analyzed per item.\n"
        "3. **Classify each region** — a frozen MobileNetV2 backbone produces "
        "features, and a multinomial logistic-regression classifier predicts the "
        "food class with top-3 probabilities.\n"
        "4. **Estimate the portion** — food pixel area is scaled against a typical "
        "serving weight from the database. You can correct the weight with a slider.\n"
        "5. **Calculate nutrition** — per-100 g values are scaled to the portion and "
        "shown as calories, protein, carbohydrate, fat, fiber, sugar, and sodium.\n"
        "6. **Save to the diary** — items are stored in a local SQLite database with "
        "timestamps and meal type (breakfast, lunch, dinner, snack)."
    )

    st.divider()

    # ── What it recognizes ──────────────────────────────────────────────────
    st.subheader("What it recognizes")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**12 classifier classes**")
        st.markdown(
            "- Pizza, burger, french fries, sandwich, omelette\n"
            "- Samosa, dosa, idli, biryani, fried rice\n"
            "- Apple, banana"
        )
        st.caption(
            "Trained on real photographs from " + _TRAINING_DATASETS + "."
        )

    with right:
        st.markdown("**Nutrition database**")
        st.markdown(
            "- 504 food entries, each with calories and macros per 100 g\n"
            "- Covers Indian dishes, street food, Western meals, fruits, "
            "vegetables, drinks, and desserts\n"
            "- Stored locally as JSON/CSV, so it works fully offline"
        )
        st.caption(
            "The database is a reference table. Recognition is limited to the "
            "12 classes above."
        )

    st.divider()

    # ── Model ───────────────────────────────────────────────────────────────
    st.subheader("Model")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top-1 accuracy", "91.1%")
    c2.metric("Top-3 accuracy", "98.6%")
    c3.metric("Held-out photos", "562")

    st.markdown(
        "- **Backbone:** MobileNetV2 pre-trained on ImageNet with all layers "
        "frozen. The final global average pooling layer yields 1280 features.\n"
        "- **Classifier:** a `StandardScaler` followed by multinomial logistic "
        "regression over 12 classes. The scaler and weights are folded into a "
        "single `Dense` layer, so the deployed model is one self-contained "
        "`.keras` file.\n"
        "- **Evaluation:** 562 real photographs (47 per class) that were never "
        "used for training or validation.\n"
        "- **Non-food gate:** each image is compared with a bank of training "
        "features (mean top-5 cosine similarity, threshold 0.57). Photos of "
        "faces or objects are reported as \"not one of the 12 supported foods\" "
        "instead of receiving invented calories."
    )
    st.caption(
        "Full per-class report, confusion matrix, and metrics are available in "
        "the `results/` folder and on the Model tab."
    )

    st.divider()

    # ── Limitations ─────────────────────────────────────────────────────────
    st.subheader("Limitations")
    st.markdown(
        "- **Only 12 classes.** Anything outside them is either rejected by the "
        "gate or mapped to the nearest supported class.\n"
        "- **Portion size is approximate.** A 2D photo has no depth or metric "
        "scale, so area-based weight estimates are heuristics. Correct them with "
        "the slider when accuracy matters.\n"
        "- **Dataset limitations.** The training photos come from a handful of "
        "public datasets, mostly Western and Indian dishes. Lighting, plating, "
        "occlusion, and mixed dishes can reduce accuracy in the real world.\n"
        "- **The optional custom CNN is not always trained.** The 4-block CNN "
        "trains slowly on CPU and is a comparison model; the app falls back to "
        "MobileNetV2 when its checkpoint is absent.\n"
        "- **Approximate nutrition.** Database values are typical averages and "
        "do not account for recipe variation or added oil."
    )

    st.divider()

    # ── Tech stack ──────────────────────────────────────────────────────────
    st.subheader("Tech stack")
    st.markdown(
        "- Streamlit for the web interface\n"
        "- TensorFlow / Keras for MobileNetV2 and the optional CNN\n"
        "- scikit-learn for the logistic-regression head and metrics\n"
        "- OpenCV and Pillow for image preprocessing and region detection\n"
        "- NumPy and Pandas for data handling\n"
        "- SQLite for the persistent food diary"
    )

    st.divider()

    # ── Viva Q&A ────────────────────────────────────────────────────────────
    with st.expander("Viva Q&A"):
        st.markdown("**Why use a linear probe instead of fine-tuning?**")
        st.markdown(
            "With roughly 2,350 training photos, fine-tuning the full MobileNetV2 "
            "overfits. Frozen ImageNet features plus a regularized linear "
            "classifier scored higher here and trains in seconds."
        )

        st.markdown("**Why MobileNetV2?**")
        st.markdown(
            "Its depthwise separable convolutions keep the model small (about "
            "3.4M parameters versus 138M for VGG16), so inference is fast on CPU "
            "and its ImageNet features transfer well to food images."
        )

        st.markdown("**How do you estimate quantity from a single 2D image?**")
        st.markdown(
            "A 2D RGB image has no depth scale. OpenCV measures the foreground "
            "food area, which is scaled against a typical serving weight from the "
            "database. The result is labeled as an approximation and can be "
            "adjusted with a slider."
        )

        st.markdown("**What happens if the photo is not food?**")
        st.markdown(
            "A 12-class softmax always returns a label. The non-food gate "
            "compares the image's features with the training feature bank; if the "
            "mean top-5 cosine similarity is below 0.57, the app reports that the "
            "photo is not a supported food and shows no nutrition values."
        )

        st.markdown("**What is the CNN versus the ANN in this project?**")
        st.markdown(
            "The CNN answers \"what food is this?\" from pixels (12 classes). The "
            "ANN is a separate tabular regressor that takes a one-hot food class "
            "plus normalized weight and predicts calories, protein, carbohydrate, "
            "and fat (test MAE 9.06 kcal)."
        )

        st.caption(
            "The same facts are used across the README, `viva_qa.py`, and the "
            "Model tab."
        )
