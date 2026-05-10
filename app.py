"""
app.py
------
Streamlit web app for the Handwritten Digit Recognizer.
Loads the pre-trained CNN model and lets users draw a digit on a canvas,
then predicts what digit it is with confidence scores.

Usage:
    streamlit run app.py

Requirements:
    Make sure you have run  `python model/train_model.py`  first.
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✏️",
    layout="centered",
)


# ── LOAD MODEL (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_model():
    """
    Loads the saved Keras model. @st.cache_resource means this function
    runs only once — the model stays in memory for all subsequent reruns.
    """
    model = tf.keras.models.load_model("model/saved/best_model.keras")
    return model


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_canvas(canvas_image: np.ndarray) -> np.ndarray:
    """
    The canvas returns a (400, 400, 4) RGBA numpy array.
    We need to turn it into a (1, 28, 28, 1) float32 array for the model.

    Steps:
        1. Convert RGBA → grayscale (L mode)
        2. Invert colors (canvas: white digit on black → MNIST: white digit on black ✓)
        3. Resize to 28×28 (MNIST input size)
        4. Normalize pixel values to 0.0–1.0
        5. Add batch & channel dimensions
    """
    # Step 1: RGBA numpy array → PIL Image → Grayscale
    img = Image.fromarray(canvas_image.astype("uint8"), "RGBA").convert("L")

    # Step 2: Invert — the drawable canvas gives white strokes on black,
    # which already matches MNIST convention (no inversion needed here),
    # but we apply to be safe if background differs.
    img = ImageOps.invert(img)

    # Step 3: Resize to 28×28 using LANCZOS (best quality for downscaling)
    img = img.resize((28, 28), Image.LANCZOS)

    # Step 4: Numpy + normalize
    img_array = np.array(img, dtype="float32") / 255.0

    # Step 5: (28,28) → (1, 28, 28, 1)  [batch=1, height, width, channels=1]
    img_array = img_array[np.newaxis, ..., np.newaxis]

    return img_array


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("✏️ Handwritten Digit Recognizer")
    st.markdown(
        "Draw a digit **(0–9)** on the canvas below and hit **Predict** to see what the CNN thinks."
    )

    st.divider()

    # Load the model
    try:
        model = load_model()
    except Exception:
        st.error(
            "Model not found! Please run  `python model/train_model.py`  first to train and save the model."
        )
        st.stop()

    # Layout: canvas on left, results on right
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Draw here")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",       # background color
            stroke_width=18,                       # brush thickness
            stroke_color="#FFFFFF",                # white strokes
            background_color="#000000",            # black background
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )

        predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")
        clear_note  = st.caption("Hit 'Clear' (top-right of canvas) to reset.")

    with col2:
        st.subheader("Prediction")

        if predict_btn:
            if canvas_result.image_data is None:
                st.warning("Please draw a digit first.")
            else:
                # Check the canvas isn't blank (all zeros = nothing drawn)
                pixel_sum = canvas_result.image_data[:, :, :3].sum()
                if pixel_sum < 100:
                    st.warning("Canvas looks empty. Draw something first!")
                else:
                    with st.spinner("Thinking..."):
                        # Preprocess
                        img_input = preprocess_canvas(canvas_result.image_data)
                        # Predict — returns array of shape (1, 10)
                        predictions = model.predict(img_input, verbose=0)[0]

                    predicted_digit = int(np.argmax(predictions))
                    confidence      = float(predictions[predicted_digit]) * 100

                    # Main result
                    st.metric(
                        label="Predicted Digit",
                        value=str(predicted_digit),
                        delta=f"{confidence:.1f}% confidence",
                    )

                    st.divider()

                    # Confidence bar for every digit
                    st.markdown("**Confidence per digit**")
                    for digit, prob in enumerate(predictions):
                        pct = float(prob) * 100
                        bar_color = "🟩" if digit == predicted_digit else "⬜"
                        st.markdown(
                            f"`{digit}` {bar_color} **{pct:.1f}%**"
                        )
                        st.progress(float(prob))

        else:
            st.info("Draw a digit and click **Predict**.")

    st.divider()

    # ── Expandable: show MNIST sample images ─────────────────────────────────
    with st.expander("ℹ️ How does this work?", expanded=False):
        st.markdown(
            """
            **Model Architecture** — Convolutional Neural Network (CNN):
            - **Conv2D (32 filters)** → detects edges and simple shapes
            - **MaxPooling** → reduces image size, keeps dominant features
            - **Conv2D (64 filters)** → detects complex curves and loops
            - **MaxPooling** → further reduction
            - **Dense (128 neurons)** → combines all learned features
            - **Dropout (50%)** → prevents overfitting during training
            - **Dense (10 neurons, softmax)** → outputs probabilities for digits 0–9

            **Dataset** — MNIST: 70,000 handwritten digit images (60K train / 10K test).

            **Accuracy** — Typically **99%+** on the MNIST test set.
            """
        )


if __name__ == "__main__":
    main()
