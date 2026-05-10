import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ── 1. REPRODUCIBILITY ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── 2. LOAD & PREPROCESS DATA ─────────────────────────────────────────────────
def load_data():
    """
    Loads the MNIST dataset from Keras, normalizes pixel values to [0, 1],
    and reshapes images to include the channel dimension (28, 28, 1).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize: pixel values are 0–255, we bring them to 0.0–1.0
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Reshape: CNN expects (batch, height, width, channels)
    # MNIST is grayscale → 1 channel
    x_train = x_train[..., np.newaxis]  # shape: (60000, 28, 28, 1)
    x_test  = x_test[..., np.newaxis]   # shape: (10000, 28, 28, 1)

    print(f"Training samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")
    print(f"Image shape      : {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


# ── 3. BUILD THE CNN MODEL ────────────────────────────────────────────────────
def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Architecture:
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten
        → Dense(128) → Dropout(0.5) → Dense(10, softmax)
    """
    model = models.Sequential(name="digit_cnn")

    # --- Block 1: first convolutional block ---
    # 32 filters, each 3×3, slides across the image to detect edges/curves
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                            input_shape=input_shape, name="conv1"))
    # MaxPool halves the spatial size (28→13), keeping the dominant features
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # --- Block 2: second convolutional block ---
    # 64 filters — deeper features (curves, loops specific to each digit)
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))

    # --- Flatten: 2D feature maps → 1D vector ---
    model.add(layers.Flatten(name="flatten"))

    # --- Dense head ---
    model.add(layers.Dense(128, activation="relu", name="fc1"))
    # Dropout randomly zeros 50% of neurons during training → prevents overfitting
    model.add(layers.Dropout(0.5, name="dropout"))

    # Output layer: 10 neurons (one per digit 0–9), softmax gives probabilities
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    return model


# ── 4. COMPILE & TRAIN ────────────────────────────────────────────────────────
def train(model, x_train, y_train, x_test, y_test):
    """
    Compiles and trains the model with early stopping and model checkpointing.
    """
    model.compile(
        optimizer="adam",
        # sparse_categorical_crossentropy works with integer labels (0–9 directly)
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,           # stop if val_loss doesn't improve for 3 epochs
        restore_best_weights=True,
        verbose=1,
    )

    os.makedirs("model/saved", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath="model/saved/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,   # 10% of training data used for validation
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    return history


# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
def evaluate(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc * 100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")
    return test_acc


# ── 6. PLOT TRAINING HISTORY ──────────────────────────────────────────────────
def plot_history(history):
    """Saves a training/validation accuracy & loss plot as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    os.makedirs("model/saved", exist_ok=True)
    plt.tight_layout()
    plt.savefig("model/saved/training_history.png", dpi=150)
    print("Training plot saved → model/saved/training_history.png")
    plt.close()


# ── 7. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  MNIST Digit Recognizer — Model Training")
    print("=" * 50)

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    history = train(model, x_train, y_train, x_test, y_test)
    evaluate(model, x_test, y_test)
    plot_history(history)

    print("\nBest model saved → model/saved/best_model.keras")
    print("Run  →  streamlit run app.py  to launch the app.")
