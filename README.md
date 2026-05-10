# ✏️ Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) trained on the MNIST dataset that recognizes handwritten digits (0–9) in real time via a Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99%25+-brightgreen)

---

## 📸 Features

- Draw any digit (0–9) on an interactive canvas
- CNN predicts the digit with a confidence score
- Confidence bar chart shown for all 10 digits
- Trained on 60,000 MNIST images — achieves **~99% test accuracy**

---

## 🗂️ Project Structure

```
digit-recognizer/
├── model/
│   ├── train_model.py          # CNN training script (run once)
│   └── saved/
│       ├── best_model.keras    # Saved model (auto-generated after training)
│       └── training_history.png
├── app.py                      # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/hasanjaved180598/digit-recognizer.git
cd digit-recognizer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (run once)
```bash
python model/train_model.py
```
This downloads MNIST automatically, trains the CNN, and saves the model to `model/saved/best_model.keras`.

### 5. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, draw a digit, and hit **Predict**.

---

## 🧠 Model Architecture

```
Input (28×28×1)
    │
Conv2D — 32 filters, 3×3, ReLU        ← detects edges and basic shapes
    │
MaxPooling2D — 2×2                     ← reduces spatial size
    │
Conv2D — 64 filters, 3×3, ReLU        ← detects curves and loops
    │
MaxPooling2D — 2×2
    │
Flatten
    │
Dense — 128 neurons, ReLU
    │
Dropout — 50%                          ← prevents overfitting
    │
Dense — 10 neurons, Softmax            ← probability per digit (0–9)
```

---

## 📊 Results

| Metric         | Value  |
|----------------|--------|
| Test Accuracy  | ~99%   |
| Test Loss      | ~0.03  |
| Training time  | ~3 min (CPU) |
| Dataset        | MNIST (60K train / 10K test) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | CNN model building and training |
| MNIST Dataset | 70,000 handwritten digit images |
| Streamlit | Web app UI |
| streamlit-drawable-canvas | Interactive drawing canvas |
| Pillow (PIL) | Image preprocessing |
| NumPy | Array operations |
| Matplotlib | Training history plots |

---

## 📚 What I Learned

- Building and training CNNs with Keras
- Understanding Conv2D, MaxPooling, Dropout layers
- Preprocessing image data for ML (normalization, reshaping)
- Using callbacks (EarlyStopping, ModelCheckpoint)
- Deploying ML models with Streamlit
- Bridging a drawn canvas image to a model-ready tensor

---

## 👤 Author

**Hasan Javed** — Flutter & ML Developer  
[GitHub](https://github.com/hasanjaved180598) · [LinkedIn](https://linkedin.com/in/hasanjaved180598)
