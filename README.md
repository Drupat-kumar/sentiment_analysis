# 🖋️ Emotion Lens

A dark, editorial-styled Streamlit app that reads the emotional undertone of a sentence using a GRU recurrent neural network trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

<p align="center">
  <em>Anger · Fear · Joy · Love · Sadness · Surprise</em>
</p>

---

## ✨ What it does

Type any sentence and the app will:

1. Clean and tokenize the text exactly as it was preprocessed during training
2. Run it through a trained GRU network
3. Show the top predicted emotion with a confidence score
4. Break down the full probability distribution across all six emotions
5. Keep a running history of your last few readings

---

## 📦 Project structure

```
.
├── app.py              # Streamlit application (UI + inference)
├── gru.keras           # Trained GRU model (Keras 3 native format)
├── tokenizer.pkl        # Fitted Keras Tokenizer used at training time
└── requirements.txt     # Pinned dependencies
```

> **Note:** `gru.keras` and `tokenizer.pkl` must sit in the same directory as `app.py` — the app loads them by relative path.

---

## 🚀 Getting started

### 1. Clone / download the project files

Make sure `app.py`, `gru.keras`, `tokenizer.pkl`, and `requirements.txt` are all in the same folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app at `http://localhost:8501`.

---

## 🧠 Model details

| | |
|---|---|
| **Task** | 6-class emotion classification |
| **Architecture** | `Embedding → GRU(128) → Dropout → GRU(64) → Dropout → Dense(64, relu) → Dense(6, softmax)` |
| **Vocabulary size** | 10,000 words |
| **Sequence length** | 100 tokens (padded/truncated) |
| **Dataset** | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| **Format** | Native Keras 3 (`.keras`), saved with Keras 3.13.2 |

**Label order matters:** the model's output layer follows the order produced by scikit-learn's `LabelEncoder`, which sorts class names alphabetically — **not** the dataset's original label order:

```
0 → anger
1 → fear
2 → joy
3 → love
4 → sadness
5 → surprise
```

### Text preprocessing (must match training)

Before tokenizing, input text is cleaned the same way it was during training:

- Punctuation removed
- Digits removed
- HTML tags stripped
- URLs stripped
- Emoji characters removed

This logic lives in `clean_text()` inside `app.py` — if you retrain the model with different preprocessing, update it there too.

---

## 🛠️ Requirements

See [`requirements.txt`](./requirements.txt). Versions are pinned rather than left open-ended:

- `streamlit==1.38.0`
- `tensorflow==2.18.0` + `keras==3.13.2` — matched to the Keras version the model was saved with, since loading a `.keras` file with a mismatched Keras version can fail
- `numpy==1.26.4`

---

## ⚠️ Known limitations

- Trained on short, first-person English sentences (Twitter-style text) — accuracy will drop on long-form, multilingual, or heavily sarcastic text.
- The model predicts a single dominant emotion; it doesn't currently support multi-label output (e.g. text that's both angry *and* sad).
- No authentication, rate limiting, or logging — this is built for local/demo use, not production deployment as-is.

---


