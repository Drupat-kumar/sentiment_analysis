import re
import string
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Lens",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_LEN = 100  # must match training (see notebook cell 10)

# Alphabetical order — this is the order sklearn's LabelEncoder assigns to
# the dair-ai/emotion label strings, so it's the order the model outputs.
LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

EMOTION_META = {
    "anger":    {"emoji": "😠", "color": "#E4572E", "desc": "Sharp, hot, and ready to push back."},
    "fear":     {"emoji": "😨", "color": "#6A4C93", "desc": "On edge, bracing for what's next."},
    "joy":      {"emoji": "😄", "color": "#F2B705", "desc": "Light, warm, unmistakably upbeat."},
    "love":     {"emoji": "❤️", "color": "#D6336C", "desc": "Tender, close, and full of warmth."},
    "sadness":  {"emoji": "😢", "color": "#3A6EA5", "desc": "Heavy-hearted, quiet, and low."},
    "surprise": {"emoji": "😲", "color": "#2E8B57", "desc": "Caught off guard, wide-eyed."},
}

# ─────────────────────────────────────────────────────────────────────────
# Classy CSS — deep charcoal canvas, ivory text, muted gold accents
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #1c1f26 0%, #0f1115 55%, #0a0b0d 100%);
    color: #ECE7DD;
}

/* Hide default streamlit chrome for a cleaner feel */
#MainMenu, footer, header {visibility: hidden;}

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: linear-gradient(120deg, #E8C874 0%, #F5E6C8 45%, #C9A24B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #A7A199;
    letter-spacing: 0.3px;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Glass panel */
.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(232, 200, 116, 0.18);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(232, 200, 116, 0.25) !important;
    border-radius: 12px !important;
    color: #ECE7DD !important;
    font-size: 1.02rem !important;
    line-height: 1.55;
}
.stTextArea textarea:focus {
    border: 1px solid #E8C874 !important;
    box-shadow: 0 0 0 1px #E8C874 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #E8C874, #C9A24B);
    color: #14161a;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(232, 200, 116, 0.25);
    color: #14161a;
}

/* Result block */
.result-emoji {
    font-size: 4.2rem;
    line-height: 1;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    text-transform: capitalize;
    margin: 0.3rem 0 0.1rem 0;
}
.result-desc {
    color: #A7A199;
    font-size: 0.98rem;
    margin-bottom: 0.4rem;
}
.result-conf {
    font-size: 0.95rem;
    color: #E8C874;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.55rem 0;
}
.prob-label {
    width: 88px;
    font-size: 0.88rem;
    color: #D8D3C8;
    text-transform: capitalize;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}
.prob-track {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
}
.prob-pct {
    width: 46px;
    text-align: right;
    font-size: 0.85rem;
    color: #A7A199;
}

/* Divider */
.thin-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,200,116,0.35), transparent);
    margin: 1.6rem 0;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f13;
    border-right: 1px solid rgba(232, 200, 116, 0.12);
}
section[data-testid="stSidebar"] * {
    color: #D8D3C8 !important;
}

/* History chip */
.history-chip {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading (cached)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the model...")
def load_artifacts():
    model = load_model("gru.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def clean_text(text: str) -> str:
    """Mirrors the preprocessing used in the training notebook."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    return text


def predict_emotion(text, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs = model.predict(padded, verbose=0)[0]
    return probs


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🖋️ Emotion Lens")
    st.markdown(
        "A GRU recurrent network trained on the "
        "**dair-ai/emotion** dataset to read the emotional "
        "tone beneath a sentence."
    )
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("**Recognises**")
    for lbl in LABELS:
        meta = EMOTION_META[lbl]
        st.markdown(
            f"<span class='history-chip' style='background:{meta['color']}22; "
            f"color:{meta['color']}; border:1px solid {meta['color']}55;'>"
            f"{meta['emoji']} {lbl.capitalize()}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("**Architecture**")
    st.markdown(
        "Embedding → GRU(128) → Dropout → GRU(64) → Dropout → "
        "Dense(64, relu) → Dense(6, softmax)"
    )
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    if st.session_state.get("history"):
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Emotion Lens</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Type a sentence — a GRU network reads its emotional undertone.</div>",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### Your text")
    text_input = st.text_area(
        label="text_input",
        label_visibility="collapsed",
        placeholder="e.g. \"I can't believe how happy I am right now, this is amazing!\"",
        height=160,
    )
    examples = st.columns(3)
    example_texts = [
        "I feel so alone and hopeless today.",
        "I am furious they cancelled the trip last minute.",
        "I was shocked by the unexpected gift!",
    ]
    for col, ex in zip(examples, example_texts):
        if col.button(ex[:22] + "…", use_container_width=True, key=ex):
            text_input = ex
            st.session_state["_prefill"] = ex

    if "_prefill" in st.session_state and not text_input:
        text_input = st.session_state["_prefill"]

    analyze = st.button("✦ Analyze emotion", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### Reading")

    if analyze and text_input.strip():
        try:
            model, tokenizer = load_artifacts()
            probs = predict_emotion(text_input, model, tokenizer)
            top_idx = int(np.argmax(probs))
            top_label = LABELS[top_idx]
            top_conf = float(probs[top_idx])
            meta = EMOTION_META[top_label]

            st.session_state.history.insert(0, (text_input.strip(), top_label, top_conf))
            st.session_state.history = st.session_state.history[:6]

            st.markdown(
                f"""
                <div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
                    <div class="result-emoji">{meta['emoji']}</div>
                    <div class="result-label" style="color:{meta['color']}">{top_label}</div>
                    <div class="result-desc">{meta['desc']}</div>
                    <div class="result-conf">{top_conf*100:.1f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
            st.markdown("###### Full distribution")

            order = np.argsort(probs)[::-1]
            for i in order:
                lbl = LABELS[i]
                m = EMOTION_META[lbl]
                pct = probs[i] * 100
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <div class="prob-label">{m['emoji']} {lbl}</div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{pct}%; background:{m['color']};"></div>
                        </div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Something went wrong while reading that text: {e}")

    elif analyze:
        st.warning("Type something first — the page is listening, not guessing.")
    else:
        st.markdown(
            "<div style='color:#7A756C; text-align:center; padding: 2.4rem 0;'>"
            "Your emotional reading will appear here.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("##### Recent readings")
    for txt, lbl, conf in st.session_state.history:
        meta = EMOTION_META[lbl]
        snippet = (txt[:80] + "…") if len(txt) > 80 else txt
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <span class="history-chip" style="background:{meta['color']}22;
                      color:{meta['color']}; border:1px solid {meta['color']}55;">
                    {meta['emoji']} {lbl.capitalize()} · {conf*100:.0f}%
                </span>
                <span style="color:#A7A199; font-size:0.9rem;">{snippet}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
