import streamlit as st
import pandas as pd
import re
import string
import os
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from deep_translator import GoogleTranslator

# ----------------- Page Config & Styling -----------------
st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")

st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(to bottom, #0f172a, #1e293b);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextInput > label, .stTextArea > label, .stMarkdown, .stSubheader, .stCaption {
            color: #e2e8f0;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #2563eb !important;
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #1d4ed8 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Sidebar Hamburger Menu -----------------
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar-title, .sidebar-text {
                color: white !important;
                font-weight: bold;
            }
        </style>
        <div class="sidebar-text">Download the full OTT industry report:</div>
    """, unsafe_allow_html=True)

    with open("OTT.pdf", "rb") as pdf_file:
        st.download_button(
            label="📄 Download OTT Report (PDF)",
            data=pdf_file,
            file_name="OTT_Report.pdf",
            mime="application/pdf"
        )

# ----------------- Stopwords -----------------
stop_words = {
    "a", "an", "the", "and", "or", "is", "are", "to", "in", "of", "that", "this", "with", "on",
    "for", "as", "was", "were", "at", "by", "be", "has", "have", "from", "but", "not", "no", "it's",
    "you", "your", "we", "they", "i", "he", "she", "them", "it", "do", "does", "did", "will", "would",
    "should", "could", "my", "our", "their", "if", "than", "then", "about", "just", "so", "because",
    "can", "been", "also", "out", "very", "there", "what", "who", "whom", "when", "where", "how",
    "more", "most", "some", "any", "such", "into", "up", "down", "off", "over", "under", "again", "further",
    "now", "only", "once", "here", "why", "don't", "isn't", "wasn't", "couldn't", "shouldn't", "wouldn't"
}
negation_words = {"not", "no", "never", "don’t", "isn’t", "wasn’t", "couldn’t", "wouldn’t", "shouldn’t"}
stop_words -= negation_words

# ----------------- Preprocessing -----------------
def preprocess_review(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# ----------------- Load Model and Tokenizer -----------------
@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    model_path = hf_hub_download(repo_id="Vansh1128/IMDB", filename="lstm_sentiment_model.h5")
    return load_model(model_path, compile=False)

@st.cache_data(show_spinner=True)
def load_tokenizer_data():
    csv_path = hf_hub_download(repo_id="Vansh1128/IMDB", filename="IMDB Dataset.csv")
    df = pd.read_csv(csv_path)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["review"])
    return tokenizer

model = load_sentiment_model()
tokenizer = load_tokenizer_data()
max_len = 500

# ----------------- Save to CSV -----------------
def save_to_history(movie, review, sentiment):
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Movie": movie,
        "Review": review,
        "Sentiment": sentiment,
    }
    file = "sentiment_history.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(file, index=False)

# ----------------- App UI -----------------
st.title("🎬 Movie Review Sentiment Analyzer")

col1, _ = st.columns([3, 1])
with col1:
    movie_name = st.text_input("🎞️ Enter Movie Name")

user_review = st.text_area("✍️ Enter Your Review (Any Language Supported)")

if st.button("🔍 Analyze Sentiment"):
    if not movie_name or not user_review.strip():
        st.warning("Please enter both movie name and review.")
    else:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(user_review)
            cleaned = preprocess_review(translated)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=max_len)
            prob = model.predict(padded, verbose=0)[0][0]
            sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"

            st.subheader(f"🧠 Sentiment: {sentiment}")
            save_to_history(movie_name, user_review, sentiment)

        except Exception as e:
            st.error(f"Translation or prediction failed: {e}")
