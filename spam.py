import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import re
import time

# Set up the page
st.set_page_config(
    page_title="Spam Shield | Message Classifier",
    page_icon="🛡️",
    layout="centered"
)

# ------------- Custom CSS -------------
st.markdown("""
    <style>
        html, body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }

        .hero {
            text-align: center;
            padding: 3rem 1rem 2rem 1rem;
        }

        .hero h1 {
            font-size: 3rem;
            color: #2C3E50;
            margin-bottom: 0.2em;
        }

        .hero p {
            font-size: 1.2rem;
            color: #566573;
            max-width: 600px;
            margin: auto;
        }

        .section {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            margin-top: 1.5rem;
        }

        .stTextArea textarea {
            border: 1px solid #ccc;
            border-radius: 10px;
            font-size: 16px;
        }

        .stButton>button {
            background-color: #2980B9;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 0.5em 1.5em;
            border-radius: 8px;
        }

        .stButton>button:hover {
            background-color: #1F618D;
        }

        .result {
            text-align: center;
            font-size: 1.2rem;
            padding: 1.2rem;
            margin-top: 2rem;
            border-radius: 10px;
        }

        .spam {
            background-color: #FADBD8;
            color: #C0392B;
            border: 1px solid #E6B0AA;
        }

        .not-spam {
            background-color: #D4EFDF;
            color: #1E8449;
            border: 1px solid #A9DFBF;
        }

        .footer {
            text-align: center;
            font-size: 13px;
            color: #888;
            margin-top: 4rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------- Hero Section (Intro) -------------
st.markdown("""
    <div class='hero'>
        <h1>🛡️ Spam Shield</h1>
        <p>Protect your inbox! Spam Shield uses machine learning to detect whether a message is <strong>spam</strong> or <strong>not spam</strong>. Paste a message below and let AI do the rest.</p>
    </div>
""", unsafe_allow_html=True)

# ------------- Load and Prepare Model -------------
data = pd.read_csv("mail_dataSet.csv")
data.drop_duplicates(inplace=True)
data["Category"] = data["Category"].replace(["ham", "spam"], ["Not Spam", "Spam"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text

data["Message"] = data["Message"].apply(preprocess_text)

mess = data["Message"]
cat = data["Category"]
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words="english")
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)

def predict(message):
    message = preprocess_text(message)
    input_features = cv.transform([message]).toarray()
    return model.predict(input_features)[0]

# ------------- Main Input Section -------------
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    with st.form("input_form"):
        input_message = st.text_area("✉️ Enter your message", height=150, placeholder="e.g., Congratulations! You've won a $1000 gift card. Click here to claim it.")
        submit = st.form_submit_button("🚀 Analyze Message")

        if submit:
            if input_message.strip() == "":
                st.warning("⚠️ Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing message..."):
                    time.sleep(1.2)
                    result = predict(input_message)

                if result == "Spam":
                    st.markdown("<div class='result spam'>🚫 This message is classified as <strong>SPAM</strong>.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result not-spam'>✅ This message is classified as <strong>NOT SPAM</strong>.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------- Footer -------------
st.markdown("<div class='footer'>© 2025 Spam Shield — Built with 💙 using Streamlit & Scikit-learn</div>", unsafe_allow_html=True)
