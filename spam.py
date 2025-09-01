import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import re
import time

# Page setup
st.set_page_config(page_title="Spam Classifier Terminal", page_icon="💾", layout="centered")

# ------------------ Custom CSS for Terminal Style ------------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: black !important;
            color: #33FF33 !important;
            font-family: 'Courier New', monospace !important;
        }

        .terminal-title {
            font-size: 36px;
            font-weight: bold;
            color: #33FF33;
            text-align: center;
            margin-bottom: 0.3em;
        }

        .terminal-subtitle {
            font-size: 16px;
            text-align: center;
            color: #66FF66;
            margin-bottom: 3em;
        }

        .stTextArea textarea {
            background-color: black !important;
            color: #33FF33 !important;
            border: 1px solid #33FF33 !important;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }

        .stButton>button {
            background-color: #33FF33;
            color: black;
            border: none;
            border-radius: 0px;
            padding: 0.5em 1.5em;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            margin-top: 1em;
        }

        .stButton>button:hover {
            background-color: #66FF66;
            color: black;
        }

        .terminal-result {
            border: 1px dashed #33FF33;
            padding: 1.2em;
            margin-top: 2em;
            font-size: 18px;
            text-align: center;
        }

        .spam {
            color: red;
        }

        .not-spam {
            color: #33FF33;
        }

        .blinker {
            animation: blink 1s infinite;
        }

        @keyframes blink {
            50% {
                opacity: 0;
            }
        }

        .footer {
            margin-top: 4em;
            text-align: center;
            font-size: 12px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Terminal Intro ------------------
st.markdown("<div class='terminal-title'>🖥️ SPAM CLASSIFIER TERMINAL</div>", unsafe_allow_html=True)
st.markdown("<div class='terminal-subtitle'>Initializing AI systems... <span class='blinker'>█</span><br><br>Type a message below to analyze if it's SPAM or NOT SPAM.</div>", unsafe_allow_html=True)

# ------------------ Load Model ------------------
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

# ------------------ Input and Analysis ------------------
with st.form("terminal_form"):
    input_message = st.text_area(">>", placeholder="e.g., You won a free iPhone! Click here...", height=150, key="input_message")
    submitted = st.form_submit_button("RUN")

    if submitted:
        if input_message.strip() == "":
            st.warning("⚠️ No input detected. Please enter a message.")
        else:
            with st.spinner("Analyzing message... please wait..."):
                time.sleep(1.5)
                result = predict(input_message)

            if result == "Spam":
                st.markdown("<div class='terminal-result spam'>🚫 Result: This message is classified as <strong>SPAM</strong>.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='terminal-result not-spam'>✅ Result: This message is classified as <strong>NOT SPAM</strong>.</div>", unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("<div class='footer'>Terminal UI ⌨️ | Built using Streamlit & Scikit-learn | 2025</div>", unsafe_allow_html=True)
