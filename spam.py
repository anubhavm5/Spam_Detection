import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import re
import time

# ------------------ Custom CSS for Beautiful UI ------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.markdown("""
    <style>
        /* General app background and font */
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }

        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #2E86C1;
            margin-bottom: 0.1em;
        }

        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #566573;
            margin-bottom: 2em;
        }

        .input-area textarea {
            border: 2px solid #3498DB !important;
            border-radius: 10px !important;
            padding: 10px;
            font-size: 16px;
        }

        .stButton>button {
            background-color: #3498DB;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5em 2em;
            font-size: 16px;
            font-weight: bold;
        }

        .stButton>button:hover {
            background-color: #2E86C1;
            color: white;
        }

        .result-box {
            margin-top: 2em;
            padding: 1.2em;
            border-radius: 10px;
            font-size: 20px;
            text-align: center;
            font-weight: bold;
        }

        .spam {
            background-color: #FADBD8;
            color: #C0392B;
        }

        .not-spam {
            background-color: #D4EFDF;
            color: #1E8449;
        }

        .footer {
            margin-top: 4em;
            font-size: 13px;
            text-align: center;
            color: #95A5A6;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Title Section ------------------
st.markdown("<div class='title'>📩 Spam Message Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Type or paste a message to check if it's <b>Spam</b> or <b>Not Spam</b>.</div>", unsafe_allow_html=True)

# ------------------ Load and Train Model ------------------
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

# ------------------ UI Input ------------------
with st.form("spam_form"):
    input_message = st.text_area("✏️ Enter your message here", height=150, key="input", placeholder="e.g., You have won a $1000 Walmart gift card. Click here to claim now!")
    submitted = st.form_submit_button("🔍 Check Message")

    if submitted:
        if input_message.strip() == "":
            st.warning("⚠️ Please enter a message to check.")
        else:
            with st.spinner("Analyzing your message..."):
                time.sleep(1.5)
                result = predict(input_message)

            if result == "Spam":
                st.markdown("<div class='result-box spam'>🚫 This message is likely <b>SPAM</b></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box not-spam'>✅ This message is <b>NOT SPAM</b></div>", unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Scikit-learn</div>", unsafe_allow_html=True)
