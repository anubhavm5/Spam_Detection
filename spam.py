import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import re
import time

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            padding: 2rem;
        }
        .title {
            font-size: 42px;
            font-weight: 800;
            color: #2E4053;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .subtitle {
            font-size: 18px;
            color: #566573;
            text-align: center;
            margin-bottom: 2em;
        }
        .input-box input {
            background-color: #ffffff;
            border: 2px solid #5DADE2;
            border-radius: 10px;
            padding: 0.75em;
            font-size: 16px;
            color: #2c3e50;
        }
        .stButton > button {
            background-color: #5DADE2;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 0.6em 1.5em;
            border-radius: 10px;
            margin-top: 1em;
        }
        .stButton > button:hover {
            background-color: #3498DB;
        }
        .result-box {
            background-color: #D5F5E3;
            color: #1D8348;
            padding: 1em;
            border-radius: 10px;
            margin-top: 2em;
            font-weight: bold;
            font-size: 18px;
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📩 Spam Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect whether a message is <strong>Spam</strong> or <strong>Not Spam</strong></div>", unsafe_allow_html=True)


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
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words="english")
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)

def predict(message):
    message = preprocess_text(message)
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

with st.container():
    input_mess = st.text_input("💬 Enter your message below:", key="input_text")
    
    if st.button("🔍 Check Message"):
        if input_mess.strip():
            with st.spinner("Analyzing..."):
                time.sleep(1.5)
                output = predict(input_mess)
                if output == "Spam":
                    st.markdown(f"<div class='result-box spam'>🚫 This message is likely <strong>SPAM</strong></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box'>✅ This message is <strong>NOT SPAM</strong></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a message to classify.")


