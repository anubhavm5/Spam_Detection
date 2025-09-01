import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import re
import time

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            background-color: #fff5f5;
            color: #333;
            font-size: 16px;
            border: 1px solid #ff4b4b;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #d43838;
        }
        h1, h2, h3, h4, h5 {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

data = pd.read_csv("mail_dataSet.csv")
data.drop_duplicates(inplace=True)
data["Category"] = data["Category"].replace(["ham", "spam"],["Not Spam", "Spam"])

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


st.header("📩 Message Classifier")
input_mess = st.text_input("Enter the Message")

if st.button("Validate"):
    if input_mess.strip():
        with st.spinner("🔍 Checking Message Type..."):
            time.sleep(2)
            output = predict(input_mess)
            if output == "Spam":
                st.markdown("<h5 style='color: red;'>⚠️ Message: Spam</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='color: green;'>✅ Message: Not Spam</h5>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to validate.")
