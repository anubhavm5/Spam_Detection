# 📧 Spam Detection (Naive Bayes + Streamlit)

A machine learning project that classifies SMS/email messages as **Spam** or **Not Spam** using a **Multinomial Naive Bayes** classifier and a **bag-of-words** model, wrapped in an interactive **Streamlit** web app for real-time predictions.


---

## 🚀 Features

- Cleans and preprocesses raw message text (lowercasing, punctuation removal, whitespace normalization)
- Extracts features using `CountVectorizer` (bag-of-words, English stop-words removed)
- Trains a `MultinomialNB` classifier on the SMS Spam Collection dataset
- Provides an interactive Streamlit UI with custom CSS styling for real-time message classification

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Data handling | pandas |
| ML / Modeling | scikit-learn (`MultinomialNB`, `CountVectorizer`, `train_test_split`) |
| Web app / UI | Streamlit |
| Text processing | `re` (regex) |
| Deployment | Streamlit Community Cloud |

---

## 📊 Model Performance

Evaluated on an 80/20 train-test split of the deduplicated dataset (5,158 messages: 4,516 ham / 642 spam).

| Metric | Score |
|---|---|
| **Accuracy** | 97.9% |
| **Precision (Spam)** | 97.5% |
| **Recall (Spam)** | 86.2% |
| **F1-score (Spam)** | 91.5% |

**Confusion Matrix**

| | Predicted: Not Spam | Predicted: Spam |
|---|---|---|
| **Actual: Not Spam** | 891 | 3 |
| **Actual: Spam** | 19 | 119 |

> Note: The dataset is imbalanced (~87% ham / ~13% spam), so accuracy alone is optimistic. Precision is very high (few false alarms), while recall is the main area for improvement — about 14% of spam messages are missed. Potential next steps: class weighting, TF-IDF vectorization, or `ComplementNB` for imbalanced text data.

---

## 📂 Project Structure

```
Spam_Detection/
├── spam.py              # Main app: preprocessing, model training, Streamlit UI
├── mail_dataSet.csv     # SMS Spam Collection dataset (Category, Message)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/anubhavm5/Spam_Detection.git
cd Spam_Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run spam.py
```

---

## 📦 Requirements

```
pandas~=2.2.3
streamlit~=1.47.1
scikit-learn~=1.7.1
```

---

## 📈 Future Improvements

- Persist the trained model (e.g., `pickle`/`joblib`) instead of retraining on every app run
- Add TF-IDF vectorization as an alternative to raw counts
- Address class imbalance (class weights, oversampling, or `ComplementNB`)
- Add automated evaluation output (accuracy/precision/recall) directly in the app or a notebook
