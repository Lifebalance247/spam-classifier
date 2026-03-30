import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train model
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("📧 Spam Message Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
