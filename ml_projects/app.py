import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, "vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)
with open(os.path.join(base_path, "model.pkl"), "rb") as f:
    model = pickle.load(f)

st.title("Email/SMS Spam Predictor")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    
    transformed_sms = transform_text(input_sms)
   
    vector_input = tfidf.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
