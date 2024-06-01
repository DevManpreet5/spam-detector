import pickle
import numpy as np
import pandas as pd
import streamlit as st
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define text processing functions
def lowercase(data):
    return data.lower()

def remove_url(data):
    data = re.sub(r"https?://\S+|www\.\S+", '', data)
    data = re.sub(r'<.*?>', '', data)
    return data

def remove_pun(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_sym(text):
    bad_chars = [';', ':', '!', "*", "^", "&", "(", ")", "$", "[", "]"]
    for char in bad_chars:
        text = text.replace(char, '')
    return text

def replace_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word not in stop_words]

ps = PorterStemmer()

def stem_words(text):
    return [ps.stem(word) for word in text]

def join_back(list_input):
    return " ".join(list_input)

def process_text(text):
    text = lowercase(text)
    text = remove_url(text)
    text = remove_pun(text)
    text = remove_sym(text)
    text = replace_multiple_spaces(text)
    text = tokenize_text(text)
    text = remove_stopwords(text)
    text = stem_words(text)
    text = join_back(text)
    return text

# Load the models
try:
    loaded_model1 = pickle.load(open("cv.pkl", 'rb'))
    loaded_model2 = pickle.load(open("model.pkl", 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Streamlit UI
st.title("Spam Detector")
text1 = st.text_area("Message", placeholder="Enter the message")
btn = st.button("Evaluate")

if btn:
        processed_text = process_text(text1)
        vector = loaded_model1.transform([processed_text])
        result = loaded_model2.predict(vector)[0]
        if result == "spam":
            st.error("Spam detected")
        else:
            st.success("No spam detected")

