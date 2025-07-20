import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os
st.write("File exists:", os.path.exists("RNN.keras"))
st.write("File size:", os.path.getsize("RNN.keras"))

word_index = imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}
from keras.models import load_model
model = load_model("RNN.keras", compile=False)


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review =[word_index.get(word,2) for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500, padding='post', truncating='post')
    return padded_review

st.title("Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative.')

user_input=st.text_area("Moive Review")

if st.button("Classify"):
    preprocessed_input=preprocess_text(user_input)

    prediction =model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence Score: {prediction[0][0]}")

