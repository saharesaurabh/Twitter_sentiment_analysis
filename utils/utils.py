import re
import contractions
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle


spell = SpellChecker()

# Load the trained model and tokenizer
model = load_model('C:\\Users\\Jerry\\Desktop\\Projects\\Twitter_sentiment_analysis\\models\\sentiment_lstm_model.keras')
with open('C:\\Users\\Jerry\\Desktop\\Projects\\Twitter_sentiment_analysis\\models\\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def cleaning_data_(data):
    # Remove URLs
    data = re.sub(r'http\S+|www\S+|https\S+', '', data, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from tweet
    data = re.sub(r'\@\w+|\#', '', data)

    data = contractions.fix(data)

    # Remove punctuations
    data = re.sub(r'[^\w\s]', '', data)

    words = word_tokenize(data)

    words = [spell.correction(word) for word in words if spell.correction(word)]

    data = ' '.join(words)

    # Convert to lowercase
    data = data.lower()
    
    return data


# Function to preprocess input text
def preprocess_text(text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences to the same length as the training data
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
    return padded_sequences

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    # Use direct call instead of model.predict() for faster single-sample inference
    prediction = model(preprocessed_text, training=False)
    predicted_class = np.argmax(prediction.numpy(), axis=1)[0]
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted_class]
