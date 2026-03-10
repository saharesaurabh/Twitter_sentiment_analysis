import os
import sys

# make the project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
from utils.utils import cleaning_data_, predict_sentiment


# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a tweet to analyze its sentiment.")
    
    user_input = st.text_area("Tweet:")
    
    if st.button("Predict Sentiment"):
        if user_input:
            cleaned_input = cleaning_data_(user_input)
            sentiment = predict_sentiment(cleaned_input)
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Please enter a tweet to analyze.")

if __name__ == "__main__":
    main()