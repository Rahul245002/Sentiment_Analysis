import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Define Flask API endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

# Title of the app
st.title("Text Sentiment Predictor")

# Upload CSV file for bulk prediction
uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction (must contain a 'Sentence' column)",
    type="csv",
)

# Text input for individual sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Button for triggering predictions
if st.button("Predict"):
    if uploaded_file is not None:
        # Handling bulk prediction via file
        file = {"file": uploaded_file}
        try:
            response = requests.post(prediction_endpoint, files=file)
            if response.status_code == 200:
                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)

                st.write("Predictions for your file:")

                st.dataframe(response_df)  # Display the predictions

                # Button to download the predictions
                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    key="result_download_button",
                )
            else:
                st.error("Error in predicting sentiments. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    elif user_input:
        # Handling individual text prediction
        response = requests.post(prediction_endpoint, json={"text": user_input})
        if response.status_code == 200:
            response_json = response.json()
            st.write(f"Predicted sentiment: {response_json['prediction']}")
        else:
            st.error("Error in predicting sentiment. Please try again.")
    else:
        st.warning("Please provide text or upload a CSV file for prediction.")

