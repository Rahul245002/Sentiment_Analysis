from io import BytesIO
from flask import Flask, request, jsonify, send_file, render_template
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Enable CORS after app initialization
CORS(app)

# Verify model directory exists
MODEL_DIR = "./sentiment_model"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found!")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

# 🔹 Function: Predict sentiment for a single input string
def single_prediction(text_input):
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
    return "Positive" if predicted_class_id == 1 else "Negative"

# 🔹 Test route for checking if service is running
@app.route("/test", methods=["GET"])
def test():
    return "✅ Test request received successfully. Sentiment Analysis service is running."

# 🔹 Home route for HTML form input
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

# 🔹 Predict route (supports text and CSV file)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            # CSV Bulk Prediction
            file = request.files["file"]
            
            # Check if file is a CSV
            if not file.filename.endswith('.csv'):
                return jsonify({"error": "Please upload a valid CSV file."})

            data = pd.read_csv(file)

            # Check if 'Sentence' column is present in the CSV
            if "Sentence" not in data.columns:
                return jsonify({"error": "CSV file must contain a 'Sentence' column."})

            predictions = bulk_prediction(data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            return response

        elif request.is_json and "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})
        
        else:
            return jsonify({"error": "No valid input provided. Expecting JSON with 'text' or CSV file upload."})

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔹 Bulk prediction for CSV input
def bulk_prediction(data):
    corpus = []
    for sentence in data["Sentence"]:
        cleaned = re.sub("[^a-zA-Z]", " ", str(sentence)).lower()
        corpus.append(cleaned)

    predictions = [single_prediction(text) for text in corpus]
    data["Predicted sentiment"] = predictions

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    return predictions_csv

# 🔹 Start the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Use debug=False in production





