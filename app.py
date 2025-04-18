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
CORS(app)  # Enable CORS

# Verify model directory exists
MODEL_DIR = "./sentiment_model"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found!")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

# 🔹 Predict a single string
def single_prediction(text_input):
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
    return "Positive" if predicted_class_id == 1 else "Negative"

# 🔹 Bulk prediction from DataFrame
def bulk_prediction(data, column_name):
    corpus = []
    for sentence in data[column_name]:
        cleaned = re.sub("[^a-zA-Z]", " ", str(sentence)).lower()
        corpus.append(cleaned)

    predictions = [single_prediction(text) for text in corpus]
    data["Predicted sentiment"] = predictions
    return data

# 🔹 Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")

# 🔹 Test route
@app.route("/test", methods=["GET"])
def test():
    return "✅ Sentiment Analysis service is up."

# 🔹 Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # File upload path
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]

            if not file.filename.endswith(".csv"):
                return jsonify({"error": "Please upload a valid CSV file."})

            data = pd.read_csv(file)

            sentence_column = "Sentence" if "Sentence" in data.columns else "text" if "text" in data.columns else None
            if not sentence_column:
                return jsonify({"error": "CSV must contain a 'Sentence' or 'text' column."})

            data = bulk_prediction(data, sentence_column)

            output = BytesIO()
            data.to_csv(output, index=False)
            output.seek(0)

            return send_file(output, mimetype="text/csv", as_attachment=True, download_name="Predictions.csv")

        # Text input path
        elif "text" in request.form and request.form["text"].strip():
            text_input = request.form["text"].strip()
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "No valid input. Please enter text or upload a CSV."})

    except Exception as e:
        return jsonify({"error": f"Exception occurred: {str(e)}"})

# 🔹 Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)









