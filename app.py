from flask import Flask, request, jsonify, render_template, send_from_directory
from textblob import TextBlob
import pandas as pd
import os

app = Flask(__name__)

# Set up the folder to store uploaded CSV files if needed
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("landing.html")  # Renders the landing.html page

@app.route("/predict", methods=["POST"])
def predict():
    # Handle file upload and text prediction
    if 'file' in request.files:
        file = request.files["file"]
        
        # Save the file to a directory if necessary
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Read the CSV file and predict sentiment for each row
        try:
            df = pd.read_csv(file_path)
            # Ensure 'text' column exists in the CSV for sentiment prediction
            if 'text' not in df.columns:
                return jsonify({"error": "CSV must contain a 'text' column"}), 400
            
            # Predict sentiment (positive, negative, or neutral)
            df["prediction"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df["prediction"] = df["prediction"].apply(
                lambda x: "positive" if x > 0 else "negative" if x < 0 else "neutral"
            )
            
            results = df[["text", "prediction"]].to_dict(orient="records")
            
            # Provide a downloadable file with predictions
            output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "Predictions.csv")
            df.to_csv(output_file_path, index=False)

            return jsonify({"results": results, "download_url": f"/download/{os.path.basename(output_file_path)}"})

        except Exception as e:
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 500

    elif request.is_json:
        # Handle text input for sentiment prediction
        data = request.get_json()
        text = data.get("text", "")
        
        if text.strip() == "":
            return jsonify({"error": "No text provided"}), 400
        
        sentiment = TextBlob(text).sentiment.polarity
        label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        return jsonify({"prediction": label})

    return jsonify({"error": "Invalid request"}), 400

# Route to download the predictions CSV file
@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)





