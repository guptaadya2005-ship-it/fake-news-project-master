from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# import joblib
# import re
# from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io

from services.pipeline_service import predict_with_evidence

app = Flask(__name__)
CORS(app)

# --- CRUCIAL WINDOWS SETUP FOR OCR ---
# If you are on Windows, you MUST install Tesseract-OCR software on your PC.
# Tell Python where it is installed (uncomment and fix the path below if needed):
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# print("Loading AI Model...")
# model = joblib.load('saved_models/model.pkl')
# vectorizer = joblib.load('saved_models/vectorizer.pkl')
# stop_words = set(stopwords.words('english'))
# print("AI Model Loaded Successfully!")

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     words = text.split()
#     cleaned_words = [word for word in words if word not in stop_words]
#     return ' '.join(cleaned_words)

# Helper function to scrape websites
def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.text for p in paragraphs])
    except:
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_to_analyze = ""

    # 1. Check if the user uploaded an image (Screenshot)
    if 'image' in request.files:
        image_file = request.files['image']
        img = Image.open(image_file.stream)
        text_to_analyze = pytesseract.image_to_string(img)
        
    # 2. Otherwise, check the text input box
    else:
        incoming_data = request.form.get('text', '')
        
        # 3. Check if the text is a link (URL)
        if incoming_data.startswith('http://') or incoming_data.startswith('https://'):
            text_to_analyze = scrape_url(incoming_data)
            if not text_to_analyze:
                return jsonify({
                    "prediction": "Error",
                    "confidence": 0,
                    "message": "No readable text found."
                })
        # 4. It's just normal text
        else:
            text_to_analyze = incoming_data

    # Safety check
    if not text_to_analyze.strip():
         return jsonify({'prediction': "Error: No readable text found."})

    # Run the complete pipeline: the style classifier is used as a fallback
    # when web evidence is missing or inconclusive.
    try:
        result = predict_with_evidence(text_to_analyze)
    except Exception:
        app.logger.exception("Full-pipeline prediction failed")
        return jsonify({
            "prediction": "Error",
            "confidence": 0,
            "message": "Analysis could not be completed. Please try again."
        }), 503

    result["confidence"] = round(result["confidence"], 2)
    result["style_analysis"]["confidence"] = round(
        result["style_analysis"]["confidence"], 2
    )
    result["evidence"]["confidence"] = round(result["evidence"]["confidence"], 2)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
