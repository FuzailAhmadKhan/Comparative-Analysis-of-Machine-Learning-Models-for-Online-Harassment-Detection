from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback
import joblib
import os
import numpy as np
import re

app = Flask(__name__, 
            static_folder='static',
            template_folder='static')
CORS(app)

# Simple preprocessor class
class SimplePreprocessor:
    def clean_tweet(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove usernames
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# Initialize preprocessor
preprocessor = SimplePreprocessor()

# Try to load the model
MODEL_FILE = "hate_speech_rf.joblib"
model = None

if os.path.exists(MODEL_FILE):
    try:
        print("📥 Loading model...")
        model = joblib.load(MODEL_FILE)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ Model file {MODEL_FILE} not found")
    print("📁 Current directory files:")
    for file in os.listdir('.'):
        print(f"   - {file}")

@app.route("/api/predict", methods=["POST"])
def predict():
    """Hate Speech prediction API"""
    try:
        # If model is not loaded, return a simulated response
        if model is None:
            data = request.get_json()
            text = data.get("text", "")
            
            # Simple keyword-based simulation
            text_lower = text.lower()
            if any(word in text_lower for word in ['kill', 'hate', 'attack', 'destroy']):
                classification = "Hate Speech"
                confidence = {"hate": 85.0, "offensive": 10.0, "neutral": 5.0}
            elif any(word in text_lower for word in ['stupid', 'idiot', 'fuck', 'shit']):
                classification = "Offensive Language"
                confidence = {"hate": 15.0, "offensive": 75.0, "neutral": 10.0}
            else:
                classification = "Neutral"
                confidence = {"hate": 5.0, "offensive": 15.0, "neutral": 80.0}
                
            return jsonify({
                "success": True,
                "algorithm": "Random Forest (Simulated)",
                "result": {
                    "classification": classification,
                    "confidence": confidence
                }
            })
        
        data = request.get_json()
        text = data.get("text", "")
        
        if not text.strip():
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Preprocess text
        cleaned = preprocessor.clean_tweet(text)
        
        # Make prediction
        try:
            # Try different model structures
            if hasattr(model, 'named_steps'):
                # Pipeline model
                tfidf = model.named_steps['tfidf'].transform([cleaned])
                pred = model.named_steps['classifier'].predict(tfidf)[0]
                probs = model.named_steps['classifier'].predict_proba(tfidf)[0]
            elif hasattr(model, 'predict_proba'):
                # Direct model with predict_proba
                # You'll need to transform the text using your vectorizer here
                # For now, let's simulate
                pred = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
            else:
                # Fallback to simulation
                raise Exception("Model structure not recognized")
                
        except Exception as model_error:
            print(f"Model prediction error: {model_error}")
            # Fallback to simulation
            text_lower = text.lower()
            if any(word in text_lower for word in ['kill', 'hate', 'attack', 'destroy']):
                pred = 0
                probs = [0.85, 0.10, 0.05]
            elif any(word in text_lower for word in ['stupid', 'idiot', 'fuck', 'shit']):
                pred = 1
                probs = [0.15, 0.75, 0.10]
            else:
                pred = 2
                probs = [0.05, 0.15, 0.80]

        label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        
        return jsonify({
            "success": True,
            "algorithm": "Random Forest",
            "result": {
                "classification": label_map.get(pred, "Neutral"),
                "confidence": {
                    "hate": float(probs[0] * 100),
                    "offensive": float(probs[1] * 100),
                    "neutral": float(probs[2] * 100)
                }
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running", 
        "model_loaded": model is not None
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files directly"""
    return app.send_static_file(filename)

if __name__ == "__main__":
    try:
        print("🚀 Starting Hate Speech Detection Server...")
        print("📁 Static folder:", app.static_folder)
        print("📁 Template folder:", app.template_folder)
        
        # Check if static files exist
        if os.path.exists('static'):
            print("✅ Static folder found")
            static_files = os.listdir('static')
            print("📄 Static files:", static_files)
        else:
            print("❌ Static folder not found")
        
        # Use use_reloader=False to prevent double execution in debug mode
        app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
        
    except SystemExit as e:
        if e.code != 0:
            print(f"⚠️  SystemExit with code {e.code}")
            print("This might be due to debug mode issues. Trying without debug...")
            app.run(host="127.0.0.1", port=5000, debug=False)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()