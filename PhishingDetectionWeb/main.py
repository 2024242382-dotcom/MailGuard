from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import traceback
import os

app = Flask(__name__)
# Enable CORS so your local HTML file is allowed to send requests to this server
CORS(app) 

# --- SERVE HOME PAGE ---
@app.route('/')
def home():
    # Serves index.html located in the same folder as main.py
    return send_file('index.html')

# --- PATH & FILE CHECKING ---
MODEL_PATH = 'naive_bayes_model.pkl'
VEC_PATH = 'tfidf_vectorizer.pkl'

model = None
vectorizer = None

print("\n" + "="*50)
print("             SYSTEM STARTUP CHECK")
print("="*50)

# Check if model files exist before trying to load them
if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    try:
        print(f"Checking for ML assets in: {os.getcwd()}")
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(VEC_PATH, 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        print("✓ SUCCESS: Naive Bayes Model & TF-IDF Vectorizer loaded!")
    except Exception as e:
        print(f"✗ ERROR loading files: {e}")
        print("Falling back to simulated prediction mode for safety.")
else:
    print("⚠️ WARNING: Machine Learning model files (.pkl) not found in this folder!")
    print(f"Current Directory: {os.getcwd()}")
    print("Please make sure 'naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl'")
    print("are copied into this folder next to this Python file.")
    print("-> System will run in DEMO/BACKUP MODE until files are added.")

print("="*50 + "\n")

# --- BACKUP PREDICTION RULE ENGINE ---
def run_backup_prediction(text):
    """
    If the ML model isn't loaded yet, this function provides a deterministic
    rule-based response so the user can test the interface connection.
    """
    text_lower = text.lower()
    # High-signal phishing keywords
    phish_signals = ['urgent', 'suspend', 'verify', 'password', 'login', 'click here', 'bank', 'paypal', 'account']
    # Spam keywords
    spam_signals = ['buy now', 'off', 'discount', 'free', 'promo', 'subscribe', 'click below']
    
    # Simple counting rule
    phish_count = sum(1 for word in phish_signals if word in text_lower)
    spam_count = sum(1 for word in spam_signals if word in text_lower)
    
    if phish_count >= 2 or (phish_count >= 1 and ('http' in text_lower or 'www' in text_lower)):
        return 'Phishing', {"Safe": 10.0, "Spam": 15.0, "Phishing": 75.0}
    elif spam_count >= 1:
        return 'Spam', {"Safe": 15.0, "Spam": 70.0, "Phishing": 15.0}
    else:
        return 'Safe', {"Safe": 85.0, "Spam": 10.0, "Phishing": 5.0}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive data from the HTML form
        data = request.json
        sender = data.get('sender', '')
        subject = data.get('subject', '')
        body = data.get('body', '')

        print(f"\n[Incoming Request] analyzing email from: {sender}")
        combined_text = f"{subject} {body}"

        # 2. Check if we should use the Real Model or the Demo Model
        if model is not None and vectorizer is not None:
            # --- REAL ML PIPELINE ---
            # Transform text to TF-IDF features
            vectorized_text = vectorizer.transform([combined_text])
            
            # Predict Class
            prediction_raw = model.predict(vectorized_text)[0]
            
            # Predict Probabilities
            probabilities = model.predict_proba(vectorized_text)[0]
            classes = model.classes_ 
            
            # Map predictions to dictionary scores
            scores = {}
            for i, class_name in enumerate(classes):
                # Ensure the key is converted to string for JSON compatibility
                scores[str(class_name)] = round(probabilities[i] * 100, 2)
                
            prediction = str(prediction_raw)
            print(f"[ML Prediction Result] Class: {prediction} | Scores: {scores}")
        else:
            # --- DEMO RULE ENGINE ---
            prediction, scores = run_backup_prediction(combined_text)
            print(f"[Demo Prediction Result] Class: {prediction} | Scores: {scores}")

        # 3. Send response back to frontend
        return jsonify({
            "prediction": prediction,
            "scores": scores
        })

    except Exception as e:
        print("!!! SERVER ERROR !!!")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Starts the server on http://127.0.0.1:5000
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)