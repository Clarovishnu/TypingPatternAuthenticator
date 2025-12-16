from flask import Flask, render_template, request, jsonify
import os
import json
import joblib
import numpy as np
import time
from scripts.extract_features import extract_features_from_events  # ✅ correct import

# --------------------------
# Flask App Setup
# --------------------------
app = Flask(__name__)

# --------------------------
# Load Trained Model + Scaler
# --------------------------
model_path = os.path.join('models', 'svm_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

# Load the trained model and scaler if they exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(" Model and Scaler loaded successfully!")
else:
    model = None
    scaler = None
    print(" Model or Scaler not found. Please train the model first!")

# --------------------------
# Home Page Route
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------
# Save Typing Log API
# --------------------------
@app.route('/api/save_log', methods=['POST'])
def save_log():
    """Save raw typing data to JSON file."""
    data = request.get_json()
    user_id = data.get('user_id', 'unknown')
    os.makedirs('data/raw_logs', exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"data/raw_logs/{user_id}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    return jsonify({"message": "Log saved successfully!", "file": filename})

# --------------------------
# Prediction API
# --------------------------
@app.route('/api/predict', methods=['POST'])
def predict_user():
    """Predict user based on typing pattern."""
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Train the model first!"}), 500

    data = request.get_json()
    try:
        # ✅ Correct function call
        features = extract_features_from_events(data['events'])

        # Arrange features in the correct order
        X = np.array([[
            features['dwell_mean'], features['dwell_std'], features['dwell_min'],
            features['dwell_max'], features['flight_mean'], features['flight_std'],
            features['flight_min'], features['flight_max'], features['n_keys']
        ]])

        # Scale + Predict
        X_scaled = scaler.transform(X)
        predicted_user = int(model.predict(X_scaled)[0])

        return jsonify({"predicted_user": predicted_user})

    except Exception as e:
        print(f" Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

# --------------------------
# Run the Flask App
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
