# backend/app.py
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os # Import os module for path manipulation

app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')

# --- Firebase Initialization ---
SERVICE_ACCOUNT_KEY_PATH = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
DATABASE_URL = 'https://health-3c965-default-rtdb.firebaseio.com' # Replace with YOUR actual Firebase URL

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase initialized successfully!")
except FileNotFoundError:
    print(f"Error: Service account key file not found at {SERVICE_ACCOUNT_KEY_PATH}. Please check the path.")
    # In a production environment, you might want to log this and potentially exit or disable Firebase features.
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    # In a production environment, you might want to log this and potentially exit or disable Firebase features.

ref = db.reference('structural_data')

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: model.pkl not found at {MODEL_PATH}. Make sure you've run your Jupyter notebook to save the model.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Serve React App's index.html for the root route
@app.route('/')
def serve_react_app():
    return render_template('index.html')

# Serve static files from the React build (CSS, JS, etc.)
# This is usually handled by the default static_folder configuration
# but explicitly defining it can be useful for debugging or specific setups.
@app.route('/static/<path:filename>')
def serve_static(filename):
    # This route is usually automatically handled by Flask's static_folder config.
    # However, if you explicitly need to serve static files from a specific path
    # you can define it like this.
    # The default 'static_folder' in Flask(__name__, static_folder='path')
    # already points to the 'build/static' folder in the React app.
    # So, this function might be redundant, but good to know for custom setups.
    return send_from_directory(app.static_folder, filename)


# Your existing prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # If model loading failed, return an internal server error
        return jsonify({'error': 'Model not loaded on the server.'}), 500

    data = request.json
    try:
        ax_g = float(data['ax_g'])
        ay_g = float(data['ay_g'])
        az_g = float(data['az_g'])
        vibration = float(data['vibration'])
        bending = float(data['bending'])
    except KeyError as e:
        return jsonify({'error': f'Missing data field: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data type for field: {e}'}), 400

    totalAccel = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)

    input_df = pd.DataFrame([{
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending
    }])

    pred = model.predict(input_df)[0]
    status = "DANGER" if pred == 1 else "SAFE"

    # Save data to Firebase Realtime Database
    timestamp = datetime.datetime.now().isoformat()
    prediction_data = {
        'timestamp': timestamp,
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending,
        'predicted_status': status
    }
    try:
        if 'firebase_admin' in globals() and firebase_admin._apps: # Check if Firebase was initialized
            ref.push(prediction_data)
            print(f"Data saved to Firebase: {prediction_data}")
        else:
            print("Firebase not initialized. Skipping data save.")
    except Exception as e:
        print(f"Error saving data to Firebase: {e}")

    return jsonify({'status': status})

if __name__ == '__main__':
    # When running locally for development:
    # 1. Run Flask app (backend/app.py)
    # 2. In a separate terminal, run React app (cd frontend && npm start)
    # The React dev server (usually on :3000) will proxy API calls to Flask (:5000)
    # For production, Flask serves the React build directly.
    app.run(debug=True, port=5000)