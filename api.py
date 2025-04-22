from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# Create the Flask app
app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON input
        data = request.get_json()
        temperature = data.get('temperature')
        vibration = data.get('vibration')
        humidity = data.get('humidity')

        # Check if any input is missing
        if None in (temperature, vibration, humidity):
            return jsonify({"error": "Missing one or more required fields: temperature, vibration, humidity"}), 400

        # Prepare the input for prediction
        features = np.array([[temperature, vibration, humidity]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (faulty)

        # Respond with result
        return jsonify({
            "prediction": int(prediction),
            "probability_faulty": round(float(proba), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
