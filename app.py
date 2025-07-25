from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model once at startup
model = load_model("kidney_disease_model.h5")

# Features in the order your model expects
feature_order = [
    "Age", "Creatinine_Level", "BUN", "Diabetes",
    "Hypertension", "GFR", "Urine_Output", "Dialysis_Needed"
]

# Initialize scaler with training data stats (mean/std)
# You should replace these with the actual scaler parameters from training
# For demo, I'll just create a dummy scaler with mean=0 std=1 (no scaling)
scaler = StandardScaler()
scaler.mean_ = np.array([50, 1.0, 30, 0, 0, 60, 1200, 0])
scaler.scale_ = np.array([15, 0.5, 10, 1, 1, 20, 500, 1])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features in right order
    try:
        features = [float(data[feat]) for feat in feature_order]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    # Scale features
    scaled = (np.array(features) - scaler.mean_) / scaler.scale_
    input_array = np.array([scaled])

    # Predict
    pred = model.predict(input_array)
    pred_class = int(np.argmax(pred))

    # Return result
    return jsonify({
        "prediction": pred_class,
        "probability": float(np.max(pred))
    })

if __name__ == '__main__':
    app.run(debug=True)
