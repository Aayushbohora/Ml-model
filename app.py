from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("kidney_disease_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([list(data.values())])
    prediction = model.predict(features)[0][0]
    return jsonify({'result': float(prediction)})

if __name__ == '__main__':
    app.run(debug=False)
