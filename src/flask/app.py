from flask import Flask, request, jsonify
import pandas as pd
from src.utils.helpers import load_model, data_for_k_mean
from src.utils.constant import model_path

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the input data from the request in JSON format
        json_ = request.json
        query_df = pd.DataFrame(json_)
        # if query_df.columns =
        query_df = data_for_k_mean(query_df)
        model = load_model(model_path)
        prediction = model.predict(query_df)
        # Return the prediction as a JSON response
        return jsonify({'prediction': "customer in this  cluster number "+str(prediction[0])}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 400
