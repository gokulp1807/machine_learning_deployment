from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    # Check whether the request contains JSON
    if not request.is_json:
        return jsonify({
            'error': 'Request must be JSON'
        }), 400

    # Get JSON data
    data = request.get_json()

    # Check whether 'minute' is present
    if 'minute' not in data:
        return jsonify({
            'error': "Missing 'minute' in input data"
        }), 400

    try:
        # Convert input data into DataFrame
        input_df = pd.DataFrame(
            [[data['minute']]],
            columns=['minute']
        )

        # Scale the input using the same scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Get prediction probabilities
        prediction_proba = model.predict_proba(scaled_input)

        # Return result as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# Run the Flask application
if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
