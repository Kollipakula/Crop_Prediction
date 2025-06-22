from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
model = joblib.load('crop_model1.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Configure the Google Gemini API
API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    pH = data['pH']
    rainfall = data['rainfall']

    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    predicted_label = model.predict(input_data)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({'predicted_crop': predicted_crop})

@app.route('/get_crop_info', methods=['POST'])
def get_crop_info():
    crop_name = request.json.get('crop_name')
    customer_prompt = (
        f"Provide detailed information about {crop_name} in the following format:"
        f"\n\n- Uses: (List the uses in bullet points)"
        f"\n- Health Benefits: (List the health benefits in bullet points)"
        f"\n- Trends in Selling: (List any trends in selling this crop in bullet points)"
    )
    try:
        response = gemini_model.generate_content(customer_prompt)
        gemini_output = response.text
        return jsonify({'crop_info': gemini_output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
