# app.py - Flask Web Application for Insurance Cost Predictor

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('insurance_cost_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None

@app.route('/')
def home():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Validate inputs
        if age < 18 or age > 100:
            return jsonify({'error': 'Age must be between 18 and 100'}), 400
        
        if bmi < 10 or bmi > 50:
            return jsonify({'error': 'BMI must be between 10 and 50'}), 400
        
        if children < 0 or children > 10:
            return jsonify({'error': 'Number of children must be between 0 and 10'}), 400
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_cost = round(prediction, 2)
        
        return render_template('result.html', 
                             prediction=predicted_cost,
                             age=age,
                             sex=sex,
                             bmi=bmi,
                             children=children,
                             smoker=smoker,
                             region=region)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [data['age']],
            'sex': [data['sex']],
            'bmi': [data['bmi']],
            'children': [data['children']],
            'smoker': [data['smoker']],
            'region': [data['region']]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_cost = round(prediction, 2)
        
        return jsonify({
            'predicted_cost': predicted_cost,
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)