from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load the trained model with error handling"""
    model_path = 'insurance_cost_model.pkl'
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully!")
            return model
        else:
            logger.error(f"Model file {model_path} not found")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Try to load the model at startup
model = load_model()

@app.route('/')
def home():
    """Render the main form page"""
    if model is None:
        return render_template('error.html', 
                             error="Model not available. Please contact administrator.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data with validation
        try:
            age = int(request.form['age'])
            sex = request.form['sex']
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']
        except (ValueError, KeyError) as e:
            return jsonify({'error': f'Invalid input: {str(e)}'}), 400
        
        # Validate inputs
        if not (18 <= age <= 100):
            return jsonify({'error': 'Age must be between 18 and 100'}), 400
        
        if not (10 <= bmi <= 50):
            return jsonify({'error': 'BMI must be between 10 and 50'}), 400
        
        if not (0 <= children <= 10):
            return jsonify({'error': 'Number of children must be between 0 and 10'}), 400
        
        if sex not in ['male', 'female']:
            return jsonify({'error': 'Sex must be male or female'}), 400
            
        if smoker not in ['yes', 'no']:
            return jsonify({'error': 'Smoker must be yes or no'}), 400
            
        if region not in ['southwest', 'southeast', 'northwest', 'northeast']:
            return jsonify({'error': 'Invalid region'}), 400
        
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
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
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
            'input_data': data,
            'success': True
        })
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else 'None'
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use Railway's PORT environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting app on port {port}")
    logger.info(f"Model loaded: {model is not None}")
    app.run(debug=False, host='0.0.0.0', port=port)