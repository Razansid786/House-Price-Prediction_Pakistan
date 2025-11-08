from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import yaml

app = Flask(__name__)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model_artifacts():
    """Load trained model and preprocessing objects"""
    try:
        params = load_params()
        model_path = params['training']['save_model_path']
        
        model = joblib.load(model_path)
        scaler = joblib.load('models/preprocessors/scaler.pkl')
        label_encoders = joblib.load('models/preprocessors/label_encoders.pkl')
        
        # Load feature columns to know the expected order
        features_df = pd.read_csv('data/processed/features.csv')
        feature_columns = features_df.columns.tolist()
        
        return model, scaler, label_encoders, feature_columns
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None, None

model, scaler, label_encoders, feature_columns = load_model_artifacts()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Create a full feature dictionary with defaults
        full_data = {
            'property_id': 0,  # Dummy value
            'location_id': 0,  # Dummy value
            'page_url': 0,  # Will be encoded
            'property_type': data.get('property_type', 'House'),
            'location': data.get('location', ''),
            'city': data.get('city', 'Lahore'),
            'province_name': data.get('province_name', 'Punjab'),
            'latitude': data.get('latitude', 31.5204),  # Default Lahore
            'longitude': data.get('longitude', 74.3587),  # Default Lahore
            'baths': int(data.get('baths', 2)),
            'area': f"{data.get('Area Size', 5)} {data.get('Area Type', 'Marla')}",
            'purpose': data.get('purpose', 'For Sale'),
            'bedrooms': int(data.get('bedrooms', 3)),
            'date_added': '01-01-2024',  # Dummy date
            'agency': 'Unknown',
            'agent': 'Unknown',
            'Area Type': data.get('Area Type', 'Marla'),
            'Area Size': float(data.get('Area Size', 5)),
            'Area Category': data.get('Area Category', '0-5 Marla')
        }
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([full_data])
        print(f"Full input DataFrame:\n{input_df}")
        
        # Encode categorical variables using the same encoders from training
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError as e:
                    print(f"Warning: {col} has unseen value, using default")
                    # Use the first class as default for unseen values
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure columns are in the same order as during training
        input_df = input_df[feature_columns]
        
        # Scale numerical features
        input_df_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_df_scaled)[0]
        print(f"Prediction: {prediction}")
        
        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        params = load_params()
        
        # Get unique values for categorical fields from label encoders
        categorical_options = {}
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in ['property_type', 'city', 'province_name', 'purpose', 'Area Type', 'Area Category']:
                    categorical_options[col] = encoder.classes_.tolist()[:20]  # Limit to 20 options
        
        return jsonify({
            'model_type': params['model']['type'],
            'model_loaded': model is not None,
            'preprocessors_loaded': scaler is not None and label_encoders is not None,
            'categorical_options': categorical_options
        })
    except Exception as e:
        print(f"Error in model_info: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not loaded. Please ensure model files exist.")
    else:
        print("Model loaded successfully!")
        print(f"Expected features: {feature_columns}")
    app.run(debug=True, host='0.0.0.0', port=5000)