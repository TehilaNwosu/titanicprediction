from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Configuration for model files
MODEL_FILES = {
    'model': 'model.pkl',
    'scaler': 'scaler.pkl',
    'features': 'features.pkl'
}

# Dictionary to hold loaded ML resources
ml_artifacts = {
    'model': None,
    'scaler': None,
    'features': []
}

def load_ml_resources():
    """Attempts to load the ML artifacts into memory on startup."""
    try:
        # Check if files exist before loading to avoid generic errors
        for key, filename in MODEL_FILES.items():
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")
            
            ml_artifacts[key] = joblib.load(filename)
        
        print("✓ Model, Scaler, and Features loaded successfully.")
        return True

    except Exception as e:
        print(f"⚠️ Warning: Failed to load ML artifacts. Error: {str(e)}")
        print("Please ensure 'train_model.py' has been run.")
        return False

# Initialize resources
resources_loaded = load_ml_resources()

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'prediction': None,
        'probability': None,
        'error': None,
        'features': ml_artifacts['features'] if ml_artifacts['features'] is not None else []
    }

    # Handle GET Request (just render the form)
    if request.method == 'GET':
        return render_template('index.html', **context)

    # Handle POST Request (Form Submission)
    if not resources_loaded or ml_artifacts['model'] is None:
        context['error'] = "Model not trained. Please run train_model.py first."
        return render_template('index.html', **context)

    try:
        # 1. Extract and Validate Input
        feature_values = []
        required_features = ml_artifacts['features']
        
        for feature_name in required_features:
            user_input = request.form.get(feature_name)
            
            if not user_input:
                context['error'] = f"Missing feature: {feature_name}"
                return render_template('index.html', **context)
            
            # Convert to float immediately to catch ValueErrors early
            feature_values.append(float(user_input))

        # 2. Preprocess Data
        # Reshape to (1, n_features) as required by Scikit-Learn
        input_array = np.array(feature_values).reshape(1, -1)
        scaled_input = ml_artifacts['scaler'].transform(input_array)

        # 3. Prediction
        model = ml_artifacts['model']
        prediction_class = model.predict(scaled_input)[0]
        prediction_probs = model.predict_proba(scaled_input)[0]

        # 4. Calculate Confidence/Probability
        # If prediction is 1, take index 1, otherwise index 0
        raw_probability = prediction_probs[1] if prediction_class == 1 else prediction_probs[0]
        
        context['prediction'] = prediction_class
        context['probability'] = raw_probability * 100

    except ValueError:
        context['error'] = "Please enter valid numerical values for all features."
    except Exception as e:
        context['error'] = f"An error occurred: {str(e)}"

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)