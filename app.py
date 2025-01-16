from flask import Flask, render_template, request
import pandas as pd
import joblib
import os  

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('pcos_rf.pkl')

# Define the expected columns for input
expected_columns = [
    'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)', 
    'Fast food (Y/N)', 'Pimples(Y/N)', 'Hair loss(Y/N)', 'Weight (Kg)', 
    'Waist(inch)', 'Hip(inch)', 'Reg.Exercise(Y/N)', 'Pregnant(Y/N)', 
    'Cycle length(days)', ' Age (yrs)', 'Marriage Status (Yrs)'
]

def map_yes_no(value):
    """Maps yes/no inputs to 1/0."""
    if value.lower() in ['oui', 'y']:
        return 1
    elif value.lower() in ['non', 'n']:
        return 0
    return 0

@app.route('/', methods=['GET'])
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request and renders the result."""
   
    # Collect form data and map it
    form_data = {
        'Skin darkening (Y/N)': map_yes_no(request.form['skin_darkening']),
        'hair growth(Y/N)': map_yes_no(request.form['hair_growth']),
        'Weight gain(Y/N)': map_yes_no(request.form['weight_gain']),
        'Cycle(R/I)': 2 if request.form['cycle'].lower() == 'régulier' else 4,
        'Fast food (Y/N)': map_yes_no(request.form['fast_food']),
        'Pimples(Y/N)': map_yes_no(request.form['pimples']),
        'Hair loss(Y/N)': map_yes_no(request.form['hair_loss']),
        'Weight (Kg)': float(request.form['weight']),
        'Waist(inch)': float(request.form['waist']),
        'Hip(inch)': float(request.form['hips']),
        'Reg.Exercise(Y/N)': map_yes_no(request.form['exercise']),
        'Pregnant(Y/N)': map_yes_no(request.form['pregnant']),
        'Cycle length(days)': float(request.form['cycle_length']),
        ' Age (yrs)': float(request.form['age']),
        'Marriage Status (Yrs)': float(request.form['marriage']),
    }

    # Convert form data to a DataFrame
    input_data = pd.DataFrame([form_data])
    input_data = input_data[expected_columns]

    # Make prediction using the trained model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)   
    certainty = prediction_proba[0][1] * 100  

    result = 'Oui' if prediction[0] == 1 else 'Non'

    # Return the result page with prediction and certainty
    return render_template('result.html', result=result, certainty=certainty)

if __name__ == '__main__':
    # Get the PORT environment variable (default to 5000 if not set)
    port = int(os.environ.get("PORT", 5000))
    
    # Run the app on all interfaces (0.0.0.0) and the dynamic port
    app.run(host='0.0.0.0', port=port, debug=False)
