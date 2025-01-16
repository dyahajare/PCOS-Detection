from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('pcos_rf.pkl')


expected_columns = [
    'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)', 
    'Fast food (Y/N)', 'Pimples(Y/N)', 'Hair loss(Y/N)', 'Weight (Kg)', 
    'Waist(inch)', 'Hip(inch)', 'Reg.Exercise(Y/N)', 'Pregnant(Y/N)', 
    'Cycle length(days)', ' Age (yrs)', 'Marriage Status (Yrs)'
]


def map_yes_no(value):
    if value.lower() in ['oui', 'y']:
        return 1
    elif value.lower() in ['non', 'n']:
        return 0
    return 0

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
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


    input_data = pd.DataFrame([form_data])
    input_data = input_data[expected_columns]

 
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)   
    certainty = prediction_proba[0][1] * 100  

    result = 'Oui' if prediction[0] == 1 else 'Non'

    return render_template('result.html', result=result, certainty=certainty)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
