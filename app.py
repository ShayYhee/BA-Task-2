from flask import Flask, render_template, request
import pandas as pd
import pickle
import xgboost as xgb

app = Flask(__name__)

# Load the trained XGBoost model
xg_model = xgb.XGBClassifier()
xg_model.load_model('cust_book_pred.json')

# Load the saved one-hot encoder
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    route = request.form['route']
    bkOrg = request.form['bkOrg']
    flDur = float(request.form['flDur'])
    exBagg = int(request.form['exBagg'])  # Ensure this is an integer
    lenStay = float(request.form['lenStay'])

    # Prepare the input data
    input_data = pd.DataFrame({
        'route': [route],
        'booking_origin': [bkOrg],
        'flight_duration': [flDur],
        'wants_extra_baggage': [exBagg],
        'length_of_stay': [lenStay]
    })

    # Apply the same one-hot encoding
    input_data_encoded = encoder.transform(input_data)

    # Make a prediction
    prediction = xg_model.predict(input_data_encoded)[0]
    prediction_proba = xg_model.predict_proba(input_data_encoded)[0][1]  # Get probability of booking

    return render_template('index.html', prediction=f'The predicted probability of booking is {prediction_proba:.4f}')

if __name__ == '__main__':
    app.run(debug=True)
