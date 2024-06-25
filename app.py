from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the trained XGBoost model
xg_model = xgb.XGBClassifier(objective='reg:squarederror', random_state=42)
xg_model.load_model('cust_book_pred.json')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    route = request.form['route']
    bkOrg = request.form['bkOrg']
    flDur = float(request.form['flDur'])
    exBagg = request.form['exBagg']
    lenStay = float(request.form['lenStay'])

#     features = ['Route', 'Booking_Origin', 'Flight_Duration', 'Extra_Baggage', 'Length_of_Stay']

    features = ['route', 'booking_origin', 'flight_duration', 'wants_extra_baggage', 'length_of_stay']
    
    # Prepare the input data    
#     input_data = [[route, bkOrg, flDur, exBagg, lenStay]]
    input_data = pd.DataFrame({
        'route': [route],
        'booking_origin': [bkOrg],
        'flight_duration': [flDur],
        'wants_extra_baggage': [exBagg],
        'length_of_stay': [lenStay]
    })
    
    for col in input_data.select_dtypes("object"):
#     for col in input_data:
        if isinstance(col, str):
            input_data[0,col],_ = input_data[0,col].factorize()
    
    # Make sure that all values are valid
    if -1 in input_data.values:
        return render_template('index.html', prediction='Invalid input. Please check your categorical values.')
    
    # Prepare the input data
    data = [[route, bkOrg, flDur, exBagg, lenStay]]
    dmatrix = xgb.DMatrix(data)

    # Make predictions
#     prediction = bst.predict(dmatrix)

#     Make a prediction
    prediction = xg_model.predict(input_data)[0]

    return render_template('index.html', prediction=f'The predicted customer prediction is {prediction:.4f}')

if __name__ == '__main__':
    app.run(debug=True)