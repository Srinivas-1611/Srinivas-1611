from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from prediction import predict_new_data, retrain_model
from sqlalchemy import create_engine

app = Flask(__name__)

# Database configuration
DB_URI = "mysql+pymysql://root:srini123@localhost/fmdc_db"
engine = create_engine(DB_URI)

@app.route('/')
def index():
    # Load dropdown options from the database
    query = "SELECT DISTINCT classification, code, implanted, name_device, name_manufacturer FROM fmdc_data"
    df = pd.read_sql(query, engine)

    dropdown_options = {
        'classification': sorted(df['classification'].dropna().unique().tolist()),
        'code': sorted(df['code'].dropna().unique().tolist()),
        'implanted': sorted(df['implanted'].dropna().unique().tolist()),
        'name_device': sorted(df['name_device'].dropna().unique().tolist()),
        'name_manufacturer': sorted(df['name_manufacturer'].dropna().unique().tolist()),
    }
    return render_template('index.html', dropdown_options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = {
        'classification': request.form.get('classification'),
        'code': request.form.get('code'),
        'implanted': request.form.get('implanted'),
        'name_device': request.form.get('name_device'),
        'name_manufacturer': request.form.get('name_manufacturer'),
    }

    new_data = pd.DataFrame([form_data])

    # Make prediction using the encoded data
    predicted_class, description, suggestion = predict_new_data(new_data.copy())

    # Add the original new data (categorical format) with the predicted class to the database
    new_data['risk_class'] = predicted_class
    new_data.to_sql('fmdc_data', engine, if_exists='append', index=False)

    # Retrain the model with the updated data
    retrain_model()

    # Render the results page with the prediction details
    return render_template('result.html', predicted_class=predicted_class, description=description, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
