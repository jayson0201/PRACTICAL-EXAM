from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    age = float(request.form["age"])
    hypertension = float(request.form["hypertension"])
    heart_disease = float(request.form["heart_disease"])
    bmi = float(request.form["bmi"])
    HbA1c_level = float(request.form["HbA1c_level"])
    blood_glucose_level = float(request.form["blood_glucose_level"])

    # Make predictions using the loaded model
    prediction = model.predict([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
