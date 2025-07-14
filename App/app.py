import webbrowser
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/fine_tune.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        week = int(request.form['week'])
        state_code = int(request.form['state_code'])
        state_name = int(request.form['state_name'])
        disease_code = int(request.form['disease_code'])
        incidence_per_capita = float(request.form['incidence_per_capita'])

        input_features = np.array([[week, state_code, state_name, disease_code, incidence_per_capita]])
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            output = "⚠️ Disease Outbreak Detected"
        else:
            output = "✅ No Disease Outbreak Detected"

        return render_template('index.html',
                               prediction_text=output,
                               week=week,
                               state_code=state_code,
                               state_name=state_name,
                               disease_code=disease_code,
                               incidence_per_capita=incidence_per_capita)

if __name__ == '__main__':
    webbrowser.open_new("http://127.0.0.1:5000/")
    app.run(debug=False)
