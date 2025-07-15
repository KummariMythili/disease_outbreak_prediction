from flask import Flask, render_template, request
import joblib
import numpy as np
import webbrowser

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model/fine_tune.pkl')  # or 'model/model.pkl'
scaler = joblib.load('model/scaler.pkl')    # required if scaling was used

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        week = int(request.form['week'])
        state_code = int(request.form['state_code'])
        state_name = int(request.form['state_name'])
        disease_code = int(request.form['disease_code'])
        incidence_per_capita = float(request.form['incidence_per_capita'])

        # Prepare feature array
        input_data = np.array([[week, state_code, state_name, disease_code, incidence_per_capita]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_text = "⚠️ Disease Outbreak Detected" if prediction == 1 else "✅ No Disease Outbreak Detected"

        # Render result and input data
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            week=week,
            state_code=state_code,
            state_name=state_name,
            disease_code=disease_code,
            incidence_per_capita=incidence_per_capita
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False)



