🦠 Disease Outbreak Prediction Using Machine Learning
This project predicts the likelihood of a disease outbreak based on various factors using supervised machine learning algorithms. It also includes a Flask web application where users can input data and get real-time predictions.

📁 Project Structure
cpp
Copy code
Project/
├── App/
│   ├── app.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── fine_tune.pkl
│   │   └── scaler.pkl
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── main_css.css
│   │   └── main.js
├── Data/
│   ├── disease_outbreak_balanced_2k.csv
│   └── preprocessed_data.csv
├── Training/
│   ├── preprocess_data.ipynb
│   └── training_notebook.ipynb
├── Evaluation/
│   ├── evaluation_and_tuning.ipynb
│   └── best_model_saving.ipynb
├── README.md
├── requirements.txt
└── python_version.txt
🚀 Project Overview
The goal of this project is to build a machine learning system that predicts whether a disease outbreak will occur based on input features such as:

Week

State Code (Encoded)

State Name (Encoded)

Disease Code (Encoded)

Incidence per Capita

The system uses Logistic Regression as the best-performing algorithm, with feature scaling to improve accuracy.

🎯 Features
✅ Data Preprocessing with Label Encoding
✅ Training with 5 Supervised Learning Algorithms
✅ Best Model Selection and Saving (model.pkl)
✅ Hyperparameter Tuning & Fine-tuning (fine_tune.pkl)
✅ Flask Web App with User-friendly Frontend
✅ Real-time Disease Outbreak Prediction

⚙️ Machine Learning Models Used
Logistic Regression (Best Model Selected) ✅

Support Vector Machine (SVM)

Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

🛠 Technology Stack
Python (scikit-learn, pandas, numpy, Flask)

HTML5, CSS3, JavaScript (Frontend)

Joblib (Model & Scaler Serialization)

📝 How to Run the Project
1. Install Dependencies
bash
Copy code
pip install -r requirements.txt
2. Run the Flask App
bash
Copy code
cd App
python app.py
The app will be available at:
➡️ http://127.0.0.1:5000/

🖥 Sample Inputs (For Testing)
Feature	Example Value
Week	15
State Code (Encoded)	27
State Name (Encoded)	5
Disease Code (Encoded)	2
Incidence per Capita	0.85

✅ Low values → No Outbreak
⚠️ High values → Disease Outbreak Detected

🎯 Purpose of the Project
To build a predictive model that can help authorities and health organizations identify potential outbreak risks early.

To demonstrate the use of machine learning algorithms for classification tasks in the healthcare domain.

To integrate the ML model into a web-based application for easy access and usability.

📌 Future Improvements
Use of time-series models for better temporal predictions.

Adding geospatial visualizations.

Deploying the app online using platforms like Heroku or AWS.

📄 Requirements
Python 3.x

Flask

scikit-learn

pandas

numpy

joblib

See full dependencies in requirements.txt.