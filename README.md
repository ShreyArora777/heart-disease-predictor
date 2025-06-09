# heart-disease-predictor
# Heart Disease Predictor App

This project is a simple web app that predicts whether a person is at risk of heart disease. It uses a logistic regression model trained on real medical data and is built with Python and Streamlit.

## What this app does

- Takes basic health inputs from the user (like age, blood pressure, cholesterol, etc.)
- Uses a trained machine learning model to make a prediction
- Tells the user whether they might have heart disease or not
- Shows how confident the prediction is (in percentage)

## Files in this project

- `heart.csv` – the dataset used to train the model
- `main.py` – the Python script where the model is trained and saved
- `heart_model.pkl` – the trained machine learning model saved using pickle
- `heart_app.py` – the Streamlit app where users can enter their data and get predictions
