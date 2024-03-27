import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Caching the data loading using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

# Main function
def main():
    # App title
    st.title("Heart Disease Prediction")

    # Load and display dataset information
    heart_data = load_data()
    st.subheader("Explore Dataset")
    if st.checkbox("Show dataset"):
        st.write(heart_data)

    # Display dataset statistics
    if st.checkbox("Show statistics"):
        st.write(heart_data.describe())

    # Data preparation
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Model training
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Accuracy on Training Data
    st.subheader("Model Performance")
    training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
    st.write("Accuracy on Training Data: ", training_data_accuracy)

    # User Inputs for Prediction
    st.subheader("Make Prediction")
    # Collecting user inputs for prediction
    age = st.number_input("Age:", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex (0: female, 1: male):", options=[0, 1])
    cp = st.number_input("Chest Pain Type:", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure:", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol:", min_value=50, max_value=400, value=200)
    fbs = st.number_input("Fasting Blood Sugar (1: >120 mg/dl, 0: <=120 mg/dl):", min_value=0, max_value=1, value=0)
    restecg = st.number_input("Resting Electrocardiographic Results (0, 1, 2):", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved:", min_value=50, max_value=250, value=150)
    exang = st.number_input("Exercise Induced Angina (1: yes, 0: no):", min_value=0, max_value=1, value=0)
    oldpeak = st.number_input("Oldpeak (ST Depression induced by exercise relative to rest):", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.number_input("Slope of the peak exercise ST segment (0, 1, 2):", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3):", min_value=0, max_value=3, value=0)
    thal = st.number_input("Thalassemia (3: normal, 6: fixed defect, 7: reversible defect):", min_value=3, max_value=7, value=3)

    # Use input data for prediction
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    # Displaying prediction outcome
    if st.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success('The Person does not have a Heart Disease.')
        else:
            st.error('The person has Heart Disease.')

if __name__ == '__main__':
    main()
