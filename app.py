import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Define a function for prediction
def predict_class(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = classifier.predict(input_df)
    return prediction[0]

# Streamlit App
st.title('Titanic Passenger Class Prediction')

# Input fields for the features
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sib_sp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=30.0)
sex_male = st.selectbox('Sex', ['male', 'female'], key='sex') == 'male'
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'], key='embarked')
embarked_Q = embarked == 'Q'
embarked_S = embarked == 'S'

# Convert categorical inputs to numeric
sex_male = 1 if sex_male else 0
embarked_Q = 1 if embarked_Q else 0
embarked_S = 1 if embarked_S else 0

# Create a dictionary for the input data
input_data = {
    'Age': age,
    'SibSp': sib_sp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}

# Prediction button
if st.button('Predict Class'):
    prediction = predict_class(input_data)
    st.write(f'The predicted class is: {prediction}')
