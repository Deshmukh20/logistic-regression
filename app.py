import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load the data
train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')

# Preprocess the data (shortened for simplicity)
train_data.drop(columns=['Survived', 'Cabin'], inplace=True)
test_data.drop(columns=['Cabin'], inplace=True)

imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

train_data.drop(columns=['Name', 'Ticket'], inplace=True)
test_data.drop(columns=['Name', 'Ticket'], inplace=True)

# Train a simple model
X_train = train_data.drop(columns=['Pclass'])
y_train = train_data['Pclass']

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

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
sex_male = st.selectbox('Sex', ['male', 'female']) == 'male'
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
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

