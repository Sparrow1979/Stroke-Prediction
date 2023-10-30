import streamlit as st
import pickle
import numpy as np
import pandas as pd
from stroke_functions import encode_categorical

with open('model_steps.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']

st.title("Stroke Prediction")
st.write("""### Please, input all the requered data sincerely for the prediction to be as accurate as possible""")


hypertension = ('No', 'Yes')
ever_married = ('Yes', 'No')
smoking_status = ('formerly smoked', 'never smoked', 'smokes', 'Unknown')
work_type = ('Private', 'Self-employed',
             'Govt_job', 'children', 'Never_worked')
heart_disease = ('Yes', 'No')

st.sidebar.header("User Inputs")

age = st.sidebar.number_input("Age:", min_value=10, max_value=110)
height = st.sidebar.number_input("Height (cm):", min_value=100, max_value=230)
weight = st.sidebar.number_input("Weight (kg):", min_value=20, max_value=300)
glucose = st.sidebar.number_input("Average Glucose Level (mg/dL):",
                                  min_value=50, max_value=250,
                                  value=140, help="Normal is considered 100 - 140, if you do not know, leave it to 140")
hyper = st.sidebar.selectbox("High Blood Pressure:", hypertension)
marital = st.sidebar.selectbox("Marital Status:", ever_married)
smoking = st.sidebar.selectbox("Smoking Status:", smoking_status)
work = st.sidebar.selectbox("Job Type:", work_type)
heart = st.sidebar.selectbox("Heart Disease:", heart_disease)

bmi = weight/((height/100)**2)

user_input_dict = {
    'age': age,
    'bmi': bmi,
    'avg_glucose_level': glucose,
    'hypertension': hyper,
    'ever_married': marital,
    'smoking_status': smoking,
    'work_type': work,
    'heart_disease': heart
}

default_dict = {'stroke': 'No'}
merged_dict = user_input_dict.copy()
merged_dict.update(default_dict)
cat_columns = ['hypertension', 'ever_married',
               'smoking_status', 'work_type', 'heart_disease', 'stroke']

ok = st.button('Predict')

if ok:
    user_input_df = pd.DataFrame([merged_dict])
    encoded_data = encode_categorical(user_input_df, cat_columns)
    encoded_data = encoded_data.drop(columns=['stroke'])
    y_pred = model.predict(encoded_data)
    if y_pred[0] == 1:
        answer = "You have a high stroke risk"
    else:
        answer = "You do not have a significant stroke risk"
    st.subheader(answer)
