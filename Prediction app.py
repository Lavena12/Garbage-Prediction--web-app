import numpy as np
import joblib
import streamlit as st

# Load the model using joblib
loaded_model = joblib.load("train_model.joblib")  # Relative path

# Function for prediction
def garbage_prediction(input_data):
    weight_input, material_code_input = input_data
    predicted = loaded_model.predict([[weight_input, material_code_input]])
    return predicted[0]
    
# Main Streamlit app
def main():
    st.title('Garbage Prediction Web App')
    st.write("Welcome to the waste category predictor!")

    Weight_grams = st.number_input('Enter weight in grams', min_value=0)
    Material_code = st.number_input('Enter material code', min_value=0)

    Prediction = ''

    if st.button("Predict Category"):
        Prediction = garbage_prediction([Weight_grams, Material_code])
        st.success(f"Predicted Category: {Prediction}")
        
if __name__ == '__main__':
    main()

