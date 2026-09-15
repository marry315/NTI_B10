import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model pipeline
# We use caching so the model doesn't reload every time the user interacts with the app
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please make sure it is in the same folder.")
        return None

pipeline = load_model()

# 2. Set up the Streamlit App UI
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺")
st.title("🩺 Health Risk Prediction App")
st.write("Please enter the patient's details below to predict their risk.")

# 3. Create input fields for the user
# We split them into two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100, step=1)

with col2:
    st.subheader("Medical History & Demographics")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    smoking = st.selectbox(
        "Smoking History", 
        ["No Info", "current", "ever", "former", "never", "not current"]
    )
    
    # Using checkboxes for binary features (True = 1, False = 0)
    hypertension = st.checkbox("Has Hypertension?")
    heart_disease = st.checkbox("Has Heart Disease?")

# 4. Make prediction when the button is clicked
if st.button("Predict Risk", type="primary"):
    if pipeline is not None:
        # Create a dictionary with the user's input
        input_data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [int(hypertension)], # Convert boolean to 0 or 1
            'heart_disease': [int(heart_disease)],
            'smoking_history': [smoking],
            'bmi': [bmi],
            'HbA1c_level': [hba1c],
            'blood_glucose_level': [glucose]
        }

        # Convert the dictionary to a Pandas DataFrame
        input_df = pd.DataFrame(input_data)

        try:
            # Make the prediction using the pipeline
            prediction = pipeline.predict(input_df)[0]
            probabilities = pipeline.predict_proba(input_df)[0]

            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error("⚠️ **High Risk Detected**")
                st.write(f"The model predicts a high risk with a probability of **{probabilities[1]*100:.2f}%**.")
            else:
                st.success("✅ **Low Risk Detected**")
                st.write(f"The model predicts a low risk with a probability of **{probabilities[0]*100:.2f}%**.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
