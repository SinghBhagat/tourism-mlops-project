import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from huggingface_hub import hf_hub_download
import os
import sys # Import sys for potential logging

# Add this line at the very top, after imports
print("--- app.py script started processing ---")
print(f"Python version: {sys.version}") # Log Python version
print(f"Streamlit version: {st.__version__}") # Log Streamlit version

# Project Configuration (Ensure these match your notebook configuration)
HF_USERNAME = "bhagat26singh" # 👈 REPLACE with your actual Hugging Face username
PROJECT_NAME = "tourism-mlops-project"
MODEL_REPO = f"{HF_USERNAME}/{PROJECT_NAME}-model" # Define MODEL_REPO here

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders"""
    st.write("Attempting to load model and encoders...") # Added log
    print("Attempting to load model and encoders (print)...") # Added print log

    try:
        # Load from Hugging Face Hub using unified repository structure
        st.write(f"Attempting hf_hub_download from repo: {MODEL_REPO}") # Added log
        print(f"Attempting hf_hub_download from repo (print): {MODEL_REPO}") # Added print log

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="models/best_model.pkl",
            repo_type="model"
        )
        st.write(f"Model file downloaded to: {model_path}") # Added log
        print(f"Model file downloaded to (print): {model_path}") # Added print log

        encoders_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="preprocessing/label_encoders.pkl",
            repo_type="model"
        )
        st.write(f"Encoders file downloaded to: {encoders_path}") # Added log
        print(f"Encoders file downloaded to (print): {encoders_path}") # Added print log


        model = joblib.load(model_path)
        st.write("Model loaded successfully.") # Added log
        print("Model loaded successfully (print).") # Added print log

        label_encoders = joblib.load(encoders_path)
        st.write("Label encoders loaded successfully.") # Added log
        print("Label encoders loaded successfully (print).") # Added print log


        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}") # Modified log
        print(f"Error loading model from Hugging Face Hub (print): {e}") # Added print log
        st.info("Fallback: Trying to load from local files...")
        print("Fallback: Trying to load from local files (print)...") # Added print log
        try:
            # Fallback to local files - these are copied by the Dockerfile
            local_model_path = "/app/models/best_model.pkl" # Adjusted path assumption
            local_encoders_path = "/app/preprocessing/label_encoders.pkl" # Adjusted path assumption

            st.write(f"Attempting to load from local path: {local_model_path}") # Added log
            print(f"Attempting to load from local path (print): {local_model_path}") # Added print log
            model = joblib.load(local_model_path)
            st.write("Model loaded successfully from local path.") # Added log
            print("Model loaded successfully from local path (print).") # Added print log


            st.write(f"Attempting to load encoders from local path: {local_encoders_path}") # Added log
            print(f"Attempting to load encoders from local path (print): {local_encoders_path}") # Added print log
            label_encoders = joblib.load(local_encoders_path)
            st.write("Label encoders loaded successfully from local path.") # Added log
            print("Label encoders loaded successfully from local path (print).") # Added print log


            return model, label_encoders
        except Exception as local_error:
            st.error(f"Error loading local model: {local_error}") # Modified log
            print(f"Error loading local model (print): {local_error}") # Added print log
            return None, None

def main():
    st.write("--- App Started ---") # Added log at the beginning of main
    print("--- App Started (print) ---") # Added print log at the beginning of main
    st.title("Tourism Package Prediction")
    st.markdown("Predict whether a customer will purchase the Wellness Tourism Package")

    # Load model and encoders
    model, label_encoders = load_model_and_encoders()

    if model is None:
        st.error("Model could not be loaded. Please check the logs for details.") # Modified message
        return

    st.success("Model and encoders loaded successfully! App is ready.") # Added success message
    st.write("--- Model and encoders loaded ---") # Added log after loading
    print("--- Model and encoders loaded (print) ---") # Added print log after loading


    # Create input form
    st.header("Enter Customer Information")
    st.write("--- Creating input form ---") # Added log before form
    print("--- Creating input form (print) ---") # Added print log before form


    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
        number_of_person_visiting = st.number_input("NumberOf Persons Visiting", min_value=1, max_value=10, value=2) # Corrected typo
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

    with col2:
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        number_of_trips = st.number_input("Number of Trips (annually)", min_value=0, max_value=20, value=2)

    with col3:
        passport = st.selectbox("Has Passport", ["Yes", "No"])
        pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        own_car = st.selectbox("Owns Car", ["Yes", "No"])
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=1000000, value=50000)

    if st.button("Predict Package Purchase", type="primary"):
        st.write("--- Predict button clicked ---") # Added log on button click
        print("--- Predict button clicked (print) ---") # Added print log on button click
        try:
            # Prepare input data
            input_data = {
                'Age': age,
                'TypeofContact': type_of_contact,
                'CityTier': city_tier,
                'DurationOfPitch': duration_of_pitch,
                'Occupation': occupation,
                'Gender': gender,
                'NumberOfPersonVisiting': number_of_person_visiting,
                'NumberOfFollowups': number_of_followups,
                'ProductPitched': product_pitched,
                'PreferredPropertyStar': preferred_property_star,
                'MaritalStatus': marital_status,
                'NumberOfTrips': number_of_trips,
                'Passport': 1 if passport == "Yes" else 0,
                'PitchSatisfactionScore': pitch_satisfaction_score,
                'OwnCar': 1 if own_car == "Yes" else 0,
                'NumberOfChildrenVisiting': number_of_children_visiting,
                'Designation': designation,
                'MonthlyIncome': monthly_income
            }

            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            st.write("--- Input DataFrame created ---") # Added log after df creation
            print("--- Input DataFrame created (print) ---") # Added print log after df creation


            # Apply label encoding
            categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
            for col in categorical_columns:
                if col in label_encoders:
                    try:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                        st.write(f"--- Encoded column: {col} ---") # Added log for each encoded column
                        print(f"--- Encoded column (print): {col} ---") # Added print log for each encoded column
                    except ValueError:
                        st.error(f"Unknown category in {col}: {input_data[col]}")
                        st.write(f"--- Error encoding column: {col} ---") # Added log for encoding error
                        print(f"--- Error encoding column (print): {col} ---") # Added print log for encoding error
                        return


            # Make prediction
            st.write("--- Making prediction ---") # Added log before prediction
            print("--- Making prediction (print) ---") # Added print log before prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            st.write("--- Prediction made ---") # Added log after prediction
            print("--- Prediction made (print) ---") # Added print log after prediction


            # Display results
            st.header("Prediction Results")
            st.write("--- Displaying results ---") # Added log before displaying results
            print("--- Displaying results (print) ---") # Added print log before displaying results


            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.success(" Customer is likely to purchase the package!")
                    st.metric("Prediction", "Will Purchase")
                else:
                    st.warning(" Customer is unlikely to purchase the package")
                    st.metric("Prediction", "Will Not Purchase")

            with col2:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")

                # Show probability breakdown
                st.write("Probability Breakdown:")
                st.write(f"- Will Not Purchase: {prediction_proba[0]:.3f}")
                st.write(f"- Will Purchase: {prediction_proba[1]:.3f}")

            # Save prediction data (optional)
            prediction_data = input_data.copy()
            prediction_data['prediction'] = int(prediction)
            prediction_data['confidence'] = float(confidence)

            # You can save this to a database or file for tracking

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write(f"--- Error during prediction process: {e} ---") # Added log for prediction error
            print(f"--- Error during prediction process (print): {e} ---") # Added print log for prediction error


if __name__ == "__main__":
    main()
