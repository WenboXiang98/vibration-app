import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Page config
st.set_page_config(page_title="Vibration Analysis", layout="wide")

# Title
st.title("Vibration Analysis Predictor")
st.write("Upload a CSV file with vibration data to get predictions.")

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

def extract_features(df):
    window_features = []
    
    for axis in ['X-axis vibration speed(mm/s)', 'Y-axis vibration speed(mm/s)', 'Z-axis vibration speed(mm/s)']:
        window_dict = {
            f'{axis}_mean': df[axis].mean(),
            f'{axis}_var': df[axis].var(),
            f'{axis}_max': df[axis].max(),
            f'{axis}_min': df[axis].min(),
            f'{axis}_rms': np.sqrt(np.mean(np.square(df[axis]))),
            f'{axis}_peak_to_peak': df[axis].max() - df[axis].min()
        }
        window_features.append(window_dict)
    
    features = {}
    for d in window_features:
        features.update(d)
    
    return pd.DataFrame([features])

try:
    model, scaler = load_model()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Show the raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # Make prediction
        features = extract_features(df)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {prediction[0]}")
        
        # Show probabilities
        st.subheader("Class Probabilities")
        probs_df = pd.DataFrame({
            'Class': model.classes_,
            'Probability': probabilities[0]
        })
        probs_df['Probability'] = probs_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(probs_df)
        
        # Visualize probabilities
        st.subheader("Probability Distribution")
        chart_data = pd.DataFrame({
            'Class': model.classes_,
            'Probability': probabilities[0]
        })
        st.bar_chart(chart_data.set_index('Class'))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please make sure the uploaded CSV file has the correct format with columns: 'X-axis vibration speed(mm/s)', 'Y-axis vibration speed(mm/s)', 'Z-axis vibration speed(mm/s)'")