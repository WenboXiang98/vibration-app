import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

# Page config
st.set_page_config(page_title="Vibration Analysis Classifier", layout="wide")

# Load the saved components
@st.cache_resource
def load_model_components():
    model = load_model('vibration_model.h5')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model, scaler, label_encoder

def extract_features(df):
    """Extract features from DataFrame"""
    # Convert time to datetime
    try:
        df['Time'] = pd.to_datetime(df['Time'], format=' %H:%M:%S.%f')
    except:
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')
        except:
            df['Time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='s')
    
    # Create 4-second windows
    window_features = []
    start_time = df['Time'].min()
    end_time = df['Time'].max()
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + pd.Timedelta(seconds=4)
        window_data = df[(df['Time'] >= current_time) & (df['Time'] < window_end)]
        
        if not window_data.empty:
            window_dict = {}
            for axis in ['X-axis vibration speed(mm/s)', 
                        'Y-axis vibration speed(mm/s)', 
                        'Z-axis vibration speed(mm/s)']:
                values = window_data[axis]
                window_dict.update({
                    f'{axis}_mean': values.mean(),
                    f'{axis}_var': values.var(),
                    f'{axis}_max': values.max(),
                    f'{axis}_min': values.min(),
                    f'{axis}_rms': np.sqrt(np.mean(np.square(values))),
                    f'{axis}_peak_to_peak': values.max() - values.min()
                })
            window_features.append(window_dict)
        current_time = window_end
    
    return pd.DataFrame(window_features)

# Main app
st.title("Vibration Analysis Classifier")
st.write("Upload a CSV file to classify the vibration pattern")

try:
    # Load model components
    model, scaler, label_encoder = load_model_components()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read and display the raw data
        df = pd.read_csv(uploaded_file)
        
        with st.expander("View Raw Data"):
            st.dataframe(df.head())
            st.line_chart(df[['X-axis vibration speed(mm/s)', 
                            'Y-axis vibration speed(mm/s)', 
                            'Z-axis vibration speed(mm/s)']])
        
        # Process the data
        features = extract_features(df)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
        
        # Calculate prediction statistics
        from collections import Counter
        prediction_counts = Counter(predicted_labels)
        total_windows = len(predicted_labels)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Distribution")
            # Create DataFrame for plotting
            dist_df = pd.DataFrame({
                'Condition': list(prediction_counts.keys()),
                'Count': list(prediction_counts.values()),
                'Percentage': [count/total_windows*100 for count in prediction_counts.values()]
            })
            
            fig = px.bar(dist_df, 
                        x='Condition', 
                        y='Percentage',
                        title='Distribution of Predictions')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Prediction Summary")
            st.write(f"Total windows analyzed: {total_windows}")
            
            # Show final prediction
            final_prediction = prediction_counts.most_common(1)[0][0]
            st.write(f"Final prediction: **{final_prediction}**")
            
            # Show detailed breakdown
            st.write("\nDetailed breakdown:")
            for condition, count in prediction_counts.items():
                percentage = (count / total_windows) * 100
                st.write(f"- {condition}: {count} windows ({percentage:.1f}%)")
            
            # Calculate and display average probabilities
            avg_probs = np.mean(predictions, axis=0)
            st.write("\nAverage probabilities:")
            for condition, prob in zip(label_encoder.classes_, avg_probs):
                st.write(f"- {condition}: {prob:.3f}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please ensure all model files (vibration_model.h5, scaler.joblib, and label_encoder.joblib) are present in the app directory.")
