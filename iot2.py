import streamlit as st
import pandas as pd
import numpy as np
import librosa
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import streamlit.components.v1 as components
import io
import os
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🫁 Respiratory Sound Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}

# Header
st.markdown("""
<div class="main-header">
    <h1>🫁 Respiratory Sound Classification</h1>
    <p>AI-Powered Medical Diagnosis Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Model Controls")
    
    # Model loading section
    st.markdown("#### 🤖 Load Trained Model")
    
    # File uploader for model
    uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=['pkl'])
    
    if uploaded_model is not None:
        try:
            # Load the model from uploaded file
            st.session_state.model = joblib.load(uploaded_model)
            st.session_state.model_loaded = True
            
            # Extract model information
            if hasattr(st.session_state.model, 'classes_'):
                st.session_state.model_info['classes'] = st.session_state.model.classes_
                st.session_state.model_info['n_classes'] = len(st.session_state.model.classes_)
            
            if hasattr(st.session_state.model, 'n_features_in_'):
                st.session_state.model_info['n_features'] = st.session_state.model.n_features_in_
            else:
                st.session_state.model_info['n_features'] = 13  # Default MFCC features
            
            st.success("✅ Model loaded successfully!")
            
            # Display model info
            if st.session_state.model_info:
                st.markdown("#### 📊 Model Information")
                st.write(f"**Classes:** {st.session_state.model_info.get('n_classes', 'Unknown')}")
                st.write(f"**Features:** {st.session_state.model_info.get('n_features', 'Unknown')}")
                
                if 'classes' in st.session_state.model_info:
                    st.write("**Disease Classes:**")
                    for i, cls in enumerate(st.session_state.model_info['classes']):
                        st.write(f"  • {cls}")
                        
        except Exception as e:
            st.session_state.model_loaded = False
            st.error(f"❌ Failed to load model: {str(e)}")
    
    # Alternative: Manual path input
    st.markdown("#### 📁 Or Enter Model Path")
    model_path = st.text_input("Model file path:", value="")
    
    if st.button("🔄 Load Model from Path") and model_path:
        try:
            if os.path.exists(model_path):
                st.session_state.model = joblib.load(model_path)
                st.session_state.model_loaded = True
                
                # Extract model information
                if hasattr(st.session_state.model, 'classes_'):
                    st.session_state.model_info['classes'] = st.session_state.model.classes_
                    st.session_state.model_info['n_classes'] = len(st.session_state.model.classes_)
                
                if hasattr(st.session_state.model, 'n_features_in_'):
                    st.session_state.model_info['n_features'] = st.session_state.model.n_features_in_
                else:
                    st.session_state.model_info['n_features'] = 13
                
                st.success("✅ Model loaded from path!")
            else:
                st.error("❌ File path does not exist!")
        except Exception as e:
            st.session_state.model_loaded = False
            st.error(f"❌ Failed to load model: {str(e)}")
    
    # Audio upload
    st.markdown("#### 🎵 Audio Analysis")
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    
    # Info
    st.markdown("#### ℹ️ About")
    if st.session_state.model_loaded and 'classes' in st.session_state.model_info:
        st.info(f"""
        **Conditions Detected:**
        {chr(10).join([f"• {cls}" for cls in st.session_state.model_info['classes']])}
        """)
    else:
        st.info("""
        **Please load a trained model to see available conditions**
        """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Analysis", "🎵 Audio", "📡 Live Data"])

with tab1:
    if st.session_state.model_loaded:
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Classes</h3>
                <h2>{st.session_state.model_info.get('n_classes', 'N/A')}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔧 Features</h3>
                <h2>{st.session_state.model_info.get('n_features', 'N/A')}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            model_type = type(st.session_state.model).__name__
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 Model</h3>
                <h2>{model_type}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Status</h3>
                <h2>Ready</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Disease classes visualization
        if 'classes' in st.session_state.model_info:
            st.markdown("### 🏥 Disease Classifications")
            
            # Create a simple visualization of classes
            classes = st.session_state.model_info['classes']
            class_data = pd.DataFrame({
                'Disease': classes,
                'Index': range(len(classes))
            })
            
            fig_classes = px.bar(
                class_data,
                x='Disease',
                y='Index',
                color='Disease',
                title="Available Disease Classifications",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_classes.update_layout(showlegend=False)
            st.plotly_chart(fig_classes, use_container_width=True)
        
        # Model information
        st.markdown("### 🔍 Model Details")
        model_details = {
            'Model Type': type(st.session_state.model).__name__,
            'Features Expected': st.session_state.model_info.get('n_features', 'Unknown'),
            'Output Classes': st.session_state.model_info.get('n_classes', 'Unknown'),
        }
        
        if hasattr(st.session_state.model, 'n_estimators'):
            model_details['Estimators'] = st.session_state.model.n_estimators
        
        details_df = pd.DataFrame(list(model_details.items()), columns=['Property', 'Value'])
        st.dataframe(details_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("⚠️ Please load a trained model first using the sidebar!")
        st.markdown("""
        ### 📋 How to Use:
        1. **Upload Model**: Use the sidebar to upload your trained .pkl model file
        2. **Upload Audio**: Upload a respiratory sound file (.wav or .mp3)
        3. **Get Prediction**: The model will classify the respiratory condition
        4. **View Results**: Check the analysis and confidence scores
        """)

with tab2:
    if st.session_state.model_loaded:
        st.markdown("### 🔮 Manual Feature Prediction")
        st.markdown("Enter audio feature values for prediction:")
        
        n_features = st.session_state.model_info.get('n_features', 13)
        
        # Create dynamic sliders based on model's expected features
        col1, col2, col3 = st.columns(3)
        
        features = []
        for i in range(n_features):
            col_idx = i % 3
            if col_idx == 0:
                with col1:
                    val = st.slider(f"Feature {i+1}", -5.0, 5.0, 0.0, key=f"feature_{i}")
            elif col_idx == 1:
                with col2:
                    val = st.slider(f"Feature {i+1}", -5.0, 5.0, 0.0, key=f"feature_{i}")
            else:
                with col3:
                    val = st.slider(f"Feature {i+1}", -5.0, 5.0, 0.0, key=f"feature_{i}")
            features.append(val)
        
        if st.button("🎯 Predict Condition"):
            try:
                # Reshape features for prediction
                feature_array = np.array(features).reshape(1, -1)
                prediction = st.session_state.model.predict(feature_array)[0]
                
                # Get prediction probabilities if available
                if hasattr(st.session_state.model, 'predict_proba'):
                    probabilities = st.session_state.model.predict_proba(feature_array)[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"🎯 **Predicted Condition:** {prediction}")
                        confidence = max(probabilities)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col2:
                        # Probability chart
                        prob_df = pd.DataFrame({
                            'Condition': st.session_state.model.classes_,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        fig_prob = px.bar(
                            prob_df, 
                            x='Condition', 
                            y='Probability',
                            color='Probability',
                            color_continuous_scale='Plasma',
                            title="Prediction Probabilities"
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.success(f"🎯 **Predicted Condition:** {prediction}")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        
        # Feature importance (if available)
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            importance = st.session_state.model.feature_importances_
            feature_names = [f'Feature_{i+1}' for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Analysis",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    else:
        st.warning("⚠️ Please load a trained model first!")

with tab3:
    st.markdown("### 🎵 Audio File Analysis")
    
    if uploaded_file is not None:
        st.success("✅ Audio file uploaded!")
        
        # Play audio
        st.audio(uploaded_file)
        
        try:
            # Load and process audio
            audio_data, sr = librosa.load(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Audio Properties")
                st.metric("Sample Rate", "44100 Hz")
                st.metric("Duration", f"{len(audio_data)/sr:.2f} s")
                st.metric("Max Amplitude", f"{np.max(np.abs(audio_data)):.3f}")
            
            with col2:
                st.markdown("#### 🌊 Waveform")
                fig, ax = plt.subplots(figsize=(10, 4))
                time = np.linspace(0, len(audio_data)/sr, len(audio_data))
                ax.plot(time, audio_data, color='#667eea', alpha=0.8)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Audio Waveform")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Extract features based on model requirements
            st.markdown("#### 🔬 Feature Extraction")
            n_features = st.session_state.model_info.get('n_features', 13)
            
            if n_features == 13:
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
                audio_features = np.mean(mfccs.T, axis=0)
                feature_names = [f'MFCC_{i+1}' for i in range(13)]
            else:
                # Extract generic spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
                
                # Create feature vector
                audio_features = []
                audio_features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
                audio_features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
                audio_features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
                audio_features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
                
                # Add MFCC features if more are needed
                if n_features > 8:
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_features-8)
                    audio_features.extend(np.mean(mfccs.T, axis=0))
                
                # Ensure we have the right number of features
                audio_features = np.array(audio_features[:n_features])
                feature_names = [f'Feature_{i+1}' for i in range(len(audio_features))]
            
            # Display features
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': audio_features
            })
            
            fig_features = px.bar(
                feature_df, 
                x='Feature', 
                y='Value',
                color='Value',
                color_continuous_scale='Viridis',
                title="Extracted Audio Features"
            )
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Predict if model is loaded
            if st.session_state.model_loaded:
                st.markdown("#### 🎯 AI Prediction")
                
                try:
                    # Reshape for prediction
                    feature_array = audio_features.reshape(1, -1)
                    prediction = st.session_state.model.predict(feature_array)[0]
                    
                    if hasattr(st.session_state.model, 'predict_proba'):
                        probabilities = st.session_state.model.predict_proba(feature_array)[0]
                        confidence = max(probabilities)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if confidence > 0.8:
                                st.success(f"🎯 **Condition:** {prediction}")
                                st.success(f"**Confidence:** {confidence:.1%}")
                            elif confidence > 0.6:
                                st.warning(f"🔶 **Condition:** {prediction}")
                                st.warning(f"**Confidence:** {confidence:.1%}")
                            else:
                                st.error(f"🔴 **Condition:** {prediction}")
                                st.error(f"**Confidence:** {confidence:.1%}")
                        
                        with col2:
                            # Probability distribution
                            prob_df = pd.DataFrame({
                                'Condition': st.session_state.model.classes_,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            fig_prob = px.bar(
                                prob_df, 
                                x='Condition', 
                                y='Probability',
                                color='Probability',
                                color_continuous_scale='Plasma',
                                title="Prediction Confidence"
                            )
                            st.plotly_chart(fig_prob, use_container_width=True)
                    else:
                        st.success(f"🎯 **Predicted Condition:** {prediction}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
            else:
                st.warning("⚠️ Load a model to get predictions!")
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
    
    else:
        st.info("📁 Upload an audio file (.wav or .mp3) to begin analysis")
        
        # Demo audio generation
        if st.button("🎼 Generate Demo Audio"):
            # Create synthetic audio signal
            duration = 3  # seconds
            sr = 22050
            t = np.linspace(0, duration, duration * sr)
            
            # Create a simple respiratory-like signal
            freq = 2  # breathing frequency
            audio_signal = np.sin(2 * np.pi * freq * t) * np.exp(-t/2)
            audio_signal += 0.1 * np.random.normal(0, 1, len(t))  # Add noise
            
            # Save as temporary file for demo
            st.audio(audio_signal, sample_rate=sr)
            st.success("🎵 Demo respiratory audio generated!")

with tab4:
    st.markdown("### 📡 Live IoT Data Stream from ThingSpeak")

    iframe_code = """
    <iframe width="450" height="260" style="border: 1px solid #cccccc;" 
    src="https://thingspeak.com/channels/2989026/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15">
    </iframe>
    """

    components.html(iframe_code, height=280)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           border-radius: 10px; color: white; margin-top: 2rem;">
    <h4>🫁 Respiratory Sound Classification Dashboard</h4>
    <p>Built with Streamlit • Machine Learning • Audio Processing</p>
    <p><em>⚠️ For Educational Purposes Only</em></p>
</div>
""", unsafe_allow_html=True)