import streamlit as st
import pandas as pd
import numpy as np
import librosa
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import streamlit.components.v1 as components
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
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

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
    
    # Data loading section
    # st.markdown("#### 📁 Data Loading")
    # if st.button("🔄 Load Demo Data"):
    #     # Create synthetic respiratory data
    #     np.random.seed(42)
    #     n_samples = 1000
        
    #     # Generate features for different conditions
    #     data = []
    #     conditions = ['Normal', 'Wheeze', 'Crackle', 'Both']
        
    #     for condition in conditions:
    #         for _ in range(250):
    #             if condition == 'Normal':
    #                 features = np.random.normal(0, 0.5, 13)  # Tighter cluster
    #             elif condition == 'Wheeze':
    #                 features = np.random.normal(3, 0.5, 13)  # Further from Normal
    #             elif condition == 'Crackle':
    #                 features = np.random.normal(-3, 0.5, 13)  # Clearly separated
    #             else:  # Both (mixed)
    #                 features = np.random.normal(1.5, 0.5, 13)
                
    #             row = list(features) + [condition]
    #             data.append(row)
        
    #     columns = [f'mfcc_{i+1}' for i in range(13)] + ['diagnosis']
    #     st.session_state.df = pd.DataFrame(data, columns=columns)
    #     st.session_state.data_loaded = True
    #     st.success("✅ Demo data loaded!")
    model_path = "C:/Users/poorv/Downloads/respiratory_disease_model.pkl"  # Path to your trained model

try:
    st.session_state.model = joblib.load(model_path)
    st.session_state.model_loaded = True
    st.success("✅ Trained model loaded successfully!")
except Exception as e:
    st.session_state.model_loaded = False
    st.error(f"❌ Failed to load model: {str(e)}")

    # Model training
    # if st.session_state.data_loaded:
    #     st.markdown("#### 🤖 Model Training")
    #     if st.button("🚀 Train Model"):
    #         df = st.session_state.df
            
    #         # Prepare data
    #         X = df.drop('diagnosis', axis=1)
    #         y = df['diagnosis']
            
    #         # Split data
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, random_state=42
    #         )
            
    #         # Train model
    #         model = RandomForestClassifier(n_estimators=100, random_state=42)
    #         model.fit(X_train, y_train)
            
    #         # Evaluate
    #         y_pred = model.predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
            
    #         # Store in session
    #         st.session_state.model = model
    #         st.session_state.accuracy = accuracy
    #         st.session_state.X_test = X_test
    #         st.session_state.y_test = y_test
    #         st.session_state.y_pred = y_pred
            
    #         st.success(f"✅ Model trained! Accuracy: {accuracy:.3f}")
    
    # Audio upload
    st.markdown("#### 🎵 Audio Analysis")
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    
    # Info
    st.markdown("#### ℹ️ About")
    st.info("""
    **Conditions Detected:**
- 🫁 COPD
- 🤧 URTI
- 🌬️ LRTI
- ✅ Healthy
...
""")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Analysis", "🎵 Audio", "📡 Live Data"])

with tab1:
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Samples</h3>
                <h2>1,000</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h2>{:.1%}</h2>
            </div>
            """.format(st.session_state.accuracy), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🏷️ Classes</h3>
                <h2>4</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 Features</h3>
                <h2>13</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Class Distribution")
            class_counts = df['diagnosis'].value_counts()
            fig_pie = px.pie(
                values=class_counts.values, 
                names=class_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Distribution of Respiratory Conditions"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 🎨 Feature Correlation")
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig_heatmap = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                title="MFCC Feature Correlations"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Sample data preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
    else:
        st.warning("⚠️ Please load demo data first using the sidebar!")

with tab2:
    if st.session_state.model is not None:
        
        # Performance metrics
        st.markdown("### 🎯 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            labels = sorted(st.session_state.df['diagnosis'].unique())
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels, y=labels,
                color_continuous_scale='Blues',
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Feature importance
            importance = st.session_state.model.feature_importances_
            feature_names = [f'MFCC_{i+1}' for i in range(13)]
            
            fig_importance = px.bar(
                x=importance, 
                y=feature_names,
                orientation='h',
                title="Feature Importance",
                color=importance,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Classification report
        st.markdown("### 📋 Detailed Classification Report")
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Manual prediction
        st.markdown("### 🔮 Manual Prediction")
        st.markdown("Enter audio feature values for prediction:")
        
        col1, col2, col3 = st.columns(3)
        
        features = []
        for i in range(13):
            col_idx = i % 3
            if col_idx == 0:
                with col1:
                    val = st.slider(f"MFCC {i+1}", -5.0, 5.0, 0.0, key=f"mfcc_{i}")
            elif col_idx == 1:
                with col2:
                    val = st.slider(f"MFCC {i+1}", -5.0, 5.0, 0.0, key=f"mfcc_{i}")
            else:
                with col3:
                    val = st.slider(f"MFCC {i+1}", -5.0, 5.0, 0.0, key=f"mfcc_{i}")
            features.append(val)
        
        if st.button("🎯 Predict Condition"):
            prediction = st.session_state.model.predict([features])[0]
            probabilities = st.session_state.model.predict_proba([features])[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🎯 **Predicted Condition:** {prediction}")
                confidence = max(probabilities)
                st.metric("Confidence", f"{confidence:.3f}")
            
            with col2:
                # Probability chart
                prob_df = pd.DataFrame({
                    'Condition': st.session_state.model.classes_,
                    'Probability': probabilities
                })
                
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
        st.warning("⚠️ Please train the model first!")

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
                st.metric("Sample Rate", f"{sr} Hz")
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
            
            # Extract MFCC features
            st.markdown("#### 🔬 Feature Extraction")
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_features = np.mean(mfccs.T, axis=0)
            
            # Display features
            feature_df = pd.DataFrame({
                'Feature': [f'MFCC_{i+1}' for i in range(13)],
                'Value': mfcc_features
            })
            
            fig_features = px.bar(
                feature_df, 
                x='Feature', 
                y='Value',
                color='Value',
                color_continuous_scale='Viridis',
                title="Extracted MFCC Features"
            )
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Predict if model is trained
            if st.session_state.model is not None:
                st.markdown("#### 🎯 AI Prediction")
                
                prediction = st.session_state.model.predict([mfcc_features])[0]
                probabilities = st.session_state.model.predict_proba([mfcc_features])[0]
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