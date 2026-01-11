import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Cancer Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e7ff;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-label {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Stats box */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .stat-value {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Alert boxes */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD MODEL FUNCTION
# ===========================
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        # Cari file model terbaru
        model_files = [f for f in os.listdir('.') if f.startswith('cancer_svm_model_') and f.endswith('.pkl')]
        scaler_files = [f for f in os.listdir('.') if f.startswith('cancer_scaler_') and f.endswith('.pkl')]
        
        if not model_files or not scaler_files:
            return None, None, "Model files not found! Please train the model first."
        
        # Ambil file terbaru
        model_file = sorted(model_files)[-1]
        scaler_file = sorted(scaler_files)[-1]
        
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        return model, scaler, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# ===========================
# PREDICTION FUNCTION
# ===========================
def predict_image(image, model, scaler):
    """Predict cancer type from image"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Handle grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Resize to 224x224x3
        img_resized = resize(img_array, (224, 224, 3))
        
        # Normalize [0, 1]
        img_normalized = img_resized.astype(np.float32)
        if img_normalized.max() > 1:
            img_normalized = img_normalized / 255.0
        
        # Flatten and reshape
        img_flat = img_normalized.flatten().reshape(1, -1)
        
        # Scale
        img_scaled = scaler.transform(img_flat)
        
        # Predict
        prediction = model.predict(img_scaled)[0]
        probabilities = model.predict_proba(img_scaled)[0]
        
        categories = ['GANAS', 'JINAK', 'NON KANKER']
        
        return {
            'success': True,
            'class': categories[prediction],
            'class_index': int(prediction),
            'confidence': float(probabilities[prediction] * 100),
            'probabilities': {cat: float(prob*100) for cat, prob in zip(categories, probabilities)},
            'processed_image': img_resized
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ===========================
# MAIN APP
# ===========================
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏥 Cancer Classification System</h1>
        <p class="header-subtitle">Advanced AI-Powered Medical Image Analysis using Support Vector Machine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("💡 Please run the training notebook first to generate the model files.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=80)
        st.markdown("### 📊 System Information")
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Model Type</p>
            <p class="stat-value">SVM Linear</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Image Resolution</p>
            <p class="stat-value">224×224×3</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Classes</p>
            <p class="stat-value">3 Types</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Classification Types")
        st.markdown("""
        - 🔴 **GANAS** - Malignant (Cancerous)
        - 🟡 **JINAK** - Benign (Non-cancerous tumor)
        - 🟢 **NON KANKER** - Non-cancer (Healthy)
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses **Support Vector Machine (SVM)** 
        with linear kernel trained on medical imaging dataset.
        
        **Features:**
        - Real-time prediction
        - Confidence scoring
        - Professional visualization
        - High accuracy classification
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Image Classification", "📈 Batch Processing", "📊 Model Analysis", "📚 Information"])
    
    # TAB 1: Single Image Classification
    with tab1:
        st.markdown("### Upload Medical Image for Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file (JPG, PNG, JPEG)",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a medical image for cancer classification"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Predict button
                if st.button("🔬 Analyze Image", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        result = predict_image(image, model, scaler)
                    
                    if result['success']:
                        # Store result in session state
                        st.session_state['last_result'] = result
                    else:
                        st.error(f"❌ Error: {result['error']}")
        
        with col2:
            if 'last_result' in st.session_state:
                result = st.session_state['last_result']
                
                # Prediction result
                class_colors = {
                    'GANAS': '#dc3545',
                    'JINAK': '#ffc107', 
                    'NON KANKER': '#28a745'
                }
                
                class_icons = {
                    'GANAS': '🔴',
                    'JINAK': '🟡',
                    'NON KANKER': '🟢'
                }
                
                predicted_class = result['class']
                
                st.markdown(f"""
                <div style="background: {class_colors[predicted_class]}; color: white; 
                            padding: 2rem; border-radius: 10px; text-align: center;">
                    <p style="font-size: 1.2rem; margin: 0;">Prediction Result</p>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">
                        {class_icons[predicted_class]} {predicted_class}
                    </h1>
                    <p style="font-size: 1.5rem; margin: 0;">
                        Confidence: {result['confidence']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("### 📊 Probability Distribution")
                
                probs = result['probabilities']
                df_probs = pd.DataFrame({
                    'Class': list(probs.keys()),
                    'Probability (%)': list(probs.values())
                })
                
                # Plotly bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_probs['Class'],
                        y=df_probs['Probability (%)'],
                        marker_color=['#dc3545', '#ffc107', '#28a745'],
                        text=[f"{v:.1f}%" for v in df_probs['Probability (%)']],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Class Probabilities",
                    xaxis_title="Cancer Type",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 105],
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                st.markdown("### 📋 Detailed Analysis")
                for class_name, prob in probs.items():
                    icon = class_icons[class_name]
                    st.markdown(f"**{icon} {class_name}**: {prob:.2f}%")
                    st.progress(prob / 100)
    
    # TAB 2: Batch Processing
    with tab2:
        st.markdown("### Batch Image Processing")
        st.info("📁 Upload multiple images for batch classification")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple medical images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images uploaded**")
            
            if st.button("🚀 Process All Images", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    image = Image.open(file)
                    result = predict_image(image, model, scaler)
                    
                    if result['success']:
                        results.append({
                            'Filename': file.name,
                            'Prediction': result['class'],
                            'Confidence (%)': f"{result['confidence']:.2f}",
                            'GANAS (%)': f"{result['probabilities']['GANAS']:.2f}",
                            'JINAK (%)': f"{result['probabilities']['JINAK']:.2f}",
                            'NON KANKER (%)': f"{result['probabilities']['NON KANKER']:.2f}"
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                st.markdown("### 📊 Batch Results")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"cancer_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.markdown("### 📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ganas_count = len(df_results[df_results['Prediction'] == 'GANAS'])
                    st.metric("🔴 GANAS", ganas_count)
                
                with col2:
                    jinak_count = len(df_results[df_results['Prediction'] == 'JINAK'])
                    st.metric("🟡 JINAK", jinak_count)
                
                with col3:
                    non_kanker_count = len(df_results[df_results['Prediction'] == 'NON KANKER'])
                    st.metric("🟢 NON KANKER", non_kanker_count)
    
    # TAB 3: Model Analysis
    with tab3:
        st.markdown("### 📊 Model Performance Analysis")
        
        # Load model metadata and info
        try:
            metadata_files = [f for f in os.listdir('.') if f.startswith('cancer_model_metadata_') and f.endswith('.pkl')]
            info_files = [f for f in os.listdir('.') if f.startswith('cancer_model_info_') and f.endswith('.txt')]
            
            if metadata_files and info_files:
                # Load metadata
                import pickle
                metadata_file = sorted(metadata_files)[-1]
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Load info text
                info_file = sorted(info_files)[-1]
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = f.read()
                
                # Parse metrics from info file
                lines = model_info.split('\n')
                metrics = {}
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            metrics[key] = value
                
                # Display Overview
                st.markdown("#### 🎯 Model Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = metrics.get('Accuracy', 'N/A')
                    if accuracy != 'N/A':
                        acc_val = float(accuracy.split('(')[1].replace('%)', '').strip()) if '(' in accuracy else 0
                        st.metric("🎯 Accuracy", accuracy.split('(')[0].strip(), f"{acc_val:.1f}%")
                    else:
                        st.metric("🎯 Accuracy", "N/A")
                
                with col2:
                    training_samples = metrics.get('Training Samples', 'N/A')
                    st.metric("📊 Training Samples", training_samples)
                
                with col3:
                    test_samples = metrics.get('Test Samples', 'N/A')
                    st.metric("✅ Test Samples", test_samples)
                
                with col4:
                    augmentation = metadata.get('augmentation_factor', 'N/A')
                    st.metric("🔄 Augmentation", f"{augmentation}x")
                
                st.markdown("---")
                
                # Dataset Distribution
                st.markdown("#### 📈 Dataset Distribution")
                categories = metadata.get('categories', ['GANAS', 'JINAK', 'NON KANKER'])
                
                # Create sample distribution (from metadata)
                training_samples_count = int(training_samples.replace(',', '')) if training_samples != 'N/A' else 1000
                
                # Estimate per-class distribution (assuming balanced after augmentation)
                samples_per_class = training_samples_count // len(categories)
                
                fig_dist = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=[samples_per_class] * len(categories),
                        marker=dict(
                            color=['#dc3545', '#ffc107', '#28a745'],
                            line=dict(color='black', width=2)
                        ),
                        text=[samples_per_class] * len(categories),
                        textposition='auto',
                    )
                ])
                
                fig_dist.update_layout(
                    title="Training Samples per Class (After Augmentation)",
                    xaxis_title="Class",
                    yaxis_title="Number of Samples",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
                
                # Per-Class Performance Metrics
                st.markdown("#### 📊 Per-Class Performance Metrics")
                
                # Parse per-class metrics from info file
                per_class_metrics = {}
                current_class = None
                
                for line in lines:
                    line = line.strip()
                    if line in categories:
                        current_class = line
                        per_class_metrics[current_class] = {}
                    elif current_class and ':' in line and any(metric in line for metric in ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            metric_name = parts[0].strip()
                            metric_value = float(parts[1].strip())
                            per_class_metrics[current_class][metric_name] = metric_value
                
                if per_class_metrics:
                    # Create dataframe for visualization
                    metrics_data = []
                    for cat in categories:
                        if cat in per_class_metrics:
                            metrics_data.append({
                                'Class': cat,
                                'Precision': per_class_metrics[cat].get('Precision', 0),
                                'Recall': per_class_metrics[cat].get('Recall', 0),
                                'F1-Score': per_class_metrics[cat].get('F1-Score', 0),
                                'ROC-AUC': per_class_metrics[cat].get('ROC-AUC', 0)
                            })
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    
                    # Display as table
                    st.dataframe(
                        df_metrics.style.format({
                            'Precision': '{:.4f}',
                            'Recall': '{:.4f}',
                            'F1-Score': '{:.4f}',
                            'ROC-AUC': '{:.4f}'
                        }).background_gradient(cmap='RdYlGn', subset=['Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
                        use_container_width=True
                    )
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart for Precision, Recall, F1
                        fig_prf = go.Figure()
                        
                        fig_prf.add_trace(go.Bar(
                            name='Precision',
                            x=df_metrics['Class'],
                            y=df_metrics['Precision'],
                            marker_color='#FF6B6B'
                        ))
                        
                        fig_prf.add_trace(go.Bar(
                            name='Recall',
                            x=df_metrics['Class'],
                            y=df_metrics['Recall'],
                            marker_color='#4ECDC4'
                        ))
                        
                        fig_prf.add_trace(go.Bar(
                            name='F1-Score',
                            x=df_metrics['Class'],
                            y=df_metrics['F1-Score'],
                            marker_color='#45B7D1'
                        ))
                        
                        fig_prf.update_layout(
                            title="Precision, Recall & F1-Score Comparison",
                            xaxis_title="Class",
                            yaxis_title="Score",
                            barmode='group',
                            height=400,
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig_prf, use_container_width=True)
                    
                    with col2:
                        # ROC-AUC bar chart
                        fig_roc = go.Figure(data=[
                            go.Bar(
                                x=df_metrics['Class'],
                                y=df_metrics['ROC-AUC'],
                                marker=dict(
                                    color=df_metrics['ROC-AUC'],
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    line=dict(color='black', width=2)
                                ),
                                text=df_metrics['ROC-AUC'].apply(lambda x: f'{x:.3f}'),
                                textposition='auto',
                            )
                        ])
                        
                        fig_roc.update_layout(
                            title="ROC-AUC Score per Class",
                            xaxis_title="Class",
                            yaxis_title="ROC-AUC Score",
                            height=400,
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                st.markdown("---")
                
                # Advanced Metrics
                st.markdown("#### 🎯 Advanced Statistical Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    kappa = metrics.get("Cohen's Kappa", 'N/A')
                    st.metric("Cohen's Kappa", kappa)
                    st.caption("Measures inter-rater reliability (0.6-0.8 = Good)")
                
                with col2:
                    mcc = metrics.get('MCC', 'N/A')
                    st.metric("Matthews Corr. Coef.", mcc)
                    st.caption("Correlation between predicted & actual (-1 to 1)")
                
                with col3:
                    roc_auc = metrics.get('ROC-AUC (Macro)', 'N/A')
                    st.metric("Macro ROC-AUC", roc_auc)
                    st.caption("Overall model discrimination ability")
                
                st.markdown("---")
                
                # Model Configuration
                st.markdown("#### ⚙️ Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Architecture:**
                    - Algorithm: Support Vector Machine (SVM)
                    - Kernel: Linear
                    - Probability Estimates: Enabled
                    - Class Weight: Balanced
                    
                    **Input Processing:**
                    - Image Size: 224 × 224 pixels
                    - Color Channels: 3 (RGB)
                    - Total Features: 150,528
                    - Normalization: [0, 1] range
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Training Configuration:**
                    - Data Split: 80% Train / 20% Test
                    - Random State: 77
                    - Stratified Sampling: Yes
                    - Augmentation Factor: {augmentation}x
                    
                    **Augmentation Techniques:**
                    - Rotation: ±30 degrees
                    - Horizontal Flip
                    - Vertical Flip
                    - Brightness Adjustment: ±20%
                    """)
                
                st.markdown("---")
                
                # Full Training Report
                with st.expander("📋 View Complete Training Report"):
                    st.code(model_info, language='text')
                
            else:
                st.warning("⚠️ Model training information not found.")
                st.info("💡 The model metadata files are generated during training in the Jupyter notebook.")
                
                st.markdown("""
                ### Expected Metrics Display:
                
                When model metadata is available, this page will show:
                
                - **Overall Metrics**: Accuracy, Kappa, MCC, ROC-AUC
                - **Dataset Distribution**: Training/test split visualization
                - **Per-Class Performance**: Precision, Recall, F1-Score, ROC-AUC for each class
                - **Advanced Statistics**: Cohen's Kappa, Matthews Correlation Coefficient
                - **Model Configuration**: Architecture and training parameters
                - **Complete Training Report**: Full detailed metrics
                """)
                
        except Exception as e:
            st.error(f"Error loading model analysis: {str(e)}")
            st.info("Please ensure model metadata files are available in the application directory.")
    
    # TAB 4: Information
    with tab4:
        st.markdown("### 📚 System Documentation")
        
        st.markdown("""
        ## 🎯 About This System
        
        This **Cancer Classification System** uses advanced Machine Learning techniques 
        to classify medical images into three categories:
        
        - **GANAS (Malignant)**: Cancerous tumors that can spread
        - **JINAK (Benign)**: Non-cancerous tumors
        - **NON KANKER (Non-cancer)**: Healthy tissue
        
        ## 🔬 Technology Stack
        
        - **Algorithm**: Support Vector Machine (SVM) with Linear Kernel
        - **Image Processing**: scikit-image
        - **Framework**: Streamlit
        - **Visualization**: Plotly, Matplotlib
        - **Model Training**: scikit-learn
        
        ## � Model Specifications
        
        - **Input Size**: 224×224×3 RGB images
        - **Features**: 150,528 features per image
        - **Preprocessing**: 
          - Automatic grayscale to RGB conversion
          - Image normalization [0, 1]
          - Standard scaling with StandardScaler
        - **Training**: 
          - Data augmentation (4x factor)
          - Rotation, flip, brightness adjustments
          - Class weight balancing
        
        ## 🚀 How to Use
        
        ### Single Image Classification:
        1. Go to the "Image Classification" tab
        2. Upload a medical image (JPG, PNG)
        3. Click "Analyze Image"
        4. View prediction results and confidence scores
        
        ### Batch Processing:
        1. Go to the "Batch Processing" tab
        2. Upload multiple images
        3. Click "Process All Images"
        4. Download results as CSV
        
        ## ⚠️ Important Notes
        
        - This system is for **research and educational purposes**
        - Always consult with medical professionals for diagnosis
        - Results should be verified by qualified healthcare providers
        - Not intended as a replacement for professional medical advice
        
        ## 📞 Support
        
        For questions or issues, please contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b;">
            <p>Developed with ❤️ using Python & Streamlit</p>
            <p>© 2026 Cancer Classification System</p>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    main()
