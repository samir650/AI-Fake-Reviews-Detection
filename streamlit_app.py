import streamlit as st
import pandas as pd
import pickle
import os
import time
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import logging
import numpy as np

# Import our optimized detection system
try:
    from fake_review_detection import GroqClient, ReviewKnowledgeBase, RAGDetector, preprocess_data, ReviewAnalysis
except ImportError as e:
    st.error(f"Failed to import detection modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Powered Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        border-radius: 10px;
    }
    .review-box {
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafafa;
        color: #000000 !important;  /* لون نص أسود */
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .fake-review {
        border-left: 5px solid #ff4b4b;
        background-color: #fff5f5;
        color: #000000 !important;  /* لون نص أسود */
    }
    .real-review {
        border-left: 5px solid #00c851;
        background-color: #f5fff5;
        color: #000000 !important;  /* لون نص أسود */
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="main-header">
    <h1>🔍 AI-Powered Fake Review Detector</h1>
    <p>Advanced fake review detection using RAG and Large Language Models</p>
    <p><small>Powered by Groq AI • Built with Streamlit</small></p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model Configuration
with st.sidebar.expander("🤖 Model Settings"):
    model_type = st.selectbox(
        "Select Model",
        ["fast", "smart", "balanced", "gemma"],
        index=1,
        help="Smart model provides best accuracy"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=5.0,
        help="Minimum confidence to consider prediction reliable"
    )

# Dataset Management
with st.sidebar.expander("📊 Dataset Management"):
    dataset_file = st.file_uploader(
        "Upload Training Dataset",
        type=['csv'],
        help="CSV file with 'text_' and 'label' columns"
    )
    
    if dataset_file is not None:
        try:
            df = pd.read_csv(dataset_file)
            st.success(f"Dataset loaded: {df.shape[0]} reviews")
            
            if st.button("🧠 Build Knowledge Base"):
                with st.spinner("Building knowledge base..."):
                    processed_df = preprocess_data(df)
                    kb = ReviewKnowledgeBase(processed_df)
                    kb.save("knowledge_base.pkl")
                    st.success("Knowledge base built successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

# Set the API key directly in the code (replace with your actual Groq API key)
API_KEY = "Insert Your Groq API key"  # غيرها بمفتاحك الحقيقي

# Initialize detector
def initialize_detector():
    """Initialize the RAG detector with knowledge base"""
    try:
        # Initialize Groq client with fixed API key
        client = GroqClient(API_KEY)
        
        # Load or create knowledge base
        kb_path = "knowledge_base.pkl"
        if Path(kb_path).exists():
            kb = ReviewKnowledgeBase.load(kb_path)
        else:
            # Try to load default dataset
            default_dataset = "fake reviews dataset.csv"
            if Path(default_dataset).exists():
                df = pd.read_csv(default_dataset)
                processed_df = preprocess_data(df)
                kb = ReviewKnowledgeBase(processed_df)
                kb.save(kb_path)
            else:
                st.error("No knowledge base found. Please upload a dataset first.")
                return None
        
        detector = RAGDetector(client, kb)
        return detector
    
    except Exception as e:
        st.error(f"Failed to initialize detector: {str(e)}")
        return None

# Main application tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Review Analysis", "📊 Batch Analysis", "📈 Analytics", "ℹ️ About"])

# Tab 1: Single Review Analysis
with tab1:
    st.header("Single Review Analysis")
    st.write("Analyze individual reviews for authenticity using AI-powered detection.")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "Sample Reviews"],
        horizontal=True
    )
    
    if input_method == "Text Input":
        review_text = st.text_area(
            "Enter review text:",
            height=150,
            placeholder="Paste the review you want to analyze here..."
        )
    else:
        # Sample reviews
        sample_reviews = {
            "Potentially Fake #1": "This product is absolutely amazing!!! Best purchase ever made, 100% recommend to everyone! Five stars definitely worth every penny amazing quality!",
            "Potentially Fake #2": "Terrible product, worst quality ever, completely useless waste of money, do not buy this garbage!!!",
            "Potentially Real #1": "I bought this for my daughter's birthday and she seems to enjoy it. The quality is decent for the price, though the instructions could be clearer. Setup took about 20 minutes.",
            "Potentially Real #2": "Good product overall. Had some issues with delivery timing but customer service was helpful. The item works as expected, though it's a bit smaller than I anticipated from the photos."
        }
        
        selected_sample = st.selectbox("Choose a sample review:", list(sample_reviews.keys()))
        review_text = sample_reviews[selected_sample]
        
        st.text_area("Selected review:", value=review_text, height=100, disabled=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if not review_text.strip():
                st.warning("Please enter a review to analyze")
            else:
                # Initialize detector if not already done
                if st.session_state.detector is None:
                    with st.spinner("Initializing detector..."):
                        st.session_state.detector = initialize_detector()
                
                if st.session_state.detector:
                    with st.spinner("Analyzing review..."):
                        try:
                            result = st.session_state.detector.predict_single(review_text.strip())
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                'timestamp': time.time(),
                                'review': review_text.strip(),
                                'result': result
                            })
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📋 Analysis Results")
                            
                            # Main prediction
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if result.prediction.lower() == 'fake':
                                    st.error(f"🚨 **FAKE REVIEW**")
                                else:
                                    st.success(f"✅ **REAL REVIEW**")
                            
                            with col2:
                                confidence_color = "red" if result.confidence < confidence_threshold else "green"
                                st.metric("Confidence", f"{result.confidence:.1f}%")
                            
                            with col3:
                                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                            
                            # Detailed analysis
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("🧠 AI Reasoning")
                                reasoning_class = "fake-review" if result.prediction.lower() == 'fake' else "real-review"
                                st.markdown(f'<div class="review-box {reasoning_class}">{result.reasoning}</div>', 
                                          unsafe_allow_html=True)
                                
                                if result.key_indicators:
                                    st.subheader("🔍 Key Indicators")
                                    for indicator in result.key_indicators:
                                        st.write(f"• {indicator}")
                            
                            with col2:
                                # Similarity score
                                if result.similarity_score > 0:
                                    st.subheader("📊 Similarity Score")
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = result.similarity_score * 100,
                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                        title = {'text': "Similarity %"},
                                        gauge = {
                                            'axis': {'range': [None, 100]},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [0, 50], 'color': "lightgray"},
                                                {'range': [50, 80], 'color': "yellow"},
                                                {'range': [80, 100], 'color': "green"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 90
                                            }
                                        }
                                    ))
                                    fig.update_layout(height=250)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Review statistics
                                st.subheader("📝 Review Stats")
                                word_count = len(review_text.split())
                                char_count = len(review_text)
                                exclamation_count = review_text.count('!')
                                question_count = review_text.count('?')
                                
                                st.metric("Words", word_count)
                                st.metric("Characters", char_count)
                                st.metric("Exclamations", exclamation_count)
                                st.metric("Questions", question_count)
                        
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Review Analysis")
    st.write("Upload multiple reviews for batch processing.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with reviews",
        type=['csv'],
        help="CSV should have a column with review texts"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {batch_df.shape[0]} reviews")
            
            # Column selection
            text_column = st.selectbox(
                "Select the column containing review texts:",
                batch_df.columns.tolist()
            )
            
            max_reviews = st.slider(
                "Maximum reviews to analyze:",
                min_value=1,
                max_value=min(50, len(batch_df)),
                value=min(10, len(batch_df)),
                help="Limited to prevent API overuse"
            )
            
            if st.button("🔍 Analyze Batch", type="primary"):
                # Initialize detector
                if st.session_state.detector is None:
                    with st.spinner("Initializing detector..."):
                        st.session_state.detector = initialize_detector()
                    
                if st.session_state.detector:
                    with st.spinner(f"Analyzing {max_reviews} reviews..."):
                        try:
                            review_texts = batch_df[text_column].head(max_reviews).tolist()
                            results = st.session_state.detector.predict_batch(review_texts, max_reviews)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    'review': text,
                                    'prediction': result.prediction,
                                    'confidence': result.confidence,
                                    'processing_time': result.processing_time
                                }
                                for text, result in zip(review_texts, results)
                            ])
                            
                            # Display summary
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                fake_count = sum(1 for r in results if r.prediction.lower() == 'fake')
                                st.metric("Fake Reviews", fake_count)
                            
                            with col2:
                                real_count = len(results) - fake_count
                                st.metric("Real Reviews", real_count)
                            
                            with col3:
                                avg_confidence = np.mean([r.confidence for r in results])
                                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                            
                            with col4:
                                total_time = sum(r.processing_time for r in results)
                                st.metric("Total Time", f"{total_time:.1f}s")
                            
                            # Results visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart
                                fig_pie = px.pie(
                                    values=[fake_count, real_count],
                                    names=['Fake', 'Real'],
                                    title="Prediction Distribution",
                                    color_discrete_map={'Fake': '#ff4b4b', 'Real': '#00c851'}
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                # Confidence distribution
                                fig_hist = px.histogram(
                                    results_df,
                                    x='confidence',
                                    color='prediction',
                                    title="Confidence Distribution",
                                    nbins=20,
                                    color_discrete_map={'fake': '#ff4b4b', 'real': '#00c851'}
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Detailed results table
                            st.subheader("📊 Detailed Results")
                            
                            # Add styling to dataframe
                            def style_prediction(val):
                                color = '#ff4b4b' if val.lower() == 'fake' else '#00c851'
                                return f'background-color: {color}; color: white; font-weight: bold;'
                            
                            styled_df = results_df.style.applymap(
                                style_prediction, subset=['prediction']
                            ).format({
                                'confidence': '{:.1f}%',
                                'processing_time': '{:.2f}s'
                            })
                            
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # Download results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv_data,
                                file_name=f"review_analysis_results_{int(time.time())}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Batch analysis failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        # Sample batch analysis option
        st.info("💡 **Tip:** Upload a CSV file with review texts for batch analysis, or try our sample data.")
        
        if st.button("📝 Try Sample Batch Analysis"):
            sample_data = [
                "This product is absolutely amazing!!! Best purchase ever made, 100% recommend!",
                "I bought this last week and it works well. Good value for the money.",
                "Terrible quality, complete waste of money, worst product ever!!!",
                "Decent product, arrived on time. The build quality could be better but it serves its purpose.",
                "AMAZING AMAZING AMAZING!!! Everyone should buy this RIGHT NOW!!!"
            ]
            
            if st.session_state.detector is None:
                with st.spinner("Initializing detector..."):
                    st.session_state.detector = initialize_detector()
            
            if st.session_state.detector:
                with st.spinner("Analyzing sample reviews..."):
                    try:
                        results = st.session_state.detector.predict_batch(sample_data)
                        
                        for i, (review, result) in enumerate(zip(sample_data, results)):
                            st.write(f"**Review {i+1}:** {review}")
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if result.prediction.lower() == 'fake':
                                    st.error("🚨 FAKE")
                                else:
                                    st.success("✅ REAL")
                            with col2:
                                st.write(f"Confidence: {result.confidence:.1f}%")
                            
                            st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Sample analysis failed: {str(e)}")

# Tab 3: Analytics
with tab3:
    st.header("Analysis History & Analytics")
    
    if st.session_state.analysis_history:
        # Convert history to dataframe
        history_df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(item['timestamp'], unit='s'),
                'review_preview': item['review'][:50] + "..." if len(item['review']) > 50 else item['review'],
                'prediction': item['result'].prediction,
                'confidence': item['result'].confidence,
                'processing_time': item['result'].processing_time
            }
            for item in st.session_state.analysis_history
        ])
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_analyses = len(history_df)
            st.metric("Total Analyses", total_analyses)
        
        with col2:
            fake_percentage = (history_df['prediction'] == 'fake').mean() * 100
            st.metric("Fake Reviews %", f"{fake_percentage:.1f}%")
        
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col4:
            avg_processing_time = history_df['processing_time'].mean()
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Timeline of analyses
            if len(history_df) > 1:
                fig_timeline = px.scatter(
                    history_df,
                    x='timestamp',
                    y='confidence',
                    color='prediction',
                    title="Analysis Timeline",
                    color_discrete_map={'fake': '#ff4b4b', 'real': '#00c851'}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Processing time distribution
            fig_time = px.box(
                history_df,
                y='processing_time',
                color='prediction',
                title="Processing Time Distribution",
                color_discrete_map={'fake': '#ff4b4b', 'real': '#00c851'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent analyses table
        st.subheader("📋 Recent Analyses")
        recent_df = history_df.tail(10).sort_values('timestamp', ascending=False)
        
        def style_prediction_analytics(val):
            color = '#ff4b4b' if val.lower() == 'fake' else '#00c851'
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        styled_recent = recent_df.style.applymap(
            style_prediction_analytics, subset=['prediction']
        ).format({
            'confidence': '{:.1f}%',
            'processing_time': '{:.2f}s',
            'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        st.dataframe(styled_recent, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear Analysis History", type="secondary"):
            st.session_state.analysis_history = []
            st.success("Analysis history cleared!")
            st.experimental_rerun()
    
    else:
        st.info("No analysis history available. Start analyzing reviews to see analytics here!")
        st.image("https://via.placeholder.com/400x200/f0f2f6/666666?text=No+Data+Available", width=400)

# Tab 4: About
with tab4:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses advanced AI techniques to detect fake reviews with high accuracy. It combines:
    - **RAG (Retrieval-Augmented Generation)** for context-aware analysis
    - **Large Language Models** (LLM) via Groq API for intelligent reasoning
    - **TF-IDF vectorization** for similarity matching
    - **Statistical analysis** for pattern recognition
    
    ### 🔧 Technical Features
    - **Real-time Analysis**: Instant fake review detection
    - **Batch Processing**: Analyze multiple reviews simultaneously  
    - **Knowledge Base**: Learns from training data patterns
    - **Confidence Scoring**: Provides reliability metrics
    - **Detailed Reasoning**: Explains detection decisions
    
    ### 📊 Detection Indicators
    
    **Fake Review Patterns:**
    - Excessive superlatives and extreme language
    - Generic or vague descriptions
    - Unusual grammar patterns
    - Inconsistent sentiment
    - Over-emphasis on specific features
    - Unnatural keyword repetition
    
    **Real Review Patterns:**
    - Specific details and personal experiences
    - Balanced sentiment (pros and cons)
    - Natural language flow
    - Contextual information
    - Appropriate length and detail
    
    ### 🚀 Getting Started
    1. **Set up API Key**: Get your free Groq API key from [console.groq.com](https://console.groq.com)
    2. **Test Connection**: Verify your API key works
    3. **Upload Dataset** (optional): Train with your own data
    4. **Start Analyzing**: Use single or batch analysis modes
    
    ### ⚡ API Limits
    - **Free Tier**: 14,400 tokens/minute, 30 requests/minute
    - **Rate Limiting**: Automatic handling with smart delays
    - **Caching**: Reduces redundant API calls
    
    ### 🛠️ Built With
    - **Streamlit**: Web application framework
    - **Groq API**: Fast LLM inference
    - **Scikit-learn**: Machine learning utilities
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation
    """)
    
    # System status
    st.subheader("🔍 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_status = "🟢 Connected"  # Assuming it's connected since key is hardcoded
        st.markdown(f"**API Status:** {api_status}")
    
    with col2:
        kb_status = "🟢 Loaded" if Path("knowledge_base.pkl").exists() else "🔴 Not Found"
        st.markdown(f"**Knowledge Base:** {kb_status}")
    
    with col3:
        detector_status = "🟢 Ready" if st.session_state.detector else "🔴 Not Initialized"
        st.markdown(f"**Detector:** {detector_status}")
    
    # Contact info
    st.markdown("---")
    st.markdown("""
    ### 📞 Support & Information
    - **Documentation**: Check the sidebar for configuration help
    - **API Issues**: Verify your Groq API key and connection
    - **Performance**: Use 'smart' model for best accuracy, 'fast' for speed
    - **Dataset**: Upload CSV with 'text_' and 'label' columns for training
    
    **⚠️ Disclaimer**: This tool is for educational and research purposes. Results should be validated with human judgment for critical applications.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with ❤️ using Streamlit • Powered by Groq AI • © 2024"
    "</div>", 
    unsafe_allow_html=True
)
