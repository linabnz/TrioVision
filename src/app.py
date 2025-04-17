import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import torch
import pandas as pd
from PIL import Image
import time
from streamlit_folium import folium_static
from streamlit_lottie import st_lottie

# Import of custom modules - use relative imports with the dot
from src.model import load_model, predict_country
from src.utils import (
    load_country_labels, preprocess_image, 
    create_beautiful_map, create_radar_chart, 
    create_similarity_chart, create_probability_distribution, 
    create_confidence_comparison, load_lottieurl
)
from src.config import (
    LABELS_PATH, MODEL_PATH, TEMP_IMAGE_PATH, DEFAULT_COORDINATES,
    COUNTRY_COLORS, RADAR_COLORS, RADAR_BORDERS, LOTTIE_ANIMATIONS,
    APP_CONFIG
)
from src.styles import get_custom_css
from src.preprocessing import Preprocessing

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# Load custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load Lottie animations
globe_animation = load_lottieurl(LOTTIE_ANIMATIONS["globe"])
analyze_animation = load_lottieurl(LOTTIE_ANIMATIONS["analyze"])
upload_animation = load_lottieurl(LOTTIE_ANIMATIONS["upload"])
success_animation = load_lottieurl(LOTTIE_ANIMATIONS["success"])

# Function to preprocess the image with our specific preprocessor
def process_image(image):
    preprocessor = Preprocessing()
    return preprocess_image(image, preprocessor, TEMP_IMAGE_PATH)

# Load labels and model
@st.cache_data
def cached_load_country_labels():
    return load_country_labels(LABELS_PATH)

@st.cache_resource
def cached_load_model(num_classes):
    return load_model(MODEL_PATH, num_classes)

# Main interface
def main():
    # Title and subtitle with enhanced style
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="premium-title">TrioVision</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Cutting-edge artificial intelligence for image geolocation.</p>', unsafe_allow_html=True)
        
        if globe_animation:
            st_lottie(globe_animation, height=180, key="globe")
    
    # Load labels and model
    countries = cached_load_country_labels()
    if countries:
        model = cached_load_model(num_classes=len(countries))
    else:
        st.error("Unable to load country labels.")
        model = None
    
    # Create premium user interface
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    col_upload, col_analyze = st.columns([2, 1])
    
    with col_upload:
        st.markdown('<h3 class="section-title">Upload an image</h3>', unsafe_allow_html=True)
        
        if upload_animation:
            st_lottie(upload_animation, height=100, key="upload")
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="", use_container_width=True)
    
    with col_analyze:
        st.markdown('<h3 class="section-title">AI Analysis</h3>', unsafe_allow_html=True)
        
        if analyze_animation:
            st_lottie(analyze_animation, height=100, key="analyze")
        
        analyze_button = st.button("Analyze the image")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results area
    results_container = st.container()
    
    # Process image if button is clicked
    if uploaded_file is not None and analyze_button:
        if model is not None and countries:
            with st.spinner(""):
                # Display custom loader
                progress_placeholder = st.empty()
                progress_placeholder.markdown("""
                <div style="text-align: center; margin: 20px 0;">
                    <div class="pulse-loader">AI</div>
                    <p style="margin-top: 10px;">Advanced analysis in progress...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress animation
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                
                # Actual prediction
                results = predict_country(image, model, countries, process_image)
                
                if results:
                    # Clear temporary elements
                    progress_placeholder.empty()
                    progress_bar.empty()
                    
                    with results_container:
                        # Display main result
                        best_country = results[0]["country"]
                        confidence = results[0]["probability"] * 100
                        
                        if success_animation:
                            st_lottie(success_animation, height=120, key="success")
                        
                        st.markdown(f"""
                        <div class="prediction-card fade-in-up">
                            <h3>Identified Country</h3>
                            <div class="prediction-country">{best_country}</div>
                            <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create tabs for visualizations
                        tab1, tab2, tab3, tab4 = st.tabs([" Map", " Analysis", " Comparison", "Statistics"])
                        
                        with tab1:
                            st.markdown('<div class="staggered delay-2">', unsafe_allow_html=True)
                            st.markdown("<h3 class='section-title'>Geographical Visualization</h3>", unsafe_allow_html=True)
                            
                            try:
                                beautiful_map = create_beautiful_map(
                                    results, 
                                    default_coordinates=DEFAULT_COORDINATES,
                                    country_colors=COUNTRY_COLORS
                                )
                                folium_static(beautiful_map, width=1000, height=500)
                            except Exception as e:
                                st.error(f"Error while creating the map: {e}")
                                import folium
                                default_map = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")
                                folium_static(default_map, width=800, height=500)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab2:
                            st.markdown('<div class="staggered delay-2">', unsafe_allow_html=True)
                            st.markdown("<h3 class='section-title'>Probability Analysis</h3>", unsafe_allow_html=True)
                            
                            radar_fig = create_radar_chart(
                                results,
                                radar_colors=RADAR_COLORS,
                                radar_borders=RADAR_BORDERS
                            )
                            st.plotly_chart(radar_fig, use_container_width=True)
                            
                            prob_fig = create_probability_distribution(results)
                            st.plotly_chart(prob_fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab3:
                            st.markdown('<div class="staggered delay-3">', unsafe_allow_html=True)
                            st.markdown("<h3 class='section-title'>Country Comparison</h3>", unsafe_allow_html=True)
                            
                            similarity_fig = create_similarity_chart(results)
                            st.plotly_chart(similarity_fig, use_container_width=True)
                            
                            confidence_fig = create_confidence_comparison(
                                results,
                                country_colors=COUNTRY_COLORS
                            )
                            st.plotly_chart(confidence_fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab4:
                            st.markdown('<div class="staggered delay-4">', unsafe_allow_html=True)
                            st.markdown("<h3 class='section-title'>Detailed Data</h3>", unsafe_allow_html=True)
                            
                            df = pd.DataFrame([
                                {"Rank": i+1, "Country": r["country"], "Probability (%)": f"{r['probability']*100:.2f}%"} 
                                for i, r in enumerate(results[:15])
                            ])
                            
                            st.dataframe(
                                df,
                                column_config={
                                    "Rank": st.column_config.NumberColumn("🏆 Rank"),
                                    "Country": st.column_config.TextColumn("🌎 Country"),
                                    "Probability (%)": st.column_config.TextColumn("📊 Probability"),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Error during prediction.")
        else:
            st.error("The model or labels were not loaded correctly.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3 style="color: #4285F4;">TrioVision</h3>
        <p>Cutting-edge artificial intelligence for image-based geolocation</p>
        <p style="font-size:0.8rem; margin-top: 10px;">
        Developed with EfficientNet and advanced visualization technologies by Lina, Sharon and Louis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
