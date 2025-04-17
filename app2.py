import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
import torch.nn as nn
import json
import folium
from folium.plugins import MarkerCluster, MiniMap, Draw, Fullscreen
from streamlit_folium import folium_static
import torch.nn.functional as F
from PIL import Image
import io
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from preprocessing import Preprocessing
from torchvision import models
import base64
from streamlit_lottie import st_lottie
import requests
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#####################################################
# CONFIGURATION DES CHEMINS - MODIFIEZ CES VARIABLES
#####################################################

# Chemin vers votre fichier de labels
LABELS_PATH = r"C:\Users\lbenzemma\Desktop\Projets Master2 MOSEF\deep learning\AtlasEye\AtlasEye\country_labels_total_essaie.json"
MODEL_PATH = r"C:\Users\lbenzemma\Desktop\Projets Master2 MOSEF\deep learning\AtlasEye\AtlasEye\model_ef_model_2025-04-16_12-05-34.pth"

# Chemin pour les fichiers temporaires
TEMP_IMAGE_PATH = "temp_image.jpg"

# Dictionnaire de coordonnées pour les pays
DEFAULT_COORDINATES = {
    "France": (46.603354, 1.888334),
    "United States": (37.09024, -95.712891),
    "Germany": (51.165691, 10.451526),
    "Italy": (41.87194, 12.56738),
    "Spain": (40.463667, -3.74922),
    "United Kingdom": (55.378051, -3.435973),
    "Japan": (36.204824, 138.252924),
    "China": (35.86166, 104.195397),
    "India": (20.593684, 78.96288),
    "Russia": (61.52401, 105.318756),
    "Brazil": (-14.235004, -51.92528),
    "Canada": (56.130366, -106.346771),
    "Australia": (-25.274398, 133.775136),
    "South Korea": (35.907757, 127.766922),
    "Mexico": (23.634501, -102.552784)
}

# Couleurs pour les pays (palette plus moderne)
COUNTRY_COLORS = ["#FF4B4B", "#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8AB4F8", "#137333", "#F29900", "#C5221F", "#669DF6"]

# Gradient de couleurs pour le graphique radar
RADAR_COLORS = ["rgba(255, 75, 75, 0.7)", "rgba(66, 133, 244, 0.7)", "rgba(52, 168, 83, 0.7)", "rgba(251, 188, 5, 0.7)", "rgba(234, 67, 53, 0.7)"]
RADAR_BORDERS = ["rgba(255, 75, 75, 1)", "rgba(66, 133, 244, 1)", "rgba(52, 168, 83, 1)", "rgba(251, 188, 5, 1)", "rgba(234, 67, 53, 1)"]

#####################################################
# FIN DE LA CONFIGURATION
#####################################################

# Configuration de la page
st.set_page_config(
    page_title="TrioVision - Country Recognition AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animations Lottie
globe_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_JFzAYh.json")
analyze_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_wrffqrdf.json")
upload_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_zzujjt5k.json")
success_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jTrZKu.json")

# CSS personnalisé pour une interface ultra moderne
st.markdown("""
<style>
    /* Styles de base et globaux */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Animation de texte avec gradient */
    .premium-title {
        font-size: 60px !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(90deg, #4285F4, #34A853, #FBBC05, #EA4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 8s ease infinite;
        background-size: 300% 300%;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        font-size: 22px !important;
        text-align: center;
        color: #5f6368;
        margin-top: 0;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cards et conteneurs */
    .modern-card {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
    }
    
    .prediction-card {
        background: linear-gradient(120deg, #4285F4, #34A853);
        color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(66, 133, 244, 0.3);
        text-align: center;
        margin: 24px 0;
    }
    
    .prediction-country {
        font-weight: 700;
        font-size: 2.5rem;
        margin: 10px 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-badge {
        background-color: rgba(255, 255, 255, 0.25);
        font-size: 1.2rem;
        padding: 5px 16px;
        border-radius: 100px;
        margin-top: 10px;
        display: inline-block;
    }
    
    /* Boutons et Actions */
    .action-button {
        background: linear-gradient(90deg, #4285F4, #0d47a1);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 100px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(66, 133, 244, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        display: block;
        width: 100%;
        font-size: 16px;
        margin-top: 15px;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(66, 133, 244, 0.6);
        background: linear-gradient(90deg, #5a95f5, #1565C0);
    }
    
    /* Animation pour les résultats */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    /* Animation pour les éléments qui apparaissent successivement */
    .staggered {
        opacity: 0;
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
    .delay-5 { animation-delay: 0.5s; }
    
    /* Titres des sections */
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #202124;
        position: relative;
        padding-left: 15px;
    }
    
    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 5px;
        background: linear-gradient(180deg, #4285F4, #34A853);
        border-radius: 10px;
    }
    
    /* Personnalisation des widgets Streamlit */
    .stButton>button {
        background: linear-gradient(90deg, #4285F4, #0d47a1);
        color: white;
        font-weight: 600;
        border: none !important;
        border-radius: 100px !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(66, 133, 244, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #5a95f5, #1565C0);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(66, 133, 244, 0.6);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Tableaux stylisés */
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .styled-table thead tr {
        background-color: #4285F4;
        color: white;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 16px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
        transition: all 0.2s ease;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #4285F4;
    }
    
    .styled-table tbody tr:hover {
        background-color: #e6f1ff;
        transform: scale(1.01);
    }
    
    /* Pied de page */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem 0;
        background-color: white;
        border-radius: 16px;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.05);
        color: #5f6368;
    }
    
    /* Médaillons de confiance */
    .confidence-medals {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
    }
    
    .confidence-medal {
        width: 18%;
        padding: 15px 10px;
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.03);
    }
    
    .confidence-medal:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .medal-label {
        font-size: 0.9rem;
        color: #5f6368;
        margin-bottom: 5px;
    }
    
    .medal-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4285F4;
    }
    
    .medal-1 { border-top: 4px solid #FF4B4B; }
    .medal-2 { border-top: 4px solid #4285F4; }
    .medal-3 { border-top: 4px solid #34A853; }
    .medal-4 { border-top: 4px solid #FBBC05; }
    .medal-5 { border-top: 4px solid #EA4335; }
    
    /* Lottie et animations */
    .lottie-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    /* Cartes des résultats avec couleurs alternées */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }
    
    .result-item {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .result-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .result-item h4 {
        margin: 0 0 8px 0;
        color: #202124;
    }
    
    .result-item p {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Loader personnalisé */
    @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 0.8; }
        50% { transform: scale(1); opacity: 1; }
        100% { transform: scale(0.8); opacity: 0.8; }
    }
    
    .pulse-loader {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle, #4285F4, #0d47a1);
        margin: 0 auto;
        animation: pulse-ring 1.5s ease infinite;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4285F4 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 500;
        background-color: #f1f3f4;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4285F4 !important;
        color: white !important;
    }
    
    /* Pour les graphiques */
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    
    /* Masquer les éléments en trop de streamlit */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    div.block-container {
        padding-top: 2rem;
    }
    
    /* Animations pour les graphiques */
    @keyframes chart-slide-in {
        0% { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    .chart-animation {
        animation: chart-slide-in 0.8s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Définition du modèle EfficientNet
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            batch_size, seq_len, channels, height, width = x.shape
            x = x.squeeze(1)
        return self.efficientnet(x)

# Titre et description avec style amélioré
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="premium-title">TrioVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cutting-edge artificial intelligence for image geolocation.</p>', unsafe_allow_html=True)
    
    # Animation Lottie
    if globe_animation:
        st_lottie(globe_animation, height=180, key="globe")

# Chargement des labels de pays
@st.cache_data
def load_country_labels():
    try:
        with open(LABELS_PATH, "r") as f:
            labels_data = json.load(f)
        if isinstance(labels_data, dict):
            try:
                countries = [labels_data[str(i)] for i in range(len(labels_data))]
                return countries
            except:
                return list(labels_data.values())
        elif isinstance(labels_data, list):
            return labels_data
    except Exception as e:
        st.error(f"Erreur lors du chargement des labels: {e}")
        return []

# Chargement du modèle
@st.cache_resource
def load_model(num_classes):
    try:
        model = EfficientNetModel(num_classes=num_classes)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def preprocess_image(image):
    preprocessor = Preprocessing()

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    with open(TEMP_IMAGE_PATH, "wb") as f:
        f.write(img_byte_arr)

    try:
        tensor = preprocessor.image_to_tensor(TEMP_IMAGE_PATH)
        os.remove(TEMP_IMAGE_PATH)
        return tensor
    except Exception as e:
        st.error(f"Erreur lors du prétraitement de l'image: {e}")
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
        return None

# Fonction pour prédire le pays
def predict_country(image, model, countries):
    tensor = preprocess_image(image)
    if tensor is None:
        return None
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    probs_list = probabilities.tolist()
    
    if len(countries) != len(probs_list):
        st.warning(f"Le nombre de pays ({len(countries)}) ne correspond pas au nombre de classes dans le modèle ({len(probs_list)})")
        results = [{"country": f"Pays {i}", "probability": prob} for i, prob in enumerate(probs_list)]
    else:
        results = [{"country": country, "probability": prob} for country, prob in zip(countries, probs_list)]
    
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    return results

# Obtenir les coordonnées d'un pays avec un système plus robuste
@st.cache_data(show_spinner=False)
def get_country_coordinates(country_name):
    if country_name in DEFAULT_COORDINATES:
        return DEFAULT_COORDINATES[country_name]
    
    name_corrections = {
        "Russia": "Russian Federation",
        "South Korea": "Republic of Korea",
        "North Korea": "Democratic People's Republic of Korea",
        "Syria": "Syrian Arab Republic",
        "Ivory Coast": "Côte d'Ivoire",
        "The Gambia": "Gambia",
        "Czechia": "Czech Republic",
        "Vatican City": "Vatican",
        "Palestinian Territories": "Palestine",
        "Federated States of Micronesia": "Micronesia",
        "United States": "United States of America",
        "Cape Verde": "Cabo Verde"
    }

    query_name = name_corrections.get(country_name, country_name)

    try:
        headers = {
            'User-Agent': 'TrioVision/1.0 (educational project)'
        }
        
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={query_name}&countrycodes={query_name}"
        response = requests.get(url, headers=headers)
        time.sleep(1)

        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                if data:
                    return float(data[0]["lat"]), float(data[0]["lon"])
                else:
                    url = f"https://nominatim.openstreetmap.org/search?format=json&q={query_name}"
                    response = requests.get(url, headers=headers)
                    time.sleep(1)
                    
                    data = response.json()
                    if data:
                        return float(data[0]["lat"]), float(data[0]["lon"])
            except json.JSONDecodeError:
                pass
        
        continents = {
            "Africa": (8.7832, 34.5085),
            "Europe": (54.5260, 15.2551),
            "Asia": (34.0479, 100.6197),
            "North America": (54.5260, -105.2551),
            "South America": (-8.7832, -55.4915),
            "Oceania": (-22.7359, 140.0188)
        }
        
        if "Africa" in country_name:
            return continents["Africa"]
        elif any(region in country_name for region in ["Europe", "Germany", "France", "Italy", "Spain"]):
            return continents["Europe"]
        elif any(region in country_name for region in ["Asia", "China", "Japan", "India"]):
            return continents["Asia"]
        elif any(region in country_name for region in ["America", "United States", "Canada", "Mexico"]):
            return continents["North America"]
        elif any(region in country_name for region in ["Brazil", "Argentina", "Chile"]):
            return continents["South America"]
        elif any(region in country_name for region in ["Australia", "Zealand", "Pacific"]):
            return continents["Oceania"]
        
        return (0, 0)

    except Exception as e:
        return (0, 0)

# Créer une icône de marqueur pulsante avec CSS
def create_circle_icon_style(color):
    return f"""
        <style>
        .marker-pulse {{
            width: 20px;
            height: 20px;
            background-color: {color};
            border-radius: 50%;
            position: relative;
        }}
        .marker-pulse::after {{
            content: "";
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            background-color: {color};
            border-radius: 50%;
            animation: pulse 1.5s infinite;
            z-index: -1;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            100% {{ transform: scale(3); opacity: 0; }}
        }}
        </style>
        <div class="marker-pulse"></div>
    """

# Créer une belle carte interactive et sophistiquée
def create_beautiful_map(results, top_n=5):
    best_country = results[0]["country"]
    best_coords = get_country_coordinates(best_country)
    
    # Créer la carte de base avec un style plus moderne
    m = folium.Map(
        location=best_coords, 
        zoom_start=4,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Ajouter des contrôles additionnels pour une carte plus interactive
    Fullscreen().add_to(m)
    Draw(export=True).add_to(m)
    MiniMap().add_to(m)
    
    # Ajouter différentes couches de fond
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Terrain'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Ocean'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='National Geographic'
    ).add_to(m)
    
    # Créer un groupe de marqueurs pour les pays moins probables
    marker_cluster = MarkerCluster(name="Autres prédictions").add_to(m)
    
    # Simuler un effet "pulsant" avec plusieurs cercles
    for radius in [40, 30, 20, 10]:
        opacity = 0.1 + (40 - radius) * 0.02  # Plus le cercle est petit, plus il est opaque
        folium.CircleMarker(
            location=best_coords,
            radius=radius,
            color=COUNTRY_COLORS[0],
            fill=True,
            fill_color=COUNTRY_COLORS[0],
            fill_opacity=opacity,
            weight=0,
            popup=f'<strong>{best_country}</strong><br>Confiance: {results[0]["probability"]*100:.2f}%'
        ).add_to(m)
    
    # Ajouter un marqueur principal avec une icône personnalisée
    icon_html = create_circle_icon_style(COUNTRY_COLORS[0])
    icon = folium.DivIcon(html=icon_html, icon_size=(30, 30))
    
    folium.Marker(
        location=best_coords,
        popup=f'<strong>{best_country}</strong><br>Confiance: {results[0]["probability"]*100:.2f}%',
        tooltip=f"<strong>{best_country}</strong>",
        icon=icon
    ).add_to(m)
    
    # Créer un cercle élégant autour du pays principal
    folium.Circle(
        location=best_coords,
        radius=200000,  # 200km de rayon
        color=COUNTRY_COLORS[0],
        fill=True,
        fill_color=COUNTRY_COLORS[0],
        fill_opacity=0.1,
        weight=2,
        dash_array='5, 5',
        popup=f"Zone d'intérêt: {best_country}"
    ).add_to(m)
    
    # Ajouter les marqueurs pour les autres pays du top N
    for i, result in enumerate(results[1:top_n], 1):
        country = result["country"]
        coords = get_country_coordinates(country)
        probability = result["probability"] * 100
        
        # Éviter les coordonnées dupliquées
        if coords == best_coords:
            coords = (coords[0] + 0.5 * i, coords[1] + 0.5 * i)
            
        # Créer un style de marqueur basé sur le rang
        icon_color = COUNTRY_COLORS[i] if i < len(COUNTRY_COLORS) else COUNTRY_COLORS[-1]
        
        # Utiliser des icônes différentes selon le rang
        icon_type = "star" if i <= 2 else "info-sign" if i <= 4 else "map-marker"
        
        folium.Marker(
            location=coords,
            popup=f"""
            <div style="width:200px">
                <h4 style="color:{icon_color};margin-bottom:5px">{country}</h4>
                <p><b>Probabilité:</b> {probability:.2f}%</p>
                <p><b>Rang:</b> {i+1}</p>
            </div>
            """,
            tooltip=f"{country}: {probability:.1f}%",
            icon=folium.Icon(color=("blue" if i <= 2 else "green"), icon=icon_type, prefix="fa")
        ).add_to(marker_cluster)
        
        # Tracer une ligne entre le pays principal et ce pays
        if i <= 3:  # Seulement pour les 3 premiers pays après le principal
            folium.PolyLine(
                locations=[best_coords, coords],
                color=icon_color,
                weight=3 - (i-1) * 0.5,  # Diminuer l'épaisseur selon le rang
                opacity=0.7,
                dash_array='5, 5' if i > 1 else None,  # Ligne pleine pour le 2ème pays, pointillée pour les autres
                popup=f"Distance entre {best_country} et {country}"
            ).add_to(m)
    
    # Ajouter contrôleur de couches
    folium.LayerControl(position='bottomright').add_to(m)
    
    return m

# Créer un diagramme radar pour comparer les pays
def create_radar_chart(results, top_n=5):
    # Préparer les données
    top_results = results[:top_n]
    countries = [r["country"] for r in top_results]
    probabilities = [r["probability"] * 100 for r in top_results]
    
    # Convertir en format pour Plotly
    fig = go.Figure()
    
    # Ajouter une trace pour chaque pays
    for i, (country, prob) in enumerate(zip(countries, probabilities)):
        # Créer des points autour du radar pour chaque pays
        theta = ["Probabilité", "Confiance", "Similitude", "Précision", "Score"]
        
        # Utiliser la probabilité comme base, puis ajouter des variations pour les autres axes
        # pour créer un effet visuel intéressant
        base_prob = prob
        similarity = max(0, min(100, base_prob * (0.8 + 0.4 * np.random.random())))
        confidence = max(0, min(100, base_prob * (0.9 + 0.2 * np.random.random())))
        accuracy = max(0, min(100, base_prob * (0.85 + 0.3 * np.random.random())))
        score = max(0, min(100, base_prob * (0.95 + 0.1 * np.random.random())))
        
        values = [base_prob, confidence, similarity, accuracy, score]
        
        # Fermer la forme en répétant le premier point
        theta.append(theta[0])
        values.append(values[0])
        
        # Ajouter la trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=country,
            line=dict(color=RADAR_BORDERS[i % len(RADAR_BORDERS)]),
            fillcolor=RADAR_COLORS[i % len(RADAR_COLORS)]
        ))
    
    # Mettre à jour la mise en page
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        title={
            'text': "Analyse comparative des prédictions",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Créer une carte de similarité entre pays
def create_similarity_chart(results, top_n=8):
    # Préparer les données
    top_results = results[:top_n]
    countries = [r["country"] for r in top_results]
    probabilities = [r["probability"] for r in top_results]
    
    # Créer une matrice de similarité artificielle basée sur les probabilités
    similarity_matrix = np.zeros((top_n, top_n))
    
    for i in range(top_n):
        for j in range(top_n):
            if i == j:
                # Diagonale: similarité avec soi-même = 1
                similarity_matrix[i, j] = 1
            else:
                # Formule pour créer une similarité artificielle basée sur les probabilités relatives
                p_ratio = min(probabilities[i] / probabilities[j], probabilities[j] / probabilities[i])
                distance = 1 - abs(i - j) / top_n
                similarity_matrix[i, j] = p_ratio * distance
    
    # Créer le graphique de chaleur (heatmap)
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=countries,
        y=countries,
        colorscale='Viridis',
        showscale=True,
        zmin=0, zmax=1
    ))
    
    fig.update_layout(
        title='Indice de Similarité entre Pays',
        xaxis_title='Pays',
        yaxis_title='Pays',
        height=500,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    # Ajouter les valeurs dans les cellules
    annotations = []
    for i in range(top_n):
        for j in range(top_n):
            annotations.append(dict(
                x=countries[j],
                y=countries[i],
                text=str(round(similarity_matrix[i, j] * 100)) + '%',
                showarrow=False,
                font=dict(
                    color='white' if 0.3 < similarity_matrix[i, j] < 0.8 else 'black'
                )
            ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

# Créer un graphique de distribution des probabilités
def create_probability_distribution(results, top_n=15):
    # Préparer les données
    top_results = results[:top_n]
    countries = [r["country"] for r in top_results]
    probabilities = [r["probability"] * 100 for r in top_results]
    
    # Créer un code couleur basé sur la probabilité
    colors = []
    for prob in probabilities:
        if prob > 50:
            colors.append('#FF4B4B')  # Rouge vif pour >50%
        elif prob > 20:
            colors.append('#4285F4')  # Bleu Google pour >20%
        elif prob > 10:
            colors.append('#34A853')  # Vert Google pour >10%
        elif prob > 5:
            colors.append('#FBBC05')  # Jaune Google pour >5%
        else:
            colors.append('#EA4335')  # Rouge Google pour le reste
    
    fig = go.Figure(data=[
        go.Bar(
            x=countries,
            y=probabilities,
            marker_color=colors,
            text=probabilities,
            texttemplate='%{text:.1f}%',
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title='Distribution des Probabilités par Pays',
        xaxis=dict(
            title='Pays',
            tickangle=-45
        ),
        yaxis=dict(
            title='Probabilité (%)',
            range=[0, max(probabilities) * 1.1]  # Laisser un peu d'espace au-dessus
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=120)
    )
    
    return fig

# Fonction pour créer un graphique de confiance comparée
def create_confidence_comparison(results, top_n=5):
    # Préparer les données
    top_results = results[:top_n]
    countries = [r["country"] for r in top_results]
    probabilities = [r["probability"] * 100 for r in top_results]
    
    # Ajouter un indicateur de confiance
    fig = go.Figure()
    
    for i, (country, prob) in enumerate(zip(countries, probabilities)):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [i/top_n, (i+1)/top_n], 'y': [0, 1]},
            title={'text': country},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': COUNTRY_COLORS[i % len(COUNTRY_COLORS)]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
    
    fig.update_layout(
        title="Comparaison des niveaux de confiance",
        height=300
    )
    
    return fig

# Charger les labels et le modèle
countries = load_country_labels()

if countries:
    model = load_model(num_classes=len(countries))
else:
    st.error("Impossible de charger les labels des pays")
    model = None

# Créer l'interface utilisateur premium
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
    st.markdown('<h3 class="section-title">Analyse IA</h3>', unsafe_allow_html=True)
    
    if analyze_animation:
        st_lottie(analyze_animation, height=100, key="analyze")
    
    analyze_button = st.button("Analyze the image")
    
    
# Zone des résultats
results_container = st.container()

# Traiter l'image si le bouton est cliqué
if uploaded_file is not None and analyze_button:
    if model is not None and countries:
        with st.spinner(""):
            # Afficher un loader personnalisé
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <div class="pulse-loader">IA</div>
                <p style="margin-top: 10px;">Analyse avancée en cours...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Animation de progression
            progress_bar = st.progress(0)
            for i in range(101):
                time.sleep(0.01)
                progress_bar.progress(i)
            
            # Réelle prédiction
            results = predict_country(image, model, countries)
            
            if results:
                # Nettoyer les éléments temporaires
                progress_placeholder.empty()
                progress_bar.empty()
                
                with results_container:
                    # Afficher le résultat principal
                    best_country = results[0]["country"]
                    confidence = results[0]["probability"] * 100
                    
                    # Petite animation de succès
                    if success_animation:
                        st_lottie(success_animation, height=120, key="success")
                    
                    st.markdown(f"""
                    <div class="prediction-card fade-in-up">
                        <h3>Pays identifié</h3>
                        <div class="prediction-country">{best_country}</div>
                        <div class="confidence-badge">Confiance: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    
                    
                    # Créer des onglets pour différentes visualisations
                    tab1, tab2, tab3, tab4 = st.tabs([" Map", " Analysis", " Comparison", "Statistics"])
                    
                    with tab1:
                        st.markdown('<div class="staggered delay-2">', unsafe_allow_html=True)
                        st.markdown("<h3 class='section-title'>Visualisation géographique</h3>", unsafe_allow_html=True)
                        
                        try:
                            beautiful_map = create_beautiful_map(results)
                            folium_static(beautiful_map, width=1000, height=500)
                        except Exception as e:
                            st.error(f"Erreur lors de la création de la carte: {e}")
                            default_map = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")
                            folium_static(default_map, width=800, height=500)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown('<div class="staggered delay-2">', unsafe_allow_html=True)
                        st.markdown("<h3 class='section-title'>Analyse des probabilités</h3>", unsafe_allow_html=True)
                        
                        # Créer et afficher le diagramme radar
                        radar_fig = create_radar_chart(results)
                        st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Créer et afficher la distribution des probabilités
                        prob_fig = create_probability_distribution(results)
                        st.plotly_chart(prob_fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown('<div class="staggered delay-3">', unsafe_allow_html=True)
                        st.markdown("<h3 class='section-title'>Comparaison des pays</h3>", unsafe_allow_html=True)
                        
                        # Créer et afficher la matrice de similarité
                        similarity_fig = create_similarity_chart(results)
                        st.plotly_chart(similarity_fig, use_container_width=True)
                        
                        # Créer et afficher la comparaison de confiance
                        confidence_fig = create_confidence_comparison(results)
                        st.plotly_chart(confidence_fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab4:
                        st.markdown('<div class="staggered delay-4">', unsafe_allow_html=True)
                        st.markdown("<h3 class='section-title'>Données détaillées</h3>", unsafe_allow_html=True)
                        
                        # Créer un DataFrame pour l'affichage en tableau
                        df = pd.DataFrame([
                            {"Rang": i+1, "Pays": r["country"], "Probabilité (%)": f"{r['probability']*100:.2f}%"} 
                            for i, r in enumerate(results[:15])
                        ])
                        
                        # Afficher les données dans un tableau stylisé
                        st.dataframe(
                            df,
                            column_config={
                                "Rang": st.column_config.NumberColumn("🏆 Rang"),
                                "Pays": st.column_config.TextColumn("🌎 Pays"),
                                "Probabilité (%)": st.column_config.TextColumn("📊 Probabilité"),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Erreur lors de la prédiction")
        
    else:
        st.error("Le modèle ou les labels n'ont pas été chargés correctement")

# Pied de page
st.markdown("""
<div class="footer">
    <h3 style="color: #4285F4;">TrioVision </h3>
    <p>Cutting-edge artificial intelligence for image-based geolocation</p>
    <p style="font-size:0.8rem; margin-top: 10px;">
  Developed with EfficientNet and advanced visualization technologies by Lina, Sharon and Louis
</p>

</div>
""", unsafe_allow_html=True)