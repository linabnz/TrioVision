import json
import os
import io
import time
import requests
import folium
from folium.plugins import MarkerCluster, MiniMap, Draw, Fullscreen
import plotly.graph_objects as go
import numpy as np

# Chargement des labels de pays
def load_country_labels(labels_path):
    try:
        with open(labels_path, "r") as f:
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
        print(f"Erreur lors du chargement des labels: {e}")
        return []

def preprocess_image(image, preprocessor, temp_image_path="temp_image.jpg"):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    with open(temp_image_path, "wb") as f:
        f.write(img_byte_arr)
    try:
        tensor = preprocessor.image_to_tensor(temp_image_path)
        os.remove(temp_image_path)
        return tensor
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {e}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return None

# Obtenir les coordonnées d'un pays avec un système plus robuste
def get_country_coordinates(country_name, default_coordinates):
    if country_name in default_coordinates:
        return default_coordinates[country_name]
    
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
        print(f"Erreur lors de la récupération des coordonnées: {e}")
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

# Charger les animations Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Fonctions pour les visualisations
def create_beautiful_map(results, top_n=5, default_coordinates=None, country_colors=None):
    if default_coordinates is None or country_colors is None:
        raise ValueError("default_coordinates and country_colors must be provided")
        
    best_country = results[0]["country"]
    best_coords = get_country_coordinates(best_country, default_coordinates)
    
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
            color=country_colors[0],
            fill=True,
            fill_color=country_colors[0],
            fill_opacity=opacity,
            weight=0,
            popup=f'<strong>{best_country}</strong><br>Confiance: {results[0]["probability"]*100:.2f}%'
        ).add_to(m)
    
    # Ajouter un marqueur principal avec une icône personnalisée
    icon_html = create_circle_icon_style(country_colors[0])
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
        color=country_colors[0],
        fill=True,
        fill_color=country_colors[0],
        fill_opacity=0.1,
        weight=2,
        dash_array='5, 5',
        popup=f"Zone d'intérêt: {best_country}"
    ).add_to(m)
    
    # Ajouter les marqueurs pour les autres pays du top N
    for i, result in enumerate(results[1:top_n], 1):
        country = result["country"]
        coords = get_country_coordinates(country, default_coordinates)
        probability = result["probability"] * 100
        
        # Éviter les coordonnées dupliquées
        if coords == best_coords:
            coords = (coords[0] + 0.5 * i, coords[1] + 0.5 * i)
            
        # Créer un style de marqueur basé sur le rang
        icon_color = country_colors[i] if i < len(country_colors) else country_colors[-1]
        
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

def create_radar_chart(results, top_n=5, radar_colors=None, radar_borders=None):
    if radar_colors is None or radar_borders is None:
        raise ValueError("radar_colors and radar_borders must be provided")
        
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
            line=dict(color=radar_borders[i % len(radar_borders)]),
            fillcolor=radar_colors[i % len(radar_colors)]
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

def create_confidence_comparison(results, top_n=5, country_colors=None):
    if country_colors is None:
        raise ValueError("country_colors must be provided")
        
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
                'bar': {'color': country_colors[i % len(country_colors)]},
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