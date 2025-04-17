def get_custom_css():
    return """
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
</style>
"""