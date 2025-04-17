# Global variables configuration for the application
import os
from pathlib import Path

# Get the root directory of the project (goes one level up from src)
ROOT_DIR = Path(__file__).parent.parent

# Relative paths based on the project root
LABELS_PATH = os.path.join(ROOT_DIR, "data", "country_labels_total_essaie.json")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_ef_model_2025-04-16_12-05-34.pth")

# Path for temporary files
TEMP_IMAGE_PATH = os.path.join(ROOT_DIR, "temp_image.jpg")

# Default coordinates for countries
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

# Country colors (modern palette)
COUNTRY_COLORS = ["#FF4B4B", "#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8AB4F8", "#137333", "#F29900", "#C5221F", "#669DF6"]

# Color gradients for the radar chart
RADAR_COLORS = ["rgba(255, 75, 75, 0.7)", "rgba(66, 133, 244, 0.7)", "rgba(52, 168, 83, 0.7)", "rgba(251, 188, 5, 0.7)", "rgba(234, 67, 53, 0.7)"]
RADAR_BORDERS = ["rgba(255, 75, 75, 1)", "rgba(66, 133, 244, 1)", "rgba(52, 168, 83, 1)", "rgba(251, 188, 5, 1)", "rgba(234, 67, 53, 1)"]

# Lottie animation URLs
LOTTIE_ANIMATIONS = {
    "globe": "https://assets1.lottiefiles.com/packages/lf20_JFzAYh.json",
    "analyze": "https://assets10.lottiefiles.com/packages/lf20_wrffqrdf.json",
    "upload": "https://assets4.lottiefiles.com/packages/lf20_zzujjt5k.json",
    "success": "https://assets1.lottiefiles.com/packages/lf20_jTrZKu.json"
}

# Streamlit page configuration
APP_CONFIG = {
    "page_title": "TrioVision - Country Recognition AI",
    "page_icon": "🌍",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}
