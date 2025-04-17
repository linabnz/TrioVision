import streamlit.web.cli as stcli
import sys
import os

# Chemin vers le fichier app.py dans src
app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())