"""
Dezip des fichiers .tar
et placement des folders dans une architecture logique
Images placées dans les folders selon le nom
"""
import tarfile
from pathlib import Path

def decompresser_fichiers_tar(dossier_source, dossier_destination=None):
    """
    Décompresse tous les fichiers .tar d'un folder à un autre
    
    Args:
        path_source (str): chemin du dossier contenant les fichiers .tar
        path_destination (str, optional): chemin du dossier où extraire les fichiers
    
    Returns:
        None
    """
    dossier_source = Path(dossier_source)
    dossier_destination = Path(dossier_destination)
    
    # création du folder de destination
    if not dossier_destination.exists():
        dossier_destination.mkdir(parents=True)
    
    fichiers_decompresses = []
    
    for fichier in dossier_source.glob('*.tar'):
        print(f"décompression et récupération de {fichier.name} :")
        
        try:
            with tarfile.open(fichier, 'r') as tar:
                tar.extractall(path=dossier_destination)
            fichiers_decompresses.append(str(fichier))
            print(f"  → Extrait dans {dossier_destination}")
        except Exception as e:
            print(f"  X Erreur lors de la décompression de {fichier.name}: {e}")
    
if __name__ == "__main__":
    # lancement du script
    input_folder = 'data/train' # chemin d'entree
    output_folder = 'data/train' # chemin de sortie
    fichiers = decompresser_fichiers_tar(input_folder, output_folder)
