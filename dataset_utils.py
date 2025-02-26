import tarfile
import sys
from pathlib import Path

class Installation:
    def __init__(self, dossier_base="data/train/"):
        """Initialise avec le dossier contenant les images."""
        self.dossier_base = Path(dossier_base)

    def decompresser_fichiers_tar(self, dossier_source, dossier_destination=None):
        """Décompresse tous les fichiers .tar d'un dossier vers un autre."""
        dossier_source = Path(dossier_source)
        dossier_destination = Path(dossier_destination or self.dossier_base)
        dossier_destination.mkdir(parents=True, exist_ok=True)

        for fichier in dossier_source.glob("*.tar"):
            print(f"Décompression : {fichier.name}")
            with tarfile.open(fichier, "r") as tar:
                tar.extractall(path=dossier_destination)
            print(f"✅ Extrait dans {dossier_destination}")

    def fetch_images(self, image_ids):
        """Trouve les chemins des images en respectant la structure des dossiers."""
        return {
            image_id: str(self.dossier_base / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg")
            if (self.dossier_base / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg").exists()
            else None
            for image_id in image_ids
        }

if __name__ == "__main__":
    install = Installation("data/train/")

    if len(sys.argv) > 1:
        mode, *args = sys.argv[1:]
        if mode == "decompress" and len(args) == 2:
            install.decompresser_fichiers_tar(*args)
        elif mode == "fetch" and args:
            for img_id, chemin in install.fetch_images(args).items():
                print(f"{img_id} → {chemin if chemin else '❌ Introuvable'}")
        else:
            print("Usage:\n  python dataset_utils.py decompress <source> <destination>\n  python dataset_utils.py fetch <image_id_1> <image_id_2> ...")
