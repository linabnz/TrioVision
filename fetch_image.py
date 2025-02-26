from pathlib import Path

def fetch_images(image_ids, dossier_base):
    """
    Trouve les chemins de plusieurs images à partir de leurs IDs en respectant la structure des dossiers.

    Args:
        image_ids (list): Liste des identifiants uniques des images.
        dossier_base (str): Le dossier racine où sont stockées les images.

    Returns:
        dict: Dictionnaire {image_id: chemin_image ou message d'erreur}.
    """
    
    resultats = {}

    for image_id in image_ids:
        if len(image_id) < 3:
            resultats[image_id] = f"⚠️ ID trop court : {image_id}"
            continue
        chemin_image = Path(dossier_base) / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"

        # on ne garde que les images trouvées
        if chemin_image.exists():
            resultats[image_id] = str(chemin_image)
        # else:
        #     resultats[image_id] = f"⚠️ Image introuvable : {chemin_image}"

    return resultats

if __name__ == "__main__":
    dossier_images = "data/train/images"   
    image_ids = ["000a0aee5e90cbaf", "0123456789abcdef", "abcdef1234567890"] 

    chemins_images = fetch_images(image_ids, dossier_images)

 
    for img_id, chemin in chemins_images.items():
        print(f"{img_id} → {chemin}")
