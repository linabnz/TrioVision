import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict

class Preprocessing:
    def __init__(self):
        """
        Classe Preprocessing qui permet de :
        - convertir les images jpg en tenseur pytorch
        - convertir les chemins d'images en dict tenseurs pytorchs
        - empiler les tenseurs pytorchs avec leur target et label 
          pour être exploiter dans pytorch
        """
        pass
    
    def image_to_tensor(self, image_path):
        """
        charge une image jpg et convertit en tenseur pytorch
        applique des transformations sur le tenseur
        
        Args:
            image_path (str): chemin de l'image jpg
        
        Returns:
            torch.Tensor: tesneur pytorch
        """

        image = Image.open(image_path)
        
        # transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # taille standard
            transforms.ToTensor(),          # conversion des valeurs entre 0 et 1
            transforms.Normalize(           # normalisation
                mean=[0.485, 0.456, 0.406],  # moyenne RGB pour ImageNet
                std=[0.229, 0.224, 0.225]     # std RGB pour ImageNet
            )
        ])
        
        tensor = transform(image)
        
        # ajout une dimension de batch
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def process_images_batch(self, batch_paths, batch_labels):
        """
        Traite un batch d'images et leurs labels associés
        
        Args:
            batch_paths: liste des chemins d'images
            batch_labels: lsite des labels correspondants
            
        Returns:
            dict: dict avec les labels comme key et listes de tenseurs comme values
        """
        dict_tensor = defaultdict(list) # dict des tenseurs

        for path, label in zip(batch_paths, batch_labels):
            tensor = self.image_to_tensor(path) # from image path to tensor
            dict_tensor[label].append(tensor)
           
        return dict_tensor
    
    def create_label_mapping(self, labels):
        """
        crée un mapping entre les labels textuels et les index
        exemple :
        {'Australia': 0,
        'Canada': 1,
        'China': 2,
        'India': 3,
        'Portugal': 4,
        'United Kingdom': 5,
        'United States': 6}
        et inversement
        Args:
            labels: liste de labels textuels (ici : pays)
            
        Returns:
            tuple: (label_to_idx, idx_to_label) dictionnaires de mapping
        """
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def stack_tensors(self, tensors_by_label):
        """
        empile les tenseurs et crée les labels correspondants
        
        Args:
            tensors_by_label: dict {label: liste_de_tenseurs}
            
        Returns:
            tuple: (tenseurs empilés, labels numériques, mapping labels)
        """
        # creation du mapping des labels vers les index et des index vers les labels
        unique_labels = list(tensors_by_label.keys())
        label_to_idx, idx_to_label = self.create_label_mapping(unique_labels)
        
        all_tensors = [] # stockage des tenseurs
        all_labels = [] # stockage des labels
        
        for label, tensors in tensors_by_label.items():
            all_tensors.extend(tensors) # ajoute le tenseur
            all_labels.extend([label_to_idx[label]] * len(tensors)) # ajoute le label
            
        # empilement des tenseurs
        try:
            images_tensor = torch.stack(all_tensors)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long) # labels en tenseur de type long
            return images_tensor, labels_tensor, idx_to_label
        except Exception as e:
            print(f"Erreur lors de l'empilement des tensors: {str(e)}")
            return None, None, None
    
    def store_image_tensors(self, df, image_path_column='image_path', label_column='country', batch_size=None):
        """
        méthode wrapper qui
        transforme les images path en tenseurs pour utilisation pytorch
        
        Args:
            df: df contenant les image path et labels
            image_path_column: colonne des liens d'images
            label_column: colonne label (pays)
            batch_size: taille du batch
            
        Returns:
            tuple: (tenseurs_images, labels_encodés, mapping_labels)
        """
       
        image_paths = df[image_path_column].tolist() # image paths
        labels = df[label_column].tolist() # countries
        
        # taille du batch
        if batch_size is None:
            batch_size = len(image_paths)
        
        # dict pour stocker les tenseurs par label
        tensors_by_label = defaultdict(list)
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # traitement d'un batch
            batch_result = self.process_images_batch(batch_paths, batch_labels)
            
            # fusion des resultats
            for label, tensors in batch_result.items():
                tensors_by_label[label].extend(tensors)
        
        return self.stack_tensors(tensors_by_label) # stacking des tenseurs