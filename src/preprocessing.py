import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict

class Preprocessing:
    def __init__(self):
        """
        Preprocessing class that allows to:
        - convert jpg images into PyTorch tensors
        - convert image paths into dict of PyTorch tensors
        - stack tensors along with their target and label 
          to be used in PyTorch
        """
        pass
    
    def image_to_tensor(self, image_path):
        """
        Load a jpg image and convert it into a PyTorch tensor,
        apply transformations on the tensor
        
        Args:
            image_path (str): path to the jpg image
        
        Returns:
            torch.Tensor: PyTorch tensor
        """

        image = Image.open(image_path)
        
        # Transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # standard size
            transforms.ToTensor(),          # convert pixel values between 0 and 1
            transforms.Normalize(           # normalization
                mean=[0.485, 0.456, 0.406],  # RGB mean for ImageNet
                std=[0.229, 0.224, 0.225]    # RGB std for ImageNet
            )
        ])
        
        tensor = transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def process_images_batch(self, batch_paths, batch_labels):
        """
        Process a batch of images with their associated labels
        
        Args:
            batch_paths: list of image paths
            batch_labels: list of corresponding labels
            
        Returns:
            dict: dictionary with labels as keys and lists of tensors as values
        """
        dict_tensor = defaultdict(list)  # dict of tensors

        for path, label in zip(batch_paths, batch_labels):
            tensor = self.image_to_tensor(path)  # from image path to tensor
            dict_tensor[label].append(tensor)
           
        return dict_tensor
    
    def create_label_mapping(self, labels):
        """
        Create a mapping between textual labels and indices
        Example:
        {'Australia': 0,
         'Canada': 1,
         'China': 2,
         'India': 3,
         'Portugal': 4,
         'United Kingdom': 5,
         'United States': 6}
        and the reverse
        
        Args:
            labels: list of textual labels (e.g., countries)
            
        Returns:
            tuple: (label_to_idx, idx_to_label) mapping dictionaries
        """
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def stack_tensors(self, tensors_by_label):
        """
        Stack tensors and create corresponding label tensors
        
        Args:
            tensors_by_label: dict {label: list_of_tensors}
            
        Returns:
            tuple: (stacked tensors, numeric labels, label mapping)
        """
        # Create mapping from labels to indices and vice versa
        unique_labels = list(tensors_by_label.keys())
        label_to_idx, idx_to_label = self.create_label_mapping(unique_labels)
        
        all_tensors = []  # store tensors
        all_labels = []   # store labels
        
        for label, tensors in tensors_by_label.items():
            all_tensors.extend(tensors)  # add tensor
            all_labels.extend([label_to_idx[label]] * len(tensors))  # add corresponding label
            
        # Stack tensors
        try:
            images_tensor = torch.stack(all_tensors)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # labels as long tensor
            return images_tensor, labels_tensor, idx_to_label
        except Exception as e:
            print(f"Error while stacking tensors: {str(e)}")
            return None, None, None
    
    def store_image_tensors(self, df, image_path_column='image_path', label_column='country', batch_size=None):
        """
        Wrapper method that
        transforms image paths into tensors for PyTorch usage
        
        Args:
            df: dataframe containing image paths and labels
            image_path_column: column with image paths
            label_column: label column (e.g., countries)
            batch_size: batch size
            
        Returns:
            tuple: (image_tensors, encoded_labels, label_mapping)
        """
       
        image_paths = df[image_path_column].tolist()  # image paths
        labels = df[label_column].tolist()  # countries
        
        # Batch size
        if batch_size is None:
            batch_size = len(image_paths)
        
        # Dict to store tensors by label
        tensors_by_label = defaultdict(list)
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Process one batch
            batch_result = self.process_images_batch(batch_paths, batch_labels)
            
            # Merge batch results
            for label, tensors in batch_result.items():
                tensors_by_label[label].extend(tensors)
        
        return self.stack_tensors(tensors_by_label)  # stack tensors
