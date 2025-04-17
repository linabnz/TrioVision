import torch
import torch.nn as nn
from torchvision import models

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

def load_model(model_path, num_classes):
    try:
        model = EfficientNetModel(num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
            
        model.eval()
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None

def predict_country(image, model, countries, preprocess_function):
    tensor = preprocess_function(image)
    if tensor is None:
        return None
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    probs_list = probabilities.tolist()
    
    if len(countries) != len(probs_list):
        print(f"Le nombre de pays ({len(countries)}) ne correspond pas au nombre de classes dans le modèle ({len(probs_list)})")
        results = [{"country": f"Pays {i}", "probability": prob} for i, prob in enumerate(probs_list)]
    else:
        results = [{"country": country, "probability": prob} for country, prob in zip(countries, probs_list)]
    
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    return results