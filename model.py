import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from config import MODEL_PATH

class PlantDiseaseModel:
    def __init__(self, model_path=None):
        model_path = model_path or MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        print(f"[INFO] Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model file: {e}")

        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            
            state_dict = checkpoint
        else:
            
            print("[INFO] Detected a full model file. Loading directly.")
            model = checkpoint
            model.to(self.device)
            model.eval()
            self.model = model
            self._setup_transform()
            return

        
        num_classes = 5  
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")

        model.to(self.device)
        model.eval()
        self.model = model

        
        self._setup_transform()

        
        self.class_names = ['Miner', 'Cercospora', 'Phoma', 'Rust', 'Health']

        print("[INFO] Model loaded and ready on", self.device)

    def _setup_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, pil_image: Image.Image):
        pil_image = pil_image.convert("RGB")
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def predict(self, pil_image: Image.Image):
        x = self.preprocess_image(pil_image)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top5_idx = np.argsort(probs)[::-1][:5]
        top5_labels = [self.class_names[i] for i in top5_idx]
        top5_probs = [float(probs[i]) for i in top5_idx]

        return {
            "label": top5_labels[0],
            "confidence": top5_probs[0],
            "top5": dict(zip(top5_labels, [round(p * 100, 2) for p in top5_probs]))
        }



_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantDiseaseModel()
    return _model_instance
