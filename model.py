
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from config import MODEL_PATH

class PlantDiseaseModel:
    def __init__(self, model_path=None):
        model_path = model_path or MODEL_PATH
        
        if os.path.isdir(model_path):
            try:
                self.model = load_model(model_path)
            except Exception:
                
                self.model = load_model(os.path.join(model_path, 'model.h5'))
        elif os.path.isfile(model_path):
            self.model = load_model(model_path)
        else:
            raise ValueError(f"Model not found at {model_path}")
        self.input_size = (224, 224)
        
        self.class_indices = None

    def preprocess_image(self, pil_image: Image.Image):
        pil_image = pil_image.convert("RGB")
        pil_image = pil_image.resize(self.input_size)
        arr = img_to_array(pil_image) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, pil_image: Image.Image):
        x = self.preprocess_image(pil_image)
        probs = self.model.predict(x)[0]
        top_idx = int(np.argmax(probs))
        return {
            "pred_index": top_idx,
            "probabilities": probs.tolist()
        }


_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantDiseaseModel()
    return _model_instance
