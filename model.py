import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import base64
from io import BytesIO
from config import MODEL_PATH


class GradCAM:
    """Grad-CAM implementation for ResNet50."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def cam_to_bbox(cam, orig_w, orig_h, threshold=0.5):
    """Convert Grad-CAM heatmap to bounding box on original image dimensions."""
    import cv2

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_resized = np.array(Image.fromarray(cam_uint8).resize((orig_w, orig_h), Image.BILINEAR))

    binary = (cam_resized >= threshold * 255).astype(np.uint8)

    # Find contours to get bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, cam_resized

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h), cam_resized


def overlay_heatmap(pil_image, cam_resized, bbox, alpha=0.4):
    """Overlay Grad-CAM heatmap and bounding box on image."""
    import cv2

    orig_np = np.array(pil_image.convert("RGB"))
    h, w = orig_np.shape[:2]

    # Colormap heatmap
    cam_norm = cam_resized.astype(np.float32)
    heatmap_bgr = cv2.applyColorMap((cam_norm).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    blended = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    result_img = Image.fromarray(blended)

    # Draw bounding box
    if bbox:
        draw = ImageDraw.Draw(result_img)
        draw.rectangle(bbox, outline=(255, 50, 50), width=4)

    return result_img


def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
            self._setup_gradcam()
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
        self._setup_gradcam()

        self.class_names = ['Miner', 'Cercospora', 'Phoma', 'Rust', 'Health']

        print("[INFO] Model loaded and ready on", self.device)

    def _setup_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _setup_gradcam(self):
        """Setup Grad-CAM targeting the last ResNet layer."""
        try:
            target_layer = self.model.layer4[-1]
            self.gradcam = GradCAM(self.model, target_layer)
        except Exception as e:
            print(f"[WARNING] Grad-CAM setup failed: {e}")
            self.gradcam = None

    def preprocess_image(self, pil_image: Image.Image):
        pil_image = pil_image.convert("RGB")
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def predict(self, pil_image: Image.Image, generate_bbox=False):
        pil_image = pil_image.convert("RGB")
        orig_w, orig_h = pil_image.size

        x = self.preprocess_image(pil_image)

        # Need grad for Grad-CAM
        x.requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()[0]

        top5_idx = np.argsort(probs)[::-1][:5]
        top5_labels = [self.class_names[i] for i in top5_idx]
        top5_probs = [float(probs[i]) for i in top5_idx]
        predicted_idx = int(top5_idx[0])

        result = {
            "label": top5_labels[0],
            "confidence": top5_probs[0],
            "top5": dict(zip(top5_labels, [round(p * 100, 2) for p in top5_probs])),
            "bbox": None,
            "annotated_image": None,
        }

        # Generate Grad-CAM + bbox
        if generate_bbox and self.gradcam is not None:
            try:
                cam = self.gradcam.generate(x, predicted_idx)
                bbox, cam_resized = cam_to_bbox(cam, orig_w, orig_h, threshold=0.45)
                annotated = overlay_heatmap(pil_image, cam_resized, bbox)
                result["bbox"] = list(bbox) if bbox else None
                result["annotated_image"] = image_to_base64(annotated)
            except Exception as e:
                print(f"[WARNING] Grad-CAM generation failed: {e}")

        return result


_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = PlantDiseaseModel()
    return _model_instance