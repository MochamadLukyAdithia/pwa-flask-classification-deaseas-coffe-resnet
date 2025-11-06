import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "saved_model/final_resnet50.pth"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
RATE_LIMIT = "20/minute"
DEBUG = False
