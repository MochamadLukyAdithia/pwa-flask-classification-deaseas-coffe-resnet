import os
import logging
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, RATE_LIMIT, DEBUG
from model import get_model
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['DEBUG'] = DEBUG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)
CORS(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@limiter.limit("10/minute")
def predict_api():
    if 'image' not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({"error": "invalid image file"}), 400

    model = get_model()
    result = model.predict(img)

    return jsonify({
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 2),
        "top5": result["top5"]
    })


@app.route('/predict-page', methods=['POST'])
def predict_page():
    logger.info("predict-page endpoint called")
    
    if 'image' not in request.files:
        logger.warning("No image file in request")
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        logger.warning(f"Invalid filename: {file.filename}")
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    try:
        img = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    model = get_model()
    result = model.predict(img)
    
    logger.info(f"Prediction result: {result}")

    return render_template(
        'result.html',
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["top5"]
    )


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "file too large"}), 413


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "too many requests"}), 429
