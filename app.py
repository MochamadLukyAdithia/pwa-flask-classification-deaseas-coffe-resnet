import os
import logging
from io import BytesIO
from flask import (
    Flask, render_template, request, jsonify,
)
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

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)
CORS(app)

def allowed_file(filename):
    """Cek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@limiter.limit("10/minute")  
def predict():
    """Endpoint API JSON untuk prediksi gambar."""
    if 'image' not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image file"}), 400

    model = get_model()
    result = model.predict(img)

    return jsonify({
        "prediction_index": result["pred_index"],
        "probabilities": result["probabilities"]
    })


@app.route('/predict-page', methods=['POST'])
def predict_page():
    """Endpoint untuk render HTML hasil prediksi."""
    if 'image' not in request.files:
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    try:
        img_bytes = file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return render_template('result.html', label=None, confidence=0, probabilities=None)

    model = get_model()
    result = model.predict(img)

    pred_index = result["pred_index"]
    probs = result["probabilities"]
    confidence = probs[pred_index]

    
    label = f"Kelas {pred_index}"

    return render_template(
        'result.html',
        label=label,
        confidence=confidence,
        probabilities=probs
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})



@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "file too large"}), 413


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "too many requests"}), 429



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
