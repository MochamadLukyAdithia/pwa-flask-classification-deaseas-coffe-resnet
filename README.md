# Plant Disease PWA (Flask)

## Setup (local)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Train (generate saved_model/)
python train.py

## Run dev
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

## Build docker
docker build -t plant-detect .
docker run -p 5000:5000 -v $(pwd)/saved_model:/app/saved_model plant-detect
# pwa-flask-classification-deaseas-coffe-resnet
