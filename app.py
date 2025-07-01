from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import requests
import os
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Class names (corrected for Windows path handling)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def load_model():
    try:
        # Windows path handling
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.h5')
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile()  # Recompile for Windows compatibility
        return model
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        return None

def predict_image(model, img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        img = img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize for Windows TF
        predictions = model.predict(img_array)
        return {
            "class": CLASS_NAMES[np.argmax(predictions[0])],
            "confidence": float(np.max(predictions[0])),
            "class_index": int(np.argmax(predictions[0]))
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": str(e)}

# Load model at startup
model = load_model()

@app.route('/')
def home():
    return jsonify({
        "message": "Plant Disease Prediction API (Windows)",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Handle both JSON and form-data
            if request.is_json:
                data = request.get_json()
                image_url = data.get('image_url')
            else:
                image_url = request.form.get('image_url')
            
            img_bytes = None
            
            if not image_url:
                if 'file' not in request.files:
                    return jsonify({"error": "No image provided"}), 400
                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "No selected file"}), 400
                img_bytes = file.read()
            else:
                if not image_url.startswith(('http://', 'https://')):
                    return jsonify({"error": "Invalid URL"}), 400
                try:
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    img_bytes = response.content
                except requests.exceptions.RequestException as e:
                    return jsonify({"error": f"URL fetch error: {str(e)}"}), 400
            
            if not img_bytes:
                return jsonify({"error": "No image data received"}), 400
            
            if model is None:
                return jsonify({"error": "Model not loaded"}), 500
            
            result = predict_image(model, img_bytes)
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "system": "Windows"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Windows-specific configuration
    app.run(
        host='0.0.0.0', 
        port=port,
        threaded=True,  # Better for Windows
        debug=False    # Disable debug mode for production
    )