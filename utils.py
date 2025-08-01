
class_name = ['Apple___Apple_scab',
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
                                'Tomato___healthy']


def load_model():
    try:
        model = tf.keras.models.load_model('../trained_model.h5')
        model._make_predict_function()  # For thread safety
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
        predictions = model.predict(img_array)
        return {
            "class": CLASS_NAMES[np.argmax(predictions[0])],
            "confidence": float(np.max(predictions[0])),
            "class_index": int(np.argmax(predictions[0]))
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": str(e)}