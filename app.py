from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

app = Flask(__name__)


# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models
vgg16_model = load_model('vgg16_model.h5')
svm_model = joblib.load('svm_model.pkl')

# Prediction function
def predict_image(image_path):
    img_size = 224
    class_names = ['Cat', 'Dog']
    
    # Preprocess image
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = vgg16_model.predict(img_array)
    features_flat = features.reshape((1, 7 * 7 * 512))
    
    # Predict class
    prediction = svm_model.predict(features_flat)
    predicted_class = class_names[int(prediction[0])]
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    
    # Make prediction
    predicted_class = predict_image(file_path)
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
