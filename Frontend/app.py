from flask import Flask, request, jsonify
import os
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

app.secret_key = "your_secret_key"
model = load_model('D:\\tumour detection\\BrainTumor10EpochsCategorical.h5')

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def model_predict(img_path):
    image = cv2.imread(img_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)

    if result[0][1] > result[0][0]:
        return "Yes, tumor detected"
    else:
        return "No tumor detected"

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = model_predict(file_path)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
