from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with your secret key
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
    img = np.expand_dims(img, axis=0)  # Expanding to match the input shape of the model (1, 64, 64, 3)

    # Predict the class
    result = model.predict(img)

    if result[0][0] > 0.5:
        return "Yes, tumor detected"
    else:
        return "No tumor detected"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = model_predict(file_path)
            return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
