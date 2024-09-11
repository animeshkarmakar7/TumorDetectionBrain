from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
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

    # Assuming output is two neurons, result[0][0] -> no tumor, result[0][1] -> tumor
    if result[0][1] > result[0][0]:
        return "Yes, tumor detected"
    else:
        return "No tumor detected"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
       
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
      
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
