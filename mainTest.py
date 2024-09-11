import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('D:\\tumour detection\\pred\\pred55.jpg')

img=Image.fromarray(image)

img=img.resize((64 , 64))

img=np.array(img)

img = img / 255.0
img = np.expand_dims(img, axis=0)  # Expanding to match the input shape of the model (1, 64, 64, 3)

# Predict the class
result = model.predict(img)

# For binary classification
if result[0][1] > result[0][0]:
    print("Yes, tumor detected")
else:
    print("No tumor detected")



