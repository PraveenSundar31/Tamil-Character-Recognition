

import os
import cv2
from flask import Flask, render_template, request, redirect, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
from tensorflow.keras.layers import Normalization


# FLASK APP INSTANCE

app = Flask(__name__, template_folder = 'templates', static_folder = 'static')

# FLASK APP CONFIG

app.secret_key = "secret_key"
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# INPUT CONFIGURATION

# valid_format = ['png', 'jpg', 'jpeg']
# image_size = (120, 120)


# DEFINING THE CALSSES

# classes = {0:'EOSINOPHIL', 1:'LYMPHOCYTE', 2:'MONOCYTE', 3:'NEUTROPHIL'}

df = pd.read_csv('TamilChar.csv', header=0)
unicode_list = df["Unicode"].tolist()
char_list = []

for element in unicode_list:
    code_list = element.split()
    chars_together = ""
    for code in code_list:
        hex_str = "0x" + code
        char_int = int(hex_str, 16)
        character = chr(char_int)
        chars_together += character
    char_list.append(chars_together)

train_dir = 'processed - png (155)/train'

class_dict = dict()
classes = []


for folder in os.listdir(train_dir):
  index = int(folder)
  char = char_list[index]
  class_dict[index] = char


sorted_dict = sorted(class_dict)
for i in sorted_dict:
  classes.append(class_dict[i])
# print(classes)
# print(class_dict)

 

image_size = (74, 74)

# USER-DEFINED FUNCTIONS FOR INPUT IMAGE PRE-PROCESSING

def preprocess(img):
        converted = img.convert("L")
        inverted = ImageOps.invert(converted)
        thick = inverted.filter(ImageFilter.MaxFilter(5))
        ratio = 48.0 / max(thick.size)
        new_size = tuple([int(round(x*ratio)) for x in thick.size])
        res = thick.resize(new_size, Image.LANCZOS)
        
        arr = np.asarray(res)
        com = ndimage.measurements.center_of_mass(arr)
        result = Image.new("L", (74, 74))
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
        result.paste(res, box)
        return result

images = []

cv_img = cv2.imread("processed - png (155)/test/6/00385.png")
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(cv_img)
img = preprocess(img)
img = np.asarray(img)

images.append(img)


images=np.array(images, dtype = 'float32')
images = images / 255.0
layer = Normalization(axis = None)
layer.adapt(images)



def predict(image_path):

    # LOADING THE TRAINED MODEL
    model = load_model('static\model', custom_objects = None, compile = True)

    pred = model.predict(images)

    pred = pred.argmax(axis = -1)

    return class_dict[int(pred)]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def upload_image():
        
    result = predict("processed - png (155)/test/6/00385.png")
    return render_template('index.html', result = result)


if __name__ == "__main__":
    app.run(debug = True)




