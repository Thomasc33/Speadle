from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
from pathlib import PurePosixPath
from utils import Preprocess, Filters
import re
import cv2
import pathlib
import numpy as np
# from fastai.vision.all import PILImage
from PIL import Image
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.PurePosixPath
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Predict function for CNN Model


def predict_cnn(m, inputs):
    # Cast data to numpy array's so tensorflow stops yelling at me
    inputs = np.array(inputs)
    predictions = []
    labels = [25, 30, 35, 40, 45, 50]

    # Predict
    outputs = m.predict(Preprocess.reshape_data(inputs))

    # Iterate through outputs
    for output in outputs:
        # Round outputs
        output = np.round_(output)

        # Convert one hot encoded output to an integer prediction and add to array
        label_idx = np.argmax(output)
        predictions.append(labels[label_idx])

    # Return predictions
    return predictions


# load the learner
learn_grouped = tf.keras.models.load_model('Models/GroupedCL')
learn_cnn = tf.keras.models.load_model('Models/CNN')


def predict_single(img_file, learn, preprocessData=False):
    'function to take image and return prediction'
    if preprocessData:
        image = np.asarray(Image.open(img_file))
        shaped_image = np.array(cv2.resize(image, dsize=(64,64))).astype('float64')
        shaped_image /= 255.0  # Normalize image
        shaped_image = Filters.grayscale(shaped_image)
        return {'pred': predict_cnn(learn, shaped_image)[0]}
    else:
        labels = [25, 30, 35, 40, 45, 50, 60]
        image = np.asarray(Image.open(img_file))
        shaped_image = np.array(cv2.resize(image, dsize=(224,224)))
        shaped_image = np.reshape(shaped_image, [-1, 224, 224, 3])
        return {'pred': labels[np.argmax(learn.predict(shaped_image))]}


# route for prediction
@app.route('/predict', methods=['POST'])
@cross_origin(origin='*')
def predict():
    selected_model = request.form.get('model')
    if selected_model == 'groupedcl':  # Thomas
        return jsonify(predict_single(request.files['image'], learn_grouped))
    if selected_model == 'cnn':  # kyle
        return jsonify(predict_single(request.files['image'], learn_cnn, preprocessData=True))
    return "model not found", 400


@app.route('/')
def home():
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5972, debug=True, ssl_context=('C:\\Certbot\\live\\a.vibot.tech\\cert.pem', 'C:\\Certbot\\live\\a.vibot.tech\\privkey.pem'))
