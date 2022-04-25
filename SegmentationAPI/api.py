import flask
from flask import request
import cv2
import numpy as np
import os

from segmentation import SETR_MLA
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/segmentation', methods=['GET'])
def main():
	return "Welcome to segmentation app"

@app.route('/segmentation/test', methods=['GET'])
def test():
	return {"test": "segmentation app is running"}

@app.route('/segmentation/test/inference', methods=['GET'])
def test_inference():
    return {"test_inference":  SETR_MLA.predict(os.path.abspath('imgs/test.jpg'))}

@app.route('/segmentation/inference', methods=['POST'])
def inference():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    imgBGR = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    prediction = SETR_MLA.predict(imgBGR)
    visualization = None
    return {"prediction": prediction, "visualization": visualization}


app.run(host='0.0.0.0', port=5000)




