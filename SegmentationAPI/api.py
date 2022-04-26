import flask
from flask import request
import cv2
import numpy as np
import os
from segmentation import utils


from segmentation import SETR_MLA
app = flask.Flask(__name__)
app.config["DEBUG"] = True

### Tests
@app.route('/segmentation', methods=['GET'])
def main():
	return "Welcome to segmentation app"

@app.route('/segmentation/test', methods=['GET'])
def test():
	return {"test": "segmentation app is running"}

@app.route('/segmentation/test/inference', methods=['GET'])
def test_inference():
    return {"test_inference":  SETR_MLA.predict(os.path.abspath('imgs/test.jpg'))}

### Classes
@app.route('/segmentation/classes/id2label', methods=['GET'])
def get_id2label():
    return {'id2label': utils.id2label}

@app.route('/segmentation/classes/label2id', methods=['GET'])
def get_label2id():
    return {'label2id': utils.label2id}

### Inference And Visualization
@app.route('/segmentation/inference', methods=['POST'])
def inference():
    file = request.files['image']
    if (file.filename == ""):
        return '''<h1>No Image Received</h1>''', 500
    byte_arr = file.read()
    img_numpy = np.frombuffer(byte_arr, np.uint8)
    imgBGR = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)
    prediction = SETR_MLA.predict(imgBGR)
    base64Image = SETR_MLA.visualization.getVisualization(prediction, base64=True)
    return {"prediction": prediction, "visualization": base64Image}

app.run(host='0.0.0.0', port=5000)




