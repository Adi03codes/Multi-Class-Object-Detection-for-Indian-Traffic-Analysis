
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from detector import ObjectDetector

app = Flask(__name__)
CORS(app)

detector = ObjectDetector("yolov8_model/best.onnx")

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = detector.detect(img)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

