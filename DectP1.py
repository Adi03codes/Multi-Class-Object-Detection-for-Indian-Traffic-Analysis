# backend/detector.py
import onnxruntime as ort
import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.4, iou_thresh=0.5):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img /= 255.0
        return np.expand_dims(img, axis=0)

    def detect(self, image):
        input_tensor = self.preprocess(image)
        preds = self.session.run(None, {self.input_name: input_tensor})[0]
        detections = self.postprocess(preds[0], image.shape[:2])
        return detections

    def postprocess(self, preds, shape):
        boxes = []
        for pred in preds:
            conf = pred[4]
            if conf > self.conf_thresh:
                x1, y1, x2, y2 = (pred[0:4] * [shape[1], shape[0], shape[1], shape[0]]).astype(int)
                label = int(pred[5])
                boxes.append({'bbox': [x1, y1, x2, y2], 'label': label, 'confidence': float(conf)})
        return boxes
