# Multi-Class-Object-Detection-for-Indian-Traffic-Analysis
 This project develops a deep learning-based multi-class object detection system tailored to Indian traffic scenarios. It detects and classifies diverse objects such as vehicles, pedestrians, potholes, and traffic signs in real-time to support AI-powered traffic management and mobility solutions.









## Project Overview

This project develops a deep learning-based multi-class object detection system tailored to Indian traffic scenarios. It detects and classifies diverse objects such as vehicles, pedestrians, potholes, and traffic signs in real-time to support AI-powered traffic management and mobility solutions.

Unique challenges addressed include handling traffic congestion, mixed vehicle types, poor lighting conditions, occlusions, shadows, and noisy images typical of Indian roads.

---

## Features

- Custom YOLOv8 model trained on annotated datasets reflecting Indian traffic diversity.
- Advanced image preprocessing and augmentation using Albumentations for robustness.
- Real-time detection dashboard with Flask backend and React.js frontend.
- Low-latency inference deployed on AWS EC2 using ONNX Runtime.
- Extensible modular codebase for training, inference, and deployment.

---

## Technologies Used

| Category       | Tools & Libraries                    |
| -------------- | ---------------------------------- |
| Programming    | Python, JavaScript                  |
| Deep Learning  | PyTorch, Ultralytics YOLOv8        |
| Image Processing | OpenCV, Albumentations            |
| Backend        | Flask                             |
| Frontend       | React.js                         |
| Deployment     | AWS EC2, Docker, ONNX Runtime      |

---

## Repository Structure

/indian-traffic-object-detection/
│
├── data/ # Dataset images and annotations
│ ├── images/ # Raw and test images
│ ├── annotations/ # YOLO-format label files
│ └── indian_traffic.yaml # Dataset config with classes and paths
│
├── models/ # Training, inference scripts and configs
│ ├── train.py
│ ├── inference.py
│ └── yolov8_config.yaml
│
├── dashboard/ # Web dashboard source code
│ ├── backend/ # Flask API server
│ └── frontend/ # React.js app
│
├── deployment/ # Deployment and docker files
│ ├── Dockerfile
│ └── inference_server.py
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # License file

yaml
Copy
Edit

---

## Dataset Classes

| Class ID | Object Type     |
| -------- | --------------- |
| 0        | Vehicle         |
| 1        | Pedestrian      |
| 2        | Pothole         |
| 3        | Traffic Sign    |
