# Face Mask Detection using Deep Learning

## Overview

This project is a Face Mask Detection system developed using Deep Learning and Computer Vision techniques. The model can detect whether a person is wearing a face mask or not in real-time using a webcam feed.

The system classifies faces into two categories:

- With Mask 😷
- Without Mask 😷❌

This project can be used in public places, offices, hospitals, schools, and transportation systems to monitor mask compliance.

---

## Features

- Real-time face mask detection using webcam
- Binary classification (Mask / No Mask)
- Deep Learning-based image classification
- Face detection using OpenCV
- Easy-to-use interface
- Trained on labeled face mask dataset

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Scikit-learn

---

## Dataset

The dataset contains images of people:

- Wearing Face Masks
- Not Wearing Face Masks

Images are preprocessed and resized before training.

Dataset Structure:

```
dataset/
│
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Model Architecture

The model is built using Convolutional Neural Networks (CNN).

Typical Layers:

1. Convolution Layer
2. ReLU Activation
3. Max Pooling Layer
4. Convolution Layer
5. ReLU Activation
6. Max Pooling Layer
7. Flatten Layer
8. Dense Layer
9. Sigmoid Output Layer

Output:

- 0 → Without Mask
- 1 → With Mask

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python train.py
```

The trained model will be saved as:

```bash
face_mask_model.h5
```

---

## Real-Time Detection

Run:

```bash
python detect_mask.py
```

The webcam will open and display:

- Mask detected
- No Mask detected
- Prediction confidence score

Example:

```
Mask (0.95)
No Mask (0.87)
```

---

## Project Workflow

1. Collect Dataset
2. Preprocess Images
3. Train CNN Model
4. Save Trained Model
5. Capture Webcam Feed
6. Detect Face using OpenCV
7. Predict Mask Status
8. Display Result in Real-Time

---

## Results

The model successfully detects face masks in real-time and can distinguish between masked and unmasked faces with good accuracy under normal lighting conditions.

---

## Future Improvements

- Improve accuracy with larger datasets
- Support multiple face detection
- Deploy as a web application
- Mobile application integration
- Edge device deployment (Raspberry Pi)

---

## Applications

- Hospitals and Healthcare Centers
- Airports and Railway Stations
- Educational Institutions
- Corporate Offices
- Public Safety Monitoring

---

## Author

Prabhat Rai

B.Tech Student | Machine Learning & Computer Vision Enthusiast

---

## License

This project is licensed under the MIT License.
