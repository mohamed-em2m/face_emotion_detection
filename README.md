# Emotion Detection using Deep Learning

## Overview
This project implements real-time facial emotion detection using deep learning techniques. It employs TensorFlow, Keras, OpenCV, and dlib to process images and videos, detect faces, and classify emotions using a pre-trained convolutional neural network (CNN).

## Project Demo
<video src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7065980829142781955?compact=1" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>
## Features
- Detects faces in images and videos using dlib and OpenCV's Haar Cascade classifier.
- Classifies facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Utilizes a deep learning model (`fernet (1).h5`) trained on a facial emotion dataset.
- Supports real-time emotion detection via webcam or video input.
- Saves processed video with emotion labels.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- dlib
- NumPy
- Pandas
- Scikit-learn
- XGBoost

### Install Dependencies
```bash
pip install tensorflow keras opencv-python dlib numpy pandas scikit-learn xgboost seaborn matplotlib
```

## Usage
### Detect Emotion in Video
Run the script to process a video and detect emotions:
```bash
python detect_emotion_video.py
```
Modify `vid=cv2.VideoCapture(0)` to use a video file instead of a webcam.

### Detect Emotion in Image
Run the script to detect emotion in a static image:
```bash
python detect_emotion_image.py
```
Replace `frame=cv2.imread("path/to/image.jpg")` with your image file.

## Model
The project uses a CNN model trained on facial expressions data. The architecture includes:
- Convolutional layers (Conv2D)
- MaxPooling layers
- Fully connected (Dense) layers
- Dropout and Batch Normalization

The model is saved as `fernet (1).h5` and loaded using:
```python
model = keras.models.load_model('fernet (1).h5')
```

## Face Detection Methods
1. **dlib HOG-based detector**
2. **OpenCV Haar Cascade classifier**

Example usage:
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```

## Output
Detected faces are marked with rectangles and labeled with predicted emotions. The results can be displayed in real-time or saved as a video file.

## Example Labels and Colors
```python
labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}
colors = {0:(0,0,255), 1:(100,0,100), 2:(200,50,123), 3:(0,255,255), 4:(0,255,0), 5:(255,255,0), 6:(255,0,0)}
```

## Contributions
Feel free to fork this repository, submit issues, or open pull requests.

## License
This project is licensed under the MIT License.

