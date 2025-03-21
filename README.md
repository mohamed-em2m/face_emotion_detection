Facial Emotion Detection Project

This project focuses on detecting human facial emotions in real-time from images, video files, and webcam feeds using various face detection algorithms and a pre-trained deep learning model. The detected emotions are classified into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise.
Features

    Real-time Emotion Detection: Works with webcam feeds, video files, and static images.

    Multiple Face Detectors:

        dlib's HOG-based Detector

        OpenCV Haar Cascades

        MTCNN (Multi-Task Cascaded Convolutional Networks)

    Emotion Classification: Uses a pre-trained TensorFlow/Keras model (fernet.h5) to predict emotions.

    Visual Output: Bounding boxes and labels around detected faces with emotion-specific colors.

Installation
Dependencies

Ensure the following libraries are installed:
bash
Copy

pip install numpy matplotlib pandas seaborn tensorflow scikit-learn xgboost opencv-python dlib mtcnn

Notes:

    dlib: If installation fails, install CMake and a C++ compiler first. On Windows, use:
    bash
    Copy

    conda install -c conda-forge dlib

    MTCNN: Requires mtcnn library (pip install mtcnn).

Usage
1. Emotion Detection in Videos/Webcam
python
Copy

# Using dlib's HOG detector
python main.py --video_path "path/to/video.mp4" --detector dlib

# Using Haar Cascade (webcam)
python main.py --video_path 0 --detector haar

# Using MTCNN (video file)
python main.py --video_path "input.mp4" --detector mtcnn

2. Emotion Detection in Images
python
Copy

python detect_image.py --image_path "path/to/image.jpg" --detector [dlib|haar|mtcnn]

Arguments:

    --video_path: Path to video file or 0 for webcam.

    --image_path: Path to image file.

    --detector: Face detector algorithm (dlib, haar, or mtcnn).

Project Structure
Copy

.
├── main.py               # Main script for video/webcam detection
├── detect_image.py       # Script for image detection
├── fernet.h5             # Pre-trained emotion classification model
├── haarcascade_frontalface_default.xml  # Haar Cascade XML file
└── README.md

Model Details

    Architecture: Custom CNN trained on FER2013 or similar dataset.

    Input Shape: (48, 48, 1) (grayscale images).

    Output: Probabilities for 7 emotion classes.

Example Output

Emotion Detection Example
Contributing

    Fork the repository.

    Create a feature branch (git checkout -b feature/your-feature).

    Commit changes (git commit -m 'Add some feature').

    Push to the branch (git push origin feature/your-feature).

    Open a Pull Request.

License

This project is licensed under the MIT License. See LICENSE for details.
Acknowledgements

    Face detection using dlib, OpenCV, and MTCNN.

    Emotion classification model inspired by FER2013 dataset training.
