facial-emotion-recognition
This project implements a deep learning-based Facial Emotion Recognition (FER) system for real-time and image-based emotion classification. It uses a custom optimized residual neural network architecture combined with OpenCV-based face detection to predict human emotions under varying conditions.

Project Structure

facial_emotion_recognition/: Core implementation scripts.
train_model.py: Main script for training the deep learning model.
realtime_emotion.py: Real-time webcam-based emotion detection.
image_emotion.py: Emotion detection from static images.
models/: Directory where trained models are stored.
data/: Dataset directory (e.g., FER2013).
haarcascade/: Face detection classifier files.
docs/: Documentation and architecture diagrams.

Installation

Clone the repository:

git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition

Install the required dependencies:

pip install tensorflow keras opencv-python numpy

Usage

To train the emotion recognition model:

python train_model.py

To run real-time emotion detection using a webcam:

python realtime_emotion.py

To detect emotion from an image:

python image_emotion.py

Ensure that the trained model file and Haar cascade classifier are available in the project directory before running inference scripts.

Requirements

Python 3.8+
TensorFlow / Keras
NumPy
OpenCV
Matplotlib (optional for visualization)

Dataset

The model is trained using the FER2013 dataset consisting of grayscale facial images categorized into seven emotion classes.

Emotion Classes

Angry  
Disgust  
Fear  
Happy  
Neutral  
Sad  
Surprise  

