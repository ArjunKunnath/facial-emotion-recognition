# facial-emotion-recognition

This project implements a deep learning-based Facial Emotion Recognition (FER) system for real-time and image-based emotion classification. It uses a custom optimized residual neural network architecture combined with OpenCV-based face detection to predict human emotions under varying conditions.

## Project Structure

- `facial_emotion_recognition/`: Core implementation scripts.
    - `train_model.py`: Main script for training the deep learning model.
    - `realtime_emotion.py`: Real-time webcam-based emotion detection.
    - `image_emotion.py`: Emotion detection from static images.
    - `models/`: Directory where trained models are stored.
- `data/`: Dataset directory (e.g., FER2013).
- `haarcascade/`: Face detection classifier files.
- `docs/`: Documentation and architecture diagrams.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/facial-emotion-recognition.git
   cd facial-emotion-recognition
