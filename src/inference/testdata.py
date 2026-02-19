import cv2
import numpy as np
import os
from keras.models import load_model

# Load model with error handling
try:
    model = load_model('optimized_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceDetect.empty():
    print("Error: Could not load face cascade classifier")
    exit(1)

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Use absolute path for the image
image_path = "c:\\S2\\MFC&EOC\\Facial Emotion Modified\\face1.jpg"
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Could not read image from {image_path}")
    exit(1)

print(f"Successfully loaded image: {image_path}")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

if len(faces) == 0:
    print("No faces detected in the image")

for x, y, w, h in faces:
    sub_face_img = gray[y:y+h, x:x+w]
    resized = cv2.resize(sub_face_img, (48, 48))
    normalize = resized/255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    print(f"Detected emotion: {labels_dict[label]}")
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
    cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
