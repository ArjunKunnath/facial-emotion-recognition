import cv2
import numpy as np
import time
from keras.models import load_model

# Load the updated ResNet-34 model with error handling
try:
    model = load_model('optimized_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load face cascade with error handling
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Cascade classifier file not found or invalid")
    print("Face cascade loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    exit(1)

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Allow user to select camera source
camera_id = 0  # Default camera
try:
    video = cv2.VideoCapture(camera_id)
    if not video.isOpened():
        print(f"Could not open camera {camera_id}, trying alternative...")
        video = cv2.VideoCapture(camera_id + 1)  # Try another camera
    
    if not video.isOpened():
        raise Exception("Could not open any camera")
except Exception as e:
    print(f"Error accessing camera: {e}")
    exit(1)

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved face detection parameters for better detection
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # More gradual scaling for better detection
        minNeighbors=5,   # Higher value for fewer false positives
        minSize=(30, 30)  # Minimum face size to detect
    )
    
    if len(faces) > 0:
        # Get the largest face for emotion analysis
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        confidence = result[0][label] * 100  # Convert to percentage
        
        # Draw rectangle and emotion label with confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        emotion_text = f"{labels_dict[label]}: {confidence:.1f}%"
        cv2.putText(frame, emotion_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display FPS
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    cv2.imshow("Facial Expression Recognition", frame)
    
    # Check for key press with more options
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save current frame
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"emotion_capture_{timestamp}.jpg", frame)
        print(f"Saved frame as emotion_capture_{timestamp}.jpg")

video.release()
cv2.destroyAllWindows()
