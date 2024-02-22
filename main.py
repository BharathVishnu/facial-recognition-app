import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained Facial Expression Recognition model
model = load_model('./model.h5')

# Load the face cascade classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess and predict facial expression
def predict_expression(frame, model):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image to match the input size of the model (48x48 pixels)
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0

        # Expand dimensions to match the input shape expected by the model
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Make predictions using the FER model
        predictions = model.predict(face_roi)

        # Get the predicted emotion label
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_index} - {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Open a video capture object (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Process the frame to detect faces and predict emotions
    frame = predict_expression(frame, model)

    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
