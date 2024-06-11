import cv2
import dlib
import numpy as np
import pyttsx3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import threading
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
from PIL import Image

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

def alert():
    engine.say("Alert!!!!")
    engine.runAndWait()
    
# Load your own trained YOLOv5 model
modely = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=r'best.pt')  # ,
# force_reload=True)

# Function to predict on an image
def predict(image_path):
    # print(image_path)
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Run inference
    results = modely(image)

    # Process results
    predictions = []
    for pred in results.pred:
        for det in pred:
            predictions.append({
                'class': det[-1],
                'confidence': float(det[4]),
                'box': det[:4].tolist()
            })

    return predictions


data_frame = pd.read_csv("drowsi.csv")

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data_frame["Close/Open"])

X_train, X_test, y_train, y_test = train_test_split(data_frame[["lear", "rear"]], data_frame["Close/Open"],
                                                    test_size=0.25, random_state=42)

# Build a Support Vector Machine (SVM) classifier
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the SVM model
model = SVC(kernel='rbf', C=1.0)

# Assuming 'data_frame' is your DataFrame containing "lear", "rear", and "Close/Open" columns
labels = label_encoder.fit_transform(data_frame["Close/Open"])
X_train, X_test, y_train, y_test = train_test_split(data_frame[["lear", "rear"]], labels, test_size=0.25,
                                                    random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Support Vector Machine (SVM) classifier
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


def eye_aspect_ratio(eye):
    # Extract coordinates of vertical eye landmarks
    v1 = np.array([eye[1].x, eye[1].y])
    v5 = np.array([eye[5].x, eye[5].y])
    v2 = np.array([eye[2].x, eye[2].y])
    v4 = np.array([eye[4].x, eye[4].y])

    # Extract coordinates of horizontal eye landmarks
    v0 = np.array([eye[0].x, eye[0].y])
    v3 = np.array([eye[3].x, eye[3].y])

    # Calculate the Euclidean distances
    vertical_dist1 = np.linalg.norm(v1 - v5)
    vertical_dist2 = np.linalg.norm(v2 - v4)
    horizontal_dist = np.linalg.norm(v0 - v3)

    # Calculate the EAR
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    return ear


# Initialize variables for drowsiness detection
drowsy_frames = 0
drowsy_threshold = 6  # You can adjust this threshold

# Open a connection to the camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)
import time

while True:
    time.sleep(.1)
    # Read a frame from the camera
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        cv2.imwrite('temp.jpg', gray)
        image_path = 'temp.jpg'
        predictions = predict(image_path)
        # print('preds',predictions[0], type(predictions[0]))
        if len(predictions) > 0:
            print('---')
            box = predictions[0]['box']
            print(box[0], box[1], box[2], box[3])
            x = int(box[0])
            y = int(box[1])
            w = int(box[2] / 2)
            h = int(box[3] / 2)

            # center_x = int(x * width)
            # center_y = int(y * height)
            # w = int(w * width)
            # h = int(h * height)
            # x = int(center_x - w / 2)
            # y = int(center_y - h / 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print('test 1 ', x, y, w, h)
    except Exception as exp:
        print('testing..', exp)

    # Detect faces in the frame
    faces = detector(gray)
    landmarks = []
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        landmarks.append(shape)

    # Extract left and right eye landmarks for each shape in the list
    left_eyes = [shape.parts()[36:42] for shape in landmarks]
    right_eyes = [shape.parts()[42:48] for shape in landmarks]

    # Iterate through each set of left and right eyes
    for left_eye, right_eye in zip(left_eyes, right_eyes):
        # Calculate EAR for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Scale the features
        features = np.array([[left_ear, right_ear]])
        features_scaled = scaler.transform(features)

        # Use the trained SVM model to predict drowsiness
        prediction = model.predict(features_scaled)

        # Display the result on the frame
        if prediction[0] == 1:  # Assuming 1 represents drowsy in your model
            cv2.putText(frame, "Not Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            drowsy_frames = 0

            # Increment drowsy_frames counter

        else:
            drowsy_frames += 1
            if drowsy_frames >= drowsy_threshold:
                cv2.putText(frame, " Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Trigger an alert or take some action for continuous drowsiness
                print("Continuous Drowsiness Detected!")
            # Reset drowsy_frames counter

    # Check if a face is detected but no seatbelt
    if len(faces) > 0 and len(predictions) == 0:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, " No SeatBelt", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            threading.Thread(target=alert).start()


    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
