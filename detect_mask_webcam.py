# =============================================
# FILE 2: detect_mask_webcam.py
# PURPOSE: Live webcam face mask detection
# TOPICS: OpenCV, OOP, NumPy Slicing,
#         Loops, Pandas CSV, File Handling
# =============================================

import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

# ---- OOP: MaskDetector Class ----
class MaskDetector:

    def __init__(self):
        # Load trained ML model
        print("Loading model...")
        try:
            with open("model/mask_detector_model.pkl", "rb") as f:
                self.model = pickle.load(f)     # File Handling
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")  # Exception Handling
            exit()

        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.log = []                            # List to store log data

    # ---- METHOD: Preprocess Face Image ----
    def preprocess_face(self, face_img):
        face = cv2.resize(face_img, (64, 64))   # OpenCV resize
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return np.array(face).flatten().reshape(1, -1)  # NumPy

    # ---- METHOD: Predict Mask or No Mask ----
    def predict(self, face_img):
        try:
            features   = self.preprocess_face(face_img)
            prediction = self.model.predict(features)[0]
            return "Mask" if prediction == 1 else "No Mask"
        except:
            return "Unknown"

    # ---- METHOD: Save Log to CSV using Pandas ----
    def save_log(self):
        if len(self.log) > 0:
            df = pd.DataFrame(self.log, columns=["Time", "Status"])
            df.to_csv("detection_log.csv", index=False)
            print(f"\nLog saved! Total detections: {len(self.log)}")
            print(df["Status"].value_counts())  # Pandas value_counts

    # ---- METHOD: Run Live Detection ----
    def run(self):
        cap = cv2.VideoCapture(0)               # Open webcam

        if not cap.isOpened():
            print("Error: Webcam not found!")
            return

        print("\nWebcam started! Press Q to quit.")
        print("GREEN box = Mask | RED box = No Mask\n")

        while True:                             # Conditional Looping
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )

            for (x, y, w, h) in faces:          # Loop through faces
                face_roi = frame[y:y+h, x:x+w]  # NumPy Array Slicing!
                label    = self.predict(face_roi)

                # Conditionals — color by result
                if label == "Mask":
                    color = (0, 255, 0)          # Green
                else:
                    color = (0, 0, 255)          # Red

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Log result
                self.log.append([
                    datetime.now().strftime("%H:%M:%S"), label
                ])

            # Show frame count on screen
            cv2.putText(frame,
                f"Detections: {len(self.log)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

            cv2.imshow("Face Mask Detector - TechNade", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_log()


# ---- Run Program ----
if __name__ == "__main__":
    try:
        detector = MaskDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")                    # Exception Handling