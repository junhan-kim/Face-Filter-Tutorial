import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


image_path = r"C:\Users\kjune\Downloads\img\002_origin.jpg"
image = cv2.imread(image_path)

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    left_eye_indices = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, # 눈 아래쪽
        466, 388, 387, 386, 385, 384, 398 # 눈 위쪽
    ]
    right_eye_full_indices = [
        133, 173, 157, 158, 159, 160, 161, 246, 33, # 눈 아래쪽
        7, 163, 144, 145, 153, 154, 155 # 눈 위쪽
    ]

    for facial_landmarks in results.multi_face_landmarks:
        for index in left_eye_indices:
            landmark = facial_landmarks.landmark[index]
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
        for index in right_eye_full_indices:
            landmark = facial_landmarks.landmark[index]
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Image", image)
cv2.waitKey(0)