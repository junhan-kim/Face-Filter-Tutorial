import cv2
import numpy as np
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


left_eye_indices = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, # 눈 아래쪽
    466, 388, 387, 386, 385, 384, 398 # 눈 위쪽
]
right_eye_indices = [
    133, 173, 157, 158, 159, 160, 161, 246, 33, # 눈 아래쪽
    7, 163, 144, 145, 153, 154, 155 # 눈 위쪽
]


def enlarge_eyes(image, scale=1.1):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return image

    for facial_landmarks in results.multi_face_landmarks:
        # 눈 주위 랜드마크 인덱스 (눈꺼풀 포함)
        # 이 인덱스는 눈 전체 영역과 눈꺼풀을 커버합니다.

        # 왼쪽 눈 영역 추출
        left_eye_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in left_eye_indices])
        # 오른쪽 눈 영역 추출
        right_eye_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in right_eye_indices])

        # 각 눈 영역을 확대하고 원본 이미지에 합성
        image = enlarge_and_blend(image, left_eye_contour, scale)
        image = enlarge_and_blend(image, right_eye_contour, scale)

    return image


def enlarge_and_blend(image, contour, scale, padding=20, feather=15):
    # 컨투어로 경계 사각형을 계산하고 패딩을 적용합니다.
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding

    # 이미지 경계를 초과하지 않도록 조정합니다.
    x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1]-x), min(h, image.shape[0]-y)

    # 지정된 영역과 주변 영역을 포함하여 확대합니다.
    enlarged_area = cv2.resize(image[y:y+h, x:x+w], None, fx=scale, fy=scale)

    # 확대된 이미지의 새로운 크기를 계산합니다.
    eh, ew = enlarged_area.shape[:2]

    # 원본 이미지에 합성할 위치를 계산합니다.
    startx = x + w // 2 - ew // 2
    starty = y + h // 2 - eh // 2
    endx = min(startx + ew, image.shape[1])
    endy = min(starty + eh, image.shape[0])
    startx = max(0, startx)
    starty = max(0, starty)

    # 확대된 이미지의 경계를 조정합니다.
    cropped_enlarged_area = enlarged_area[0:(endy-starty), 0:(endx-startx)]

    # 원본 이미지의 대응하는 영역에 가우시안 블러를 적용합니다.
    image[starty:endy, startx:endx] = cv2.GaussianBlur(image[starty:endy, startx:endx], (feather, feather), 0)

    # 합성할 영역에 대한 마스크를 생성합니다.
    mask = np.zeros((endy-starty, endx-startx, 3), dtype=np.uint8)
    cv2.rectangle(mask, (feather//2, feather//2), (mask.shape[1]-feather//2, mask.shape[0]-feather//2), (255, 255, 255), -1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    # 확대된 이미지와 마스크를 사용하여 원본 이미지에 부드럽게 합성합니다.
    for c in range(0, 3):
        image[starty:endy, startx:endx, c] = image[starty:endy, startx:endx, c] * (1 - mask[:, :, c]/255.0) + cropped_enlarged_area[:, :, c] * (mask[:, :, c]/255.0)

    return image



image_path = r"C:\Users\kjune\Downloads\img\001_origin.jpg"
image = cv2.imread(image_path)
adjusted_image = enlarge_eyes(image.copy())

result_path = image_path.replace("_origin", "_result")
result_image = cv2.imread(result_path)

# resize
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
adjusted_image = cv2.resize(adjusted_image, (0, 0), fx=0.5, fy=0.5)
result_image = cv2.resize(result_image, (0, 0), fx=0.5, fy=0.5)

os.makedirs('output', exist_ok=True)

cv2.imwrite('original.jpg', image)
cv2.imwrite('adjusted.jpg', adjusted_image)
cv2.imwrite('result.jpg', result_image)