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
nose_indices = [
    # 코 중앙선
    1, 2, 3, 4, 5, 
    # 코의 윗부분과 아래부분
    19, 94, 370, 462,
    # 코 주변과 코끝 주변
    168, 6, 197, 195,
    # 코 양쪽 가장자리
    129, 209, 358
]


def enlarge_contour(contour, x_scale, y_scale):
    center = np.mean(contour, axis=0)  # 컨투어의 중심점 계산
    
    # 각 점과 중심점 사이의 거리 계산
    distances = np.sqrt(((contour - center)**2).sum(axis=1))
    
    # 거리에 기반한 가중치 계산 (거리가 가까울수록 더 큰 가중치)
    weights_y = 1 + (y_scale - 1) * (np.max(distances) - distances) / np.max(distances)
    weights_x = 1 + (x_scale - 1) * (np.max(distances) - distances) / np.max(distances)
    
    # 가중치를 적용하여 컨투어 확장
    enlarged_y = center[1] + weights_y * (contour[:, 1] - center[1])
    enlarged_x = center[0] + weights_x * (contour[:, 0] - center[0])
    
    enlarged_contour = contour.copy()
    enlarged_contour[:, 1] = enlarged_y
    enlarged_contour[:, 0] = enlarged_x
    
    return enlarged_contour.astype(np.int32)


def enlarge_eyes(image, scale=1.05):
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

        # 컨투어 영역 확장
        left_eye_contour = enlarge_contour(left_eye_contour, 4.0, 3.0)
        right_eye_contour = enlarge_contour(right_eye_contour, 4.0, 3.0)

        # 각 눈 영역을 확대하고 원본 이미지에 합성
        image = enlarge_and_blend(image, left_eye_contour, scale)
        image = enlarge_and_blend(image, right_eye_contour, scale)

    return image


def shrink_nose(image, scale=0.9):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return image

    for facial_landmarks in results.multi_face_landmarks:
        # 코 영역 추출
        nose_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in nose_indices])

        # 코 영역을 축소하고 원본 이미지에 합성
        image = shrink_and_blend(image, nose_contour, scale)

    return image


def enlarge_and_blend(image, contour, scale, padding=30):
    # 컨투어보다 약간 큰 영역을 추출하고 확대
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
    x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1]-x), min(h, image.shape[0]-y)

    roi = image[y:y+h, x:x+w]
    enlarged_area = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 컨투어에 완전히 일치하는 마스크를 생성합니다.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # 컨투어를 사용하여 마스크 내부를 채웁니다.
    mask = cv2.resize(mask[y:y+h, x:x+w], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)  # 확대
    dilated_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=10)  # 마스크를 확대합니다.

    # 영역의 중심 좌표 계산
    center = (x + w//2, y + h//2)

    # seamlessClone을 사용하여 이미지 합성
    # cv2.NORMAL_CLONE 또는 cv2.MIXED_CLONE 중 선택할 수 있음
    result = cv2.seamlessClone(enlarged_area, image, dilated_mask, center, cv2.NORMAL_CLONE)

    return result


def shrink_and_blend(image, contour, scale, padding=7, feather=17):
    # 컨투어로 경계 사각형을 계산하고 패딩을 적용합니다.
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding

    # 이미지 경계를 초과하지 않도록 조정합니다.
    x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1]-x), min(h, image.shape[0]-y)

    # 지정된 영역과 주변 영역을 포함하여 축소합니다.
    shrunk_area = cv2.resize(image[y:y+h, x:x+w], None, fx=scale, fy=scale)

    # 축소된 이미지의 새로운 크기를 계산합니다.
    sh, sw = shrunk_area.shape[:2]

    # 원본 이미지에 합성할 위치를 계산합니다.
    startx = x + w // 2 - sw // 2
    starty = y + h // 2 - sh // 2
    endx = min(startx + sw, image.shape[1])
    endy = min(starty + sh, image.shape[0])
    startx = max(0, startx)
    starty = max(0, starty)

    # 원본 이미지의 대응하는 영역에 가우시안 블러를 적용합니다.
    image[starty:endy, startx:endx] = cv2.GaussianBlur(image[starty:endy, startx:endx], (feather, feather), 0)

    # 합성할 영역에 대한 마스크를 생성합니다.
    mask = np.zeros((endy-starty, endx-startx, 3), dtype=np.uint8)
    cv2.rectangle(mask, (feather//2, feather//2), (mask.shape[1]-feather//2, mask.shape[0]-feather//2), (255, 255, 255), -1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    # 축소된 이미지와 마스크를 사용하여 원본 이미지에 부드럽게 합성합니다.
    for c in range(0, 3):
        image[starty:endy, startx:endx, c] = image[starty:endy, startx:endx, c] * (1 - mask[:, :, c]/255.0) + shrunk_area[:, :, c] * (mask[:, :, c]/255.0)

    return image



image_path = r"C:\Users\kjune\Downloads\img\001_origin.jpg"
image = cv2.imread(image_path)
eye_adjusted_image = enlarge_eyes(image.copy())
nose_adjusted_image = shrink_nose(image.copy())
adjusted_image = enlarge_eyes(shrink_nose(image.copy()))

result_path = image_path.replace("_origin", "_result")
result_image = cv2.imread(result_path)

# resize
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
eye_adjusted_image = cv2.resize(eye_adjusted_image, (0, 0), fx=0.5, fy=0.5)
nose_adjusted_image = cv2.resize(nose_adjusted_image, (0, 0), fx=0.5, fy=0.5)
adjusted_image = cv2.resize(adjusted_image, (0, 0), fx=0.5, fy=0.5)
result_image = cv2.resize(result_image, (0, 0), fx=0.5, fy=0.5)

# cv2.imshow('original', image)
# cv2.imshow('eye_adjusted', eye_adjusted_image)
# # cv2.imshow('result', result_image)
# # cv2.imshow('diff', cv2.absdiff(image, eye_adjusted_image))
# cv2.waitKey(0)

os.makedirs('output', exist_ok=True)

cv2.imwrite('original.jpg', image)
# cv2.imwrite('eye_adjusted.jpg', eye_adjusted_image)
# cv2.imwrite('nose_adjusted.jpg', nose_adjusted_image)
cv2.imwrite('adjusted.jpg', adjusted_image)
cv2.imwrite('result.jpg', result_image)

cv2.destroyAllWindows()