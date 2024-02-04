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
        # 왼쪽 눈 영역 추출
        left_eye_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in left_eye_indices])
        # 오른쪽 눈 영역 추출
        right_eye_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in right_eye_indices])

        # 컨투어 영역 확장
        left_eye_contour = enlarge_contour(left_eye_contour, 4.0, 3.0)
        right_eye_contour = enlarge_contour(right_eye_contour, 4.0, 3.0)

        # 각 눈 영역을 확대하고 원본 이미지에 합성
        image = apply_bulge_effect(image, left_eye_contour)
        image = apply_bulge_effect(image, right_eye_contour)

    return image


def shrink_nose(image, scale=0.9):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return image

    for facial_landmarks in results.multi_face_landmarks:
        # 코 영역 추출
        nose_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in nose_indices])

        # center는 코 중앙점 (landmark 4번)
        center = (int(facial_landmarks.landmark[4].x * image.shape[1]), int(facial_landmarks.landmark[4].y * image.shape[0]))

        # 코 영역을 축소하고 원본 이미지에 합성
        image = apply_shrink_effect(image, nose_contour, center)

    return image


# def apply_bulge_effect(image, contour, exp=0.1, scale=1.1):
def apply_bulge_effect(image, contour, exp=0.1, scale=1.1):
    rows, cols = image.shape[:2]

    # 컨투어 중심과 반경 계산
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    radius = max(w, h) // 2 * scale

    # 매핑 배열 생성
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            # 각 픽셀에서 컨투어 중심까지의 거리 계산
            dx = (col - cx)
            dy = (row - cy)
            distance = np.sqrt(dx**2 + dy**2)

            # 컨투어 중심에서의 거리에 따라 왜곡 적용
            if distance < radius:
                # 왜곡 영역 내 픽셀에 대한 새 위치 계산
                factor = 1.0 if distance == 0 else np.power(distance / radius, exp)
                new_x = cx + dx * factor
                new_y = cy + dy * factor
                map_x[row, col] = new_x
                map_y[row, col] = new_y
            else:
                map_x[row, col] = col
                map_y[row, col] = row

    # 재매핑 변환
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted



def apply_shrink_effect(image, contour, center, exp=-0.2, scale=0.9):
    rows, cols = image.shape[:2]

    # 컨투어 중심과 반경 계산
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2

    # 바운딩 박스에 외접하는 원의 반지름 계산
    diagonal_length = np.sqrt(w**2 + h**2)
    radius = (diagonal_length / 2) * scale

    # 매핑 배열 생성
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            # 각 픽셀에서 컨투어 중심까지의 거리 계산
            dx = (col - cx)
            dy = (row - cy)
            distance = np.sqrt(dx**2 + dy**2)

            # 컨투어 중심에서의 거리에 따라 왜곡 적용
            if distance < radius:
                # 왜곡 영역 내 픽셀에 대한 새 위치 계산
                factor = 1.0 if distance == 0 else np.power(distance / radius, exp)
                new_x = cx + dx * factor
                new_y = cy + dy * factor
                map_x[row, col] = new_x
                map_y[row, col] = row  # y축 변화 없음으로 설정
            else:
                map_x[row, col] = col
                map_y[row, col] = row

    # 재매핑 변환으로 축소 효과 적용
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted


if __name__ == '__main__':

    IMAGE_PATH = r"C:\Users\kjune\Downloads\img\001_origin.jpg"

    image = cv2.imread(IMAGE_PATH)
    # render
    adjusted_image = enlarge_eyes(shrink_nose(image.copy()))

    # display
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    adjusted_image = cv2.resize(adjusted_image, (0, 0), fx=0.5, fy=0.5)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'origin.jpg'), image)
    cv2.imwrite(os.path.join(output_dir, 'adjusted_image.jpg'), adjusted_image)

