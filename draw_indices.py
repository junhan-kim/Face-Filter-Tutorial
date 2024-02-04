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

# nose_indices = [
#     # 코 중앙선
#     1, 2, 3, 4, 5, 
#     # 코의 윗부분과 아래부분
#     19, 94, 370, 462,
#     # 코 주변과 코끝 주변
#     168, 6, 197, 195,
#     # 코 양쪽 가장자리
#     129, 209, 358
# ]

# nose_indices = [
#     # 코 중앙선
#     # 1, 2, 3, 4, 5, 
#     4, 5,
#     # 코의 윗부분과 아래부분
#     19, 94, 370, 462,
#     # 370, 462,
#     # 코 주변과 코끝 주변
#     # 168, 6, 197, 195,
#     # 코 양쪽 가장자리
#     129, 209, 358
# ]


# nose_indices = [
#     4, 5
# ]



# def enlarge_contour(contour):
#     # 눈 영역에서 쌍커풀 영역을 포함하기 위해 살짝 확장된 컨투어 영역을 구함
#     # 좌우보다는 상하로 더 확장된 영역을 반환 (y축 스케일링을 더 강하게 적용)
#     center = np.mean(contour, axis=0)
#     enlarged_y = center[1] + 2.2 * (contour[:, 1] - center[1])
#     enlarged_x = center[0] + 1.3 * (contour[:, 0] - center[0])
#     enlarged_contour = contour.copy()
#     enlarged_contour[:, 1] = enlarged_y
#     enlarged_contour[:, 0] = enlarged_x
#     return enlarged_contour.astype(np.int32)

def enlarge_contour(contour):
    center = np.mean(contour, axis=0)  # 컨투어의 중심점 계산
    
    # 각 점과 중심점 사이의 거리 계산
    distances = np.sqrt(((contour - center)**2).sum(axis=1))
    
    # 거리에 기반한 가중치 계산 (거리가 가까울수록 더 큰 가중치)
    weights_y = 1 + (3.0 - 1) * (np.max(distances) - distances) / np.max(distances)
    weights_x = 1 + (5.0 - 1) * (np.max(distances) - distances) / np.max(distances)
    
    # 가중치를 적용하여 컨투어 확장
    enlarged_y = center[1] + weights_y * (contour[:, 1] - center[1])
    enlarged_x = center[0] + weights_x * (contour[:, 0] - center[0])
    
    enlarged_contour = contour.copy()
    enlarged_contour[:, 1] = enlarged_y
    enlarged_contour[:, 0] = enlarged_x
    
    return enlarged_contour.astype(np.int32)



def enlarge_eyes(image):
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
        # 코 영역 추출
        nose_contour = np.array([[int(facial_landmarks.landmark[i].x * image.shape[1]), int(facial_landmarks.landmark[i].y * image.shape[0])] for i in nose_indices])

        # 눈 영역에서 쌍커풀 영역을 포함하기 위해 살짝 확장된 컨투어 영역을 구함
        left_eye_contour = enlarge_contour(left_eye_contour)
        right_eye_contour = enlarge_contour(right_eye_contour)

        # 코 영역 bounding box 그리기
        x, y, w, h = cv2.boundingRect(nose_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # contour 영역 그리기
        for i in range(len(left_eye_contour)):
            cv2.circle(image, (left_eye_contour[i][0], left_eye_contour[i][1]), 2, (0, 255, 0), -1)
        for i in range(len(right_eye_contour)):
            cv2.circle(image, (right_eye_contour[i][0], right_eye_contour[i][1]), 2, (0, 255, 0), -1)
        for i in range(len(nose_contour)):
            cv2.circle(image, (nose_contour[i][0], nose_contour[i][1]), 10, (0, 0, 255), -1)

    return image


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\kjune\Downloads\img\001_origin.jpg")
    image = enlarge_eyes(image)

    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()