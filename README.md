# 장비 스펙
CPU: intel cpu (i5-14600kf)
RAM: 32GB
GPU: RTX 4070 TI (GPU가 필수는 아님)
OS: Windows 11

# 파이썬 버전
python version: 3.11.7  (mediapipe 라이브러리는 3.11 이하에서만 동작합니다)

# 설치
루트 경로에서
pip install -r requirements.txt

# 실행
main.py 내부 IMAGE_PATH를 원본 이미지 경로로 지정합니다.
그 후 `python main.py` 으로 실행합니다.
결과는 `output` 아래에 저장됩니다. 
origin: 원본 이미지
adjusted_image: 필터링 이미지


