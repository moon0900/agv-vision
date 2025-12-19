from vision.detector import PlateNumberDetector
import cv2

# 그냥 카메라에서 프레임 1장 캡처해오는 함수 테스트용
def get_frame():
    # 카메라 열기
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # 카메라 열기 실패 체크
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        exit(1)

    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        assert ()

    return frame


# 번호판 인식 객체 로드
# 주의: OCR 테스트를 많이 하고 싶으면 model을 paddle로 지정해서 로컬 모델로 돌릴 것!
# clova ocr api 호출 수 너무 늘어나지 않게 주의
detector = PlateNumberDetector(
    model="paddle",                 # OCR할 엔진 선택 ("paddle" 또는 "clova")
    plate_similarity_thresh=82,     # 타겟 번호판과 일치하는지 판단할 유사도 임계치(기본값 80)
    debug_mode=True,                # 디버그 모드 : 사전 OCR 데이터(data/demo/에 존재) 불러오기 -> API 호출 수 줄이거나 모델 로드 리소스 생략 가능
)


# 카메라에서 불러온 프레임
frame = get_frame()

# 현재 프레임에서 찾고자하는 번호판이 있는지
result = detector.detect(frame, target='630모8800')

# 만약 임계치를 넘는 인식 결과가 없으면 result는 None
# 임계치를 넘는 결과가 있으면 다음과 같은 정보가 저장됨
# result.text           -> 인식된 글자
# result.similarity     -> 타겟 번호판과의 유사도 점수
# result.bbox           -> 글자 인식 위치

