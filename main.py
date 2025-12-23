from vision.detector import PlateNumberDetector
import cv2


# 그냥 카메라에서 프레임 1장 캡처해오는 함수 테스트용
def get_frame():
    frame = cv2.imread("./data/demo/test1.png")
    return frame


# 번호판 인식 객체 로드
# ! apply_preprocess 옵션 활용 시 기본 전처리 기법이 적용되며,
#   별도의 전처리 프로세스를 따르고 싶을 때는 이를 False로 변경한 후,
#   vision/utils/image_proc.py의 전처리 함수들을 활용 가능
detector = PlateNumberDetector(
    model="clova",                 # OCR할 엔진 선택 ("paddle" 또는 "clova")
    plate_similarity_thresh=82,     # 타겟 번호판과 일치하는지 판단할 유사도 임계치(기본값 80)
    apply_preprocess=True,          # 전처리 적용 여부 (CLACHE 기법)
    debug_mode=True,                # 디버그 모드 : 사전 OCR 데이터(data/demo/에 존재) 불러오기 -> API 호출 수 줄이거나 모델 로드 리소스 생략 가능
)

# 현재 프레임에서 찾고자하는 번호판이 있는지
frame = get_frame()
result = detector.detect(frame, target='630모8800')

# 만약 임계치를 넘는 인식 결과가 없으면 result는 None
# 임계치를 넘는 결과가 있으면 다음과 같은 정보가 저장됨
# result.text           -> 인식된 글자
# result.similarity     -> 타겟 번호판과의 유사도 점수
# result.bbox           -> 글자 인식 위치

