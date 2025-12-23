# agv-vision

- AGV 환경에서 사용하기 위한 **Vision 모듈 패키지**입니다.
- 차량 번호판 인식(OCR) 및 특정 색상 영역 탐지 기능을 제공합니다.
- 로직은 `vision` 모듈 내에 있고 실행 예제는 `main.py`, `color_recognition.py`를 제공합니다.


## 핵심 기능

* **차량 번호판 인식 (OCR)**
  * `paddle`: 로컬 OCR (API 호출 없음)
  * `clova`: Clova OCR API
  * 타겟 문자열과의 유사도 기반 필터링 지원
* **색상 영역 탐지 (Color Recognition)**
  * CLAHE 및 White Balance 전처리를 통한 강인한 색상 인식
  * ROI(관심 영역) 설정 및 최소 면적 비율 기반 필터링
* **이미지 전처리 유틸리티**
  * 밝기/대비 보정, 감마 보정, 적응형 이진화 등 공통 도구 제공


## 디렉토리 구조

```
agv-vision/
├─ vision/   # Vision / OCR 모듈
│ ├─ utils/
│ │ └─ image_proc.py    # 공통 이미지 전처리 함수
│ ├─ color.py           # 색상 인식 로직 (ColorRecognizer)
│ ├─ detector.py        # 번호판 인식 로직 (PlateNumberDetector)
│ └─ result.py          # 결과 데이터 구조 (Dataclasses)
├─ data/ 
│ └─ demo/                # 디버그용 이미지 및 사전 OCR 결과
├─ main.py                # 번호판 인식 사용 예제
├─ color_recognition.py   # 스레딩 기반 색상 인식 활용 예제
├─ requirements.txt
└─ README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 그 외 환경 설정
### Clova OCR 사용 시
Clova OCR 엔진(`model="clova"`)을 사용하는 경우, API 호출을 위해 환경 변수 설정이 필요합니다.
`.env` 파일을 프로젝트 루트에 생성하고 아래 항목을 설정해야 합니다.

```env
CLOVA_OCR_API_URL=your_clova_api_url
CLOVA_OCR_API_KEY=your_clova_api_key
```
* `.env` 파일은 **Git에 커밋하지 않도록 주의** (`.gitignore`에 포함 권장).

### Paddle OCR 사용 시

* Paddle OCR(`model="paddle"`) 사용 시 [PaddleOCR 원본 저장소](https://github.com/PaddlePaddle/PaddleOCR)의 공식문서 설명에 따른 환경 설정이 필요합니다.
* JetBot 환경에서는 CUDA, Python 버전과의 호환성으로 인해 활용하기 어려울 수 있습니다.


## 사용 예시
### 1. 차량 번호판 인식
`main.py`에서 주석으로 더 자세히 설명하기 때문에, 
아래 코드로 설명이 부족하다면 그 쪽을 참고해 주시기 바랍니다.
```python
from vision.detector import PlateNumberDetector
import cv2

... # 이미지 로드 생략

# 번호판 탐지 객체 생성
detector = PlateNumberDetector(
    model="paddle",             # "paddle" or "clova"
    plate_similarity_thresh=82, # 지정 번호판과의 유사도 판단 기준
    apply_preprocess=True,       # 사전 지정된 전처리 기법 적용 여부
    debug_mode=True,            # data/demo 사용
)

# OCR을 통한 번호판 인식(target 번호판만을 탐지 결과로 반환)
result = detector.detect(frame, target="630모8800")

if result:
    print(f"Match: {result.text}, Similarity: {result.similarity}, Bbox: {result.bbox}")
```
### 2. 색상 인식
`color_recognition.py`의 예제를 참고해주시기 바랍니다.


## Debug Mode

* `debug_mode=True`일 경우
  * 실제 OCR 엔진 호출 없이 `data/demo/`에 저장된 사전 OCR 결과를 사용합니다.
  * API 비용 절감 및 빠른 로직 테스트에 유용합니다.
