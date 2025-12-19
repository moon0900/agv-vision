# agv-vision

- AGV 환경에서 사용하기 위한 **차량 번호판 인식(Vision) 모듈**입니다.
- 입력 이미지에서 번호판을 인식하고,
지정한 타겟 번호판과의 유사도를 기준으로 탐지 결과를 반환합니다.
- 현재 이 레포는 Vision 모듈 및 디버그 파이프라인 구현에 초점을 두고 있습니다.


## 핵심 기능

* OCR 기반 차량 번호판 인식
* OCR 엔진 선택
  * `paddle` : 로컬 OCR (API 호출 없음)
  * `clova` : Clova OCR API
* 타겟 번호판 문자열과의 유사도 기반 필터링
* 디버그 모드 지원 (사전 OCR 결과 사용, `data/demo`위치에 있는 데이터를 대신 사용)


## 디렉토리 구조

```
agv-vision/
├─ vision/          # Vision / OCR 모듈
├─ data/
│  └─ demo/         # 디버그용 이미지 및 사전 OCR 결과
├─ main.py          # 사용 예제
├─ requirements.txt
└─ README.md
```



## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수 설정 (Clova OCR 사용 시)

Clova OCR 엔진(`model="clova"`)을 사용하는 경우, API 호출을 위해 환경 변수 설정이 필요합니다.
`.env` 파일을 프로젝트 루트에 생성하고 아래 항목을 설정해야 합니다.

```env
CLOVA_OCR_API_URL=your_clova_api_url
CLOVA_OCR_API_KEY=your_clova_api_key
```

* Paddle OCR(`model="paddle"`) 사용 시에는 별도의 환경 변수 설정이 필요하지 않음.
* `.env` 파일은 **Git에 커밋하지 않도록 주의** (`.gitignore`에 포함 권장).



## 사용 예시
`main.py`에서 주석으로 더 자세히 설명하기 때문에, 
아래 코드로 설명이 부족하다면 그 쪽을 참고해 주시기 바랍니다.
```python
from vision.detector import PlateNumberDetector
import cv2

...
frame = get_frame() # 이미지 읽어오는 부분 생략

# 번호판 탐지 객체 생성
detector = PlateNumberDetector(
    model="paddle",             # "paddle" or "clova"
    plate_similarity_thresh=82, # 지정 번호판과의 유사도 판단 기준
    debug_mode=True,            # data/demo 사용
)

# 특정 프레임에 대한 
result = detector.detect(frame, target="630모8800")

if result:
    print(result.text, result.similarity, result.bbox)
```


## Debug Mode

* `debug_mode=True`일 경우

  * 실제 OCR 엔진 호출 없음
  * `data/demo/`에 저장된 사전 OCR 결과 사용
* 디버깅 시 API 호출 최소화 및 모델 로딩 과정 생략 목적

