from vision.color_recognizer import ColorRecognizer
import cv2
import os, glob
from queue import Queue
import time

# Road Following에 사용했던 데이터를 사용한 예제(실제로는 camra에서 값 읽어오는 것으로 변경 필요)
data_dir = "./data/xy_dataset_edited/"
images = glob.glob(os.path.join(data_dir, "*.jpg"))

frame_queue = Queue(maxsize=1)  # 최신 1개의 프레임만 유지하도록 설정

# 스레드 시작
recognizer = ColorRecognizer(
    frame_queue,
    color_area_size_thresh = 0.09,      # 눈 앞에 보이는 색상 영역이 전체 화면 중 어떤 비율 이상을 차지할 때 색상을 인식한 것으로 처리할지 임계값
    roi_top = 0.25,                     # 항상 관심영역에서 배제할 상단 영역 비율(뒷 배경 제거 목적)
    roi_bottom = 0.5                    # 항상 관심영역으로 유지할 하단 영역 비율
)
recognizer.start()


for image in images:
    frame = cv2.imread(image)

    # 큐가 꽉 찼다면 이전 프레임을 버리고 최신 프레임 삽입 (딜레이 방지)
    if frame_queue.full():
        frame_queue.get_nowait()
    frame_queue.put(frame)

    time.sleep(0.5)

    # 결과 가져오기 (메인 스레드에서는 출력만 담당)
    color = recognizer.recognized_color
    # 화면에 결과 표시 (예시)
    cv2.putText(frame, f"Recognized: {color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

recognizer.stop()
cv2.destroyAllWindows()