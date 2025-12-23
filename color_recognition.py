import threading
import time
import cv2

# 색상 인식 스레드 워커 클래스 정의 예제
class ColorRecognitionWorker(threading.Thread):
    def __init__(self, recognizer, input_queue, ):
        super().__init__()

        self.recognizer = recognizer  # ColorRecognizer 주입
        self.input_queue = input_queue

        self.result = None
        self.daemon = True
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                self.result = self.recognizer.recognize(frame)
            else:
                time.sleep(0.01)

    def stop(self):
        self.stopped = True


# ==========================================
# cv2.VideoCapture를 이용한 실행 예제
# ==========================================
from vision.color import ColorRecognizer
from queue import Queue

# 1. 초기화 (카메라 인덱스는 보통 0)
cap = cv2.VideoCapture(0)
frame_queue = Queue(maxsize=1)
recognizer = ColorRecognizer(min_detection_area_ratio=0.088)
worker = ColorRecognitionWorker(recognizer, frame_queue)

worker.start()

print("인식을 시작합니다. 'q'를 누르면 종료합니다.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 큐가 비어있을 때만 새 프레임을 넣음 (병목 방지)
        if not frame_queue.full():
            frame_queue.put(frame)

        # 워커의 최신 결과 가져오기
        res = worker.result

        # 결과 시각화
        display_frame = frame.copy()
        if res:
            text = f"{res.color} (area_ratio: {res.area_ratio:.2%})"
            cv2.putText(display_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Color Detection Test', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()