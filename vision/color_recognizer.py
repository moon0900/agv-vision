import cv2
import threading
from queue import Queue
import time
import numpy as np

class ColorRecognizer(threading.Thread):

    def __init__(
            self,
            input_queue,
            color_area_size_thresh = 0.09,
            roi_top = 0.25,
            roi_bottom = 0.5
     ):
        super().__init__()
        self.input_queue = input_queue
        self.latest_results = {}  # 마스크 영역 크기 등 결과 저장
        self.processed_frame = None
        self.stopped = False
        self.daemon = True  # 프로그램 종료 시 함께 종료

        # roi 영역(상단은 반드시 제외할 높이 비율, 하단은 반드시 유지할 높이 비율)
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom

        # 추출된 색상 영역의 크기 임계값(이 크기를 넘어가면 해당 색상 영역 위에 있는 것으로 판정)
        self.color_area_size_thresh = color_area_size_thresh

        # 색상 hsv 범위
        self.recognized_color = None
        self.recognized_color_area_size = None
        self.white_range = (np.array([0, 0, 163]), np.array([179, 50, 255]))
        self.color_ranges = {
            'red': [(0, 70, 115), (7, 255, 255), (167, 80, 83), (179, 255, 255)],
            'orange': [(9, 45, 87), (19, 255, 255)],
            'yellow': [(21, 68, 121), (34, 255, 255)],
            'green': [(43, 83, 114), (88, 255, 255)],
            'blue': [(100, 56, 116), (116, 255, 255)],
            'purple': [(120, 61, 82), (151, 255, 255)]
        }

    def balance_white(self, img, p=0.5):
        """
        각 채널별로 밝기 분포를 분석하여 톤을 균일하게 맞춤 (White Patch 가설 기반)
        p: 각 채널에서 흰색으로 간주할 상위 퍼센트
        """
        assert img.dtype == np.uint8
        out = img.astype(np.float32)

        # 각 채널별로 독립적인 scaling 적용
        for c in range(3):
            channel = out[:, :, c]

            # 각 채널의 하위 p%와 상위 p% 지점 찾기
            low = np.percentile(channel, p)
            high = np.percentile(channel, 100 - p)

            # 채널별로 0~255로 꽉 채워 확장 (이 과정에서 톤이 중립화됨)
            if high > low:
                channel = (channel - low) / (high - low) * 255

            out[:, :, c] = channel

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_floor_mask(self, hsv, h):
        # 흰색 바닥 추출
        floor_mask = cv2.inRange(hsv, self.white_range[0], self.white_range[1])

        # 상단 영역은 강제로 제외 (배경 노이즈 제거)
        top_limit = int(h * self.roi_top)
        bottom_limit = int(h * self.roi_bottom)  # 하단 일부는 무조건 유지
        floor_mask[:top_limit, :] = 0
        floor_mask[bottom_limit:, :] = 255

        # 바닥 마스크 내부의 구멍(타일)을 채우기
        # 1. 마스크에서 윤곽선 찾기
        contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(floor_mask)
        if contours:
            # 2. 그중 가장 면적이 넓은 컨투어(바닥일 확률 높음) 선택
            largest_contour = max(contours, key=cv2.contourArea)

            # 3. 해당 컨투어 안쪽을 흰색(255)으로 꽉 채움 (두께 -1이 채우기 옵션)
            cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return filled_mask

    def get_color_masks(self, hsv):
        masks = {}
        for color_name, boundaries in self.color_ranges.items():
            if color_name == 'red': # 빨간색은 두 범위를 합침
                mask1 = cv2.inRange(hsv, boundaries[0], boundaries[1])
                mask2 = cv2.inRange(hsv, boundaries[2], boundaries[3])
                masks[color_name] = cv2.bitwise_or(mask1, mask2)
            else:
                masks[color_name] = cv2.inRange(hsv, boundaries[0], boundaries[1])
        return masks

    def run(self):
        while not self.stopped:
            if not self.input_queue.empty():
                frame = self.input_queue.get()

                # 톤 밸런싱 적용
                img = self.balance_white(frame)

                # HSV 값 구하기
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, w = img.shape[:2]

                # 바닥 ROI 마스크 생성
                floor_roi_mask = self.get_floor_mask(hsv, h)

                # 색상별 마스크 생성
                color_masks = self.get_color_masks(hsv)

                # 색상 영역을 추출하여 가장 큰 색상 영역 찾기
                largest_area_size = 0
                largest_color = None
                color_area = None
                for color_name, color_mask in color_masks.items():
                    roi_color_mask = cv2.bitwise_and(floor_roi_mask, color_mask)
                    area_size = cv2.countNonZero(roi_color_mask)
                    area_size_ratio = area_size * 1.0 / (h * w)
                    print(color_name, area_size, area_size_ratio)
                    if area_size_ratio > largest_area_size:
                        largest_area_size = area_size_ratio
                        largest_color = color_name
                        color_area = cv2.bitwise_and(img, img, mask=roi_color_mask)

                if self.color_area_size_thresh <= largest_area_size:
                    self.recognized_color = largest_color
                    self.recognized_color_area_size = largest_area_size
                else:
                    self.recognized_color = None
                    self.recognized_color_area_size = None

            else:
                time.sleep(0.01)  # 큐가 비었을 때 CPU 점유율 방지

    def stop(self):
        self.stopped = True