import numpy as np
import cv2


class ImagePreprocessor:
    def __init__(
            self, *,
            gamma=1.5,
            clahe_clip_limit=2.0,
            contrast_alpha=1.5,
            contrast_beta=0
    ):
        self.gamma = gamma
        self.clahe_clip_limit = clahe_clip_limit
        self.contrast_alpha = contrast_alpha
        self.contrast_beta = contrast_beta

    # 흑백 변환
    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이진화
    def binarize(self, img, threshold=127):
        _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary_img

    # 대비 조절
    def adjust_contrast(self, img, alpha=1.5, beta=0):
        # alpha: 대비 조정, beta: 밝기 조정
        return cv2.convertScaleAbs(img, alpha=self.contrast_alpha, beta=self.contrast_beta)

    # 밝기(감마) 조절
    def gamma_correction(self, img):
        inv = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    # 지역별 히스토그램 균등화
    def apply_clahe(self, img):
        # YCrCb 변환
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # CLAHE 적용
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8)
        )
        y_clahe = clahe.apply(y)

        # 합치기
        ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
        return cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

