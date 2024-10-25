import cv2
import numpy as np
import imutils

from PIL import Image, ImageEnhance
from scipy.ndimage import interpolation as inter


def sharpen(img: np.ndarray, factor: float):
    """Hàm xử lý làm nét ảnh
    Args:
        img (numpy.ndarray): Ảnh sau khi đọc bằng opencv.
        factor (float): Độ làm nét.

    Returns:
        enhancer (numpy.ndarray): Ảnh sau khi xử lý làm nét.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    enhancer = np.array(enhancer)
    return enhancer


def correct_skew(image: np.ndarray, delta=1, limit=5):
    """Hàm xử lý xoay ảnh
    Args:
        image (numpy.ndarray): Ảnh sau khi đọc bằng opencv.

    Returns:
        corrected (numpy.ndarray): Ảnh sau khi xử lý xoay.
    """

    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    img_new = sharpen(image, 100)
    img_new = imutils.resize(img_new, height=680)
    gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    print('best_angle ', best_angle)
    if best_angle == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return corrected


# if __name__ == '__main__':
#     # Đọc ảnh bằng opencv
#     img_path = "image01.png"
#     img = cv2.imread(img_path)
#
#     # Gọi hàm xoay ảnh
#     new_img = correct_skew(img, 2)
#
#     # Hiển thị kết quả
#     cv2.imshow('Before', img)
#     cv2.imshow('After', new_img)
#     cv2.waitKey(0)
