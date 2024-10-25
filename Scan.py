import glob

import cv2
import numpy as np
import imutils
import os

from PIL import Image, ImageEnhance
from util import correct_skew
from DetecInfoBoxes.GetBoxes import Detect
from config import scan_opt

getDictionary = Detect(scan_opt)


class Scan:
    @staticmethod
    def preprocess(img, factor: float):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(img).enhance(factor)
        if gray.std() < 30:
            enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
        return np.array(enhancer)

    @staticmethod
    def get_conner(image, imgsz, stride, device, half, model, names):
        (h, w, d) = image.shape
        dicts = getDictionary.prediction(image, imgsz, stride, device, half, model, names)
        dicts = getDictionary.dict_processing(dicts)

        print(dicts)

        # getDictionary.show_result(dicts, image)
        # cv2.imshow('a', image)
        # cv2.waitKey(0)

        if dicts.get('conner') is None:
            conners = [[0, 0], [w, 0], [w, h], [0, h]]
        else:
            if len(dicts.get('conner')) != 4:
                conners = [[0, 0], [w, 0], [w, h], [0, h]]
            else:
                conner_boxes = dicts.get('conner')
                sorted_conner = sorted(conner_boxes, key=lambda item: item[2])
                top_point = sorted_conner[0:2]
                bot_point = sorted_conner[2:4]
                top_left, top_right = sorted(top_point, key=lambda item: item[1])
                bot_left, bot_right = sorted(bot_point, key=lambda item: item[1])

                conners = [top_left[1: 3], top_right[1: 3], bot_right[1: 3], bot_left[1: 3]]

        return conners

    @staticmethod
    def sort_file(imageFolder):
        allImgPath = glob.glob(imageFolder + "/*")
        list_path = list(allImgPath)
        list_path.sort(key=lambda x: os.path.getmtime(x))
        return list_path

    def scan(self, list_path, imgsz, stride, device, half, model, names):
        scan_path = [''] * len(list_path)
        for i, path_img in enumerate(list_path):
            image = cv2.imread(path_img)
            image = imutils.resize(image, height=780)
            (h, w, d) = image.shape

            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

            approx = self.get_conner(image, imgsz, stride, device, half, model, names)

            approx = np.asarray(approx, dtype=np.float32)
            op = cv2.getPerspectiveTransform(approx, pts)
            dst = cv2.warpPerspective(image, op, (w, h))
            dst = cv2.resize(dst, (w, h))
            dst = self.preprocess(dst, 1.7)

            dst = correct_skew(dst)

            # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            scan_path[i] = list_path[i].replace('.jpg', '_scan.jpg')
            # cv2.imwrite(scan_path[i], dst)
            # os.remove(list_path[i])
            # cv2.imshow('2', dst)
            # cv2.waitKey(0)
        return scan_path
