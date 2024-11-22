import random
import cv2
import os
import time

from PIL import Image
from DetecInfoBoxes.GetBoxes import Detect
from util import correct_skew, sharpen
from config import opt

get_dictionary = Detect(opt)


class ReadInfo:
    def __init__(self, imgsz, stride, device, half, model, names, ocr_predictor):
        self.imgsz = imgsz
        self.stride = stride
        self.device = device
        self.half = half
        self.model = model
        self.names = names
        self.opt = opt
        self.ocrPredictor = ocr_predictor

    @staticmethod
    def get_the_most_confident_bbox(page_boxes: dict):
        for key in page_boxes:
            value = page_boxes.get(key)
            value = sorted(value, key=lambda item: item[4])
            page_boxes.update({key: [value[-1]]})

        return page_boxes

    @staticmethod
    def arrange_info(infos: list):
        sorted_infos = sorted(infos, key=lambda item: item[1])
        return sorted_infos

    def ocr_info(self, img, info: list):
        x_min = info[0] - int(int(info[2]) / 2)
        y_min = info[1] - int(int(info[3]) / 2)
        w = info[2]
        h = info[3]

        crop_img = img[y_min:y_min + h, x_min:x_min + w]
        name = random.random()
        cv2.imwrite('Img/OcrImg/' + str(name) + '.jpg', crop_img)
        image_pill = Image.open('Img/OcrImg/' + str(name) + '.jpg')
        text = self.ocrPredictor.predict(image_pill)
        os.remove('Img/OcrImg/' + str(name) + '.jpg')

        return text

    def get_all_info(self, img_path):
        st = time.time()
        img = cv2.imread(img_path)
        img = correct_skew(img)
        # img = sharpen(img, 5)

        page_boxes = get_dictionary.prediction(img, self.imgsz, self.stride, self.device, self.half, self.model,
                                              self.names)
        page_boxes = get_dictionary.dict_processing(page_boxes)

        fields = ["id", "full_name", "date_of_birth", "sex", "nationality", "place_of_origin", "place_of_residence",
                  "date_of_expiry", "qr_code"]
        # fields = ["id", "name", "birthDay", "birthPlace", "partyDate", "official", "issuePlace",
        #           "date", "man", "woman"]

        user_info_dict = {}
        for field in fields:
            if field == "qr_code":
                continue

            infos = page_boxes.get(field)

            if infos:
                if len(infos) != 1:
                    infos = self.arrange_info(infos)
                    all_text = ''
                    for info in infos:
                        text = self.ocr_info(img, info)
                        all_text += text + ' '
                else:
                    all_text = self.ocr_info(img, infos[0])
                user_info_dict.update({field: all_text.strip()})

            else:
                user_info_dict.update({field: ''})

        print('Full Time: ', time.time() - st)
        get_dictionary.new_draw_boxes(page_boxes, img)
        cv2.imwrite(img_path, img)

        return user_info_dict

    def get_vehicle_registration_info(self, img_path):
        st = time.time()
        img = cv2.imread(img_path)
        img = correct_skew(img)

        page_boxes = get_dictionary.prediction(img, self.imgsz, self.stride, self.device, self.half, self.model,
                                              self.names)
        page_boxes = self.get_the_most_confident_bbox(get_dictionary.dict_processing(page_boxes))

        fields = ["name", "addr", "brand", "model", "engine", "chassis", "color", "plate", "type", "seat_capacity",
                  "capacity", "origin", "dob"]

        user_info_dict = {}
        for field in fields:
            infos = page_boxes.get(field)

            if infos:
                all_text = self.ocr_info(img, infos[0])
                user_info_dict.update({field: all_text.strip()})

            else:
                user_info_dict.update({field: ''})

        print('Full Time: ', time.time() - st)
        get_dictionary.new_draw_boxes(page_boxes, img)
        cv2.imwrite(img_path, img)
        return user_info_dict