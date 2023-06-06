import random
import cv2
import os
import time

from PIL import Image
from DetecInfoBoxes.GetBoxes import GetDictionary

getDictionary = GetDictionary()


class ReadInfo:
    def __init__(self, imgsz, stride, device, half, model, names, opt, ocrPredictor):
        self.imgsz = imgsz
        self.stride = stride
        self.device = device
        self.half = half
        self.model = model
        self.names = names
        self.opt = opt
        self.ocrPredictor = ocrPredictor

    @staticmethod
    def arrange_info(infos: list):
        sorted_infos = sorted(infos, key=lambda item: item[2])
        return sorted_infos

    def ocr_info(self, img, info: list):
        x_min = info[1] - int(info[3] / 2)
        y_min = info[2] - int(info[4] / 2)
        w = info[3]
        h = info[4]

        crop_img = img[y_min:y_min + h, x_min:x_min + w]
        name = random.random()
        cv2.imwrite('Img/OcrImg/' + str(name) + '.jpg', crop_img)
        imagePill = Image.open('Img/OcrImg/' + str(name) + '.jpg')
        text = self.ocrPredictor.predict(imagePill)
        os.remove('Img/OcrImg/' + str(name) + '.jpg')

        return text

    def get_all_info(self, img_path):
        st = time.time()
        img = cv2.imread(img_path)
        page_boxes = getDictionary.prediction(img, self.imgsz, self.stride, self.device, self.half, self.model, self.names, self.opt)
        page_boxes = getDictionary.dict_processing(page_boxes)

        fields = ["id", "full_name", "date_of_birth", "sex", "nationality", "place_of_origin", "place_of_residence", "date_of_expiry", "qr_code"]

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

        return user_info_dict
