import cv2
import torch
import numpy as np

from DetecInfoBoxes.models.experimental import attempt_load
from DetecInfoBoxes.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from DetecInfoBoxes.utils.torch_utils import select_device


class GetDictionary:
    @staticmethod
    def load_model(weights, opt):
        with torch.no_grad():
            imgsz = opt['img-size']
            set_logging()
            device = select_device(opt['device'])
            half = device.type != 'cpu'
            model = attempt_load(weights, map_location=device)
            stride = int(model.stride.max())
            imgsz = check_img_size(imgsz, s=stride)
            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        return imgsz, stride, device, half, model, names

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):

        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, ratio, (dw, dh)

    @staticmethod
    def dict_processing(dictionary: dict):
        new_dictionary = {}
        for item in dictionary.items():
            new_bbox = []
            for bbox in item[1]:
                new_bbox.append([item[0], int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])
            new_dictionary.update({item[0]: new_bbox})
        return new_dictionary

    def prediction(self, imgOfficial, imgsz, stride, device, half, model, names, opt):

        img = self.letterbox(imgOfficial, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes,
                                   agnostic=False)
        dictsLabel = {}
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgOfficial.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'
                    # confident = f'{conf:.2f}'

                    if label not in dictsLabel.keys():
                        dictsLabel.update({label: [[xyxy[0], xyxy[1], xyxy[2], xyxy[3]]]})
                    else:
                        values = dictsLabel[label]
                        values.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                        dictsLabel.update({label: values})
        return dictsLabel

    # Used for testing
    @staticmethod
    def show_result(dictsLabel, image_draw):
        for label in dictsLabel:
            horizontal_list = dictsLabel.get(label)
            for bbox in horizontal_list:
                x_center = int(bbox[1])
                y_center = int(bbox[2])
                w = int(bbox[3])
                h = int(bbox[4])
                if bbox[0] == 'choosen_ans':
                    # cv2.putText(image_draw, str(bbox[0]), (x_center - int(w/2) + 1, y_center - int(h/2) - 10),
                    #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0))
                    cv2.rectangle(image_draw, (x_center - int(w/2), y_center - int(h/2)), (x_center + int(w/2), y_center + int(h/2)), (0, 255, 0), 2)
                elif bbox[0] == 'not_choosen_ans':
                    # cv2.putText(image_draw, str(bbox[0]), (x_center - int(w / 2) + 1, y_center - int(h / 2) - 10),
                    #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0))
                    cv2.rectangle(image_draw, (x_center - int(w / 2), y_center - int(h / 2)),
                                  (x_center + int(w / 2), y_center + int(h / 2)), (255, 0, 0), 2)
                elif bbox[0] == 'question':
                    # cv2.putText(image_draw, str(bbox[0]), (x_center - int(w / 2) + 1, y_center - int(h / 2) - 10),
                    #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255))
                    cv2.rectangle(image_draw, (x_center - int(w / 2), y_center - int(h / 2)),
                                  (x_center + int(w / 2), y_center + int(h / 2)), (0, 0, 255), 2)
                else:
                    # cv2.putText(image_draw, str(bbox[0]), (x_center - int(w / 2) + 1, y_center - int(h / 2) - 10),
                    #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255))
                    cv2.rectangle(image_draw, (x_center - int(w / 2), y_center - int(h / 2)),
                                  (x_center + int(w / 2), y_center + int(h / 2)), (0, 0, 0), 2)
