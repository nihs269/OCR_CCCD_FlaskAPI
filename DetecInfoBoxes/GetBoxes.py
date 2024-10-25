import cv2
import torch
import numpy as np

from DetecInfoBoxes.models.experimental import attempt_load
from DetecInfoBoxes.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from DetecInfoBoxes.utils.torch_utils import select_device


class Detect:
    def __init__(self, opt):
        self.opt = opt

    def load_model(self, weights):
        """Hàm load model YOLO V7

                Args:
                    weights (str): Đường dẫn đến file trọng số model YOLO V7 (.pt) hoặc (.pth).

                Returns:
                    imgsz, stride, device, half, model, names: Các tham số đầu ra phục vụ cho quá trình predict của model.

                """
        with torch.no_grad():
            imgsz = self.opt['img-size']
            set_logging()
            device = select_device(self.opt['device'])
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
        """Hàm chuyển đổi dict từ dạng {label: [x_min, y_min, x_max, y_max, conf]} sang dạng {label: [x_center, y_center, width, height, conf, label]}.

        Args:
            dictionary (dict): dictionary đầu vào.

        Returns:
            new_dictionary (dict): dictionary sau khi được chuyển đổi sang dạng mới (new_dictionary).

        """
        new_dictionary = {}
        for item in dictionary.items():
            new_bbox = []
            for bbox in item[1]:
                new_bbox.append([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2), int(bbox[2] - bbox[0]),
                                 int(bbox[3] - bbox[1]), bbox[4], item[0]])
            new_dictionary.update({item[0]: new_bbox})
        return new_dictionary

    def prediction(self, img_official: np.ndarray, imgsz, stride, device, half, model, names):
        """Hàm đưa ra dự đoán của model YOLO

        Args:
            img_official (np.ndarray): Ảnh đầu vào cho model dự đoán.
            imgsz, stride, device, half, model, names: Các tham số có được từ quá trình load model.

        Returns:
            dictionary (dict): dictionary dạng {label: [x_min, y_min, x_max, y_max, conf]}.
        """
        img = self.letterbox(img_official, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

        classes = None
        pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes=classes,
                                   agnostic=False)
        dictsLabel = {}
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_official.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'
                    confident = f'{conf:.2f}'

                    if label not in dictsLabel.keys():
                        dictsLabel.update({label: [[xyxy[0], xyxy[1], xyxy[2], xyxy[3], confident]]})
                    else:
                        values = dictsLabel[label]
                        values.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], confident])
                        dictsLabel.update({label: values})
        return dictsLabel

    # Used for testing
    @staticmethod
    def new_draw_boxes(new_dicts: dict, image_draw: np.ndarray):
        """Hàm vẽ kết quả detect của model YOLO

        Args:
            new_dicts (dict): Dictionary (dạng {label: [x_center, y_center, width, height, conf, label]}) thu được sau khi chuyển
            đổi thông qua hàm dict_processing().
            image_draw (np.ndarray): Ảnh để vẽ dự đoán của model.

        Returns:
            None
        """
        for label in new_dicts:
            horizontal_list = new_dicts.get(label)
            for bbox in horizontal_list:
                x_center = int(bbox[0])
                y_center = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                cv2.putText(image_draw, str(bbox[5]),
                            (x_center - int(w / 2) + 1, y_center - int(h / 2) - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(image_draw, (x_center - int(w / 2), y_center - int(h / 2)),
                              (x_center + int(w / 2), y_center + int(h / 2)), (0, 0, 255), 1)

    @staticmethod
    def draw_boxes(dicts: dict, image_draw: np.ndarray):
        """Hàm vẽ kết quả dự đoán của model YOLO

        Args:
            dicts (dict): Dictionary (dạng {label: [x_min, y_min, x_max, y_max, conf]}) thu được sau khi model dự đoán.
            image_draw (np.ndarray): Ảnh để vẽ dự đoán của model.

        Returns:
            None
        """
        for label in dicts:
            horizontal_list = dicts.get(label)
            for bbox in horizontal_list:
                x_min = int(bbox[0])
                y_min = int(bbox[1])
                x_max = int(bbox[2])
                y_max = int(bbox[3])
                cv2.putText(image_draw, str(label) + ' ' + str(bbox[4]), (x_min + 1, y_min - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 255))
                cv2.rectangle(image_draw, (x_min, y_min),
                              (x_max, y_max), (255, 0, 255), 1)