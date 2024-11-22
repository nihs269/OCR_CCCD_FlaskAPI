import random
import os
import sys
import time

from flask import Flask, request, render_template
from readInfoIdCard import ReadInfo
from DetecInfoBoxes.GetBoxes import Detect
from Vocr.tool.predictor import Predictor
from Vocr.tool.config import Cfg as Cfg_vietocr
from config import opt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

get_dictionary = Detect(opt)
sys.path.insert(0, 'DetecInfoBoxes')


# Load Ocr model
config_vietocr = Cfg_vietocr.load_config_from_file('Vocr/config/vgg-seq2seq.yml')
config_vietocr['weights'] = 'Models/seq2seqocr.pth'
# config_vietocr['weights'] = 'Models/OCR_Vehicle_Registration_0.95.pt'
config_vietocr['device'] = 'cpu'
ocr_predictor = Predictor(config_vietocr)

# Load Yolo model
scan_weight = 'Models/cccdYoloV7.pt'
# scan_weight = 'Models/TheDang.pt'
imgsz, stride, device, half, model, names = get_dictionary.load_model(scan_weight)
read_info = ReadInfo(imgsz, stride, device, half, model, names, ocr_predictor)

# vehicle_registration_yolo_weight = 'Models/DangKyXe.pt'
# imgsz, stride, device, half, model, names = get_dictionary.load_model(vehicle_registration_yolo_weight)
# vehicle_registration_reader = ReadInfo(imgsz, stride, device, half, model, names, ocr_predictor)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/extract', methods=['POST', 'GET'])
def process():
    try:
        save_path = 'static/' + str(random.random()) + '.jpg'
        image = request.files['myfile']

        if image:
            # t1 = time.time()
            image.save(save_path)
            # dicts = vehicle_registration_reader.get_vehicle_registration_info(save_path)
            dicts = read_info.get_all_info(save_path)
            # os.remove(save_path)
            # t2 = time.time()
            # print('TIME: ', t2 - t1)
        else:
            dicts = 'Bạn chưa chọn ảnh!'
        return render_template("img_submit.html", results=dicts, img_path=save_path)
    except:
        return render_template("home.html")


if __name__ == '__main__':
    app.run(debug=True, port=6123)
