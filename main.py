import random
import os
import sys
import time

from flask import Flask, request, render_template
from readInfoIdCard import ReadInfo
from DetecInfoBoxes.GetBoxes import GetDictionary
from Vocr.tool.predictor import Predictor
from Vocr.tool.config import Cfg as Cfg_vietocr

app = Flask(__name__)

getDictionary = GetDictionary()
sys.path.insert(0, 'DetecInfoBoxes')

opt = {
    "img-size": 800,
    "conf-thres": 0.5,
    "iou-thres": 0.15,
    "device": 'cpu',
}

# Load Ocr model
config_vietocr = Cfg_vietocr.load_config_from_file('Vocr/config/vgg-seq2seq.yml')
config_vietocr['weights'] = 'Models/seq2seqocr.pth'
config_vietocr['device'] = 'cpu'
ocrPredictor = Predictor(config_vietocr)

# Load Yolo model
scan_weight = 'Models/cccdYoloV7.pt'
imgsz, stride, device, half, model, names = getDictionary.load_model(scan_weight, opt)

readInfo = ReadInfo(imgsz, stride, device, half, model, names, opt, ocrPredictor)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/extract', methods=['POST'])
def process():
    image = request.files['myfile']

    if image:
        # t1 = time.time()
        save_path = 'Img/OriginalImage/' + str(random.random()) + '.jpg'
        image.save(save_path)

        dicts = readInfo.get_all_info(save_path)
        os.remove(save_path)
        # t2 = time.time()
        # print('TIME: ', t2 - t1)
    else:
        dicts = 'Bạn chưa chọn ảnh!'
    return render_template("img_submit.html", results=dicts)


if __name__ == '__main__':
    app.run(debug=True)
