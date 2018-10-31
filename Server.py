#import numpy as np
from flask import Flask, request
import re
import sys
import os
import cv2
import re
import requests
import base64
import numpy as np
import keras_video_classifier
import json
import asyncio
import tensorflow as tf
from multiprocessing import Process
from keras import backend as K


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello'

# @app.route('/download')
# def download_test():
#     download_path = patch_path('downloads')
#     if not os.path.exists(download_path):
#         os.makedirs(download_path)
#     url = request.args.get('url')
#     file_path = download_file(url, download_path)
#     return file_path

@app.route('/classify')
def classify_video():
    download_path = patch_path('downloads')
    if not os.path.exists(download_path):
        os.makedirs(download_path)    
    url = request.args.get('url')
    file_path = download_file(url, download_path)
    print(file_path)
    result, face_images = classify(file_path)
    data = {}
    data['video_id'] = str(url[url.rfind("/")+1:]).split('.')[0]
    data['result'] = result
    data['images'] = len(face_images)
    for name, img in face_images.items():
        success, buffer = cv2.imencode('.jpg',img)
        if success:
            cv2.imshow("img",img)
            # cv2.waitKey(0)
            # print(success,buffer)
            face_images[name] = buffer
    response = json.dumps(data)

    p = Process(target=send_images, args = (face_images, data,))
    p.start()
    os.remove(file_path)
    # json.dumps(np.array(face_images).tolist())
    # K.clear_session()
    # tf.reset_default_graph()
    return response

def send_images(images, data):
    url = 'http://localhost/api/visual-data/'
    files = []
    for k, v in images.items():        
        files.append((k,base64.b64encode(v)))
    print('Sending images to ' + url)
    r = requests.post(url,files=files, data=data)
    print(r)

def classify(file_path):
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier

    vgg16_include_top = True
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models/dataset')

    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    np.random.seed(42)
    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)
    predicted_label, face_images = predictor.predict(file_path)

    return predicted_label, face_images


def download_file(url, download_path):
    r = requests.get(url, allow_redirects=True)
    filename = url[url.rfind("/")+1:]
    print('downloading url:',url )
    open(os.path.join(download_path , filename), 'wb').write(r.content)
    return os.path.join(download_path, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

