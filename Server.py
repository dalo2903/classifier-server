#import numpy as np
from flask import Flask, request, abort, Response
import re
import sys
import os
import cv2
import re
import requests
import numpy as np
import keras_video_classifier
import json
import asyncio
# import tensorflow as tf
# from keras import backend as K
SECRET_KEY = 'hK0Oeiy2zft0ZZtbmaawNW5zT0Ebcybn'
ADMIN_ID = '5b86b4a8a96c6a000446705b'
GODSEYE_URL = 'http://localhost/api/visual-data/classified'

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'hello'

# @app.route('/download')
# def download_test():
#     download_path = patch_path('downloads')
#     if not os.path.exists(download_path):
#         os.makedirs(download_path)
#     url = request.args.get('url')
#     file_path = download_file(url, download_path)
#     return file_path
# @app.route('/test')
# def test():
#     img = cv2.imread('45173636_284087462439146_441652943041593344_n.jpg')
#     name = "hahaha.jpg"
#     success, buffer = cv2.imencode('.jpg',img)
#     face_images = {}
#     data = {}
#     if success:
#         # cv2.imshow("img",img)
#         # cv2.waitKey(0)
#         # print(success,buffer)
#         face_images[name] = buffer.tostring()
#     response = json.dumps(data)

#     p = Process(target=send_images, args = (face_images, data,))
#     p.start()
#     return "done"
#     p.join()

@app.route('/classify')
def classify_video():
    secret = request.args.get('secret')
    if(secret != SECRET_KEY):
        error = {}
        error['code'] = 403
        error['message'] = 'Invalid secret key'
        return json.dumps(error),403
    download_path = patch_path('downloads')
    if not os.path.exists(download_path):
        os.makedirs(download_path)    
    url = request.args.get('url')
    file_path = download_file(url, download_path)
    print(file_path)
    result, face_images = classify(file_path)
    data = {}
    data['id'] = ADMIN_ID
    data['video_id'] = str(url[url.rfind("/")+1:]).split('.')[0]
    data['result'] = result
    data['images'] = len(face_images)
    for name, img in face_images.items():
        success, buffer = cv2.imencode('.jpg',img)
        if success:
            # cv2.imshow("img",img)
            # cv2.waitKey(0)
            # print(success,buffer)
            face_images[name] = buffer
    # p = Process(target=send_images, args = (face_images, data,))
    # p.start()
    send_images(face_images, data)
    os.remove(file_path)
    return json.dumps(data)

def send_images(images, data):
    files = []
    for k, v in images.items():        
        files.append((k,v))
    print('Sending images to ' + GODSEYE_URL)
    r = requests.post(GODSEYE_URL,files=files, data=data)
    print(r)
    return

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

