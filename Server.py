
#import numpy as np
from flask import Flask, request

import sys
import os
import requests

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

app = Flask(__name__)

@app.route('/classify')
def classify_video():
    download_path = patch_path('downloads')
    url = request.args.get('url', '')
    file_path = download_file(url, download_path)
    classify(file_path)


def classify(file_path):
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier

    vgg16_include_top = False
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models/dataset')

    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)


    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)


    predicted_label = predictor.predict(file_path)
    print('predicted: ' + predicted_label + ' actual: ' + label)


def download_file(url, download_path):
    r = requests.get(url, allow_redirects=True)
    filename = get_filename_from_cd(r.headers.get('content-disposition'))
    open(download_path + filename, 'wb').write(r.content)
    return os.path.join(download_path, filename)


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    file_name = re.findall('filename=(.+)', cd)
    if len(file_name) == 0:
        return None
    return file_name[0]


if __name__ == "__main__":
    app.run(host='0.0.0.0')