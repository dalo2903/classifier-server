import cv2
import os
import numpy as np
import json
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras_video_classifier.library.utility.frame_extractors.face_detector import detect_faces
import subprocess
import shlex

MAX_NB_CLASSES = 20


def get_rotation(file_path_with_file_name):
    """
    Function to get the rotation of the input video file.
    Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
    stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

    Returns a rotation None, 90, 180 or 270
    """
    cmd = "-loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
    args = shlex.split(cmd)
    print(file_path_with_file_name)
    ffprobe_location = r'C:\Users\Lenovo\Downloads\Compressed\ffmpeg-20181029-32d021c-win64-static\ffmpeg-20181029-32d021c-win64-static\bin\ffprobe.exe'
    args.insert(0, ffprobe_location)

    args.append(file_path_with_file_name)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args, shell=True).decode('utf-8')
    if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
        ffprobe_output = json.loads(ffprobe_output)
        rotation = ffprobe_output

    else:
        rotation = 0

    return rotation

def extract_vgg16_features_live(model, video_input_file_path):
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    file_name = os.path.basename(video_input_file_path).split('.')[0]
    rotation = get_rotation(video_input_file_path)
    rotation_code = {
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        0: 0
    }[rotation]
    print('rotation',rotation)
    success, image = vidcap.read()
    print(image.shape)
    features = []
    face_images = {}
    success = True

    # seconds = 1 
    count = 0
    fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))        
    # multiplier = fps * seconds 
    # print('fps: ', fps)

    while success:
        frameId = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        if rotation != 0:
            image = cv2.rotate(image, rotation_code)
        if frameId % fps == 0 and success: 
            print('extracted frame', frameId)
            if len(detect_faces(image))!=0:
                frame_name = file_name+'_frame'+str(count)+'.jpg'
                print(frame_name)
                face_images[frame_name] = image
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count += 1
    unscaled_features = np.array(features)
    return unscaled_features, face_images


def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    # seconds = 1
    fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))        
    # multiplier = fps * seconds
    # print('fps + multiplyer',seconds, fps, multiplier)
    while success:
        frameId = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        if frameId % fps == 0 and success: 
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

    if model is None:
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                print('extracting features: ',video_file_path)
                x = extract_vgg16_features(model, video_file_path, output_feature_file_path)
                y = f
                if x.any():
                    y_samples.append(y)
                    x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

