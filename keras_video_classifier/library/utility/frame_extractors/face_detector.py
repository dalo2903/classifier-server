import numpy as np
import cv2
import os

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(patch_path('haarcascade_frontalface_default.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    print('faces: ', faces)
    return faces