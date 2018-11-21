import numpy as np
import cv2
import os
# import face_recognition
# import time

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
# faces_encoding =[]
# start = time.clock()
# def detect_faces(img):
#     # image = face_recognition.load_image_file(img,mode="RGB")
#     #image_90=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
#     # cv2.imshow("IMG",image_90)
#     # cv2.waitKey()
#     face_locations = face_recognition.face_locations(img)
#     # face_encode = face_recognition.face_encodings(image,face_locations,1)[0]
#    # faces_encoding.append(face_encode)
#     print(face_locations)
#     # print(face_encode)

# recog_faces("myfile.jpg")
# #recog_faces("myfile2.jpg")
# unk_image = face_recognition.load_image_file("difftest2.jpg")
# face_locations = face_recognition.face_locations(unk_image)
# print("face loc tets:",face_locations)
# face_encode = face_recognition.face_encodings(unk_image,face_locations,1)[0]
# print("Face encode: ",face_encode)
# print(type(face_encode))
# print("dist: ",face_recognition.face_distance(faces_encoding,face_encode))
# print("Compare: ", face_recognition.compare_faces(faces_encoding,face_encode,tolerance=0.2))
# #recog_faces("myfile3.jpg")
# print(len(faces_encoding))
# print(time.clock()-start)