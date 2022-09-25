import cv2
import os
from keras.models import load_model

video_path= 'vid1.MOV'
vid= cv2.VideoCapture(video_path)
mycnn= load_model('quan_model.h5')

while True:
    ret, frames = vid.read()
    cv2.imshow('Frame', frames)
    print(frames)
    # dim = (256,256)
    # frames= cv2.resize(frames, dim)
    cv2.waitKey(20)
    prediction = mycnn.predict(frames.shape)
    if prediction < 0.5:
        print('yeu')

