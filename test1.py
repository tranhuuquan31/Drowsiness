import cv2
import os
import numpy as np
from keras.models import load_model

video_path= 'vid1.MOV'
vid= cv2.VideoCapture(video_path)
mycnn= load_model('quan_model.h5')

while True:
    ret, frames = vid.read()
    cv2.imshow('Frame', frames)
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    frames = cv2.resize(frames, (256, 256)).astype("float32")

    preds = mycnn.predict(np.expand_dims(frames, axis=0))[0]
    print(preds)
    cv2.waitKey(20)

