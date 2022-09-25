import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from playsound import playsound

mycnn = load_model('quan_model.h5')
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
vs= cv2.VideoCapture('1.mp4')
# if not args.get("input", False):
# 	print("[INFO] Starting the live stream..")
# 	vs = cv2.VideoCapture('h1.mp4') #đổi camera thay video
# else:
# 	print("[INFO] Starting the video..")
# 	vs = cv2.VideoCapture(args["input"])
OUTPUT_VIDEO_FILE = r"D:\doantotnghiep\test\1.mp4"
writer = None
(W, H) = (None, None)
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# clone the output frame, then convert it from BGR to RGB0
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (64, 64))
	# frame -= mean
	frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5


	label = mycnn.predict(frame)


		# draw the activity on the output frame
	if label != 'open':
		#playsound('pippip.wav')
		text ="KHONG TAP TRUNG LAI XE"
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
	else:
		text = "HANH DONG: {}".format(label)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0,255,0), 3)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
			(W, H), True)
	# write the output frame to disk
	writer.write(output)
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# release the file pointers
print("[INFO] cleaning up...")
#vs.release()
vs.release()

