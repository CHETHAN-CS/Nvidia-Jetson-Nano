# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
#from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from datetime import datetime
import os
#print(datetime.now())


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] loading video...")


uchoice = int(input("1.vehicle detection   2.Signal jump violation \n"))

if uchoice == 2:
	vs = cv2.VideoCapture('inputvideo2.mp4')
	while (vs.isOpened()):
		ret, frame = vs.read()
		if not ret:
			break
		(h, w) = frame.shape[:2]
		frame[0:70,0:w]=[0,0,255]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 0.007843, (600, 600), 127.5)
		cv2.line(frame,(0,h-120),(w,h-120),(0,255,255),2)
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
				obname=label.split(':')
				if((obname[0]=='car') or (obname[0]=='bicycle') or (obname[0]=='bus') or (obname[0]=='motorbike')):
					cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					cv2.putText(frame,str(datetime.now()),(800,h-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
					if(endY-15>h-120):
						cv2.line(frame,(0,h-120),(w,h-120),(0,0,255),5)
						cv2.putText(frame,"SIGNAL JUMP DETECTED @"+str(datetime.now()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
						sdate=((str(datetime.now())[0:19]).replace(' ','_')).replace(':','_')
						foldername='./violation_images/'+sdate[0:10]
						try:
							if not os.path.exists(foldername):
								os.makedirs(foldername)
						except OSError:
							print('error while creating directory')
						pname=foldername+'/'+sdate+'.jpg'
						cv2.imwrite(pname, frame)
				cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	vs.release()
	cv2.destroyAllWindows()
	exit()

else:
	vs = cv2.VideoCapture('vdvdo.mp4')
	while (vs.isOpened()):
		ret, frame = vs.read()
		if not ret:
			break
		(h, w) = frame.shape[:2]
		#frame[0:70,0:w]=[0,0,255]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 0.007843, (600, 600), 127.5)
		
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
				obname=label.split(':')
				if((obname[0]=='car') or (obname[0]=='bicycle') or (obname[0]=='bus')):
					cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					cv2.putText(frame,str(datetime.now()),(800,h-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
				cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	vs.release()
	cv2.destroyAllWindows()
	exit()

	
