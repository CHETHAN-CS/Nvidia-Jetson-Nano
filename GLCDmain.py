import cv2
import jetson.inference
import jetson.utils
from datetime import datetime

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)

vs = cv2.VideoCapture('vdvdo.mp4')
#vs.set(3, 640)
#vs.set(4, 480)

while True:
    success, frame = vs.read()
    if not success:
        break
    cv2.resize(frame, (600, 600))
    imgCuda = jetson.utils.cudaFromNumpy(frame)
    (h, w) = frame.shape[:2]
    detections = net.Detect(imgCuda)
    for d in detections:
        x1,y1,x2,y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        if(className == "car" or className == "bicycle" or className == "bus"):
            print("ClassName:: ",className)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            y = x1 - 15 if y1 - 15 > 15 else y1 + 15
            cv2.putText(frame, className, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2 )
            if(y2-15>h-120):
                cv2.line(frame,(0,h-120),(w,h-120),(0,0,255),5)
                cv2.putText(frame,"SIGNAL JUMP DETECTED @"+str(datetime.now()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    cv2.imshow("Vehicle", frame)
    cv2.waitKey(1)

