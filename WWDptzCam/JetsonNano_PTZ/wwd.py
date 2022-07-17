import cv2
import numpy as np
import time
import math as m
from tracker import *
#cap = cv2.VideoCapture('inputvideo01.mp4')
cascade = cv2.CascadeClassifier('myhaar.xml')
#length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tracker = EuclideanDistTracker()

frame_counter = 0

def detect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected = cascade.detectMultiScale(gray_frame, 1.1, 13, 0, (24, 24))
    
    return detected
    
        
def wwdmain(frame):
    frame_counter = 0
    found = 0
    corners = np.array([])
    while True:
        #ret, frame = cap.read()
        if type(frame) == type(None):
            break
        frame = cv2.resize(frame, (1000, 600))
        cv2.putText(frame, "Wrong Way Detection",(30,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        if not (frame_counter % 1):
            detections = []
            cars = detect(frame)
            i=0;
            for x, y, w, h in cars:
                detections.append([x, y, w, h])
                i+=1
            boxes_ids = tracker.update(detections)
            isWrong={}
            for i in boxes_ids.keys():
                for j in range(1,len(boxes_ids[i].track)):
                    cv2.line(frame,(boxes_ids[i].track[j-1][0],boxes_ids[i].track[j-1][1]),(boxes_ids[i].track[j][0],boxes_ids[i].track[j][1]),(boxes_ids[i].color[0],boxes_ids[i].color[1],boxes_ids[i].color[2]),2)
                if(boxes_ids[i].track[0][1]+10>boxes_ids[i].track[-1][1]):
                    isWrong[i]=True
                else:
                    isWrong[i]=False
            for i in boxes_ids.values():
                x=i.center[0]
                y=i.center[1]
                w=i.dim[0]
                h=i.dim[1]
                clr=i.color
                # print(tracks[id])
                if(abs(i.track[0][1]-i.track[-1][1])>10):
                    cv2.rectangle(frame, (x-w//2, y-w//2), (x + w//2, y + h//2), (clr[0],clr[1],clr[2]),2)
                    if(isWrong[i.id]==True):
                        cv2.putText(frame,"Wrong Way", (x-w//2, y-h//2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv2.imshow('WWD', frame)
        frame_counter += 1
        return frame
        cv2.waitKey(20)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
#if __name__ == '__main__':
#    wwdmain()
#    cv2.destroyAllWindows()
#    cap.release()
