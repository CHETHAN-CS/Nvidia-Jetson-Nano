
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
import math as maths
from datetime import datetime
import os

########################################################################################################################

net = cv2.dnn.readNet("yolo/yolov3_608.weights", "yolo/yolov3_608.cfg")

with open("yolo/yolov3_608.names", 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# NN input image size
inpWidth = 608
inpHeight = 608

# Input video size
width = None
height = None
car_ct = CentroidTracker()
car_ct.maxDisappeared = 10
truck_ct = CentroidTracker()
truck_ct.maxDisappeared = 10

# the list of trackers
trackers = []
# lists of objects we are going to track
car_trackableObjects = {}
truck_trackableObjects = {}
totalFrames = 0
frame_number = 0
old_car_trackers = None
old_truck_trackers = None
stopped_car_IDs = []
stopped_truck_IDs = []
car_counting_frames = {}
truck_counting_frames = {}
car_counting_seconds = {}
truck_counting_seconds = {}
parts = 80
frames_to_stop = 30


def draw_centroids(frame, objects, trackableObjects, long_stopped_cars):
    for (objectID, centroid) in objects.items():
        # check if a trackable objects exists for particular ID
        to = trackableObjects.get(objectID, None)

        # if it doesn't then we create a new one corresponding to the given centroid
        if to is None:
            to = TrackableObject(objectID, centroid)

        # place the trackable object into the dict.
        trackableObjects[objectID] = to

        # drawing circle and text
        if objectID in long_stopped_cars:
            text = "ID {} PARKING".format(objectID + 1)
            # if a car is not moving then we draw a large yellow centroid
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 6, (0, 255, 255), -1)
            cv2.putText(frame,"PARKING DETECTED @"+str(datetime.now()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,0),2)
            sdate=((str(datetime.now())[0:19]).replace(' ','_')).replace(':','_')
            foldername='./violation_images/'+sdate[0:10]
            try:
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
            except OSError:
                print('error while creating directory')
            pname=foldername+'/'+sdate+'.jpg'
            cv2.imwrite(pname, frame)
        else:
            text = "ID {}".format(objectID + 1)
            # else we draw a smaller green centroid
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)


# Function calculates distance between two centroids
def find_distance(c1, c2):
    c1 = c1.tolist()
    c2 = c2.tolist()
    return int(maths.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2))

# Function compares coords on the certain car's box on N and N+1 frames
# It returns a list of cars that are, PERHAPS, not moving
def compare_trackers(old_trackers, new_trackers, frame_width, stopped_car_IDs):
    for (old_objectID, old_centroid) in old_trackers.items():
        for (new_objectID, new_centroid) in new_trackers.items():
            if old_objectID == new_objectID:
                distance = find_distance(old_centroid, new_centroid)
                #print(f"Distance between centroids of car number {old_objectID+1} is {distance}")
                # If the distance between centroids is less than 1/N of width of the frame then we add it to the list
                if distance < frame_width / parts:
                    if new_objectID not in stopped_car_IDs:
                        stopped_car_IDs.append(new_objectID)
                else:
                    if new_objectID in stopped_car_IDs:
                        # If the distance is more than 1/N then it means that the car started moving again - delete it from the list
                        stopped_car_IDs.remove(new_objectID)
            # if a car has moved away from the frame and we can not see it anymore then we should
            # delete it from the list of stopped cars
            if old_objectID not in new_trackers.keys():
                if old_objectID in stopped_car_IDs:
                    stopped_car_IDs.remove(old_objectID)
    # if new_trackers are an empty array, that means that there are NO cars of a frame at all
    # so we should clear stopped_car_IDs
    if len(new_trackers.keys()) == 0:
        if stopped_car_IDs != []:
            stopped_car_IDs.clear()

    return stopped_car_IDs


# Function finds cars that were not moving long enough
def find_stopped_cars(counting_frames, frames_to_stop):
    long_stopped_cars = []
    for ID, frames in counting_frames.items():
        if frames > frames_to_stop: # this number can be changed to increase work efficiency
            long_stopped_cars.append(ID)
    return long_stopped_cars



########################################################################################################################

def noparkingdtn(frame):
    frame_number += 1
    # change frame size to increase speed a bit
    frame = imutils.resize(frame, width=900)
    (h, w) = frame.shape[:2]
    frame[0:70,0:w]=[0,0,255]
    #change colors from RGB to BGR to work in dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if width is None or height is None:
        height, width, channels = frame.shape

    # lists of bounding boxes
    car_rects = []
    truck_rects = []
    cv2.putText(frame, "Wrong Way Detection",(30,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    if totalFrames % 5 == 0:
        # empty list of trackers
        trackers = []
        # list of classes numbers
        class_ids = []

        # pass the blob-model of the frame through the NN to get boxes of detected objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # analyze boxes list
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if (class_id != 2) and (class_id != 7):  # if a car or a truck is detected - continue
                    continue
                confidence = scores[class_id]
                if confidence > 0.98:
                    # box'es center coords
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    # width of the box
                    w = int(detection[2] * width)
                    # height of the box
                    h = int(detection[3] * height)

                    # if a box is too small (for example if a car is moving very close to the edge of a frame, then we do skip it
                    if h <= (height / 10):
                        continue

                    # coords of left upper and right lower connors of the box
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = x1 + w
                    y2 = y1 + h

                    # let's make a maximum distance of centroid tracker equal to the width of a box
                    truck_ct.maxDistance = w
                    car_ct.maxDistance = w

                    # draw a box and write a detected class above it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # create a tracker for every car
                    tracker = dlib.correlation_tracker()
                    # create a dlib rectangle for a box
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    # start tracking each box
                    tracker.start_track(rgb, rect)
                    # every tracker is placed into a list
                    trackers.append(tracker)
                    class_ids.append(class_id)


    # if frame number is not N then we work with previously created list of trackers rather that boxes
    else:
        for tracker, class_id in zip(trackers, class_ids):
            # a car was detected on one frame and after that on other frames it's coords are constantly updating
            tracker.update(rgb)

            pos = tracker.get_position()

            # get box coords from each tracker
            x1 = int(pos.left())
            y1 = int(pos.top())
            x2 = int(pos.right())
            y2 = int(pos.bottom())

            # draw a box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            obj_class = CLASSES[class_id]

            if obj_class == "car":
                car_rects.append((x1, y1, x2, y2))
            elif obj_class == "truck":
                truck_rects.append((x1, y1, x2, y2))
    cars = car_ct.update(car_rects)
    trucks = truck_ct.update(truck_rects)
    # get the IDs of cars that are, perhaps, stopped
    if (old_car_trackers is not None):
        stopped_car_IDs = compare_trackers(old_car_trackers, cars, width, stopped_car_IDs)
    if (old_truck_trackers is not None):
        stopped_truck_IDs = compare_trackers(old_truck_trackers, trucks, width, stopped_truck_IDs)
        if stopped_car_IDs != []:
            for ID in stopped_car_IDs:
                # Increasing the number of frames
                if ID in car_counting_frames.keys():
                    car_counting_frames[ID] += 1
                # Adding a new car ID
                else:
                    car_counting_frames[ID] = 1
            # if any ID is IN car_counting_frames.keys() but it os NOT IN the stopped_car_IDs then we have to delete
            # from the dictionary as it means that the car is stopped and moving at the same time which is impossible
            for ID in car_counting_frames.copy().keys():
                if ID not in stopped_car_IDs:
                    car_counting_frames.pop(ID)
        else:
            # If a list is empty it means that there are no cars to process
            car_counting_frames = {}
        # same thing for trucks (you can add your classed here)
        if stopped_truck_IDs != []:
            for ID in stopped_truck_IDs:
                if ID in truck_counting_frames.keys():
                    truck_counting_frames[ID] += 1
                else:
                    truck_counting_frames[ID] = 1

            for ID in truck_counting_frames.copy().keys():
                if ID not in stopped_truck_IDs:
                    truck_counting_frames.pop(ID)
        else:
            truck_counting_frames = {}
    long_stopped_cars = find_stopped_cars(car_counting_frames, frames_to_stop)
    long_stopped_trucks = find_stopped_cars(truck_counting_frames, frames_to_stop)
    # now when we have a list of cars that are for sure stopped we can count how long (in seconds) they are not moving
    for ID in long_stopped_cars:
        if ID not in car_counting_seconds.keys():
            # if it is a new car then we pinpoint time when car stops
            start = time.asctime()
            # [0,0] will later be replaced
            car_counting_seconds[ID] = [0,0]
            car_counting_seconds[ID][0] = start
        else:
            # else if this car is already on the list then we add time to it's current time
            stop = time.asctime()
            car_counting_seconds[ID][1] = stop

    # do the same thing but for trucks
    for ID in long_stopped_trucks:
        if ID not in truck_counting_seconds.keys():
            # if it is a new car then we pinpoint time when car stops
            start = time.asctime()
            truck_counting_seconds[ID] = [0, 0]
            truck_counting_seconds[ID][0] = start
        else:
            # else if this car is already on the list then we add time to it's current time
            stop = time.asctime()
            truck_counting_seconds[ID][1] = stop


    old_car_trackers = cars.copy()
    old_truck_trackers = trucks.copy()

    # draw centroids for cars and trucks
    draw_centroids(frame, cars, car_trackableObjects, long_stopped_cars)
    draw_centroids(frame, trucks, truck_trackableObjects, long_stopped_trucks)
    totalFrames += 1
    return frame
    cv2.waitKey(20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
