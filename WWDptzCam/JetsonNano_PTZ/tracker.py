import math
import random

class Car:
    def __init__(self,x,y,id,w,h):
        self.track=[(x,y)]
        self.id=id
        self.dim=(w,h)
        self.center=(x,y)
        self.color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    def updateCenter(self,x,y,w,h):
        self.track.append((x,y))
        self.center=(x,y)
        self.dim=(w,h)


class EuclideanDistTracker:
    def __init__(self):
        # self.tracks={}
        # # Store the center positions of the objects
        # self.center_points = {}
        
        # self.colors={}
        # # Keep the count of the IDs
        # # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.cars={}

    def update(self, objects_rect):
        # Objects boxes and ids
        ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            # print(self.cars)
            for car in self.cars.values():
                # print(car.id)
                id=car.id
                dist = math.hypot(cx - car.center[0], cy - car.center[1])

                if dist < min(w/2,h/2):
                    car.updateCenter(cx, cy,w,h)
                    ids.append(id)
                    # objects_bbs_ids.append([x, y, w, h, id,self.colors[id]])
                    # self.tracks[id].append((cx,cy))
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.cars[self.id_count]=Car(cx,cy,self.id_count,w,h)
                ids.append(self.id_count)
                # self.center_points[self.id_count] = (cx, cy)
                # self.colors[self.id_count]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
                # objects_bbs_ids.append([x, y, w, h, self.id_count,self.colors[self.id_count]])
                # self.tracks[self.id_count]=[(cx,cy)]
                self.id_count += 1
        # Clean the dictionary by center points to remove IDS not used anymore
        # new_center_points = {}
        # for obj_bb_id in objects_bbs_ids:
        #     _, _, _, _, object_id,_ = obj_bb_id
        #     center = self.center_points[object_id]
        #     new_center_points[object_id] = center
        to_pop=[]
        for i in self.cars.keys():
            if i not in ids:
                to_pop.append(i)
        for i in to_pop:
            self.cars.pop(i)
        # Update dictionary with IDs not used removed
        # self.center_points = new_center_points.copy()
        return self.cars



