from ultralytics import YOLO
from django.utils import timezone
import matplotlib.pyplot as plt 
import supervision as sv
from io import BytesIO
from .models import countData
import numpy as np
import threading
import base64
import cv2
import time
import queue



from collections import deque

class objectdetection_countingregion:
    
    
    def __init__(self, url, model, initial_point, end_point, polygon1, polygon2):
        self.url = url
        self.initial_point = initial_point
        self.end_point = end_point
        self.model = model
        
        
        # set the defulat data
         
        self.frame = None
        self.annoimg = None
        
        self.xyxy = np.array([],dtype=np.float32).reshape(0,4)
        self.confidence = np.array([],dtype=np.float32)
        self.class_id = np.array([],dtype=np.int32)
                
        
        self.last_update_date = None
        
        
        # Data collection
        self.date = deque(maxlen=50)
        self.datetime = deque(maxlen=50)
        self.linecount1 = deque(maxlen=50)
        self.linecount2 = deque(maxlen=50)
        self.polygon1count = deque(maxlen=50)
        self.polygon2count = deque(maxlen=50)
        
        # Threading and queue setup
        self.lock = threading.Lock()
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)  # Single frame to reduce latency
        
        # Start processing thread
        self.thread = threading.Thread(target=self._updateframe, daemon=True)
        self.thread.start()
        
        # Supervision setup (initialize once)
        self.tracker = sv.ByteTrack()
        
        self.linezone = sv.LineZone(self.initial_point, self.end_point)
        self.polygonzone1 = sv.PolygonZone(polygon1)
        self.polygonzone2 = sv.PolygonZone(polygon2)
        self.line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.3, text_padding=0.5)
        self.poly1_annotator = sv.PolygonZoneAnnotator(zone=self.polygonzone1, thickness=1, text_thickness=1, text_scale=0.5, text_padding=1)
        self.poly2_annotator = sv.PolygonZoneAnnotator(zone=self.polygonzone2, thickness=1, text_thickness=1, text_scale=0.5, text_padding=1)
    
    def __del__(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        if self.thread.is_alive():
            self.thread.join()

    def _updateframe(self):
        
        self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print(f"Failed to open video stream: {self.url}")
            self.running = False
            return
                
        current_time = timezone.localtime()
        current_hour = current_time.hour
        

        while self.running:
            
            ret, self.frame = self.cap.read()
            
            if not ret:
                
                self.cap = cv2.VideoCapture(self.url)

                continue
            
            print("Frame captured successfully")
            
            # YOLO prediction
            time.sleep(0.1)
            anno_img = self.model.predict(self.frame)[0]  # Suppress logs
            
            self.xyxy=anno_img.boxes.xyxy.cpu().numpy()
            self.confidence=anno_img.boxes.conf.cpu().numpy()
            self.class_id=anno_img.boxes.cls.cpu().numpy().astype(int)
            
            detections = sv.Detections(
                xyxy = self.xyxy,
                confidence = self.confidence,
                class_id=self.class_id
            )
            
            # Update tracker and zones
            self.tracker.update_with_detections(detections)
            self.linezone.trigger(detections=detections)
            self.polygonzone1.trigger(detections=detections)
            self.polygonzone2.trigger(detections=detections)

            current_time =timezone.localtime() 
            
            with self.lock:
                
                if timezone.localtime().hour < current_hour and timezone.localtime().minute >= 0 and timezone.localtime().second >= 0:
                    
                    self.linecount1.append(int(self.linezone.in_count))
                    self.linecount2.append(int(self.linezone.out_count))
                    
                    self.date.append(timezone.localtime().strftime('%H:%M:%S'))
                    
                    self.linezone = sv.LineZone(self.initial_point, self.end_point)
                    current_hour = timezone.localtime().hour
                
                
                elif current_hour < timezone.localtime().hour and timezone.localtime().hour > 0 :
                    
                    if timezone.localtime().hour == 1 and  timezone.localtime().minute >= 0 and timezone.localtime().second >=0 :
                        
                        self.linecount1.clear()
                        self.linecount2.clear()
                        self.date.clear()
                    
                    self.linecount1.append(int(self.linezone.in_count))
                    self.linecount2.append(int(self.linezone.out_count))
                    
                    self.date.append(timezone.localtime().strftime("%H:%M:%S"))
                    
                    self.linezone = sv.LineZone(self.initial_point, self.end_point)               
                    current_hour = timezone.localtime().hour
                    
     
                if current_time.minute % 6 == 0 and current_time.second ==0 and  (self.last_update_date is None or current_time > self.last_update_date+timezone.timedelta(seconds=1)):
                         
                    if timezone.localtime().minute == 0 :
                        
                        self.datetime.clear()
                        self.polygon1count.clear()
                        self.polygon2count.clear()
                    
                    self.polygon1count.append(int(self.polygonzone1.current_count))
                    self.polygon2count.append(int(self.polygonzone2.current_count))
                    self.datetime.append(timezone.localtime().strftime("%H:%M:%S"))
                    
                    self.last_update_date = current_time
                    
                    
    # load data from database(model.py) 
    def _load_data(self):
        """ load data from database(model) """
        
        with self.lock:
            
            try:
                # fetch 50 data from database  
                lastest = countData.objects.all()[:24]

                for entry in reversed(lastest):

                    self.date.append(entry.date)
                    self.datetime.append(entry.datetime)
                    self.polygon1count.append(entry.polygon1count)
                    self.polygon2count.append(entry.polygon2count)
                    self.linecount1.append(entry.linecount1)
                    self.linecount2.append(entry.linecount2)
                    
                    if lastest :
                        
                        self.last_update_date = lastest[0].timestep
                    
                    else:
                        return None 
            
            
            
            except Exception as e:
                
                print(f'Error from loading data : {e}')
                
    def _save_data(self,date=None,datetime=None,polygon1count=0,polygon2count=0,linecount1=0,linecount2=0):
        
        with self.lock:
            
            try:
                
                # using data from recording function 
                 
                countData.objects.create(
                    date = self.date,
                    datetime = self.datetime,
                    polygon1count = self.polygon1count,
                    polygon2count = self.polygon2count,
                    linecount1 = self.linecount1,
                    linecount2 = self.linecount2
                )
                
                print(f"save data to database : {date},{datetime},{polygon1count},{polygon2count},{linecount1},{linecount2}")
                


            except Exception as e :
                
                print(f" save data into database ")
            
            
        
        
        
        
            
            


