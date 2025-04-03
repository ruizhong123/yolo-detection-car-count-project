from ultralytics import YOLO
from django.utils import timezone
import matplotlib.pyplot as plt 
import supervision as sv
from io import BytesIO
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
        
        
        self.frame = None
        self.annoimg = None
        
        
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
        current_time = timezone.localtime()
        current_hour = current_time.hour
        
        last_append_time = time.time()
        last_update_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap = cv2.VideoCapture(self.url)
                ret, frame = self.cap.read()
                continue
            
       
            
            # YOLO prediction
            time.sleep(0.1)
            anno_img = self.model.predict(frame)[0]  # Suppress logs
            detections = sv.Detections(
                xyxy=anno_img.boxes.xyxy.cpu().numpy(),
                confidence=anno_img.boxes.conf.cpu().numpy(),
                class_id=anno_img.boxes.cls.cpu().numpy().astype(int)
            )
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            # Update tracker and zones
            self.tracker.update_with_detections(detections)
            self.linezone.trigger(detections=detections)
            self.polygonzone1.trigger(detections=detections)
            self.polygonzone2.trigger(detections=detections)
            
            # Annotate frame
            
            annoimg = self.line_annotator.annotate(frame, self.linezone)
            annoimg = self.poly1_annotator.annotate(annoimg)
            annoimg = self.poly2_annotator.annotate(annoimg)
            annoimg = bounding_box_annotator.annotate(annoimg,detections=detections)
            
            
            with self.lock:
                self.annoimg = annoimg
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(annoimg.copy())
            

            
            # Store counting data
            currenttime = time.time()
            
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
                    
     
                if timezone.localtime().minute == 0 and timezone.localtime().second == 0:
                        
                        self.datetime.clear()
                        self.polygon1count.clear()
                        self.polygon2count.clear()
                        self.polygon1count.append(int(self.polygonzone1.current_count))
                        self.polygon2count.append(int(self.polygonzone2.current_count))
                        self.datetime.append(timezone.localtime().strftime("%H:%M:%S"))
                        
                        self.last_update_time = currenttime
                        self.last_append_time = currenttime
                elif currenttime - last_append_time >= 360 :
                    self.polygon1count.append(int(self.polygonzone1.current_count))
                    self.polygon2count.append(int(self.polygonzone2.current_count))
                    self.datetime.append(timezone.localtime().strftime("%H:%M:%S"))
                    last_append_time = currenttime

    
    def get_annotated_frame_counting_region(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.annoimg is None:
                    return None
                frame = self.annoimg.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if ret else None