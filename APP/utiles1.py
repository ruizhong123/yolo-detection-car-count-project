from ultralytics import YOLO
from django.utils import timezone
import supervision as sv
from .models import CountData_for_Line, CountData_for_polygon
import numpy as np
import threading
import cv2
import time
import queue
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ObjectDetectionCountingRegion:
    
    def __init__(self, url, model, initial_point, end_point, polygon1, polygon2):
        
        self.url = url
        self.initial_point = initial_point
        self.end_point = end_point
        self.model = model

        # Default data
        self.frame = None
        
        self.xyxy = np.array([], dtype=np.float32).reshape(0, 4)
        self.confidence = np.array([], dtype=np.float32)
        self.class_id = np.array([], dtype=np.int32)
        
        self.linecount1 = 0 
        self.linecount2 = 0
        self.polygon1count = 0
        self.polygon2count = 0
        self.last_update_date = None

        # Data collection
        self.date = deque(maxlen=24)
        self.datetime = deque(maxlen=10)
        self.linecount1 = deque(maxlen=24)
        self.linecount2 = deque(maxlen=24)
        self.polygon1count = deque(maxlen=10)
        self.polygon2count = deque(maxlen=10)
        
        # Threading and queue setup
        self.data_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        self.running = True
        
        self.frame_queue = queue.Queue(maxsize=1)  # Single frame to reduce latency
        self.data_queue = queue.Queue()  # Queue for DB operations
        

        # Supervision setup
        self.tracker = sv.ByteTrack()
        self.linezone = sv.LineZone(self.initial_point, self.end_point)
        self.polygonzone1 = sv.PolygonZone(polygon1)
        self.polygonzone2 = sv.PolygonZone(polygon2)
        self.line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.3, text_padding=0.5)
        self.poly1_annotator = sv.PolygonZoneAnnotator(zone=self.polygonzone1, thickness=1, text_thickness=1, text_scale=0.5, text_padding=1)
        self.poly2_annotator = sv.PolygonZoneAnnotator(zone=self.polygonzone2, thickness=1, text_thickness=1, text_scale=0.5, text_padding=1)

        # Start threads
        self.frame_thread = threading.Thread(target=self._update_frame, daemon=True)
        self.data_thread = threading.Thread(target=self._process_data_queue, daemon=True)
        
        self.frame_thread.start()
        self.data_thread.start()
        
        logger.info("ObjectDetectionCountingRegion initialized and threads started")
        
        


    ## prevent other threading from distributing  
    def __del__(self):
        
        self.running = False
        
        with self.frame_lock:
            
            if hasattr(self, 'cap'):
                self.cap.release()
                
            for i in (self.frame_thread, self.data_thread):
                i.join()
                
    def _update_frame(self):
        
        self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            
            logger.error(f"Failed to open video stream: {self.url}")
            self.running = False
            return
        
        current_hour = timezone.localtime().hour  # Initialize
        
        while self.running:
               
            try:
                
                logger.debug("Attempting to read frame")      
                ret, self.frame = self.cap.read()
                
                if not ret:
                
                    logger.warning("Failed to capture frame, reconnecting...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url)
                    continue
                
                logger.info("Frame captured successfully")
                # predict frame
                logger.debug("Starting YOLO prediction")
                self.anno_img = self.model.predict(self.frame)[0]
                logger.debug("Frame prediction completed")
                # Detection and zone logic
                self.xyxy = self.anno_img.boxes.xyxy.cpu().numpy()
                self.confidence = self.anno_img.boxes.conf.cpu().numpy()
                self.class_id = self.anno_img.boxes.cls.cpu().numpy().astype(int)
                detections = sv.Detections(
                    xyxy=self.xyxy,
                    confidence=self.confidence,
                    class_id=self.class_id
                )
                # Update tracker and zones
                self.tracker.update_with_detections(detections)
                self.linezone.trigger(detections=detections)
                self.polygonzone1.trigger(detections=detections)
                self.polygonzone2.trigger(detections=detections)
                time.sleep(0.01)  # Small delay to prevent CPU overload
                # Time-based data processing
                current_time = timezone.localtime()
                linedata = {
                    "date": "",
                "linecount1": 0,
                "linecount2": 0
                }
                polygondata = { 
                    "datetime": "",
                    "polygon1count": 0,
                    "polygon2count": 0
                }
                
                ## cache data from database 
                if current_time.second % 5 == 0 :
                    self._load_data()
                
                # Line zone counts
                # Hourly save at :00:00    
                if current_time.hour != current_hour and current_time.minute == 0 and current_time.second == 0:
                
                    logger.info("Performing hourly save")
                    linedata.update({
                        "date": current_time.strftime("%H:%M:%S"),
                        "linecount1": int(self.linezone.in_count),
                        "linecount2": int(self.linezone.out_count)
                    })
                    self.data_queue.put(("linedata",linedata.copy()))
                    current_hour = current_time.hour
                # 1 AM reset
                if current_time.hour == 1 and current_time.minute == 0 and current_time.second == 0:
                
                    logger.info("Performing 1 AM reset")
                    CountData_for_Line.objects.all().delete()
                    linedata.update({
                        "date": current_time.strftime("%H:%M:%S"),
                        "linecount1": int(self.linezone.in_count),
                        "linecount2": int(self.linezone.out_count)
                    })
                    self.data_queue.put(("linedata",linedata.copy()))
                    logger.info(f"Deleted all CountData records at 1 AM and restart ")
                # 6-minute polygon save
                if (current_time.minute % 6 == 0 and current_time.second == 0 and 
                    (self.last_update_date is None or current_time > self.last_update_date + timezone.timedelta(seconds=1))):
                    if current_time.minute == 0 and current_time.second == 0:
                        with self.data_lock:
                            CountData_for_polygon.objects.all().delete()
                    
                    logger.debug("Saving polygon data to queue")
                    
                    polygondata.update({
                        "datetime": current_time.strftime("%H:%M:%S"),
                        "polygon1count": int(self.polygonzone1.current_count),
                        "polygon2count": int(self.polygonzone2.current_count)
                    })
                    self.data_queue.put(("polygondata", polygondata.copy()))
                    self.last_update_date = current_time
                    
            except Exception as e:
                logger.error(f"Frame update error: {e}")
                


    
    def _process_data_queue(self):
        
        while self.running:
            try:    
                datatype,data = self.data_queue.get(timeout=1)
                
                try:
                    
                    with self.data_lock:               
                        
                        if datatype == "linedata":
                            
                            CountData_for_Line.objects.create(**data)
                            logger.info(f"Saved line data: {data}")
                        
                        if datatype == "polygondata":
                            
                            CountData_for_polygon.objects.create(**data)
                            logger.info(f"Saved polygon data: {data}")
                        
                        logger.info(f"Saved line data: {data}" if datatype == "linedata" else f"Saved polygon data: {data}")
                    
                except Exception as e:
                    logger.error(f"Data save error: {e}")
                    
                self.data_queue.task_done()
            
            except queue.Empty:
                continue
    
    def _load_data(self):
    
        try:
            
            with self.data_lock:
                logger.debug("Loading data from database")
                latest_polygon = CountData_for_polygon.objects.all()[:10]
                latest_line = CountData_for_Line.objects.all()[:24]
                
                self.date.clear()
                self.datetime.clear()
                self.linecount1.clear()
                self.linecount2.clear()
                self.polygon1count.clear()
                self.polygon2count.clear()
                
                for entry in latest_line:
                    self.date.append(entry.date)
                    
                    self.linecount1.append(entry.linecount1)
                    self.linecount2.append(entry.linecount2)
                
                for entry in latest_polygon:
                    
                    self.datetime.append(entry.datetime)
                    self.polygon1count.append(entry.polygon1count)
                    self.polygon2count.append(entry.polygon2count)
                    
                logger.info(f"Loaded {len(latest_line)} line records and {len(latest_polygon)} polygon records")
    
        except Exception as e:
            logger.error(f"Load data error: {e}")
            
    
        
            
            


