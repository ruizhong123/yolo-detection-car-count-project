from ultralytics import YOLO
from django.utils import timezone
import matplotlib.pyplot as plt 
import supervision as sv
from io import BytesIO
from .models import CountData
import numpy as np
import threading
import base64
import cv2
import time
import queue
from collections import deque



import logging
logger = logging.getLogger(__name__)
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
        self.date = deque(maxlen=24)
        self.datetime = deque(maxlen=24)
        self.linecount1 = deque(maxlen=24)
        self.linecount2 = deque(maxlen=24)
        self.polygon1count = deque(maxlen=24)
        self.polygon2count = deque(maxlen=24)
        
        # Threading and queue setup
        self.data_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)  # Single frame to reduce latency
        self.data_queue = queue.Queue()  # Queue for DB operations
        
        # Start processing thread
        self.frame_thread = threading.Thread(target=self._updateframe, daemon=True)
        self.frame_thread.start()
        

        
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
        
        with self.frame_lock:
            if hasattr(self, 'cap'):
                self.cap.release()
        
        if self.frame_thread.is_alive():
            self.frame_thread.join(timeout=5)  # Timeout to avoid infinite wait
            if self.frame_thread.is_alive():
                logger.warning("Thread did not terminate cleanly")
        

    
    
    
    def _updateframe(self):
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video stream: {self.url}")
            self.running = False
            return

        current_time = timezone.localtime()
        current_hour = current_time.hour

        while self.running:
            
            try:
                logger.debug("Attempting to read frame")
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame, reconnecting...")
                    self.cap = cv2.VideoCapture(self.url)
                    time.sleep(1)
                    continue
                
                logger.info("Frame captured successfully")
                time.sleep(0.1)
                
                logger.debug("Starting YOLO prediction")
                anno_img = self.model.predict(frame)[0]
                logger.info("YOLO prediction completed")


                # ... detection and zone logic ...
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



                with self.frame_lock:
                    self.frame = frame.copy()

                current_time = timezone.localtime()




                with self.data_lock:
                    
            
                    # Hourly save at :00:00
                    if current_time.hour != current_hour and current_time.minute == 0 and current_time.second == 0:
                        logger.info("Performing hourly save")
                        self._save_data(
                            date=current_time.strftime("%H:%M:%S"),
                            linecount1=int(self.linezone.in_count),
                            linecount2=int(self.linezone.out_count)
                        )
                        self._load_data()
                        self.linecount1.append(int(self.linezone.in_count))
                        self.linecount2.append(int(self.linezone.out_count))
                        self.date.append(current_time.strftime("%H:%M:%S"))
                        self.linezone = sv.LineZone(self.initial_point, self.end_point)
                        current_hour = current_time.hour
                    # 1 AM reset
                    if current_time.hour == 1 and current_time.minute == 0 and current_time.second == 0:
                        logger.info("Performing 1 AM reset")
                        cutoff = timezone.localtime() - timezone.timedelta(hours=24)
                        deleted = CountData.objects.filter(timestep__lt=cutoff).delete()
                        logger.info(f"Deleted {deleted[1]} old rows")
                        CountData.objects.all().update(
                            date=current_time.strftime("%H:%M:%S"),
                            linecount1=0,
                            linecount2=0
                        )
                        self.linecount1.clear()
                        self.linecount2.clear()
                        self.date.clear()
                        self._load_data()
                        
                        
                        
                    logger.debug(f"Starting DB operation at {current_time.strftime('%H:%M:%S')}")
                        # 6-minute polygon save
                    if current_time.minute % 6 == 0 and current_time.second == 0 and (self.last_update_date is None or current_time > self.last_update_date + timezone.timedelta(seconds=1)):
                        logger.info("Performing 6-minute polygon save")
                                                # Reset only at midnight
                        if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
                            logger.error
                            CountData.objects.all().update(
                                datetime=current_time.strftime("%H:%M:%S"),
                                polygon1count=0,
                                polygon2count=0
                            )
                            self.datetime.clear()
                            self.polygon1count.clear()
                            self.polygon2count.clear()
                            
                            
                            
                        logger.debug("save data to database")

                        self._save_data(
                            datetime=current_time.strftime("%H:%M:%S"),
                            polygon1count=int(self.polygonzone1.current_count),
                            polygon2count=int(self.polygonzone2.current_count)
                        )
                        self._load_data()
                        self.polygon1count.append(int(self.polygonzone1.current_count))
                        self.polygon2count.append(int(self.polygonzone2.current_count))
                        self.datetime.append(current_time.strftime("%H:%M:%S"))
                        self.last_update_date = current_time
                        
                    logger.debug(f"Finished DB operation at {timezone.localtime().strftime('%H:%M:%S')}")
                    
                    continue
                    
            except Exception as e:
                logger.error(f"Update frame error: {e}")
                time.sleep(0.01)
                




                    
    # load data from database(model.py) 
    def _load_data(self):
        """ load data from database(model) """
        
        with self.data_lock:
            try:
                logger.debug("Loading data from database")
                # fetch 24 data from database  
                latest = CountData.objects.all()[:24]
                self.date.clear()  # Clear before loading
                self.datetime.clear()
                self.linecount1.clear()
                self.linecount2.clear()
                self.polygon1count.clear()
                self.polygon2count.clear()
                

                for entry in reversed(latest):

                    self.date.append(entry.date)
                    self.datetime.append(entry.datetime)
                    self.polygon1count.append(entry.polygon1count)
                    self.polygon2count.append(entry.polygon2count)
                    self.linecount1.append(entry.linecount1)
                    self.linecount2.append(entry.linecount2)
                    
                logger.info(f"Loaded {len(latest)} records")
            except Exception as e:
                
                logger.error(f"Load data error: {e}")
                
    def _save_data(self,date=None,datetime=None,polygon1count=0,polygon2count=0,linecount1=0,linecount2=0):
        
        with self.data_lock:
            
            try:
                logger.debug("Saving data to database")
                # using data from recording function  
                CountData.objects.create(
                    date = date,
                    datetime = datetime,
                    polygon1count = polygon1count,
                    polygon2count = polygon2count,
                    linecount1 = linecount1,
                    linecount2 = linecount2
                )
                
                logger.info(f"save data to database : {date},{datetime},{polygon1count},{polygon2count},{linecount1},{linecount2}")
                logger.debug("Data saved successfully")

            except Exception as e :
                
                logger.error(f"Save data error: {e}")
        
        
            
            


