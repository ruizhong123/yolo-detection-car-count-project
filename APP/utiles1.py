from ultralytics import YOLO
from django.utils import timezone
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import supervision as sv

from io import BytesIO
import numpy as np
import threading
import base64
import cv2
import time
import queue

class objectdetection_countingregion:
    def __init__(self, url, model, initial_point, end_point, polygon1, polygon2):
        self.url = url
        self.initial_point = initial_point
        self.end_point = end_point
        self.model = model
        
        # Data collection (use deques for faster append/pop)
        from collections import deque
        self.date = deque(maxlen=50)  # Limit to 50 entries
        self.datetime = deque(maxlen=50)
        self.linecount1 = deque(maxlen=50)
        self.linecount2 = deque(maxlen=50)
        self.polygon1count = deque(maxlen=50)
        self.polygon2count = deque(maxlen=50)
        
        self.frame = None
        self.annoimg = None
        
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
            
            # Store frame
            with self.lock:
                self.annoimg = annoimg
                try:
                    if not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(annoimg.copy())
                except queue.Full:
                    pass
            
            # Store counting data
            currenttime = time.time()
            with self.lock:
                if current_hour < timezone.localtime().hour:
                    self.linecount1.append(int(self.linezone.in_count))
                    self.linecount2.append(int(self.linezone.out_count))
                    self.date.append(timezone.localtime().strftime("%H:%M:%S"))
                    self.linezone = sv.LineZone(self.initial_point, self.end_point)
                    current_hour = timezone.localtime().hour
                elif timezone.localtime().hour == 23 & timezone.localtime().minute > 0 & timezone.localtime().second > 0 :
                    
                    self.linecount1.append(int(self.linezone.in_count))
                    self.linecount2.append(int(self.linezone.out_count))
                    self.date.append(timezone.localtime().strftime('%H:%M:%S'))
                    self.linezone = sv.LineZone(self.initial_point, self.end_point)
                    current_hour = timezone.localtime().hour
                
                if currenttime - last_update_time >= 600 or (timezone.localtime().hour == 24 & timezone.localtime().minute>0 &timezone.localtime().second > 0):
                    self.datetime.clear()
                    self.polygon1count.clear()
                    self.polygon2count.clear()
                    self.polygon1count.append(int(self.polygonzone1.current_count))
                    self.polygon2count.append(int(self.polygonzone2.current_count))
                    self.datetime.append(timezone.localtime().strftime("%H:%M:%S"))
                    last_update_time = currenttime
                    last_append_time = currenttime
                elif currenttime - last_append_time >= 60 :
                    self.polygon1count.append(int(self.polygonzone1.current_count))
                    self.polygon2count.append(int(self.polygonzone2.current_count))
                    self.datetime.append(timezone.localtime().strftime("%H:%M:%S"))
                    last_append_time = currenttime
    
    def get_chart(self):
        with self.lock:
            plt.switch_backend('AGG')
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))  # Smaller figure
            ax1.bar(self.date, self.linecount2, color='b')
            ax1.set_title(' Left Road ', fontsize=12)
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.grid(True)
            ax2.bar(self.date, self.linecount1, color='r')
            ax2.set_title(' Right Road ', fontsize=12)
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.tick_params(axis='y', labelsize=10)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.grid(True)
            plt.tight_layout()
            buffer1 = BytesIO()
            plt.savefig(buffer1, format='png', dpi=80)  # Lower DPI
            buffer1.seek(0)
            graph1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
            buffer1.close()
            plt.close(fig1)
            
            fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10))
            ax3.bar(self.datetime, self.polygon1count, color='b')
            ax3.set_title(' Left Road ', fontsize=12)
            ax3.tick_params(axis='x', rotation=45, labelsize=10)
            ax3.tick_params(axis='y', labelsize=10)
            ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax3.grid(True)
            
            ax4.bar(self.datetime, self.polygon2count, color='r')
            ax4.set_title(' Right Road ', fontsize=12)
            ax4.tick_params(axis='x', rotation=45, labelsize=10)
            ax4.tick_params(axis='y', labelsize=10)
            ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax4.grid(True)
            plt.tight_layout()
            buffer2 = BytesIO()
            plt.savefig(buffer2, format='png', dpi=80)
            buffer2.seek(0)
            graph2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
            buffer2.close()
            plt.close(fig2)
            
            return graph1, graph2
    
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