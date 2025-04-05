from django.http import StreamingHttpResponse
from django.views.decorators.gzip import gzip_page
from django.shortcuts import render
from . import utiles1
from ultralytics import YOLO
import supervision as sv
import numpy as np
import json
import cv2
import base64
import time 

# Model and polygons initialization
model = YOLO('yolo12n.pt')

# Line zone
initial_point = sv.Point(70, 170)
end_point = sv.Point(320, 170)

# Polygon zones
polygon1 = np.array([[100, 240], [325, 240], [170, 150], [80, 150]])
polygon2 = np.array([[330, 240], [352, 175], [270, 150], [180, 150]])

# Initialize counter
CAMERA_URL = 'https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=13380'
counter = utiles1.objectdetection_countingregion(CAMERA_URL, model, initial_point, end_point, polygon1, polygon2)

@gzip_page
def stream_video(request):
    """Stream video with region counting functionality"""
    def generate():
        
        cap = cv2.VideoCapture(CAMERA_URL)
        
        # Check if the video stream opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video stream from {CAMERA_URL}")
            return
        
        # Get the video resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Print the resolution to the console
        if width == 0 or height == 0:
            print("Error: Could not retrieve video resolution. Stream might be unavailable.")
        else:
            print(f"Video resolution: {width}x{height}")
        
        
        while True:
            
            ret,frame = cap.read()

            if not ret :
                
                cap.release()
                cap = cv2.VideoCapture(CAMERA_URL)
                ret,frame = cap.read()
                
                
            ret,buffer = cv2.imencode('.jpg',frame)
            trans_frame= buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   trans_frame + 
                   b'\r\n')
                
    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


# make the object of annotation 

def annotate(request):
    def generate():
        try:
            while True:
                
                with counter.lock:
                    detection = {"xyxy" : counter.xyxy.tolist(),
                                 "confidence" : counter.confidence.tolist(),
                                 "class_id" : counter.class_id.tolist()
                                 }
                    
                    time.sleep(0.01)
                    
                    yield f'data:{json.dumps(detection)}\n\n' # becausse the data can be changed over time, so we need to use json.dumps to convert the data that can be dumped with json format

        except Exception as e : 
            print("Error in annotate:",e)
    
    return StreamingHttpResponse(generate(), content_type="text/event-stream")
        
    
from django.http import JsonResponse

def get_chart_data(request):
    with counter.lock:  # Ensure thread safety
        
        data = {
            "dates": list(counter.date),
            "linecount1": list(counter.linecount1),
            "linecount2": list(counter.linecount2),
            "datetime": list(counter.datetime),
            "polygon1count": list(counter.polygon1count),
            "polygon2count": list(counter.polygon2count),
        }
        time.sleep(0.015)
        
        
    return JsonResponse(data)

@gzip_page
def main_view(request):
    return render(request, 'base.html')