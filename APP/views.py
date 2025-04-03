from django.http import StreamingHttpResponse
from django.views.decorators.gzip import gzip_page
from django.shortcuts import render
from . import utiles1
from ultralytics import YOLO
import supervision as sv
import numpy as np
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
def stream_video_counting_region(request):
    """Stream video with region counting functionality"""
    def generate():
        last_time = time.time()
        while True:
            frame = counter.get_annotated_frame_counting_region()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame + 
                       b'\r\n')
            # Dynamic delay for ~20 FPS
            elapsed = time.time() - last_time
            delay = max(0, 0.05 - elapsed)  # 0.05s = 20 FPS
            time.sleep(delay)
            last_time = time.time()
    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
    
    

    
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
    return JsonResponse(data)

@gzip_page
def main_view(request):
    return render(request, 'base.html')