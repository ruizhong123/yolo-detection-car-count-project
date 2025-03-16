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
initial_point = sv.Point(0, 120)
end_point = sv.Point(300, 120)

# Polygon zones
polygon1 = np.array([[45, 240], [352, 235], [100, 100], [15, 100]])
polygon2 = np.array([[350, 220], [350, 120], [200, 90], [110, 100]])

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
    
@gzip_page
def get_bar_plot(request):
    """Get the bar plot of the counting region"""
    def generate():
        while True:
            plot1, _ = counter.get_chart()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + 
                   base64.b64decode(plot1) +
                   b'\r\n')
            time.sleep(2)  # Update every 2s to reduce load
    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

@gzip_page
def get_line_plot(request):
    """Get the line plot of the counting region"""
    def generate():
        while True:
            _, plot2 = counter.get_chart()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + 
                   base64.b64decode(plot2) +
                   b'\r\n')
            time.sleep(2)  # Update every 2s
    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

@gzip_page
def main_view(request):
    return render(request, 'base.html')