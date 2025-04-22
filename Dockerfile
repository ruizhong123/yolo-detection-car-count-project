# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system libraries needed for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1
    

# Copy the requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your app files
COPY . .

# Copy the YOLO model
COPY yolo12n.pt .

# Open port 8000 for the web server
EXPOSE 8000

# Run database migrations and start the server
CMD ["sh", "-c", "python manage.py migrate && gunicorn --bind 0.0.0.0:8000 APP_SETTING.wsgi:application"]

