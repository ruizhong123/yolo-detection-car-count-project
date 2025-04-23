# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app


# run package on linux system 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

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



# Health check
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/ || exit 1

# Run database migrations and start the server
CMD ["sh", "-c", "python manage.py migrate && gunicorn --bind 0.0.0.0:8000 APP_SETTING.wsgi:application"]



