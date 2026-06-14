# Gunakan base image Python yang stabil dan ringan
FROM python:3.10-slim

# Install dependensi sistem Linux yang wajib dibawa oleh OpenCV agar tidak crash
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set folder kerja di dalam server container cloud Render
WORKDIR /app

# Copy seluruh file projectmu (main.py, best.pt, dll) ke dalam server cloud
COPY . /app

# Install pustaka Python langsung di dalam server container
RUN pip install --no-cache-dir fastapi uvicorn ultralytics opencv-python

# Buka gerbang port 8001 di cloud
EXPOSE 8001

# Perintah untuk menghidupkan FastAPI YOLOv26 kamu
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]N