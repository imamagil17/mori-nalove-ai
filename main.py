from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import time

app = FastAPI()

# Sudah dikembalikan ke YOLOv26 Nano!
model = YOLO('yolo26n.pt') 

@app.get("/")
def home():
    return {"status": "Online", "pesan": "Server AI Flood Vision YOLOv26 Siap Menerima Video!"}

@app.post("/api/deteksi")
async def proses_video(file: UploadFile = File(...)):
    # 1. Simpan video dari Laravel ke file sementara
    temp_file = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. YOLOv26 Menganalisis Video
    results = model.predict(source=temp_file, conf=0.25)

    # 3. Kumpulkan Hasil Tebakan
    objek_terdeteksi = []
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            kelas_id = int(box.cls[0])
            nama_kelas = model.names[kelas_id]
            akurasi = float(box.conf[0])
            
            objek_terdeteksi.append({
                "kelas": nama_kelas,
                "akurasi": round(akurasi * 100, 2)
            })

    # 4. Hapus file video sementara
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # 5. Kirim kembali hasil JSON ke Laravel
    return {
        "status": "sukses",
        "file_diproses": file.filename,
        "total_objek": len(objek_terdeteksi),
        "hasil": objek_terdeteksi
    }