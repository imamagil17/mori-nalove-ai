from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import time
import cv2

app = FastAPI()

# Muat model otak YOLOv26 hasil training dari root folder
model = YOLO('best.pt') 


@app.get("/")
def home():
    return {"status": "Online", "pesan": "Server AI Flood Vision YOLOv26 Siap Menerima Video!"}

@app.post("/api/deteksi")
async def proses_video(file: UploadFile = File(...)):
    # 1. Simpan file video simulasi dari Laravel ke penyimpanan sementara
    temp_file = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Buka video menggunakan OpenCV untuk melacak per frame
    cap = cv2.VideoCapture(temp_file)
    
    # State flags untuk mencatat warna apa saja yang PERNAH terlihat di sepanjang video
    hijau_pernah_terdeteksi = False
    kuning_pernah_terdeteksi = False
    merah_pernah_terdeteksi = False
    
    # Interval frame-skipping (biar proses scan cepat, kita cek setiap 5 frame saja)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Video habis, keluar dari loop
            
        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Lewati frame ini demi efisiensi performa server
            
        # Jalankan prediksi YOLOv26 pada frame terpilih
        results = model(frame, conf=0.40, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                kelas_id = int(box.cls[0])
                nama_kelas = model.names[kelas_id]
                
                # Catat tanda warna yang tertangkap kamera
                if nama_kelas == 'tanda_hijau':
                    hijau_pernah_terdeteksi = True
                elif nama_kelas == 'tanda_kuning':
                    kuning_pernah_terdeteksi = True
                elif nama_kelas == 'tanda_merah':
                    merah_pernah_terdeteksi = True

    cap.release()

    # 3. LOGIKA EVALUASI STATUS BANJIR (EARLY WARNING SYSTEM)
    # Kita tentukan status berdasarkan kombinasi tanda warna yang tersisa/hilang
    status_final = "AMAN"
    tinggi_simulasi_cm = 180  # Default tinggi air aman standar lapangan
    
    if merah_pernah_terdeteksi and not kuning_pernah_terdeteksi and not hijau_pernah_terdeteksi:
        # Kondisi ekstrem: Hanya warna merah yang lolos/terlihat, hijau & kuning tenggelam
        status_final = "BAHAYA"
        tinggi_simulasi_cm = 550  # Menyesuaikan angka uji coba riwayat dashboardmu
    elif kuning_pernah_terdeteksi and not hijau_pernah_terdeteksi:
        # Kondisi menengah: Hijau sudah hilang tenggelam
        status_final = "SIAGA"
        tinggi_simulasi_cm = 345
    elif hijau_pernah_terdeteksi:
        # Jika hijau masih kelihatan, status dipastikan aman
        status_final = "AMAN"
        tinggi_simulasi_cm = 180

    # 4. Bersihkan file video sementara dari penyimpanan lokal
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # 5. Kirim data JSON matang yang siap dibaca Controller Laravel
    return {
        "status": "sukses",
        "file_diproses": file.filename,
        "kondisi_banjir": status_final,      # "AMAN", "SIAGA", atau "BAHAYA"
        "ketinggian_yolo": tinggi_simulasi_cm # Angka centimeter untuk kolom riwayat Laravel
    }