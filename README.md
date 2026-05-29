# 🧠 Flood-Vision AI: Engine Inferensi Ketinggian Air Berbasis YOLOv26 Nano

Repositori ini merupakan sub-sistem kecerdasan buatan (*AI Inference Server*) dari platform **Flood-Vision**. Sub-sistem ini dibangun menggunakan **Python FastAPI** dan **Ultralytics YOLO** untuk memproses data biner rekaman video/gambar rambu ukur aliran sungai secara *asynchronous*, mengekstrak objek level air, dan mengembalikan data analisis terstruktur (JSON) ke server utama (Laravel).

---

## 🛠️ Arsitektur Core AI & Teknologi

Sistem dirancang mandiri (*decoupled*) agar tidak membebani performa web server utama saat melakukan komputasi *Deep Learning* yang berat.
- **Framework API:** FastAPI (Python 3.9+)
- **ASGI Server:** Uvicorn (Hot-Reload Enabled)
- **Computer Vision Engine:** Ultralytics YOLOv26 Nano Architecture
- **Image Processing:** OpenCV Python
- **Format Pertukaran Data:** JSON Telemetry Data Payload
- **Model Weights Core:** `yolo26n.pt` (Custom-Trained Weights via PyTorch)

---

## 📁 Struktur Berkas Repositori

```text
flood-vision-ai/
├── env/                 <-- Folder Python Virtual Environment (Di-ignore dari Git)
├── __pycache__/
├── main.py              <-- Script Driver Utama FastAPI & Routing API Deteksi
└── yolo26n.pt           <-- Berkas Bobot (Weights) Hasil Training Model YOLOv26

Alur Kerja Deteksi Objek (Pipeline)
HTTP POST Request: Server Laravel mengirimkan berkas video streaming dari pos pantau ke endpoint /api/deteksi.

Temporary Storage: Berkas biner video disimpan sementara dengan nama unik berbasis timestamp Unix untuk menghindari bentrokan data.

YOLO Inference: Model yolo26n.pt mengeksekusi metode model.predict() dengan ambang batas kepercayaan (confidence score) minimum 0.25.

Data Extraction: Lapisan bounding box memetakan koordinat piksel objek dan mengambil label tingkat bahaya (Rendah, Sedang, Tinggi).

Garbage Collection: File video sementara langsung dihapus otomatis menggunakan modul os.remove() demi menjaga efisiensi ruang penyimpanan disk.

JSON Response: Hasil ekstraksi dikonversi menjadi data array terstruktur dan ditembakkan kembali ke Laravel dalam hitungan milidetik.

Panduan Instalasi & Jalur Deploy
Ikuti langkah-langkah di bawah ini untuk menyalakan Server AI di lingkungan lokal Anda:

1. Clone & Masuk ke Direktori
Buka terminal Anda, klon repositori ini, lalu masuk ke foldernya:

Bash
git clone [https://github.com/imamagil17/flood-vision-ai.git](https://github.com/imamagil17/flood-vision-ai.git)
cd flood-vision-ai
2. Aktivasi Virtual Environment (env)
Sangat disarankan menggunakan isolasi environment agar versi pustaka (library) tidak bentrok dengan proyek Python lain di laptop Anda.

Untuk Windows:

Bash
python -m venv env
env\Scripts\activate
Untuk Linux / macOS:

Bash
python3 -m venv env
source env/bin/activate
3. Instalasi Pustaka Dependensi
Pastikan pip Anda berada di versi terbaru, kemudian instal modul-modul wajib berikut:

Bash
pip install --upgrade pip
pip install fastapi uvicorn ultralytics opencv-python
4. Menjalankan Server Uvicorn AI
Nyalakan server FastAPI menggunakan Uvicorn. PENTING: Set port ke 5000 agar tidak bertabrakan dengan port bawaan Laravel (8000):

Bash
uvicorn main:app --reload --port 8001 --reload
Server AI kini aktif dan mendengarkan request pada alamat: http://127.0.0.1:8001
