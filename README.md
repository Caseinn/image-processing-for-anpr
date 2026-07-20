# ANPR — Deteksi Plat Nomor dengan Image Processing Klasik

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Astro-5-FF5D01?logo=astro" alt="Astro">
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
</p>

Pipeline deteksi plat nomor kendaraan berbasis contour detection dan geometric filtering. Pure OpenCV, zero ML.

## Latar Belakang

Buat apa susah-susah pake computer vision klasik kalo tinggal colok YOLO? Karena ini proyek edukasi — demonstrasi bahwa teknik dasar image processing masih relevan dan penting dipahami. Grayscale, adaptive histogram, edge detection, contour analysis — semuanya dijalankan tanpa training, tanpa inference, tanpa model weights.

Setiap tahap pipeline bisa divisualisasikan dan parameternya bisa diotak-atik langsung. Cocok buat mahasiswa, engineer yang mau belajar CV, atau siapapun yang penasaran gimana plate detection bekerja dari sisi algorithmic.

> **Bukan production-grade.** Ini murni rule-based. Kalo butuh akurasi tinggi, pake YOLO atau model terlatih lainnya.

## Cara Pakai

### Prasyarat

- Python 3.10+
- Node.js 18+ & npm

### Backend

```bash
cd backend
python -m venv venv
```

Aktifin virtual environment:

```bash
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
python main.py
```

Server jalan di `http://localhost:8000`.

| Endpoint | Fungsi |
|---|---|
| `POST /api/detect` | Upload gambar → JSON (annotated image, pipeline steps, bounding boxes, crop hasil) |
| `GET /` | Health check |
| `/docs` | Swagger UI |

Test pake curl:

```bash
curl -X POST -F "file=@plat.jpg" http://localhost:8000/api/detect
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Buka `http://localhost:5173`. Upload gambar atau pilih dari sample. Pipeline jalan otomatis, setiap step divisualisasikan.

### Batch Processing

Jalanin deteksi ke 100 gambar uji sekaligus:

```bash
cd backend
python -c "from detector.runner import run; run()"
```

Hasilnya:

- `output/eval.txt` — metrik global: IoU, precision, recall, F1
- `output/crops/` — region plat yang terdeteksi (file PNG)
- `output/crops.csv` — daftar source-crop pairs

Evaluasi pakai YOLO-format ground truth dari `data/labels/`. Setiap deteksi dihitung true positive kalo IoU-nya ≥ 0.5.

## Pipeline

```
Input → Grayscale → CLAHE → Gaussian Blur → Canny → Contours → Area Filter → Quadrilateral Approx → Aspect Filter → Crop
```

| # | Langkah | Penjelasan |
|---|---|---|
| 1 | **Grayscale** | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`. Warna gak relevan buat deteksi tepi, buang aja. |
| 2 | **CLAHE** | `createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`. Adaptive contrast enhancement biar karakter plat lebih keliatan, terutama kalo pencahayaan gak merata. |
| 3 | **Gaussian Blur** | `GaussianBlur(kernel_size=5)`. Buang noise frekuensi tinggi yang bisa bikin false edges pas Canny. |
| 4 | **Canny Edge** | `Canny(threshold1=50, threshold2=150)`. Piksel dengan gradien di atas 150 jadi edge, di bawah 50 dibuang, sisanya tergantung konektivitas. |
| 5 | **Find Contours** | `findContours(mode=RETR_EXTERNAL)`. Ambil contour terluar dari edge map. RETR_EXTERNAL ngambil contour paling luar aja, biar contour di dalamnya (misalnya karakter plat) gak ikut. |
| 6 | **Area Filter** | Kontur dengan luas di luar `[0.05%, 5%]` dari total area gambar langsung dibuang. |
| 7 | **Quadrilateral Approx** | `approxPolyDP(epsilon=0.02 × perimeter)`. Plat nomor bentuknya rectangle → contour harus punya tepat 4 vertices dan convex. |
| 8 | **Aspect Filter** | `2.0 < width/height < 8.0`. Bentuk yang gak sesuai rasio plat nomor kefilter. |
| 9 | **Crop** | `minAreaRect` → warp. Potong dan luruskan region plat pake perspective transform. |

## Konfigurasi

Semua parameter di `backend/detector/config.py`:

| Parameter | Default | Fungsi |
|---|---|---|
| `clahe_clip` | 2.0 | Kontrol kontras CLAHE |
| `clahe_grid` | 8 | Ukuran tile grid |
| `gauss_kernel` | 5 | Kernel size Gaussian blur (harus ganjil) |
| `canny_low` | 50 | Threshold bawah Canny |
| `canny_high` | 200 | Threshold atas Canny |
| `area_range` | (0.0005, 0.3) | Rentang luas kontur relatif terhadap frame |
| `aspect_range` | (2.0, 8.0) | Rentang rasio lebar/tinggi |
| `approx_eps` | 0.02 | Epsilon buat approxPolyDP |
| `iou_thr` | 0.5 | Threshold IoU pas evaluasi |

Kalo hasil deteksi kurang memuaskan:

| Masalah | Parameter | Arah |
|---|---|---|
| Banyak false contours | `canny_low` | Naikkin |
| Tepi plat putus-putus | `canny_low` / `canny_high` | Turunin |
| Plat kegedean/kekecilan | `area_range` | Sesuaikan |
| Objek persegi lolos | `aspect_range` | Persempit |
| Kontras kurang | `clahe_clip` | Naikkin (3.0–4.0) |
| Terlalu noisy | `gauss_kernel` | Naikkin (7 atau 9) |

## Mode Kegagalan

Ini pure contour-based, jadi banyak batasnya. Scenario yang hampir pasti fail:

| Skenario | Root Cause |
|---|---|
| Plat miring / angle jelek | approxPolyDP gagal dapet 4 corners |
| Pencahayaan minim | Canny gak dapet gradient cukup kuat |
| Overexposed / blown-out | Too many false edges dari sensor noise |
| Plat kotor / berkarat | Edges putus, contour pecah |
| Latar kompleks (pohon, pagar, iklan) | Ratusan contour lolos filter |
| Motion blur | Edges jadi terlalu lebar, Canny kacau |
| Silau lensa | CLAHE gak bisa ngembaliin blown-out highlights |
| Plat terlalu kecil / jauh | Kefilter sama area_range |
| Multiple vehicles | Hanya detek kandidat terkuat |
| Plat gelap di body gelap | Kontras terlalu rendah |

## Struktur Proyek

```
├── main.py                  # Entry point batch processing
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt
│   ├── detector/
│   │   ├── core.py          # Pipeline utama
│   │   ├── config.py        # Parameter
│   │   ├── runner.py        # Batch processing + evaluasi
│   │   ├── eval.py          # IoU, precision, recall, F1
│   │   ├── io.py            # I/O file
│   │   └── __init__.py
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── components/      # React components (kebab-case)
│   │   ├── pages/           # index.astro, 404.astro
│   │   └── styles/          # Tailwind v4 global.css
│   ├── public/samples/      # Gambar sample buat testing frontend
│   ├── package.json
│   ├── astro.config.mjs
│   └── tsconfig.json
├── data/
│   ├── images/              # 100 gambar uji (test001–test100)
│   └── labels/              # Ground truth format YOLO
└── output/                  # Hasil batch evaluation
    ├── crops/               # Region plat yang terdeteksi
    ├── crops.csv
    └── eval.txt
```
