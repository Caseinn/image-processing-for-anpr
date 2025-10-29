# Automatic Number Plate Recognition (ANPR) System
## Sistem Deteksi Plat Nomor Otomatis menggunakan Image Processing

### Overview / Gambaran Umum

This project implements an Automatic Number Plate Recognition (ANPR) system using computer vision and image processing techniques. The system can detect and extract license plate numbers from vehicle images or video frames.

Proyek ini mengimplementasikan sistem deteksi plat nomor otomatis menggunakan teknik computer vision dan image processing. Sistem dapat mendeteksi dan mengekstrak nomor plat kendaraan dari gambar atau frame video.

### Pipeline / Alur Proses

```
[Input Frame]
      ↓
[Grayscale] → reduce channels
      ↓
[Bilateral Filter] → denoise + keep edges
      ↓
[Canny Edge Detector] → binary edges
      ↓
[Find Contours] → extract shapes
      ↓
[Filter by Area + Approximate to 4 corners]
      ↓
[Filter by Aspect Ratio (2–8)]
      ↓
[Crop ROI (x, y, w, h)]
      ↓
[EasyOCR → Text]
      ↓
[Validate with Regex]
      ↓
[Annotate Frame + Save Results]
```

### Detailed Process Explanation / Penjelasan Detail Proses

#### 1. **Input Frame / Frame Input**
- Menerima gambar atau frame video sebagai input
- Format yang didukung: JPG, PNG, MP4, AVI, dll.

#### 2. **Grayscale Conversion / Konversi Grayscale**
- **Tujuan**: Mengurangi kompleksitas komputasi dengan mengubah gambar RGB (3 channel) menjadi grayscale (1 channel)
- **Metode**: Menggunakan weighted average: `Gray = 0.299*R + 0.587*G + 0.114*B`
- **Manfaat**: Mempercepat proses dan mengurangi noise

#### 3. **Bilateral Filter / Filter Bilateral**
- **Tujuan**: Menghilangkan noise sambil mempertahankan tepi (edge) yang tajam
- **Keunggulan**: 
  - Smoothing pada area uniform
  - Preservasi detail pada edges
  - Ideal untuk preprocessing sebelum edge detection

#### 4. **Canny Edge Detection / Deteksi Tepi Canny**
- **Tujuan**: Mendeteksi tepi/kontur objek dalam gambar
- **Parameter**: 
  - Lower threshold: untuk weak edges
  - Upper threshold: untuk strong edges
- **Output**: Binary image (hitam-putih) dengan tepi yang terdeteksi

#### 5. **Find Contours / Mencari Kontur**
- **Tujuan**: Mengekstrak bentuk-bentuk geometris dari binary image
- **Metode**: Menggunakan algoritma kontour detection untuk menemukan garis batas objek
- **Output**: List koordinat yang membentuk kontur

#### 6. **Area Filtering & Corner Approximation / Filter Area & Aproksimasi Sudut**
- **Filter berdasarkan area**: Menghilangkan kontur yang terlalu kecil atau terlalu besar
- **Approximation**: Menyederhanakan kontur menjadi polygon dengan 4 titik sudut
- **Tujuan**: Mengidentifikasi bentuk persegi panjang (karakteristik plat nomor)

#### 7. **Aspect Ratio Filtering / Filter Rasio Aspek**
- **Range**: 2:1 hingga 8:1 (lebar:tinggi)
- **Alasan**: Plat nomor umumnya memiliki bentuk persegi panjang horizontal
- **Contoh**: Plat Indonesia ≈ 4:1, Plat Eropa ≈ 5:1

#### 8. **ROI Cropping / Pemotongan Region of Interest**
- **Tujuan**: Mengekstrak area yang diduga sebagai plat nomor
- **Parameter**: Koordinat (x, y) dan dimensi (width, height)
- **Output**: Cropped image yang hanya berisi plat nomor

#### 9. **OCR (Optical Character Recognition)**
- **Library**: EasyOCR (mendukung berbagai bahasa)
- **Tujuan**: Mengubah gambar text menjadi string text
- **Preprocessing**: Resize, contrast enhancement, noise reduction

#### 10. **Regex Validation / Validasi dengan Regex**
- **Tujuan**: Memvalidasi format nomor plat sesuai standar
- **Contoh Pattern Indonesia**: 
  - `[A-Z]{1,2} \d{1,4} [A-Z]{1,3}` (B 1234 ABC)
  - `[A-Z]{2} \d{1,4} [A-Z]{1,2}` (AB 1234 CD)

#### 11. **Annotation & Results / Anotasi & Hasil**
- **Visualisasi**: Menggambar bounding box di sekitar plat terdeteksi
- **Text overlay**: Menampilkan nomor plat yang berhasil dibaca
- **Save results**: Menyimpan gambar hasil dan data text

### Technical Requirements / Kebutuhan Teknis

#### Libraries / Pustaka:
- **OpenCV**: Image processing dan computer vision
- **EasyOCR**: Optical Character Recognition
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization dan plotting

#### Hardware Recommendations / Rekomendasi Hardware:
- **CPU**: Multi-core processor untuk parallel processing
- **RAM**: Minimum 8GB untuk processing gambar resolusi tinggi
- **GPU**: Optional, untuk accelerated OCR processing

### Challenges & Solutions / Tantangan & Solusi

#### 1. **Variasi Pencahayaan**
- **Masalah**: Overexposed atau underexposed images
- **Solusi**: Histogram equalization, adaptive thresholding

#### 2. **Sudut Pandang (Perspective)**
- **Masalah**: Plat nomor terdistorsi karena sudut kamera
- **Solusi**: Perspective transformation menggunakan 4-point mapping

#### 3. **Plat Kotor atau Rusak**
- **Masalah**: Karakter tidak jelas atau hilang
- **Solusi**: Morphological operations, template matching

#### 4. **Multiple Plates**
- **Masalah**: Beberapa kendaraan dalam satu frame
- **Solusi**: NMS (Non-Maximum Suppression), confidence scoring

### Performance Metrics / Metrik Performa

- **Precision**: Akurasi deteksi plat nomor yang benar
- **Recall**: Persentase plat nomor yang berhasil dideteksi
- **Processing Time**: Waktu yang dibutuhkan per frame
- **Character Accuracy**: Akurasi pembacaan karakter individual

### Usage Examples / Contoh Penggunaan

```python
# Basic usage
detector = ANPRDetector()
result = detector.detect_plate("input_image.jpg")
print(f"Detected plate: {result['plate_number']}")
```

### Future Improvements / Pengembangan Selanjutnya

1. **Deep Learning Integration**: Menggunakan YOLO atau CNN untuk detection
2. **Real-time Processing**: Optimisasi untuk video streaming
3. **Multi-language Support**: Mendukung berbagai format plat internasional
4. **Database Integration**: Penyimpanan dan pencarian data plat nomor
5. **Mobile Application**: Implementasi pada smartphone