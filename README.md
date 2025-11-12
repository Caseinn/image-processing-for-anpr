# Automatic Number Plate Recognition (ANPR) System
## Sistem Deteksi Plat Nomor Otomatis menggunakan Image Processing

### Overview / Gambaran Umum

This project implements an Automatic Number Plate Recognition (ANPR) system using image processing techniques. The system can detect and extract license plate numbers from vehicle images.

Proyek ini mengimplementasikan sistem deteksi plat nomor otomatis menggunakan teknik computer vision dan image processing. Sistem dapat mendeteksi dan mengekstrak nomor plat kendaraan dari gambar atau frame video.

### Pipeline / Alur Proses

```
[Input Image]
      ↓
[Grayscale] → reduce channels
      ↓
[CLAHE] → enhance contrast
      ↓
[Gaussian Blur] → denoise + keep edges
      ↓
[Canny Edge Detector] → binary edges
      ↓
[Find Contours] → extract shapes
      ↓
[Filter by Area + Approximate to 4 corners]
      ↓
[Filter by Aspect Ratio (2–8)] → plate dimensions
      ↓
[Crop ROI (x, y, w, h)] → extract plate region
      ↓
[EasyOCR → Text] → character recognition
      ↓
[Validate with Regex] → verify plate format
      ↓
[Annotate Frame + Save Results]
```


###  Detailed Process Explanation / Penjelasan Detail Proses

#### 1. **Input Image / Input Gambar**

* **Deskripsi**: Sistem menerima gambar (JPG/PNG) atau frame dari video (MP4/AVI) sebagai masukan.
* **Tujuan**: Menjadi sumber utama untuk mendeteksi dan mengenali plat nomor kendaraan.


#### 2. **Grayscale Conversion / Konversi Grayscale**

* **Tujuan**: Mengurangi jumlah channel dari RGB (3 channel) menjadi grayscale (1 channel).
* **Metode**: Menggunakan rumus weighted average: `Gray = 0.299R + 0.587G + 0.114B`.
* **Manfaat**: Mempercepat komputasi dan meningkatkan efisiensi deteksi tepi.


#### 3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

* **Tujuan**: Meningkatkan kontras lokal agar karakter pada plat lebih jelas.
* **Keunggulan**:

  * Tidak over-enhance seperti histogram global.
  * Efektif di kondisi pencahayaan tidak merata (bayangan, pantulan).


#### 4. **Gaussian Blur / Perataan Gaussian**

* **Tujuan**: Mengurangi noise sembari mempertahankan struktur tepi (edge).
* **Metode**: Kernel Gaussian 3×3 atau 5×5.
* **Hasil**: Gambar lebih halus tanpa kehilangan kontur utama.


#### 5. **Canny Edge Detection / Deteksi Tepi Canny**

* **Tujuan**: Mengubah hasil blur menjadi **binary edge map** (hitam-putih).
* **Parameter**: Dua ambang batas (lower & upper threshold) untuk mendeteksi tepi kuat dan lemah.
* **Hasil**: Tepi objek, termasuk bentuk persegi panjang plat, menjadi terlihat jelas.

#### 6. **Find Contours / Mencari Kontur**

* **Tujuan**: Mengekstrak batas-batas bentuk dari gambar biner hasil deteksi tepi.
* **Metode**: Menggunakan algoritma `cv2.findContours()` untuk mendapatkan list koordinat tiap bentuk.
* **Output**: Daftar kontur kandidat yang mungkin berisi plat nomor.


#### 7. **Filter by Area & Approximation / Filter Area & Aproksimasi Sudut**

* **Langkah 1 – Filter Area**: Menghapus kontur yang terlalu kecil/besar dibanding ukuran frame.
* **Langkah 2 – Approximation**: Menyederhanakan kontur jadi polygon dengan 4 sudut (`cv2.approxPolyDP`).
* **Tujuan**: Menemukan bentuk persegi panjang yang menyerupai plat nomor.


#### 8. **Filter by Aspect Ratio / Filter Rasio Aspek**

* **Tujuan**: Memastikan bentuk kandidat sesuai rasio plat kendaraan.
* **Range Ideal**: 2:1 hingga 8:1 (lebar : tinggi).
* **Contoh**:

  * Plat Indonesia ≈ 4:1
  * Plat Eropa ≈ 5:1


#### 9. **Crop ROI (Region of Interest) / Pemotongan Area Plat**

* **Tujuan**: Mengambil area koordinat (x, y, w, h) dari plat yang telah lolos filter.
* **Output**: Gambar kecil berisi hanya plat nomor, siap diproses OCR.

---

### **Challenges / Tantangan**

#### 1. **Variasi Pencahayaan (Lighting Variation)**

* **Masalah**: Gambar terlalu terang (overexposed) atau terlalu gelap (underexposed).
* **Dampak pada Image Processing**: 
  * CLAHE bisa over-enhance dan menciptakan noise buatan
  * Canny edge detection gagal menemukan tepi karena kontras rendah
  * Thresholding menghasilkan binary image yang buruk

#### 2. **Sudut Pandang (Perspective Distortion)**

* **Masalah**: Plat nomor tampak miring atau terdistorsi karena posisi kamera.
* **Dampak pada Image Processing**:
  * Contour approximation gagal menghasilkan 4 corners
  * Aspect ratio filtering menolak plat yang valid
  * Karakter terdistorsi sehingga OCR gagal

#### 3. **Noise pada Gambar (Image Noise)**

* **Masalah**: Gangguan visual seperti bintik atau blur akibat pencahayaan rendah atau kamera bergerak.
* **Dampak pada Image Processing**:
  * Gaussian blur menghilangkan detail penting karakter
  * Canny edge mendeteksi false edges dari noise
  * Morphological closing tidak efektif menyambungkan edges yang benar

#### 4. **Plat Kotor atau Rusak (Dirty or Damaged Plates)**

* **Masalah**: Karakter sulit terbaca karena debu, lumpur, atau kerusakan fisik.
* **Dampak pada Image Processing**:
  * Edge detection terputus-putus
  * Morphological operations mengubah bentuk karakter asli
  * ROI yang di-crop mengandung banyak artifacts

#### 5. **Multiple Plates dalam Satu Gambar (Multiple Plates per Frame)**

* **Masalah**: Lebih dari satu kendaraan muncul dalam satu frame.
* **Dampak pada Image Processing**:
  * Find contours mendeteksi banyak kandidat
  * Perlu additional filtering logic untuk memilih plat yang benar
  * Computational cost meningkat untuk processing semua kandidat

#### 6. **Refleksi dan Glare (Reflections and Glare)**

* **Masalah**: Pantulan cahaya matahari atau lampu pada permukaan plat nomor yang mengkilap.
* **Dampak pada Image Processing**:
  * Area terang ekstrim menyebabkan loss of detail
  * CLAHE tidak efektif pada area over-saturated
  * Edge detection gagal pada region yang blown-out

#### 7. **Bayangan (Shadows)**

* **Masalah**: Bayangan dari frame plat atau objek lain menutupi sebagian karakter.
* **Dampak pada Image Processing**:
  * Kontras tidak merata antara area bayangan dan terang
  * Thresholding tidak optimal untuk seluruh region
  * Edge detection menghasilkan discontinuous edges

#### 8. **Motion Blur**

* **Masalah**: Kendaraan bergerak cepat menghasilkan blur pada plat nomor.
* **Dampak pada Image Processing**:
  * Edges menjadi tidak tajam dan lebar
  * Gaussian blur memperburuk keadaan
  * Canny threshold perlu disesuaikan tetapi menimbulkan false positives

#### 9. **Kontras Rendah antara Karakter dan Background (Low Contrast)**

* **Masalah**: Warna karakter dan background plat terlalu mirip.
* **Dampak pada Image Processing**:
  * CLAHE kurang efektif meningkatkan separation
  * Edge detection gagal menemukan boundary karakter
  * Binary thresholding tidak bisa memisahkan foreground/background

#### 10. **Ukuran Plat Terlalu Kecil atau Besar (Scale Variation)**

* **Masalah**: Plat nomor terlalu kecil (jauh) atau terlalu besar (dekat) dalam frame.
* **Dampak pada Image Processing**:
  * Area filtering menolak plat yang terlalu kecil/besar
  * Resolution ROI terlalu rendah untuk OCR
  * Edge detection parameter tidak universal untuk semua scale

#### 11. **Background Noise Kompleks (Cluttered Background)**

* **Masalah**: Background ramai dengan objek, teks, atau pattern lain.
* **Dampak pada Image Processing**:
  * Find contours mendeteksi ratusan false positives
  * Area dan aspect ratio filtering tidak cukup untuk eliminasi
  * Processing time meningkat drastis

#### 12. **Tekstur Plat yang Beragam (Plate Texture Variation)**

* **Masalah**: Plat embossed, printed, atau dengan coating berbeda.
* **Dampak pada Image Processing**:
  * Edge detection menghasilkan double edges pada embossed plates
  * Refleksi berbeda-beda memerlukan adaptive preprocessing
  * Single parameter set tidak optimal untuk semua jenis


#### 13. **Oklusif Parsial (Partial Occlusion)**

* **Masalah**: Sebagian plat tertutup bumper, stiker, atau objek lain.
* **Dampak pada Image Processing**:
  * Contour tidak membentuk rectangle lengkap
  * Area filtering bisa menolak plat yang valid
  * ROI yang di-crop incomplete

#### 14. **Variasi Warna Plat (Color Variation)**

* **Masalah**: Plat hitam, kuning, merah, putih memiliki karakteristik berbeda.
* **Dampak pada Image Processing**:
  * Grayscale conversion menghasilkan kontras berbeda per warna
  * Single threshold/parameter tidak optimal untuk semua
  * Perlu adaptive processing berdasarkan color analysis

---
