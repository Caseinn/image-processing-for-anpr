# gui/app.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressBar, QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer

from detector.config import load_config


# --------- helper conversion ---------
def to_pixmap(img):
    """Convert OpenCV BGR image to QPixmap."""
    if img is None or img.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    q = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("Automatic Number Plate Recognition")
        self.resize(1000, 620)

        # state
        self.current_img = None       # original BGR
        self.step_images = []         # list of BGR images per step
        self.current_step_index = -1  # for animation

        # step meta (title + desc)
        self.steps_info = [
            (
                "Step 1: Grayscale Conversion",
                "Pada tahap ini, gambar dikonversi menjadi grayscale untuk "
                "mereduksi jumlah kanal sehingga proses analisis berikutnya lebih sederhana."
            ),
            (
                "Step 2: CLAHE Enhancement",
                "Gambar kemudian ditingkatkan kontrasnya menggunakan CLAHE agar detail plat "
                "lebih tampak meski kondisi pencahayaan tidak merata."
            ),
            (
                "Step 3: Gaussian Blur",
                "Gaussian Blur diterapkan untuk mengurangi noise sambil tetap mempertahankan "
                "struktur tepi penting."
            ),
            (
                "Step 4: Canny Edge Detection",
                "Tahap ini mengekstraksi tepi dalam bentuk citra biner agar struktur objek seperti "
                "plat lebih mudah dikenali."
            ),
            (
                "Step 5: Find Contours",
                "Dari hasil deteksi tepi, sistem mencari kontur yang mewakili bentuk-bentuk yang "
                "muncul pada gambar."
            ),
            (
                "Step 6: Filter by Area + Approximate Corners",
                "Kontur yang terlalu kecil atau tidak relevan disaring, lalu bentuknya diaproksimasi "
                "menjadi empat sudut agar mendekati bentuk persegi panjang plat nomor."
            ),
            (
                "Step 7: Filter by Aspect Ratio (2–8)",
                "Kontur yang lolos kemudian dipilih berdasarkan rasio panjang–lebar khas plat nomor "
                "agar kandidat lebih akurat."
            ),
            (
                "Step 8: Crop ROI (x, y, w, h)",
                "Terakhir, area plat hasil seleksi dipotong sebagai Region of Interest untuk proses "
                "pengenalan lebih lanjut."
            ),
        ]

        self._build_ui()

    # --------- UI setup ---------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(16)

        # header
        title = QLabel("Automatic Number Plate Recognition")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ffffff;")
        main.addWidget(title)

        # card frame
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1f2933;
                border-radius: 18px;
                border: none;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 16)
        card_layout.setSpacing(12)

        # image area
        self.image_label = QLabel("Unggah gambar di sini")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #111827;
                border-radius: 12px;
                border: 2px dashed #4b5563;
                color: #9ca3af;
                font-size: 14px;
            }
        """)
        card_layout.addWidget(self.image_label)

        # step text
        self.step_title_label = QLabel("")
        self.step_title_label.setStyleSheet("color: #93c5fd; font-weight: 600;")
        self.step_desc_label = QLabel("Silakan unggah gambar plat kendaraan untuk memulai proses.")
        self.step_desc_label.setStyleSheet("color: #e5e7eb;")
        self.step_desc_label.setWordWrap(True)

        card_layout.addWidget(self.step_title_label)
        card_layout.addWidget(self.step_desc_label)

        # progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self.steps_info))
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #374151;
                border-radius: 8px;
                text-align: center;
                background-color: #111827;
                color: #e5e7eb;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 8px;
            }
        """)
        card_layout.addWidget(self.progress)

        # bottom controls
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.upload_btn = QPushButton("Unggah Gambar")
        self.upload_btn.clicked.connect(self.open_file)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                padding: 8px 18px;
                border-radius: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        bottom.addWidget(self.upload_btn)
        card_layout.addLayout(bottom)

        main.addWidget(card)

        # background color for whole window
        self.setStyleSheet("background-color: #111827;")

    # --------- events ---------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            self.step_desc_label.setText("Gagal memuat gambar.")
            return

        self.current_img = img
        self.step_images = self._compute_steps(img)
        self.current_step_index = -1

        # mulai animasi step-by-step
        self._next_step()

    # --------- compute pipeline ---------
    def _compute_steps(self, img):
        """Hitung semua tahap pemrosesan dan simpan sebagai list gambar BGR."""
        cfg = self.cfg
        H, W = img.shape[:2]
        step_imgs = []

        # Step 1: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        step_imgs.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        # Step 2: CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=cfg["clahe_clip"],
            tileGridSize=(cfg["clahe_grid"], cfg["clahe_grid"])
        )
        gray_eq = clahe.apply(gray)
        step_imgs.append(cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR))

        # Step 3: Gaussian Blur
        blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        step_imgs.append(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR))

        # Step 4: Canny
        edges = cv2.Canny(blur, 50, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        step_imgs.append(edges_bgr)

        # Step 5: Contours (semua)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_img = img.copy()
        cv2.drawContours(cont_img, contours, -1, (0, 255, 255), 2)
        step_imgs.append(cont_img)

        # Step 6: Filter area + approx 4 sudut
        area_img = H * W
        area_min, area_max = cfg["area_range"]
        quad_candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < area_img * area_min or a > area_img * area_max:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quad_candidates.append(approx)

        area_img_vis = img.copy()
        cv2.drawContours(area_img_vis, quad_candidates, -1, (0, 255, 0), 2)
        step_imgs.append(area_img_vis)

        # Step 7: Filter aspect ratio
        asp_min, asp_max = cfg["aspect_range"]
        final_boxes = []
        aspect_vis = img.copy()
        for c in quad_candidates:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            asp = w / float(h)
            if asp_min <= asp <= asp_max:
                final_boxes.append((c, (x, y, w, h)))
                cv2.rectangle(aspect_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        step_imgs.append(aspect_vis)

        # Step 8: Crop ROI (pakai crop pertama jika ada)
        if final_boxes:
            _, (x, y, w, h) = final_boxes[0]
            roi = img[y:y + h, x:x + w]
            crop_img = roi.copy()
        else:
            # kalau tidak ada kandidat, tampilkan lagi gambar dengan bounding box step 7
            crop_img = aspect_vis.copy()
        step_imgs.append(crop_img)

        return step_imgs

    # --------- animation over steps ---------
    def _next_step(self):
        self.current_step_index += 1
        if self.current_step_index >= len(self.step_images):
            # done
            return

        step_img = self.step_images[self.current_step_index]
        title, desc = self.steps_info[self.current_step_index]

        # update labels
        self.step_title_label.setText(title)
        self.step_desc_label.setText(desc)
        self.progress.setValue(self.current_step_index + 1)

        # show image scaled to label
        pix = to_pixmap(step_img)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 12px;
                border: 1px solid #4b5563;
                color: #9ca3af;
            }
        """)

        # schedule next step
        QTimer.singleShot(2500, self._next_step)  # 800ms per step

    # --------- responsive resize ---------
    def resizeEvent(self, event):
        # rescale current step image when window resized
        if 0 <= self.current_step_index < len(self.step_images):
            pix = to_pixmap(self.step_images[self.current_step_index])
            self.image_label.setPixmap(
                pix.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
