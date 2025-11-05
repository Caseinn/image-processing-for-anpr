import os
import csv
import re
import cv2
import numpy as np

# ----------------------------
# Optional: simple Indonesia plate validator
# e.g., B1234ABC, BE1234CD, AB123CD, etc.
# Adjust as you like.
PLATE_RE = re.compile(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$')

# ----------------------------
# OCR setup (uses GPU if available; falls back to CPU)
try:
    import easyocr
    try:
        READER = easyocr.Reader(['en'], gpu=True)
    except Exception:
        READER = easyocr.Reader(['en'], gpu=False)
except ImportError:
    READER = None  # Will warn later if not installed


def ensure_output_dirs(base_dir):
    crops_dir = os.path.join(base_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    return crops_dir

def load_images(images_dir, exts):
    images = []
    for ext in exts:
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(ext):
                images.append(os.path.join(images_dir, f))
    return images

def preprocess_frame(image, clahe_clip, clahe_grid):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    return edges

def find_plate_candidates(edges, frame_area, min_rel_area, max_rel_area):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * min_rel_area or area > frame_area * max_rel_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append(approx)
    return candidates

def filter_by_aspect_ratio(contours, aspect_min, aspect_max):
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / float(h)
        if aspect_min <= aspect <= aspect_max:
            filtered.append((cnt, (x, y, w, h)))
    return filtered

# --- light enhancement before OCR (helps EasyOCR a lot)
def enhance_for_ocr(roi_gray):
    # Resize min height to ~48px for better OCR
    h, w = roi_gray.shape[:2]
    if h < 48:
        scale = 48 / float(h)
        roi_gray = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # CLAHE + denoise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enh = clahe.apply(roi_gray)
    enh = cv2.fastNlMeansDenoising(enh, h=10)
    return enh

def run_easyocr_on_roi(roi_bgr, min_conf=0.3, validate_regex=True):
    if READER is None:
        return None, 0.0, None  # easyocr not installed
    # to grayscale & enhance
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr
    enh = enhance_for_ocr(gray)

    # readtext returns: [ [box, text, conf], ... ]
    results = READER.readtext(enh)
    if not results:
        return None, 0.0, None

    # pick the highest-confidence line that (optionally) matches plate regex
    best_text, best_conf, best_box = None, 0.0, None
    for box, txt, conf in results:
        txt = txt.strip().upper().replace(" ", "")
        if validate_regex and not PLATE_RE.match(txt):
            continue
        if conf > best_conf:
            best_text, best_conf, best_box = txt, conf, box

    # if nothing matched regex, fall back to highest conf regardless
    if best_text is None:
        for box, txt, conf in results:
            txt = txt.strip().upper().replace(" ", "")
            if conf > best_conf:
                best_text, best_conf, best_box = txt, conf, box

    return best_text, best_conf, best_box

# ----------------------------
# CROPPING (FIRST PASS) — returns list of (source_image_basename, crop_path)
def extract_and_save_crops_only(image, contour_info, base_name, crops_dir, source_image_path):
    saved_pairs = []  # (src_name, crop_path)
    for idx, (_, (x, y, w, h)) in enumerate(contour_info, start=1):
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        crop_name = f"{base_name}_{idx:02d}.png"
        crop_path = os.path.join(crops_dir, crop_name)
        cv2.imwrite(crop_path, roi)
        saved_pairs.append((os.path.basename(source_image_path), crop_path))
    return saved_pairs

def process_image_for_crops(image_path, crops_dir, area_range, aspect_range, clahe_clip, clahe_grid):
    """Detect candidates and save crops only. Returns list of (source_image, crop_path)."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Cannot read image {image_path}")
        return []

    edges = preprocess_frame(image, clahe_clip, clahe_grid)
    frame_area = image.shape[0] * image.shape[1]
    candidates = find_plate_candidates(edges, frame_area, *area_range)
    filtered = filter_by_aspect_ratio(candidates, *aspect_range)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_pairs = extract_and_save_crops_only(image, filtered, base_name, crops_dir, image_path)

    print(f"[INFO] {os.path.basename(image_path)}: edges={len(candidates)} valid={len(filtered)} saved={len(saved_pairs)}")
    return saved_pairs

# ----------------------------
# OCR (SECOND PASS) — iterate over saved crops and write CSV rows
def ocr_saved_crops_and_write_csv(saved_pairs, csv_path):
    """saved_pairs: list of (source_image_name, crop_path)"""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "crop_path", "text", "conf"])  # header

        if READER is None:
            print("[WARN] easyocr not installed. OCR columns will be empty.")
            for src_name, crop_path in saved_pairs:
                writer.writerow([src_name, crop_path, "", ""])
            return

        for src_name, crop_path in saved_pairs:
            roi = cv2.imread(crop_path)
            if roi is None or roi.size == 0:
                writer.writerow([src_name, crop_path, "", ""])
                print(f"[OCR] {os.path.basename(crop_path)}: unreadable crop")
                continue

            text, conf, _ = run_easyocr_on_roi(roi)
            writer.writerow([
                src_name,
                crop_path,
                text or "",
                f"{conf:.4f}" if conf else ""
            ])
            if text:
                print(f"[OCR] {os.path.basename(crop_path)}: text='{text}' conf={conf:.2f}")
            else:
                print(f"[OCR] {os.path.basename(crop_path)}: no confident text")

# ----------------------------
def main():
    images_dir = "data/images"
    output_dir = "output"
    exts = [".jpg", ".jpeg", ".png"]

    min_rel_area, max_rel_area = 0.0005, 0.3
    min_aspect, max_aspect = 2.0, 8.0
    clahe_clip, clahe_grid = 2, 8

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = ensure_output_dirs(output_dir)
    images = load_images(images_dir, exts)

    if not images:
        print(f"[WARN] No images found in {images_dir}")
        return

    # First pass: detect & save ALL crops
    all_saved_pairs = []  # list of (source_image_name, crop_path)
    for image_path in images:
        pairs = process_image_for_crops(
            image_path,
            crops_dir,
            (min_rel_area, max_rel_area),
            (min_aspect, max_aspect),
            clahe_clip,
            clahe_grid,
        )
        all_saved_pairs.extend(pairs)

    # Second pass: OCR all saved crops & write CSV
    csv_path = os.path.join(output_dir, "plates.csv")
    ocr_saved_crops_and_write_csv(all_saved_pairs, csv_path)

    print(f"[OK] CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
