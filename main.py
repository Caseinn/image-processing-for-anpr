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
# IoU Evaluation Functions
def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels and convert to pixel coordinates.
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # YOLO format: class x_center y_center width height (normalized)
            cls, x_center, y_center, width, height = map(float, parts[:5])
            
            # Convert to pixel coordinates
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            # Convert to x1, y1, x2, y2
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            boxes.append((x1, y1, x2, y2))
    
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes.
    box format: (x1, y1, x2, y2)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_detections_to_ground_truth(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
    """Match detected boxes to ground truth boxes using IoU.
    Returns: (matched_pairs, unmatched_detections, unmatched_ground_truths)
    matched_pairs: list of (detected_idx, gt_idx, iou)
    """
    matched_pairs = []
    matched_detections = set()
    matched_gts = set()
    
    # For each ground truth, find best matching detection
    for gt_idx, gt_box in enumerate(ground_truth_boxes):
        best_iou = 0.0
        best_det_idx = -1
        
        for det_idx, det_box in enumerate(detected_boxes):
            if det_idx in matched_detections:
                continue
            
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_det_idx = det_idx
        
        if best_det_idx >= 0:
            matched_pairs.append((best_det_idx, gt_idx, best_iou))
            matched_detections.add(best_det_idx)
            matched_gts.add(gt_idx)
    
    unmatched_detections = [i for i in range(len(detected_boxes)) if i not in matched_detections]
    unmatched_gts = [i for i in range(len(ground_truth_boxes)) if i not in matched_gts]
    
    return matched_pairs, unmatched_detections, unmatched_gts

def evaluate_iou_metrics(all_image_results, iou_threshold=0.5):
    """Calculate overall metrics from all images.
    Returns: dict with precision, recall, f1, mAP
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    
    for result in all_image_results:
        total_tp += result['true_positives']
        total_fp += result['false_positives']
        total_fn += result['false_negatives']
        all_ious.extend(result['matched_ious'])
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_detections': total_tp + total_fp,
        'total_ground_truths': total_tp + total_fn
    }

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

def process_image_for_crops(image_path, crops_dir, area_range, aspect_range, clahe_clip, clahe_grid, labels_dir=None, iou_threshold=0.5):
    """Detect candidates and save crops only. Returns list of (source_image, crop_path) and evaluation result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Cannot read image {image_path}")
        return [], None

    edges = preprocess_frame(image, clahe_clip, clahe_grid)
    frame_area = image.shape[0] * image.shape[1]
    candidates = find_plate_candidates(edges, frame_area, *area_range)
    filtered = filter_by_aspect_ratio(candidates, *aspect_range)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_pairs = extract_and_save_crops_only(image, filtered, base_name, crops_dir, image_path)

    # IoU Evaluation
    eval_result = None
    if labels_dir:
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        img_height, img_width = image.shape[:2]
        ground_truth_boxes = load_yolo_labels(label_path, img_width, img_height)
        
        # Convert detected contours to bounding boxes
        detected_boxes = [(x, y, x + w, y + h) for _, (x, y, w, h) in filtered]
        
        # Match detections to ground truth
        matched_pairs, unmatched_dets, unmatched_gts = match_detections_to_ground_truth(
            detected_boxes, ground_truth_boxes, iou_threshold
        )
        
        matched_ious = [iou for _, _, iou in matched_pairs]
        
        eval_result = {
            'image_name': os.path.basename(image_path),
            'num_detections': len(detected_boxes),
            'num_ground_truths': len(ground_truth_boxes),
            'true_positives': len(matched_pairs),
            'false_positives': len(unmatched_dets),
            'false_negatives': len(unmatched_gts),
            'matched_ious': matched_ious,
            'mean_iou': np.mean(matched_ious) if matched_ious else 0.0
        }
        
        print(f"[INFO] {os.path.basename(image_path)}: det={len(detected_boxes)} gt={len(ground_truth_boxes)} TP={len(matched_pairs)} FP={len(unmatched_dets)} FN={len(unmatched_gts)} mIoU={eval_result['mean_iou']:.3f}")
    else:
        print(f"[INFO] {os.path.basename(image_path)}: edges={len(candidates)} valid={len(filtered)} saved={len(saved_pairs)}")
    
    return saved_pairs, eval_result

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
    labels_dir = "data/labels"  # Ground truth labels
    output_dir = "output"
    exts = [".jpg", ".jpeg", ".png"]

    min_rel_area, max_rel_area = 0.0005, 0.3
    min_aspect, max_aspect = 2.0, 8.0
    clahe_clip, clahe_grid = 2, 8
    iou_threshold = 0.5  # IoU threshold for matching

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = ensure_output_dirs(output_dir)
    images = load_images(images_dir, exts)

    if not images:
        print(f"[WARN] No images found in {images_dir}")
        return

    # Check if labels directory exists
    evaluate_iou = os.path.exists(labels_dir)
    if evaluate_iou:
        print(f"[INFO] Labels found. IoU evaluation will be performed.")
    else:
        print(f"[WARN] No labels directory found at {labels_dir}. Skipping IoU evaluation.")

    # First pass: detect & save ALL crops
    all_saved_pairs = []  # list of (source_image_name, crop_path)
    all_eval_results = []  # IoU evaluation results
    
    for image_path in images:
        pairs, eval_result = process_image_for_crops(
            image_path,
            crops_dir,
            (min_rel_area, max_rel_area),
            (min_aspect, max_aspect),
            clahe_clip,
            clahe_grid,
            labels_dir if evaluate_iou else None,
            iou_threshold
        )
        all_saved_pairs.extend(pairs)
        if eval_result:
            all_eval_results.append(eval_result)

    # Second pass: OCR all saved crops & write CSV
    csv_path = os.path.join(output_dir, "plates.csv")
    ocr_saved_crops_and_write_csv(all_saved_pairs, csv_path)

    print(f"[OK] CSV saved to {csv_path}")
    
    # Save IoU evaluation results
    if all_eval_results:
        eval_csv_path = os.path.join(output_dir, "iou_evaluation.csv")
        with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name", "num_detections", "num_ground_truths", 
                "true_positives", "false_positives", "false_negatives", "mean_iou"
            ])
            
            for result in all_eval_results:
                writer.writerow([
                    result['image_name'],
                    result['num_detections'],
                    result['num_ground_truths'],
                    result['true_positives'],
                    result['false_positives'],
                    result['false_negatives'],
                    f"{result['mean_iou']:.4f}"
                ])
        
        # Calculate overall metrics
        overall_metrics = evaluate_iou_metrics(all_eval_results, iou_threshold)
        
        # Save overall metrics
        metrics_path = os.path.join(output_dir, "overall_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("=== Overall IoU Evaluation Metrics ===\n\n")
            f.write(f"IoU Threshold: {iou_threshold}\n\n")
            f.write(f"Total Detections: {overall_metrics['total_detections']}\n")
            f.write(f"Total Ground Truths: {overall_metrics['total_ground_truths']}\n")
            f.write(f"True Positives (TP): {overall_metrics['total_tp']}\n")
            f.write(f"False Positives (FP): {overall_metrics['total_fp']}\n")
            f.write(f"False Negatives (FN): {overall_metrics['total_fn']}\n\n")
            f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
            f.write(f"Recall: {overall_metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {overall_metrics['f1_score']:.4f}\n")
            f.write(f"Mean IoU: {overall_metrics['mean_iou']:.4f}\n")
        
        print(f"\n=== IoU Evaluation Results ===")
        print(f"Precision: {overall_metrics['precision']:.4f}")
        print(f"Recall: {overall_metrics['recall']:.4f}")
        print(f"F1-Score: {overall_metrics['f1_score']:.4f}")
        print(f"Mean IoU: {overall_metrics['mean_iou']:.4f}")
        print(f"TP: {overall_metrics['total_tp']}, FP: {overall_metrics['total_fp']}, FN: {overall_metrics['total_fn']}")
        print(f"\n[OK] IoU evaluation saved to {eval_csv_path}")
        print(f"[OK] Overall metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
