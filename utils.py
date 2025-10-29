# utils.py
import cv2
import numpy as np
import re
import os

def preprocess_for_detection(frame):
    """Preprocess frame for plate region detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    edged = cv2.Canny(blurred, 30, 200)
    return edged, gray


def find_plate_candidates(edged, frame_area):
    """Find quadrilateral regions that look like license plates"""
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.001 or area > frame_area * 0.25:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 8.0 and h > 20:
                candidates.append((x, y, w, h))
    return candidates

def clean_ocr_text(text):
    """Clean and normalize OCR output"""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned

def is_valid_indonesian_plate(text, pattern):
    """Validate against Indonesian format"""
    return bool(re.match(pattern, text)) and 4 <= len(text) <= 9

def debug_pipeline(frame, edged, candidates, plate_roi=None, step_frame=0, output_dir='debug'):
    """
    Save intermediate images that match the current preprocess path:
    BGR -> Gray -> CLAHE -> Gaussian -> Canny(30,200) -> Contours -> Candidates -> (Cropped ROI)
    """
    debug_dir = os.path.join(output_dir, f"frame_{step_frame}")
    os.makedirs(debug_dir, exist_ok=True)

    orig = frame.copy()
    cv2.imwrite(os.path.join(debug_dir, "1_original.jpg"), orig)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = gray.copy()
    cv2.imwrite(os.path.join(debug_dir, "2_grayscale.jpg"), g)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    e = enhanced.copy()
    cv2.imwrite(os.path.join(debug_dir, "3_clahe.jpg"), e)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    b = blurred.copy()
    cv2.imwrite(os.path.join(debug_dir, "4_blurred.jpg"), b)

    canny = cv2.Canny(blurred, 30, 200)
    c = canny.copy()
    cv2.imwrite(os.path.join(debug_dir, "5_edges.jpg"), c)

    contour_img = frame.copy()
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, "6_contours.jpg"), contour_img)

    cand_img = frame.copy()
    for (x, y, w, h) in candidates:
        cv2.rectangle(cand_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, "7_candidates.jpg"), cand_img)


    if plate_roi is not None and plate_roi.size > 0:
        cv2.imwrite(os.path.join(debug_dir, "8_cropped_plate.jpg"), plate_roi)

    print(f"🔍 Debug images saved to: {debug_dir}")
