# detector/core.py
import os
import cv2
import numpy as np

def preprocess(img, clip, grid):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.Canny(g, 50, 200)

def find_candidates(edges, area, amin, amax):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area * amin or a > area * amax:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            out.append(approx)
    return out

def filter_aspect(cands, amin, amax):
    out = []
    for c in cands:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        asp = w / float(h)
        if amin <= asp <= amax:
            out.append((c, (x, y, w, h)))
    return out

def save_crops(img, boxes, base, out_folder):
    results = []
    for i, (_, (x, y, w, h)) in enumerate(boxes, 1):
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        path = os.path.join(out_folder, f"{base}_{i:02d}.png")
        cv2.imwrite(path, roi)
        results.append((base, path))
    return results

def detect(img, cfg):
    H, W = img.shape[:2]
    edges = preprocess(img, cfg["clahe_clip"], cfg["clahe_grid"])
    area = H * W

    cands = find_candidates(edges, area, *cfg["area_range"])
    boxes = filter_aspect(cands, *cfg["aspect_range"])

    vis = img.copy()
    for _, (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return vis, boxes
