# detector/io.py
import os
import csv

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_images(folder, exts):
    if not os.path.exists(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if any(f.lower().endswith(e) for e in exts):
            out.append(os.path.join(folder, f))
    return out

def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "crop"])
        for r in rows:
            w.writerow(r)
