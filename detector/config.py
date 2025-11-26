# detector/config.py

def load_config():
    return {
        "images_dir": "data/images",
        "labels_dir": "data/labels",
        "out_dir": "output",
        "exts": [".jpg", ".jpeg", ".png"],
        "area_range": (0.0005, 0.3),
        "aspect_range": (2.0, 8.0),
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "iou_thr": 0.5,
    }
