"""Central configuration for the ANPR pipeline.

All tunable parameters are defined here so they can be adjusted
without modifying detection logic.
"""


def load_config():
    """Return a dictionary of pipeline configuration values.

    Returns:
        dict: Configuration keys including file paths, image extensions,
            area and aspect ratio ranges, CLAHE parameters, Canny edge
            thresholds, approximation epsilon, and evaluation IoU threshold.
    """
    return {
        "images_dir": "data/images",
        "labels_dir": "data/labels",
        "out_dir": "output",
        "exts": [".jpg", ".jpeg", ".png"],
        "area_range": (0.0005, 0.3),
        "aspect_range": (2.0, 8.0),
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "gauss_kernel": 5,
        "canny_low": 50,
        "canny_high": 200,
        "approx_eps": 0.02,
        "iou_thr": 0.5,
    }
