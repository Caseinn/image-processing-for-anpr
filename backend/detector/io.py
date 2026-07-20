"""File I/O utilities for the ANPR pipeline.

Handles directory creation, image discovery, and CSV export.
"""

import os
import csv


def ensure_dir(path):
    """Create a directory if it does not already exist.

    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def load_images(folder, exts):
    """List all image files in a directory with matching extensions.

    Args:
        folder (str): Directory to scan.
        exts (list[str]): Allowed file extensions (e.g.
            ``['.jpg', '.png']``).

    Returns:
        list[str]: Sorted list of full image file paths.
    """
    if not os.path.exists(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if any(f.lower().endswith(e) for e in exts):
            out.append(os.path.join(folder, f))
    return out


def write_csv(rows, path):
    """Write a list of (source, crop) pairs to a CSV file.

    Args:
        rows (list[tuple[str, str]]): Rows to write.
        path (str): Output CSV file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "crop"])
        for r in rows:
            w.writerow(r)
