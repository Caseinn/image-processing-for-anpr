"""Entry point for the ANPR batch processing pipeline.

Usage:
    python main.py

The script loads configuration from :mod:`detector.config`, processes
all images in ``data/images/``, saves detected plate crops to
``output/crops/``, and writes evaluation metrics to ``output/eval.txt``.
"""

from detector.runner import run

if __name__ == "__main__":
    run()
