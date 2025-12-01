"""
Gradio front-end to run the ANPR extraction pipeline on single images.
Upload a vehicle photo to see detections, crops, and each processing step.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np

from detector.config import load_config


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to RGB for display."""
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB image (from Gradio) to BGR for OpenCV processing."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def build_pipeline_steps(
    img_bgr: np.ndarray, cfg: dict
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
    """Run the same detection steps as runner.detect but keep every stage."""
    steps: List[Tuple[str, np.ndarray]] = []

    # 1) Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps.append(("Grayscale", to_rgb(gray)))

    # 2) CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=cfg["clahe_clip"],
        tileGridSize=(cfg["clahe_grid"], cfg["clahe_grid"]),
    )
    gray_eq = clahe.apply(gray)
    steps.append(("CLAHE", to_rgb(gray_eq)))

    # 3) Gaussian Blur
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    steps.append(("Gaussian Blur", to_rgb(blur)))

    # 4) Canny Edges
    edges = cv2.Canny(blur, 50, 200)
    steps.append(("Canny Edges", to_rgb(edges)))

    # 5) All contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_bgr.copy()
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 255), 2)
    steps.append(("All Contours", to_rgb(contour_img)))

    # 6) Area filter + quadrilateral candidates
    area_min, area_max = cfg["area_range"]
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    quad_candidates: List[np.ndarray] = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * area_min or area > img_area * area_max:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad_candidates.append(approx)

    area_vis = img_bgr.copy()
    cv2.drawContours(area_vis, quad_candidates, -1, (0, 255, 0), 2)
    steps.append(("Area Filter + 4 Corners", to_rgb(area_vis)))

    # 7) Aspect ratio filter
    asp_min, asp_max = cfg["aspect_range"]
    final_boxes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    aspect_vis = img_bgr.copy()

    for c in quad_candidates:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if asp_min <= aspect <= asp_max:
            final_boxes.append((c, (x, y, w, h)))
            cv2.rectangle(aspect_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    steps.append(("Aspect Filtered", to_rgb(aspect_vis)))

    # 8) Crop preview (first valid box or full image as fallback)
    if final_boxes:
        _, (x, y, w, h) = final_boxes[0]
        crop_preview = img_bgr[y : y + h, x : x + w]
    else:
        crop_preview = img_bgr
    steps.append(("Crop Preview", to_rgb(crop_preview)))

    return steps, final_boxes


def run_extraction(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]], List[List[float]], str]:
    """Callback for Gradio: process one image and return visuals plus metadata."""
    if image is None:
        return None, [], [], [], "Upload an image to start."

    cfg = load_config()
    bgr = to_bgr(image)

    steps, boxes = build_pipeline_steps(bgr, cfg)

    annotated = bgr.copy()
    crops: List[Tuple[np.ndarray, str]] = []
    rows: List[List[float]] = []

    # Iterate over final boxes and collect metadata
    for idx, (_, (x, y, w, h)) in enumerate(boxes, start=1):
        roi = bgr[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        aspect = w / float(h) if h else 0.0
        crops.append((to_rgb(roi), f"Crop {idx}"))
        rows.append([idx, x, y, w, h, round(aspect, 3)])

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert steps to Gallery format: (image, label)
    step_gallery = [(img, title) for title, img in steps]
    annotated_rgb = to_rgb(annotated)

    status = (
        f"Found {len(crops)} crop(s)."
        if crops
        else "No plate-like regions detected with current parameters."
    )

    return annotated_rgb, crops, step_gallery, rows, status


DESCRIPTION = """
# ANPR Plate Extraction
Upload a vehicle image to run the same processing steps used in `detector/runner.py`.
See the detections with bounding boxes, extracted plate crops, and every intermediate stage.
"""


BASE_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
EXAMPLES = [
    [os.path.join(BASE_EXAMPLE_DIR, "test007.jpg")],
    [os.path.join(BASE_EXAMPLE_DIR, "test009.jpg")],
    [os.path.join(BASE_EXAMPLE_DIR, "test082.jpg")],
]


with gr.Blocks(
    title="Plate Extraction",
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload", "clipboard"],
                height=500
            )
            run_btn = gr.Button("Run Extraction", variant="primary")
            status = gr.Markdown("Ready.", elem_id="status-text")

        with gr.Column(scale=1):
            annotated_out = gr.Image(
                label="Detections (with bounding boxes)",
                type="numpy",
                height=500,
                format="png"
            )

    with gr.Row():
        crop_gallery = gr.Gallery(
            label="Extracted Plate Crops",
            show_label=True,
            elem_id="crop-gallery",
            columns=3,
            height="auto",
            format="png"
        )

    with gr.Row():
        step_gallery = gr.Gallery(
            label="Processing Steps",
            show_label=True,
            columns=4,
            height=280,
            preview=False,
            format="png"
        )
        bbox_table = gr.Dataframe(
            headers=["id", "x", "y", "w", "h"],
            datatype=["number", "number", "number", "number", "number"],
            interactive=False,
            label="Bounding boxes",
        )

    clear_btn = gr.ClearButton(
        components=[image_input, annotated_out, crop_gallery, step_gallery, bbox_table],
        value="Clear",
    )

    run_btn.click(
        fn=run_extraction,
        inputs=image_input,
        outputs=[annotated_out, crop_gallery, step_gallery, bbox_table, status],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=image_input,
        outputs=[annotated_out, crop_gallery, step_gallery, bbox_table, status],
        fn=run_extraction,
        cache_examples=False,
        label="Try sample images",
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"),
                css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto;
            }
        """
    )
