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
from detector.core import detect


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to RGB for display.

    Args:
        img: Input image in BGR (3-channel) or grayscale (2D) format.

    Returns:
        Image in RGB format suitable for Gradio/matplotlib display.
        Returns ``None`` unchanged.
    """
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR for OpenCV processing.

    Args:
        img: Input image in RGB format (from Gradio).

    Returns:
        Image in BGR format.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def build_pipeline_steps(
    img_bgr: np.ndarray, cfg: dict
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
    """Build step-by-step visualizations from the detection pipeline.

    Delegates actual detection to :func:`detector.core.detect` and
    creates annotated images for each processing stage (grayscale,
    CLAHE, blur, edges, contours, candidates, final boxes, crop).

    Args:
        img_bgr: Input BGR image.
        cfg: Configuration dictionary.

    Returns:
        Tuple of:
            - steps: List of ``(label, RGB_image)`` for each stage.
            - boxes: List of ``(contour, (x, y, w, h))`` for detected
              plates.
    """
    _, boxes, pipeline = detect(img_bgr, cfg)

    steps: List[Tuple[str, np.ndarray]] = []

    steps.append(("Grayscale", to_rgb(pipeline["gray"])))
    steps.append(("CLAHE", to_rgb(pipeline["clahe"])))
    steps.append(("Gaussian Blur", to_rgb(pipeline["blur"])))
    steps.append(("Canny Edges", to_rgb(pipeline["edges"])))

    contour_img = img_bgr.copy()
    cv2.drawContours(contour_img, pipeline["contours"], -1, (0, 255, 255), 2)
    steps.append(("All Contours", to_rgb(contour_img)))

    area_vis = img_bgr.copy()
    cv2.drawContours(area_vis, pipeline["candidates"], -1, (0, 255, 0), 2)
    steps.append(("Area Filter + 4 Corners", to_rgb(area_vis)))

    aspect_vis = img_bgr.copy()
    for _, (x, y, w, h) in pipeline["boxes"]:
        cv2.rectangle(aspect_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    steps.append(("Aspect Filtered", to_rgb(aspect_vis)))

    if pipeline["boxes"]:
        _, (x, y, w, h) = pipeline["boxes"][0]
        crop_preview = img_bgr[y : y + h, x : x + w]
    else:
        crop_preview = img_bgr
    steps.append(("Crop Preview", to_rgb(crop_preview)))

    return steps, pipeline["boxes"]


def run_extraction(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]], List[List[float]], str]:
    """Process a single image and return all Gradio visualizations.

    This is the callback function bound to the Gradio UI's "Run
    Extraction" button.

    Args:
        image: Input RGB image from the Gradio upload component.

    Returns:
        Tuple of:
            - annotated_rgb: Image with green bounding boxes drawn on
              detections.
            - crops: List of ``(crop_RGB, label)`` for gallery display.
            - step_gallery: List of ``(step_RGB, label)`` for pipeline
              steps gallery.
            - rows: Table rows with ``[id, x, y, w, h, aspect]``.
            - status: Status message string.
    """
    if image is None:
        return None, [], [], [], "Upload an image to start."

    cfg = load_config()
    bgr = to_bgr(image)

    steps, boxes = build_pipeline_steps(bgr, cfg)

    annotated = bgr.copy()
    crops: List[Tuple[np.ndarray, str]] = []
    rows: List[List[float]] = []

    for idx, (_, (x, y, w, h)) in enumerate(boxes, start=1):
        roi = bgr[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        aspect = w / float(h) if h else 0.0
        crops.append((to_rgb(roi), f"Crop {idx}"))
        rows.append([idx, x, y, w, h, round(aspect, 3)])

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
