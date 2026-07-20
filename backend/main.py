"""FastAPI server for the ANPR plate detection pipeline."""

import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector.config import load_config
from detector.core import detect


cfg = load_config()
app = FastAPI(title="ANPR Plate Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode(img: np.ndarray) -> str:
    """Encode a numpy image array to a base64 data URI string.

    Args:
        img (np.ndarray): Input BGR image array.

    Returns:
        str: Base64 data URI in ``data:image/png;base64,...`` format.
            Returns an empty string if encoding fails.
    """
    success, buf = cv2.imencode(".png", img)
    if not success:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/api/detect")
async def detect_plate(image: UploadFile = File(...)):
    """Run plate detection on an uploaded image.

    Accepts an uploaded image, runs the OpenCV detection pipeline, and
    returns annotated visualisations, per-step pipeline images,
    bounding box metadata, and cropped plate regions.

    Args:
        image (UploadFile): Uploaded image file (JPEG, PNG, etc.).

    Returns:
        dict: JSON-serialisable dictionary with the following keys:
            - annotated: Base64 data URI of the image with green bounding
              boxes overlaid.
            - boxes: List of dicts with ``id``, ``x``, ``y``, ``w``,
              ``h``, and ``aspect`` for each detected plate.
            - crops: List of base64 data URIs of cropped plate regions.
            - pipeline: Dict of per-step visualisation data URIs
              (grayscale, clahe, blur, edges, contours, area_filter,
              aspect_filter).
    """
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    _, boxes, pipeline = detect(img, cfg)

    rows = []
    crops = []
    for idx, (_, (x, y, w, h)) in enumerate(pipeline["boxes"], 1):
        aspect = round(w / float(h), 3) if h else 0.0
        rows.append({"id": idx, "x": x, "y": y, "w": w, "h": h, "aspect": aspect})
        roi = img[y:y+h, x:x+w]
        crops.append(_encode(roi))

    vis = img.copy()
    for _, (x, y, w, h) in pipeline["boxes"]:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    contour_img = img.copy()
    cv2.drawContours(contour_img, pipeline["contours"], -1, (0, 255, 255), 2)

    area_vis = img.copy()
    cv2.drawContours(area_vis, pipeline["candidates"], -1, (0, 255, 0), 2)

    aspect_vis = img.copy()
    for _, (x, y, w, h) in pipeline["boxes"]:
        cv2.rectangle(aspect_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return {
        "annotated": _encode(vis),
        "boxes": rows,
        "crops": crops,
        "pipeline": {
            "grayscale": _encode(pipeline["gray"]),
            "clahe": _encode(pipeline["clahe"]),
            "blur": _encode(pipeline["blur"]),
            "edges": _encode(pipeline["edges"]),
            "contours": _encode(contour_img),
            "area_filter": _encode(area_vis),
            "aspect_filter": _encode(aspect_vis),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
