"""
Image utilities shared across the detection pipeline.
All functions operate on numpy arrays in BGR order (OpenCV convention)
unless explicitly converted.
"""
import cv2
import numpy as np


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes (jpg/png) into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    return image


def resize_for_streaming(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Downscale a frame for the live webcam pipeline so repeated /detect/frame
    calls stay fast. No-op if the image is already narrower than max_width.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def crop_box(image: np.ndarray, box: tuple, padding: int = 4) -> np.ndarray:
    """
    Crop a (x1, y1, x2, y2) region out of `image`, with a small pixel margin
    (helps OWLv2 see a little context around the box) clamped to bounds.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    if x2 <= x1 or y2 <= y1:
        # Degenerate box — return a 1x1 pixel rather than raising, so a bad
        # detection never crashes the whole pipeline.
        return image[0:1, 0:1]
    return image[y1:y2, x1:x2]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_crop(image: np.ndarray, path) -> None:
    """Persist a crop to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
