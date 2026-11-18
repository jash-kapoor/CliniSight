"""Stage A: Medical Lab Report Image Ingestion, Preprocessing, and Quality Assessment."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

# Register HEIC opener if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import pypdfium2 as pdfium
from logger import logger

# Maximum image dimension to cap memory and API payload sizes
MAX_IMAGE_DIMENSION = 2400
TARGET_PDF_RENDER_SCALE = 2.5  # ~180-220 DPI


@dataclass
class PageQualityInfo:
    """Quality metrics and warnings for a single report page."""
    page_number: int
    quality_score: float  # 0.0 (unusable) to 1.0 (pristine)
    blur_score: float  # Laplacian variance (higher is sharper)
    brightness: float  # Mean luminance (0-255)
    glare_ratio: float  # Fraction of saturated/overexposed pixels
    is_blurry: bool
    has_glare: bool
    edges_cropped: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class PageImageBundle:
    """Container for processed page representations."""
    page_number: int
    original_image: Image.Image
    cleaned_image: Image.Image
    quality: PageQualityInfo
    text_layer: str = ""  # Secondary hint extracted from digital PDF if available


# ---------------------------------------------------------------------------
# Image Preprocessing Functions (OpenCV + Pillow)
# ---------------------------------------------------------------------------

def auto_rotate_image(pil_img: Image.Image) -> Image.Image:
    """Auto-orient image based on EXIF tags."""
    try:
        return ImageOps.exif_transpose(pil_img) or pil_img
    except Exception:
        return pil_img


def detect_and_deskew(cv_gray: np.ndarray, cv_bgr: np.ndarray) -> np.ndarray:
    """
    Detect document text skew angle using edge contours and rotate the image.
    Caps correction to +/- 45 degrees to avoid accidental 90-degree flips.
    """
    try:
        blurred = cv2.GaussianBlur(cv_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        pts = cv2.findNonZero(thresh)
        if pts is not None and len(pts) > 150:
            rect = cv2.minAreaRect(pts)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Only correct if noticeable skew (> 0.5 deg and < 25 deg)
            if 0.5 < abs(angle) < 25.0:
                h, w = cv_gray.shape[:2]
                center = (w // 2, h // 2)
                m = cv2.getRotationMatrix2D(center, -angle, 1.0)
                rotated = cv2.warpAffine(
                    cv_bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
    except Exception as exc:
        logger.debug("Deskew failed or skipped: %s", exc)
    return cv_bgr


def correct_perspective_if_paper_detected(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Detect if the image is a phone photo of a paper document on a surface.
    If a clear 4-sided convex quadrilateral covering > 35% of the frame is found,
    warp perspective to rectify the page. Otherwise, returns original.
    """
    try:
        h, w = cv_bgr.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv_bgr

        # Sort by contour area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:3]:
            area = cv2.contourArea(c)
            # Paper must occupy between 35% and 98% of the frame
            if 0.35 * img_area < area < 0.98 * img_area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2).astype(np.float32)

                    # Order points: top-left, top-right, bottom-right, bottom-left
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    tl = pts[np.argmin(s)]
                    br = pts[np.argmax(s)]
                    tr = pts[np.argmin(diff)]
                    bl = pts[np.argmax(diff)]

                    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

                    width_a = np.linalg.norm(br - bl)
                    width_b = np.linalg.norm(tr - tl)
                    max_w = int(max(width_a, width_b))

                    height_a = np.linalg.norm(tr - br)
                    height_b = np.linalg.norm(tl - bl)
                    max_h = int(max(height_a, height_b))

                    # Aspect ratio check: typical documents are ~1.2 to 1.6
                    aspect = max_h / max(max_w, 1)
                    if 0.8 < aspect < 2.0 and max_w > 400 and max_h > 400:
                        dst = np.array(
                            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                            dtype=np.float32,
                        )
                        m = cv2.getPerspectiveTransform(ordered, dst)
                        warped = cv2.warpPerspective(
                            cv_bgr, m, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                        )
                        logger.info("Applied 4-point perspective rectification (document contour detected).")
                        return warped
    except Exception as exc:
        logger.debug("Perspective correction skipped: %s", exc)
    return cv_bgr


def remove_shadows_and_enhance(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Remove uneven illumination/shadows and enhance contrast for document text readability.
    Uses morphological background division and CLAHE on the luminance channel.
    """
    try:
        lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Estimate background illumination with large structural dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated = cv2.dilate(l, kernel)
        bg = cv2.medianBlur(dilated, 25)

        # Normalize luminance channel by background
        diff = 255 - cv2.absdiff(l, bg)
        norm_l = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply CLAHE to sharpen text contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(norm_l)

        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception as exc:
        logger.debug("Shadow removal skipped: %s", exc)
        return cv_bgr


def crop_to_content(cv_bgr: np.ndarray) -> np.ndarray:
    """Crop excessive solid border margins (e.g. scanner borders) if safe."""
    try:
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, rw, rh = cv2.boundingRect(coords)
            # Only crop if margins are significant and content area is large
            if rw > 0.6 * w and rh > 0.6 * h:
                # Add safe 15px padding
                pad = 15
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(w, x + rw + pad)
                y1 = min(h, y + rh + pad)
                return cv_bgr[y0:y1, x0:x1]
    except Exception:
        pass
    return cv_bgr


def preprocess_page_image(pil_img: Image.Image) -> Image.Image:
    """
    Execute full Stage A preprocessing pipeline:
    Auto-rotate -> Perspective -> Deskew -> Shadow Removal -> Crop.
    """
    # 1. EXIF Auto-rotate
    rotated_pil = auto_rotate_image(pil_img)

    # 2. Convert to OpenCV BGR
    cv_img = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)

    # 3. Perspective correction (for photos)
    rectified = correct_perspective_if_paper_detected(cv_img)

    # 4. Deskew
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    deskewed = detect_and_deskew(gray, rectified)

    # 5. Shadow removal and contrast enhancement
    enhanced = remove_shadows_and_enhance(deskewed)

    # 6. Crop excessive empty margins
    cropped = crop_to_content(enhanced)

    # Convert back to PIL RGB
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(rgb)

    # Cap maximum dimension to control payload cost
    if max(result_pil.size) > MAX_IMAGE_DIMENSION:
        result_pil.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

    return result_pil


# ---------------------------------------------------------------------------
# Image Quality Assessment
# ---------------------------------------------------------------------------

def assess_image_quality(pil_img: Image.Image, page_num: int = 1) -> PageQualityInfo:
    """
    Evaluate document image clarity, blur, brightness, glare, and edge cutoffs.
    Generates actionable warnings and an aggregate quality score (0.0 - 1.0).
    """
    warnings: List[str] = []
    np_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # 1. Blur score via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < 110.0
    if blur_score < 70.0:
        warnings.append(f"Page {page_num} is noticeably blurry (score: {blur_score:.1f}). Text may be difficult to read.")
    elif blur_score < 110.0:
        warnings.append(f"Page {page_num} has mild motion blur or soft focus.")

    # 2. Brightness & Glare
    brightness = float(np.mean(gray))
    if brightness < 60.0:
        warnings.append(f"Page {page_num} is very dark/underexposed (brightness: {brightness:.1f}).")
    elif brightness > 248.0 and np.std(gray) < 15.0:
        warnings.append(f"Page {page_num} appears washed out or overexposed.")

    # Glare ratio: check if substantial non-background areas are saturated
    overexposed = np.sum(gray > 252) / float(gray.size)
    has_glare = overexposed > 0.88 and np.std(gray) < 22.0
    if has_glare:
        warnings.append(f"Page {page_num} has excessive flash glare or saturated highlights.")

    # 3. Resolution & Dimensions
    if min(w, h) < 600:
        warnings.append(f"Page {page_num} resolution ({w}x{h}) is low. Small numbers and units may degrade.")

    # 4. Check for cut-off edges (text bleeding onto extreme border rows/columns)
    border_pixels = 8
    top_edge = gray[:border_pixels, :]
    bot_edge = gray[-border_pixels:, :]
    left_edge = gray[:, :border_pixels]
    right_edge = gray[:, -border_pixels:]

    edges_cropped = False
    for name, edge in [("top", top_edge), ("bottom", bot_edge), ("left", left_edge), ("right", right_edge)]:
        dark_ratio = np.sum(edge < 160) / float(edge.size)
        if dark_ratio > 0.22:
            edges_cropped = True
            warnings.append(f"{name.capitalize()} edge of page {page_num} may have cut-off content or text near the border.")
            break

    # Compute composite quality score (0.0 to 1.0)
    score = 1.0
    if is_blurry:
        score -= 0.35 if blur_score < 70.0 else 0.18
    if brightness < 60.0 or brightness > 250.0:
        score -= 0.20
    if has_glare:
        score -= 0.25
    if edges_cropped:
        score -= 0.15
    if min(w, h) < 600:
        score -= 0.15

    final_score = max(0.05, min(1.0, score))

    return PageQualityInfo(
        page_number=page_num,
        quality_score=round(final_score, 2),
        blur_score=round(blur_score, 1),
        brightness=round(brightness, 1),
        glare_ratio=round(overexposed, 3),
        is_blurry=is_blurry,
        has_glare=has_glare,
        edges_cropped=edges_cropped,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# High-Level Page Loader: PDF, JPG, PNG, HEIC & Multi-Image
# ---------------------------------------------------------------------------

def load_pdf_to_images(pdf_path_or_bytes: Union[Path, str, bytes]) -> List[Tuple[Image.Image, str]]:
    """
    Render PDF pages to PIL Images at high resolution (~200-300 DPI)
    and extract the embedded text layer as a secondary hint.
    """
    if isinstance(pdf_path_or_bytes, (str, Path)):
        doc = pdfium.PdfDocument(str(pdf_path_or_bytes))
    else:
        doc = pdfium.PdfDocument(pdf_path_or_bytes)

    pages: List[Tuple[Image.Image, str]] = []
    for idx, page in enumerate(doc):
        # Render at target scale
        bitmap = page.render(scale=TARGET_PDF_RENDER_SCALE)
        pil_img = bitmap.to_pil().convert("RGB")

        # Cap max dimensions
        if max(pil_img.size) > MAX_IMAGE_DIMENSION:
            pil_img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

        # Extract text layer as hint
        try:
            textpage = page.get_textpage()
            text_hint = textpage.get_text_range() or ""
        except Exception:
            text_hint = ""

        pages.append((pil_img, text_hint.strip()))

    return pages


def load_image_file_to_pil(file_path_or_bytes: Union[Path, str, bytes], filename: str = "") -> Image.Image:
    """Load image from path or bytes, supporting JPG, PNG, HEIC, TIFF, WEBP."""
    if isinstance(file_path_or_bytes, (str, Path)):
        img = Image.open(str(file_path_or_bytes))
    else:
        img = Image.open(io.BytesIO(file_path_or_bytes))

    img = auto_rotate_image(img).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIMENSION:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
    return img


def ingest_report_pages(
    inputs: List[Tuple[Union[Path, str, bytes], str]],
) -> List[PageImageBundle]:
    """
    Process any combination of PDF documents or individual page photographs (JPG/PNG/HEIC).
    Returns an ordered list of PageImageBundle containing both original and cleaned images,
    along with quality assessment metrics and warnings.
    """
    raw_pages: List[Tuple[Image.Image, str]] = []

    for item, filename in inputs:
        clean_ext = Path(filename).suffix.lower()
        if clean_ext == ".pdf":
            pdf_pages = load_pdf_to_images(item)
            raw_pages.extend(pdf_pages)
        else:
            # Standalone image (JPG, PNG, HEIC, etc.)
            pil_img = load_image_file_to_pil(item, filename)
            raw_pages.append((pil_img, ""))

    bundles: List[PageImageBundle] = []
    for page_idx, (orig_img, text_hint) in enumerate(raw_pages, start=1):
        quality = assess_image_quality(orig_img, page_idx)
        cleaned_img = preprocess_page_image(orig_img)

        bundle = PageImageBundle(
            page_number=page_idx,
            original_image=orig_img,
            cleaned_image=cleaned_img,
            quality=quality,
            text_layer=text_hint,
        )
        bundles.append(bundle)

    logger.info(
        "Ingested %d page(s). Total quality warnings: %d",
        len(bundles),
        sum(len(b.quality.warnings) for b in bundles),
    )
    return bundles
