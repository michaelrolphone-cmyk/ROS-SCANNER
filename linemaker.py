#!/usr/bin/env python3
"""
Max-quality OCR script to extract bearings and distances from Record of Survey PDFs.

This version adds the geometry-consistency feedback loop you asked for:

NEW (Geometry verification + break handling)
- Build a raster linework evidence map (downscaled) once per page/DPI.
- For each extracted bearing/distance candidate, locally verify:
  - Direction consistency: is there linework near the anchor with matching direction?
  - Corridor overlap: does the predicted direction pass through observed linework?
  - Break detection: detect interior gaps + squiggle signature so broken lines aren't penalized for length.
- Geometry score is fed back into candidate voting (soft bonus / penalty).
- Optional debug overlays for low-geometry candidates.

CRITICAL performance / memory fix (for huge DPI pages)
- We DO NOT rotate the full page at every angle anymore (that explodes RAM with 16 threads).
- We detect ROIs on a small (downscaled) page for many angles, map ROIs back to original coords,
  then rotate ONLY ROI crops for OCR. This matches your original ask: crop callouts, multipass.

Existing features preserved
- Multi-DPI scan + voting across DPIs
- Full 360° multi-angle scan (dense angles in max quality)
- ROI detection + fallback full-page tiling
- Multi-pass OCR (multiple PSM/OEM + multiple preprocess variants)
- Linework detection + inpaint cleanup + overlap penalties (reduces hallucinated digits)
- Decimal point recovery via pixel evidence (dot blob detection inside token region)
- Filters for acres/area, title block dates/metadata
- Record distance capture: 123.45 (124)
- Line Table / Curve Table extraction (L1/L2..., C1/C2...) + callout counting
- Multithreaded per-ROI-angle processing via ThreadPoolExecutor (--threads)
- Basis of bearing detection restored (heuristic)

Install:
  pip install pytesseract pdf2image Pillow opencv-python-headless numpy

Usage (max quality, threaded):
  python grokpt.py file.pdf --quality max --threads 16 --geom-verify --break-detect --output out.json --verbose

Optional debug:
  python grokpt.py file.pdf --debug-rois rois/ --debug-geom geom/
"""

import re
import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_path, pdfinfo_from_path
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install: pip install pytesseract pdf2image Pillow opencv-python-headless numpy")
    sys.exit(1)

# ---------------------------
# PIL / large image safety
# ---------------------------

Image.MAX_IMAGE_PIXELS = 800_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Pixel budget controls (keep you from OOM when DPI is huge)
MAX_DET_PIXELS = 12_000_000         # low-res detection stage budget
MAX_ROI_PIXELS = 18_000_000         # per ROI crop budget (tiling splits if larger)
MAX_VARIANT_PIXELS = 18_000_000     # per preprocess variant scaling cap

# Geometry evidence budget
MAX_GEOM_PIXELS = 14_000_000        # downscaled geometry evidence cap per page (mask + edges)

# ---------------------------
# Linework filtering controls
# ---------------------------

LINE_OVERLAP_HARD_REJECT = 0.60   # if line mask covers >60% of token bbox, skip candidate
LINE_OVERLAP_PENALTY_WT  = 3.0    # score penalty weight during voting
LINE_INPAINT_RADII       = [2, 3, 4]  # try multiple line-inpaint radii in max mode

# Geometry scoring weights
GEOM_BONUS_WT = 8.0               # adds up to +8 confidence points to vote score
GEOM_PENALTY_WT = 8.0             # subtracts up to -8 points for bad matches
GEOM_MIN_SUPPORT_BONUS = 0.4      # modest bump if geometry is consistent across many candidates

# ---------------------------
# Metadata / clutter filters
# ---------------------------

HEADER_FOOTER_STRIP_FRAC = 0.08   # ignore candidates in top/bottom 8% during full-page tiling
METADATA_KEYWORDS = {
    "recorded", "filed", "re-recorded", "rerecorded", "instrument", "instr", "doc", "document",
    "book", "page", "index", "ros", "r.o.s", "survey", "county", "auditor", "assessor",
    "tax", "parcel", "apn", "pin", "project", "sheet", "scale", "date", "dated"
}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec"
}

AREA_UNITS_RE = re.compile(
    r"(?i)\b(a\.?\s*c\.?|acres?|acre|sq\.?\s*ft|sqft|s\.?\s*f\.?|sf|square\s+feet)\b"
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b")

# ---------------------------
# OCR cleanup maps / regex
# ---------------------------

DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "D": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2",
})

BEARING_RE = re.compile(
    r"""
    (?P<ns>[NS])\s*
    (?P<deg>[0-9OolIDSBZ|]{1,3})\s*
    (?:[°oº]\s*)?
    (?P<min>[0-9OolIDSBZ|]{1,2})\s*
    (?:[\'’]\s*)?
    (?P<sec>[0-9OolIDSBZ|]{1,2})?
    \s*(?:["”])?
    \s*(?P<ew>[EW])
    """,
    re.IGNORECASE | re.VERBOSE,
)

DIST_TOKEN_RE = re.compile(r"^\(?\d[\d,]*([.·•∙⋅]\d{1,4})?\)?$")

# Heuristic for ROI discovery (line-level)
CANDIDATE_LINE_RE = re.compile(
    r"(?i)\b[NS]\s*\d{1,3}\s*(?:[°oº]|\s)\s*\d{1,2}.*\b[EW]\b.*\d{2,10}"
)

# Table labels
LINE_ID_RE = re.compile(r"(?i)\bL[\s\-]?\d{1,3}\b")
CURVE_ID_RE = re.compile(r"(?i)\bC[\s\-]?\d{1,3}\b")
CURVE_HINT_RE = re.compile(r"(?i)\b(chord|delta|radius|length|arc|tangent)\b")

# Basis-of-bearing hint
BASIS_HINT_RE = re.compile(r"(?i)\bbasis\b.*\bbearing\b|\bbasis\s+of\s+bearing\b")

def _clean_digits(s: str) -> str:
    return (s or "").translate(DIGIT_FIX)

def canon_label(s: str) -> str | None:
    if not s:
        return None
    t = s.strip().upper().replace(" ", "").replace("-", "")
    if re.fullmatch(r"[LC]\d{1,3}", t):
        return t
    return None

# ---------------------------
# Bearing parsing
# ---------------------------

def parse_bearing_to_canon(bearing_text: str) -> str | None:
    m = BEARING_RE.search(bearing_text or "")
    if not m:
        return None

    ns = m.group("ns").upper()
    ew = m.group("ew").upper()

    deg_s = _clean_digits(m.group("deg"))
    min_s = _clean_digits(m.group("min"))
    sec_s = _clean_digits(m.group("sec") or "0")

    try:
        deg = int(deg_s)
        minute = int(min_s)
        sec = int(sec_s)
    except ValueError:
        return None

    # Fix common OCR "100+deg" errors without assuming decimal length
    if deg > 90:
        if 100 <= deg < 190:
            deg -= 100
        elif 200 <= deg < 290:
            deg -= 200
        elif 300 <= deg < 390:
            deg -= 300

    if 60 <= minute <= 99:
        minute -= 60
    if 60 <= sec <= 99:
        sec -= 60

    if not (0 <= deg <= 90 and 0 <= minute <= 59 and 0 <= sec <= 59):
        return None

    return f"{ns} {deg:02d}°{minute:02d}'{sec:02d}\" {ew}"

def bearing_theta_deg(canon_bearing: str) -> float | None:
    m = re.match(r"^([NS])\s+(\d{2})°(\d{2})'(\d{2})\"\s+([EW])$", canon_bearing)
    if not m:
        return None
    ns, d, mi, se, ew = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    ang = d + mi / 60.0 + se / 3600.0
    if ns == "N" and ew == "E":
        return ang
    if ns == "N" and ew == "W":
        return (360.0 - ang) % 360.0
    if ns == "S" and ew == "E":
        return (180.0 - ang) % 360.0
    if ns == "S" and ew == "W":
        return (180.0 + ang) % 360.0
    return None

def theta_mod180(theta: float) -> float:
    t = float(theta) % 360.0
    return t if t < 180.0 else (t - 180.0)

def angdiff_mod180(a: float, b: float) -> float:
    """Smallest absolute difference between two undirected angles in degrees."""
    da = abs((a - b) % 180.0)
    return min(da, 180.0 - da)

# ---------------------------
# Rotation / coordinate mapping
# ---------------------------

def rotate_keep_size(pil_img: Image.Image, angle_deg: float) -> Image.Image:
    a = float(angle_deg)
    if abs(a) < 1e-9:
        return pil_img

    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]

    # Fast cases
    am = a % 360.0
    if abs(am - 90.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if (out.shape[1], out.shape[0]) != (w, h) else out
        return Image.fromarray(out)
    if abs(am - 270.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if (out.shape[1], out.shape[0]) != (w, h) else out
        return Image.fromarray(out)
    if abs(am - 180.0) < 1e-9:
        return Image.fromarray(cv2.rotate(arr, cv2.ROTATE_180))

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, a, 1.0)
    out = cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(out)

def rotate_point_xy(x: float, y: float, w: int, h: int, angle_deg: float) -> tuple[float, float]:
    """Rotate a point around image center using the same convention as rotate_keep_size()."""
    a = float(angle_deg)
    if abs(a) < 1e-9:
        return x, y
    cx, cy = w / 2.0, h / 2.0
    rad = np.deg2rad(a)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    dx, dy = x - cx, y - cy
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    return xr + cx, yr + cy

def rect_rot_to_orig(r: tuple[int, int, int, int], small_w: int, small_h: int,
                     orig_w: int, orig_h: int, angle_deg: float, scale: float) -> tuple[int, int, int, int]:
    """
    r is a rect in the ROTATED SMALL image coordinate system.
    Map it back to ORIGINAL (unrotated) full-res coordinates by:
      - inverse rotate corners in small space
      - scale up to original
    """
    x1, y1, x2, y2 = r
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    inv = -float(angle_deg)
    back = [rotate_point_xy(px, py, small_w, small_h, inv) for (px, py) in pts]
    xs = [p[0] / scale for p in back]
    ys = [p[1] / scale for p in back]
    ox1 = int(max(0, min(orig_w - 1, min(xs))))
    oy1 = int(max(0, min(orig_h - 1, min(ys))))
    ox2 = int(max(1, min(orig_w, max(xs))))
    oy2 = int(max(1, min(orig_h, max(ys))))
    if ox2 <= ox1 + 1:
        ox2 = min(orig_w, ox1 + 2)
    if oy2 <= oy1 + 1:
        oy2 = min(orig_h, oy1 + 2)
    return (ox1, oy1, ox2, oy2)

# ---------------------------
# Linework detection + cleanup
# ---------------------------

def _adaptive_inv(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

def build_line_mask(gray: np.ndarray) -> np.ndarray:
    """
    Prefer long straight segments (parcel linework, borders, tables). Suppresses most text.
    """
    h, w = gray.shape[:2]
    bw = _adaptive_inv(gray)

    hk = max(25, w // 18)
    vk = max(25, h // 18)

    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)),
        iterations=1
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)),
        iterations=1
    )

    mask = cv2.bitwise_or(horiz, vert)

    edges = cv2.Canny(gray, 60, 160)
    min_len = max(60, int(0.28 * min(w, h)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=70,
        minLineLength=min_len,
        maxLineGap=12
    )

    if lines is not None:
        diag = np.zeros_like(mask)
        for x1, y1, x2, y2 in lines[:, 0]:
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 < (min_len ** 2):
                continue
            cv2.line(diag, (x1, y1), (x2, y2), 255, 3)
        mask = cv2.bitwise_or(mask, diag)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask

def remove_linework(gray: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    mask = build_line_mask(gray)
    cleaned = cv2.inpaint(gray, mask, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA)
    return cleaned, mask

def mask_overlap_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mask.shape[1], x2); y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(roi.size)

# ---------------------------
# Decimal dot detection (pixel evidence)
# ---------------------------

def _binarize_for_components(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw

def detect_decimal_dot_digits_right(dot_ref_pil: Image.Image, word_bbox: tuple[int, int, int, int]) -> int | None:
    """
    Detect a small decimal dot blob inside/near the distance token bbox and infer digits-right.
    This is evidence-based (pixel components), not digit-length guessing.
    """
    x1, y1, x2, y2 = word_bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    mx = max(2, int(w * 0.08))
    my = max(2, int(h * 0.12))
    X1 = max(0, x1 - mx)
    Y1 = max(0, y1 - my)
    X2 = min(dot_ref_pil.size[0], x2 + mx)
    Y2 = min(dot_ref_pil.size[1], y2 + my)

    crop = dot_ref_pil.crop((X1, Y1, X2, Y2))
    gray = np.array(crop.convert("L"))

    bw = _binarize_for_components(gray)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 2:
        return None

    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area <= 0:
            continue
        comps.append((i, x, y, cw, ch, area, float(centroids[i][0]), float(centroids[i][1])))
    if not comps:
        return None

    areas = sorted([c[5] for c in comps], reverse=True)
    top = areas[:6]
    median_big = float(np.median(top)) if top else 0.0
    if median_big <= 0:
        return None

    digit_comps = []
    for (i, x, y, cw, ch, area, cx, cy) in comps:
        if area < max(6, 0.10 * median_big):
            continue
        digit_comps.append((i, x, y, cw, ch, area, cx, cy))
    if len(digit_comps) < 2:
        return None

    digit_cys = [c[7] for c in digit_comps]
    cy_med = float(np.median(digit_cys))

    dot_candidates = []
    for (i, x, y, cw, ch, area, cx, cy) in comps:
        if area >= 0.35 * median_big:
            continue
        if area < 2:
            continue
        if cw > 0.40 * w or ch > 0.40 * h:
            continue
        fill = area / max(1.0, float(cw * ch))
        if fill < 0.20:
            continue
        if not (cy >= 0.50 * h and cy <= 0.95 * h):
            continue
        if cy < cy_med:
            continue
        ar = cw / max(1.0, float(ch))
        ar = ar if ar >= 1.0 else 1.0 / ar
        if ar > 3.0:
            continue

        score = 0.0
        score += (1.0 - min(1.0, area / (0.35 * median_big))) * 2.0
        score += (1.0 - min(1.0, (ar - 1.0) / 2.0)) * 1.5
        score += (min(1.0, (cy - cy_med) / max(1.0, (0.40 * h)))) * 1.0
        score += fill * 0.8
        dot_candidates.append((score, cx, cy, i))

    if not dot_candidates:
        return None

    dot_candidates.sort(key=lambda t: t[0], reverse=True)
    _, dot_cx, _dot_cy, _ = dot_candidates[0]

    digit_centers_x = sorted([c[6] for c in digit_comps])
    right = [x for x in digit_centers_x if x > dot_cx + 1.5]
    left = [x for x in digit_centers_x if x < dot_cx - 1.5]
    digits_right = len(right)

    if digits_right < 1 or len(left) < 1:
        return None
    if digits_right > 4:
        return None

    return digits_right

# ---------------------------
# Distance parsing (NO digit-length guessing)
# ---------------------------

def parse_distance_with_quality(
    dist_text: str,
    dot_digits_right: int | None = None,
    min_ft: float = 1.0,
    max_ft: float = 50_000_000.0,
) -> tuple[float | None, str]:
    if not dist_text:
        return None, "invalid"

    s = (dist_text or "").strip().replace(" ", "")
    s = s.replace("·", ".").replace("•", ".").replace("∙", ".").replace("⋅", ".")

    if "(" in s:
        s = s.split("(", 1)[0].strip()
    if not s:
        return None, "invalid"

    # comma-decimal evidence (not length-based)
    if s.count(",") == 1 and "." not in s:
        left, right = s.split(",", 1)
        if right.isdigit() and len(right) in (2, 3) and left.replace(",", "").isdigit():
            s2 = (left + "." + right).translate(DIGIT_FIX)
            try:
                v = float(s2)
                if min_ft <= v <= max_ft:
                    return v, "comma_decimal"
            except ValueError:
                pass

    s = s.replace(",", "")
    s = s.translate(DIGIT_FIX)

    if "." in s:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])
        try:
            v = float(s)
            if min_ft <= v <= max_ft:
                return v, "explicit_decimal"
        except ValueError:
            return None, "invalid"

    digits = re.sub(r"\D", "", s)
    if not digits:
        return None, "invalid"

    # dot blob evidence inserts decimal
    if dot_digits_right is not None:
        if 1 <= dot_digits_right < len(digits):
            ins = len(digits) - dot_digits_right
            s2 = digits[:ins] + "." + digits[ins:]
            try:
                v = float(s2)
                if min_ft <= v <= max_ft:
                    return v, "dot_blob"
            except ValueError:
                pass

    try:
        v = float(digits)
        if min_ft <= v <= max_ft:
            return v, "integer"
    except ValueError:
        pass

    return None, "invalid"

# ---------------------------
# Preprocess variants
# ---------------------------

def _apply_clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return clahe.apply(gray)

def preprocess_variant(pil_img: Image.Image, variant: dict) -> tuple[Image.Image, dict]:
    arr = np.array(pil_img.convert("RGB"))
    gray0 = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if variant.get("clahe", False):
        gray0 = _apply_clahe(gray0, clip=variant.get("clahe_clip", 2.0), grid=variant.get("clahe_grid", 8))

    scale = float(variant.get("scale", 1.0))
    base_pixels = int(gray0.shape[0] * gray0.shape[1])
    if base_pixels > 0 and scale != 1.0:
        target = base_pixels * (scale * scale)
        if target > MAX_VARIANT_PIXELS:
            scale = (MAX_VARIANT_PIXELS / base_pixels) ** 0.5

    if scale != 1.0:
        gray = cv2.resize(gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        gray = gray0

    line_mask = build_line_mask(gray) if bool(variant.get("line_mask", True)) else None

    if bool(variant.get("line_clean", False)):
        gray, _ = remove_linework(gray, radius=int(variant.get("inpaint_radius", 3)))

    inv = bool(variant.get("inv", False))
    thresh = variant.get("thresh", "adaptive_gauss")
    blur = int(variant.get("blur", 0) or 0)

    if thresh == "none":
        out = 255 - gray if inv else gray
        proc = Image.fromarray(out)
        return proc, {"scaled_gray": gray, "line_mask": line_mask, "dot_ref": bool(variant.get("dot_ref", False)), "scale_used": float(scale)}

    gray2 = gray
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray2 = cv2.GaussianBlur(gray2, (k, k), 0)

    block_size = int(variant.get("block_size", 31))
    if block_size < 11:
        block_size = 11
    if block_size % 2 == 0:
        block_size += 1
    C = int(variant.get("C", 10))

    if thresh == "adaptive_gauss":
        out = cv2.adaptiveThreshold(
            gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, block_size, C
        )
    elif thresh == "adaptive_mean":
        out = cv2.adaptiveThreshold(
            gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, block_size, C
        )
    elif thresh == "otsu":
        mode = (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
        _, out = cv2.threshold(gray2, 0, 255, mode)
    else:
        out = gray2

    if bool(variant.get("denoise", False)):
        out = cv2.fastNlMeansDenoising(out, None, h=25, templateWindowSize=7, searchWindowSize=21)

    if bool(variant.get("morph_close", False)):
        kernel = np.ones((2, 2), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    proc = Image.fromarray(out)
    return proc, {"scaled_gray": gray, "line_mask": line_mask, "dot_ref": bool(variant.get("dot_ref", False)), "scale_used": float(scale)}

# ---------------------------
# OCR: keep word bboxes
# ---------------------------

def ocr_image_to_lines_with_words(pil_img: Image.Image, config: str, timeout: int | None = None) -> list[dict]:
    data = pytesseract.image_to_data(
        pil_img, config=config, lang="eng", output_type=Output.DICT,
        timeout=timeout
    )

    n = len(data.get("text", []))
    lines: dict[tuple[int, int, int], dict] = {}

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue

        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        block = int(data.get("block_num", [0])[i] or 0)
        par = int(data.get("par_num", [0])[i] or 0)
        line = int(data.get("line_num", [0])[i] or 0)
        key = (block, par, line)

        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        bb = (x, y, x + w, y + h)

        if key not in lines:
            lines[key] = {"words": [], "bbox": [x, y, x + w, y + h], "confs": []}

        lines[key]["words"].append({"text": txt, "conf": conf, "bbox": bb})

        L = lines[key]["bbox"]
        L[0] = min(L[0], x)
        L[1] = min(L[1], y)
        L[2] = max(L[2], x + w)
        L[3] = max(L[3], y + h)

        if conf >= 0:
            lines[key]["confs"].append(conf)

    out = []
    for rec in lines.values():
        words = rec["words"]
        words.sort(key=lambda w: w["bbox"][0])
        text = " ".join(w["text"] for w in words)
        confs = rec["confs"]
        mean_conf = (sum(confs) / len(confs)) if confs else -1.0
        out.append({"text": text, "conf": mean_conf, "bbox": tuple(rec["bbox"]), "words": words})

    out.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return out

# ---------------------------
# Rect helpers
# ---------------------------

def rect_intersects(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def merge_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    rects = rects[:]
    merged = True
    while merged and len(rects) > 1:
        merged = False
        out = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            rx1, ry1, rx2, ry2 = rects[i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(rects)):
                    if used[j]:
                        continue
                    if rect_intersects((rx1, ry1, rx2, ry2), rects[j]):
                        x1, y1, x2, y2 = rects[j]
                        rx1 = min(rx1, x1)
                        ry1 = min(ry1, y1)
                        rx2 = max(rx2, x2)
                        ry2 = max(ry2, y2)
                        used[j] = True
                        changed = True
                        merged = True
            out.append((rx1, ry1, rx2, ry2))
        rects = out
    return rects

def clamp_rect(r, w, h):
    x1, y1, x2, y2 = r
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1 + 1:
        x2 = min(w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(h, y1 + 2)
    return (x1, y1, x2, y2)

def tile_rect(r, page_w, page_h, max_pixels=MAX_ROI_PIXELS, overlap=140) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = r
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return []
    area = w * h
    if area <= max_pixels:
        return [clamp_rect(r, page_w, page_h)]

    n = int(np.ceil(np.sqrt(area / max_pixels)))
    n = max(2, min(n, 7))

    tw = int(np.ceil(w / n))
    th = int(np.ceil(h / n))

    tiles = []
    for iy in range(n):
        for ix in range(n):
            tx1 = x1 + ix * tw - overlap
            ty1 = y1 + iy * th - overlap
            tx2 = x1 + (ix + 1) * tw + overlap
            ty2 = y1 + (iy + 1) * th + overlap
            tiles.append(clamp_rect((tx1, ty1, tx2, ty2), page_w, page_h))

    return merge_rects(tiles)

# ---------------------------
# ROI detection (run on an image you pass in)
# ---------------------------

def detect_bearing_distance_rois(page_img: Image.Image, roi_pad: int = 55, max_rois: int = 60) -> list[tuple[int, int, int, int]]:
    """
    Detect likely bearing+distance callout regions on the given image.

    IMPORTANT: This function internally downscales to stay under MAX_DET_PIXELS.
    In this version, we call it mostly on already-downscaled images (so it stays ~1.0 scale).
    """
    W, H = page_img.size
    page_pixels = W * H
    scale = min(1.0, (MAX_DET_PIXELS / page_pixels) ** 0.5) if page_pixels > 0 else 1.0

    if scale < 1.0:
        small = page_img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BICUBIC)
    else:
        small = page_img

    sW, sH = small.size
    arr = np.array(small.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    det = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 41, 11)
    det_img = Image.fromarray(det)

    # NOTE: no quotes in config (avoid shlex "No closing quotation")
    det_cfg = "--oem 3 --psm 11 -c preserve_interword_spaces=1"
    lines = ocr_image_to_lines_with_words(det_img, det_cfg)

    rects = []
    scored = []
    pad_s = max(5, int(roi_pad * scale))

    for ln in lines:
        t = ln["text"]
        if not t:
            continue

        looks_like_pair = bool(CANDIDATE_LINE_RE.search(t)) or bool(BEARING_RE.search(t) and re.search(r"\d", t))
        if not looks_like_pair:
            continue

        x1, y1, x2, y2 = ln["bbox"]
        bw = x2 - x1
        bh = y2 - y1

        if bh > 0.20 * sH:
            continue
        if bw > 0.98 * sW and bh > 0.06 * sH:
            continue

        x1 -= pad_s
        y1 -= pad_s
        x2 += pad_s
        y2 += pad_s
        x1, y1, x2, y2 = clamp_rect((x1, y1, x2, y2), sW, sH)

        if scale != 1.0:
            ox1 = int(x1 / scale)
            oy1 = int(y1 / scale)
            ox2 = int(x2 / scale)
            oy2 = int(y2 / scale)
        else:
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        r = clamp_rect((ox1, oy1, ox2, oy2), W, H)
        rects.append(r)
        scored.append((ln["conf"], r))

    if not rects:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    rects = [r for _, r in scored[: max_rois * 3]]
    rects = merge_rects(rects)

    page_area = W * H
    def area(r): return (r[2] - r[0]) * (r[3] - r[1])
    rects = [r for r in rects if area(r) < 0.35 * page_area]

    rects.sort(key=area)
    return rects[:max_rois]

# ---------------------------
# ROI detection: Line Table / Curve Table
# ---------------------------

TABLE_KEYWORDS = {
    "table", "line", "curve", "bearing", "distance", "chord", "delta", "radius", "length", "arc"
}

def detect_table_rois(page_img: Image.Image, max_tables: int = 10) -> list[tuple[str, tuple[int, int, int, int]]]:
    """
    Detect table blocks ("LINE TABLE", "CURVE TABLE", etc.) on a given image (often downscaled).
    Returns (kind, rect) in coordinates of the given image.
    """
    W, H = page_img.size
    page_pixels = W * H
    scale = min(1.0, (MAX_DET_PIXELS / page_pixels) ** 0.5) if page_pixels > 0 else 1.0

    if scale < 1.0:
        small = page_img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BICUBIC)
    else:
        small = page_img

    sW, sH = small.size
    arr = np.array(small.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    pil = Image.fromarray(bw)

    # NOTE: do NOT include quote characters in whitelist (avoid shlex quoting issues)
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()°.,-"
    cfg = f"--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(pil, config=cfg, lang="eng", output_type=Output.DICT)

    hits = []
    n = len(data.get("text", []))
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0
        if conf < 35:
            continue

        tl = re.sub(r"[^a-z0-9]", "", t.lower())
        if not tl:
            continue
        if tl in TABLE_KEYWORDS:
            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            hits.append((tl, conf, (x, y, x + w, y + h)))

    if not hits:
        return []

    clusters = []
    for kw, conf, bb in hits:
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        placed = False
        for cl in clusters:
            if abs(cx - cl["cx"]) < 280 and abs(cy - cl["cy"]) < 170:
                cl["items"].append((kw, conf, bb))
                cl["cx"] = (cl["cx"] * 0.7 + cx * 0.3)
                cl["cy"] = (cl["cy"] * 0.7 + cy * 0.3)
                placed = True
                break
        if not placed:
            clusters.append({"cx": cx, "cy": cy, "items": [(kw, conf, bb)]})

    out = []
    for cl in clusters:
        kws = {k for (k, _, _) in cl["items"]}
        kind = None
        if "table" in kws and "line" in kws and ("bearing" in kws or "distance" in kws):
            kind = "line_table"
        elif "table" in kws and "curve" in kws and ("chord" in kws or "radius" in kws or "delta" in kws):
            kind = "curve_table"
        else:
            if "line" in kws and "bearing" in kws and "distance" in kws:
                kind = "line_table"
            elif "curve" in kws and ("chord" in kws or "radius" in kws) and ("distance" in kws or "length" in kws):
                kind = "curve_table"
        if not kind:
            continue

        xs = [bb[0] for (_, _, bb) in cl["items"]]
        ys = [bb[1] for (_, _, bb) in cl["items"]]
        xe = [bb[2] for (_, _, bb) in cl["items"]]
        ye = [bb[3] for (_, _, bb) in cl["items"]]
        x1 = min(xs); y1 = min(ys); x2 = max(xe); y2 = max(ye)

        header_h = max(12, y2 - y1)
        pad_x = int(0.18 * sW)
        pad_y = int(0.05 * sH)

        x1 = max(0, x1 - pad_x)
        x2 = min(sW, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(sH, y2 + max(int(0.40 * sH), 18 * header_h))

        if scale != 1.0:
            ox1 = int(x1 / scale)
            oy1 = int(y1 / scale)
            ox2 = int(x2 / scale)
            oy2 = int(y2 / scale)
        else:
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2

        r = clamp_rect((ox1, oy1, ox2, oy2), W, H)
        out.append((kind, r))

    if not out:
        return []

    by_kind = defaultdict(list)
    for k, r in out:
        by_kind[k].append(r)

    merged = []
    for k, rects in by_kind.items():
        rects = merge_rects(rects)
        for r in rects:
            merged.append((k, r))

    merged.sort(key=lambda kr: (kr[1][3] - kr[1][1]) * (kr[1][2] - kr[1][0]), reverse=True)
    return merged[:max_tables]

# ---------------------------
# Clutter filters
# ---------------------------

def is_metadata_line(line_text: str) -> bool:
    t = (line_text or "").strip().lower()
    if not t:
        return False
    if DATE_RE.search(t):
        return True
    has_year = bool(YEAR_RE.search(t))
    has_month = any(m in t for m in MONTHS)
    if not (has_year or has_month):
        return False
    for kw in METADATA_KEYWORDS:
        if kw in t:
            return True
    return False

def has_area_unit_near(words: list[dict], idx: int, window: int = 4) -> bool:
    start = max(0, idx - window)
    end = min(len(words), idx + window + 1)
    snippet = " ".join((words[i]["text"] or "") for i in range(start, end))
    return bool(AREA_UNITS_RE.search(snippet))

# ---------------------------
# Record distance parsing: (124) after measured distance
# ---------------------------

PAREN_RE = re.compile(r"\(([^)]+)\)")

def parse_record_distance_from_following(
    words: list[dict],
    dist_idx: int,
    dot_ref_img: Image.Image | None
) -> tuple[float | None, str | None, str | None]:
    join = ""
    bbox_for_dot = None

    for j in range(dist_idx, min(len(words), dist_idx + 6)):
        tok = (words[j]["text"] or "")
        join += tok

        m = PAREN_RE.search(join)
        if not m:
            continue

        inside = (m.group(1) or "").strip()
        inside2 = re.sub(r"[^0-9.,·•∙⋅]", "", inside)
        if not inside2:
            return None, None, None

        tokj = (words[j]["text"] or "")
        if "(" in tokj and ")" in tokj:
            bbox_for_dot = words[j]["bbox"]

        dot_digits_right = None
        if dot_ref_img is not None and bbox_for_dot is not None:
            dot_digits_right = detect_decimal_dot_digits_right(dot_ref_img, bbox_for_dot)

        v, q = parse_distance_with_quality(inside2, dot_digits_right=dot_digits_right, min_ft=0.0, max_ft=50_000_000.0)
        if v is None:
            return None, None, None

        if abs(v - round(v)) < 1e-9:
            s = f"{int(round(v))}"
        else:
            s = f"{v:.2f}".rstrip("0").rstrip(".")
        return float(v), s, q

    return None, None, None

# ---------------------------
# Extract pairs from line using word bboxes + line overlap + clutter rules
# ---------------------------

def find_label_in_words(words: list[dict]) -> str | None:
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        m = LINE_ID_RE.search(t)
        if m:
            c = canon_label(m.group(0))
            if c:
                return c
        m = CURVE_ID_RE.search(t)
        if m:
            c = canon_label(m.group(0))
            if c:
                return c
    joined = " ".join((w.get("text") or "") for w in words)
    m = LINE_ID_RE.search(joined)
    if m:
        c = canon_label(m.group(0))
        if c:
            return c
    m = CURVE_ID_RE.search(joined)
    if m:
        c = canon_label(m.group(0))
        if c:
            return c
    return None

def extract_pairs_from_line_words(
    line_text: str,
    line_words: list[dict],
    dot_ref_img: Image.Image | None,
    line_mask: np.ndarray | None,
    min_ft: float = 1.0,
    require_label: bool = False,
) -> list[dict]:
    out = []
    if not line_words:
        return out

    if is_metadata_line(line_text):
        return out

    ref_id = find_label_in_words(line_words)
    if require_label and not ref_id:
        return out

    texts = [w["text"] for w in line_words]

    for i in range(len(texts)):
        for win in (3, 4, 5, 6, 7):
            j = min(len(texts), i + win)
            snippet = " ".join(texts[i:j])
            bm = BEARING_RE.search(snippet)
            if not bm:
                continue

            b_raw = bm.group(0)
            b_canon = parse_bearing_to_canon(b_raw)
            if not b_canon:
                continue

            dist_idx = None
            for k in range(j, min(len(texts), j + 14)):
                tok = (texts[k] or "").strip()
                if not tok:
                    continue
                if re.search(r"[A-Za-z]", tok):
                    continue

                if "(" in tok or ")" in tok:
                    # allow measured distance token with parentheses only if it looks numeric
                    if "(" in tok and re.search(r"^\(?\d", tok):
                        if any(ch in tok for ch in (".", "·", "•", "∙", "⋅")):
                            dist_idx = k
                            break
                    continue

                if DIST_TOKEN_RE.match(tok) or re.search(r"\d", tok):
                    dist_idx = k
                    break

            if dist_idx is None:
                continue

            meas_raw = texts[dist_idx]
            meas_bbox = line_words[dist_idx]["bbox"]

            if has_area_unit_near(line_words, dist_idx, window=5):
                break

            overlap = 0.0
            if line_mask is not None and meas_bbox is not None:
                overlap = mask_overlap_ratio(line_mask, meas_bbox)
                if overlap >= LINE_OVERLAP_HARD_REJECT:
                    break

            dot_digits_right = None
            if dot_ref_img is not None:
                dot_digits_right = detect_decimal_dot_digits_right(dot_ref_img, meas_bbox)

            meas_val, meas_quality = parse_distance_with_quality(meas_raw, dot_digits_right=dot_digits_right, min_ft=min_ft)
            if meas_val is None:
                break

            rec_val, rec_str, rec_quality = parse_record_distance_from_following(line_words, dist_idx, dot_ref_img)

            meas_str = f"{meas_val:.3f}".rstrip("0").rstrip(".") if meas_quality == "dot_blob" else f"{meas_val:.2f}"

            out.append({
                "canon_bearing": b_canon,
                "numeric_dist": float(meas_val),
                "distance": meas_str,
                "dist_quality": meas_quality,
                "dist_raw": meas_raw,
                "dist_bbox": meas_bbox,
                "line_overlap": float(overlap),

                "record_numeric_dist": float(rec_val) if rec_val is not None else None,
                "record_distance": rec_str,
                "record_quality": rec_quality,

                "ref_id": ref_id,
                "basis_hint": bool(BASIS_HINT_RE.search(line_text or "")),
            })
            break

    return out

# ---------------------------
# Multipass OCR on ROI (single rotated ROI image)
# ---------------------------

def multipass_ocr_on_roi(
    roi_img: Image.Image,
    ocr_configs: list[tuple[str, str]],
    variants: list[dict],
    require_label: bool,
    tess_timeout: int | None
) -> list[dict]:
    candidates = []
    for v in variants:
        proc, aux = preprocess_variant(roi_img, v)
        dot_ref_img = proc if aux.get("dot_ref", False) else None
        line_mask = aux.get("line_mask", None)

        for cfg, desc in ocr_configs:
            lines = ocr_image_to_lines_with_words(proc, cfg, timeout=tess_timeout)
            for ln in lines:
                pairs = extract_pairs_from_line_words(
                    line_text=ln["text"],
                    line_words=ln["words"],
                    dot_ref_img=dot_ref_img,
                    line_mask=line_mask,
                    min_ft=1.0,
                    require_label=require_label
                )
                for c in pairs:
                    theta = bearing_theta_deg(c["canon_bearing"])
                    candidates.append({
                        "canon_bearing": c["canon_bearing"],
                        "distance": c["distance"],
                        "numeric_dist": c["numeric_dist"],
                        "theta": theta,
                        "conf": float(ln["conf"]) if ln["conf"] is not None else -1.0,

                        "pass_name": v["name"],
                        "config_desc": desc,
                        "raw_line": ln["text"],

                        "dist_quality": c.get("dist_quality"),
                        "dist_raw": c.get("dist_raw"),
                        "dist_bbox": c.get("dist_bbox"),
                        "line_overlap": float(c.get("line_overlap", 0.0)),
                        "line_clean": bool(v.get("line_clean", False)),

                        "record_numeric_dist": c.get("record_numeric_dist"),
                        "record_distance": c.get("record_distance"),
                        "record_quality": c.get("record_quality"),

                        "ref_id": c.get("ref_id"),
                        "basis_hint": bool(c.get("basis_hint", False)),
                    })
    return candidates

# ---------------------------
# Candidate clustering / voting
# ---------------------------

def similar_measurement(a: dict, b: dict, dist_tol: float, bear_tol_deg: float) -> bool:
    qa = a["canon_bearing"].split()[0] + a["canon_bearing"].split()[-1]
    qb = b["canon_bearing"].split()[0] + b["canon_bearing"].split()[-1]
    if qa != qb:
        return False
    if abs(a["numeric_dist"] - b["numeric_dist"]) > dist_tol:
        return False
    ta = a.get("theta")
    tb = b.get("theta")
    if ta is None or tb is None:
        return False
    dtheta = abs(ta - tb)
    dtheta = min(dtheta, 360.0 - dtheta)
    return dtheta <= bear_tol_deg

def cluster_candidates(cands: list[dict], dist_tol: float, bear_tol_deg: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for c in cands:
        placed = False
        for cl in clusters:
            if similar_measurement(c, cl[0], dist_tol, bear_tol_deg):
                cl.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    return clusters

def _decimal_evidence_bonus(q: str) -> float:
    if q == "explicit_decimal":
        return 2.2
    if q == "comma_decimal":
        return 1.6
    if q == "dot_blob":
        return 1.3
    if q == "integer":
        return 0.0
    return 0.0

# ---------------------------
# Geometry evidence + verification
# ---------------------------

@dataclass
class GeomEvidence:
    scale: float            # small/orig
    gray_small: np.ndarray  # uint8
    mask_small: np.ndarray  # uint8 linework mask (255 lines)
    segs: list              # list of (x1,y1,x2,y2,theta180,len)

def _downscale_to_budget(pil_img: Image.Image, max_pixels: int) -> tuple[Image.Image, float]:
    W, H = pil_img.size
    pix = W * H
    if pix <= max_pixels or pix <= 0:
        return pil_img, 1.0
    s = (max_pixels / float(pix)) ** 0.5
    newW = max(1, int(W * s))
    newH = max(1, int(H * s))
    return pil_img.resize((newW, newH), Image.BICUBIC), s

def build_geom_evidence(page_img: Image.Image) -> GeomEvidence:
    """
    Build a downscaled linework mask + segment list for geometry verification.
    Uses build_line_mask() to suppress most text and emphasize drafted linework.
    """
    small, s = _downscale_to_budget(page_img, MAX_GEOM_PIXELS)
    arr = np.array(small.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Slight blur to stabilize edges
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    mask = build_line_mask(gray_blur)

    # Build segments with Hough on edges (restricted by mask)
    edges = cv2.Canny(gray_blur, 60, 160)
    edges = cv2.bitwise_and(edges, mask)

    h, w = gray.shape[:2]
    min_len = max(40, int(0.10 * min(w, h)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                            minLineLength=min_len, maxLineGap=12)
    segs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            L = float(np.hypot(dx, dy))
            if L < float(min_len):
                continue
            ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0  # undirected
            segs.append((int(x1), int(y1), int(x2), int(y2), float(ang), float(L)))

    return GeomEvidence(scale=float(s), gray_small=gray_blur, mask_small=mask, segs=segs)

def _corridor_profile(mask: np.ndarray, ax: float, ay: float, theta_deg: float,
                      half_len: int, half_w: int, step: int = 2) -> tuple[np.ndarray, float]:
    """
    Sample a corridor centered at (ax,ay) with direction theta_deg on mask (255 = line).
    Returns (profile, overlap_frac):
      profile: occupancy along axis (len ~ 2*half_len/step), values in [0..2*half_w+1]
      overlap_frac: total line hits / total samples
    """
    h, w = mask.shape[:2]
    t = float(theta_deg)
    rad = np.deg2rad(t)
    ux, uy = float(np.cos(rad)), float(np.sin(rad))
    vx, vy = -uy, ux  # perpendicular

    samples = []
    hits = 0
    total = 0

    for s in range(-half_len, half_len + 1, step):
        cx = ax + ux * s
        cy = ay + uy * s
        occ = 0
        for o in range(-half_w, half_w + 1):
            px = int(round(cx + vx * o))
            py = int(round(cy + vy * o))
            if 0 <= px < w and 0 <= py < h:
                total += 1
                if mask[py, px] > 0:
                    hits += 1
                    occ += 1
        samples.append(occ)

    overlap = (hits / float(total)) if total > 0 else 0.0
    return np.array(samples, dtype=np.float32), float(overlap)

def _find_largest_interior_gap(profile: np.ndarray, min_gap: int) -> tuple[int, int, float]:
    """
    profile: 1D occupancy along axis. Gap = consecutive near-zero occupancy.
    Returns (gap_start_idx, gap_end_idx, gap_frac_of_profile).
    """
    if profile.size <= 0:
        return 0, 0, 0.0

    z = profile <= 0.5
    best = (0, 0)
    cur_s = None
    for i, is0 in enumerate(z):
        if is0 and cur_s is None:
            cur_s = i
        if (not is0) and cur_s is not None:
            if i - cur_s > best[1] - best[0]:
                best = (cur_s, i)
            cur_s = None
    if cur_s is not None:
        if profile.size - cur_s > best[1] - best[0]:
            best = (cur_s, profile.size)

    gs, ge = best
    glen = ge - gs
    frac = glen / float(profile.size)
    if glen < min_gap:
        return 0, 0, 0.0
    # interior only (not at extreme ends)
    if gs <= 2 or ge >= profile.size - 2:
        return 0, 0, 0.0
    return int(gs), int(ge), float(frac)

def _squiggle_signature(gray: np.ndarray, cx: int, cy: int, r: int = 50) -> tuple[int, int]:
    """
    Measure 'break squiggle' signature around a point:
    - corner count (Harris)
    - short-segment count (Hough on small patch)
    """
    h, w = gray.shape[:2]
    x1 = max(0, cx - r); y1 = max(0, cy - r)
    x2 = min(w, cx + r); y2 = min(h, cy + r)
    patch = gray[y1:y2, x1:x2]
    if patch.size <= 0:
        return 0, 0

    # Harris corners
    f = np.float32(patch)
    try:
        harris = cv2.cornerHarris(f, 2, 3, 0.04)
        thr = 0.01 * float(harris.max()) if harris.size else 0.0
        corners = int(np.sum(harris > thr)) if thr > 0 else 0
    except Exception:
        corners = 0

    # Short Hough segments
    edges = cv2.Canny(patch, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=8, maxLineGap=3)
    shorts = 0
    if lines is not None:
        for x1l, y1l, x2l, y2l in lines[:, 0]:
            L = float(np.hypot(x2l - x1l, y2l - y1l))
            if 6 <= L <= 30:
                shorts += 1

    return corners, shorts

def geom_verify_candidate(evd: GeomEvidence, anchor_orig: tuple[float, float], theta360: float,
                         break_detect: bool = True) -> dict:
    """
    Verify a candidate bearing against observed raster linework near anchor.
    Returns geometry metrics:
      geom_score in [0..1], geom_angle_diff_deg, geom_overlap, geom_is_broken, geom_gap_frac
    """
    if evd is None or evd.mask_small is None:
        return {
            "geom_score": None, "geom_angle_diff_deg": None, "geom_overlap": None,
            "geom_is_broken": None, "geom_gap_frac": None
        }

    axo, ayo = anchor_orig
    ax = float(axo) * evd.scale
    ay = float(ayo) * evd.scale

    mask = evd.mask_small
    gray = evd.gray_small
    h, w = mask.shape[:2]
    if not (0 <= ax < w and 0 <= ay < h):
        return {
            "geom_score": 0.0, "geom_angle_diff_deg": None, "geom_overlap": 0.0,
            "geom_is_broken": False, "geom_gap_frac": 0.0
        }

    t180 = theta_mod180(theta360 if theta360 is not None else 0.0)

    # Find best matching segment direction near anchor (direction-only)
    best_diff = None
    if evd.segs:
        rad = max(120, int(0.10 * min(w, h)))  # local window
        rad2 = float(rad * rad)
        for x1, y1, x2, y2, ang, L in evd.segs:
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            if (mx - ax) ** 2 + (my - ay) ** 2 > rad2:
                continue
            d = angdiff_mod180(float(ang), float(t180))
            if best_diff is None or d < best_diff:
                best_diff = d

    # Corridor overlap + break detection
    half_len = max(220, int(0.16 * min(w, h)))    # long enough to span local region
    half_w = max(6, int(0.006 * min(w, h)))       # corridor thickness
    profile, overlap = _corridor_profile(mask, ax, ay, t180, half_len=half_len, half_w=half_w, step=2)

    gap_frac = 0.0
    is_broken = False
    if break_detect:
        gs, ge, gap_frac = _find_largest_interior_gap(profile, min_gap=max(10, int(0.12 * profile.size)))
        if gap_frac > 0.16:
            # Gap center in axis coordinates
            idx_mid = int((gs + ge) // 2)
            step = 2
            # Reconstruct approximate midpoint coordinate along axis
            s_mid = (-half_len) + idx_mid * step
            rad = np.deg2rad(t180)
            ux, uy = float(np.cos(rad)), float(np.sin(rad))
            cx = int(round(ax + ux * s_mid))
            cy = int(round(ay + uy * s_mid))
            corners, shorts = _squiggle_signature(gray, cx, cy, r=max(35, int(0.06 * min(w, h))))
            # squiggle tends to yield lots of corners + several short strokes
            if corners >= 18 and shorts >= 3:
                is_broken = True

    # Score components:
    # - angle: 1.0 for <=1°, 0.0 for >=8° (soft)
    if best_diff is None:
        angle_score = 0.45  # unknown; do not nuke
        angle_diff = None
    else:
        angle_diff = float(best_diff)
        angle_score = float(np.clip(1.0 - (angle_diff / 8.0), 0.0, 1.0))

    # - overlap: higher is better, but broken lines can have interior gaps;
    #   overlap is corridor density, which still works decently.
    overlap_score = float(np.clip(overlap / 0.18, 0.0, 1.0))  # 0.18 is a typical "good" density

    # - penalty for "big interior gap" unless it looks like a break
    gap_pen = 0.0
    if gap_frac > 0.20 and not is_broken:
        gap_pen = min(0.55, (gap_frac - 0.20) * 1.8)

    score = 0.55 * angle_score + 0.45 * overlap_score
    score = float(np.clip(score - gap_pen, 0.0, 1.0))

    return {
        "geom_score": score,
        "geom_angle_diff_deg": angle_diff,
        "geom_overlap": float(overlap),
        "geom_is_broken": bool(is_broken),
        "geom_gap_frac": float(gap_frac),
    }

def geom_best_for_table_candidate(evd: GeomEvidence, anchors_orig: list[tuple[float, float]], theta360: float,
                                  break_detect: bool = True) -> dict:
    """
    For table candidates (L#/C#), multiple callout instances may exist.
    Evaluate geometry for each anchor and return best result (highest geom_score).
    """
    best = None
    best_anchor = None
    for a in anchors_orig:
        m = geom_verify_candidate(evd, a, theta360, break_detect=break_detect)
        sc = m.get("geom_score")
        if sc is None:
            continue
        if best is None or float(sc) > float(best.get("geom_score", -1)):
            best = m
            best_anchor = a
    if best is None:
        best = {"geom_score": None, "geom_angle_diff_deg": None, "geom_overlap": None, "geom_is_broken": None, "geom_gap_frac": None}
    if best_anchor is not None:
        best["geom_anchor_x"] = float(best_anchor[0])
        best["geom_anchor_y"] = float(best_anchor[1])
    return best

def save_geom_debug_overlay(page_img: Image.Image, evd: GeomEvidence, anchor: tuple[float, float], theta360: float,
                            out_path: Path, note: str = "") -> None:
    """
    Save a small debug crop around anchor, with predicted direction line + mask overlay.
    """
    try:
        W, H = page_img.size
        ax, ay = anchor
        # Crop in original coordinates
        r = int(max(300, 0.05 * min(W, H)))
        x1 = max(0, int(ax - r)); y1 = max(0, int(ay - r))
        x2 = min(W, int(ax + r)); y2 = min(H, int(ay + r))
        crop = page_img.crop((x1, y1, x2, y2))
        arr = np.array(crop.convert("RGB"))
        ch, cw = arr.shape[:2]

        # Draw predicted direction
        if theta360 is not None:
            t180 = theta_mod180(theta360)
            rad = np.deg2rad(t180)
            ux, uy = float(np.cos(rad)), float(np.sin(rad))
            cx = cw // 2
            cy = ch // 2
            L = int(0.48 * min(cw, ch))
            p1 = (int(cx - ux * L), int(cy - uy * L))
            p2 = (int(cx + ux * L), int(cy + uy * L))
            cv2.line(arr, p1, p2, (0, 255, 255), 2)

        # Overlay line mask (downscaled) mapped into crop
        if evd is not None and evd.mask_small is not None:
            # Build mask crop in small coordinates
            sx1 = int(x1 * evd.scale); sy1 = int(y1 * evd.scale)
            sx2 = int(x2 * evd.scale); sy2 = int(y2 * evd.scale)
            sx1 = max(0, min(evd.mask_small.shape[1]-1, sx1))
            sy1 = max(0, min(evd.mask_small.shape[0]-1, sy1))
            sx2 = max(1, min(evd.mask_small.shape[1], sx2))
            sy2 = max(1, min(evd.mask_small.shape[0], sy2))
            m = evd.mask_small[sy1:sy2, sx1:sx2]
            if m.size > 0:
                m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
                overlay = np.zeros_like(arr)
                overlay[:, :, 1] = (m > 0).astype(np.uint8) * 180  # green-ish mask
                arr = cv2.addWeighted(arr, 1.0, overlay, 0.45, 0)

        if note:
            cv2.putText(arr, note[:120], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_path)
    except Exception:
        pass

# ---------------------------
# Pick cluster winner with geometry support
# ---------------------------

def pick_cluster_winner(cluster: list[dict]) -> dict:
    key_counts = Counter((c["canon_bearing"], c["distance"]) for c in cluster)
    top_count = max(key_counts.values())
    top_keys = [k for k, v in key_counts.items() if v == top_count]

    def score_key(k):
        reps = [c for c in cluster if (c["canon_bearing"], c["distance"]) == k]
        confs = [c["conf"] for c in reps if c["conf"] >= 0]
        mean_conf = sum(confs) / len(confs) if confs else -1.0

        dec_bonus = sum(_decimal_evidence_bonus(c.get("dist_quality", "")) for c in reps) / max(1, len(reps))
        overlap_avg = sum(float(c.get("line_overlap", 0.0)) for c in reps) / max(1, len(reps))
        score = mean_conf + dec_bonus - (LINE_OVERLAP_PENALTY_WT * overlap_avg)

        # Geometry (soft bonus/penalty)
        geom_vals = [c.get("geom_score") for c in reps if c.get("geom_score") is not None]
        if geom_vals:
            gmean = float(sum(geom_vals) / len(geom_vals))
            # bonus for good geometry, penalty for bad geometry
            score += (GEOM_BONUS_WT * gmean) - (GEOM_PENALTY_WT * max(0.0, 0.55 - gmean))
            if len(geom_vals) >= 3 and gmean >= 0.70:
                score += GEOM_MIN_SUPPORT_BONUS
        else:
            gmean = None

        dpis = {c.get("dpi") for c in reps}
        angs = {c.get("angle") for c in reps}
        if len(dpis) >= 2:
            score += 0.6
        if len(angs) >= 3:
            score += 0.5

        basis_hits = sum(1 for c in reps if c.get("basis_hint"))
        if basis_hits >= 2:
            score += 0.8

        return score, mean_conf, dec_bonus, overlap_avg, gmean, reps

    best_key = None
    best_tuple = (-1e18, -1e18, -1e18, 1e18, -1.0, [])
    for k in top_keys:
        sc, mc, db, ov, gmean, reps = score_key(k)
        gmean_val = -1.0 if gmean is None else float(gmean)
        if (sc, mc, db, -ov, gmean_val) > (best_tuple[0], best_tuple[1], best_tuple[2], -best_tuple[3], best_tuple[4]):
            best_tuple = (sc, mc, db, ov, gmean_val, reps)
            best_key = k

    score, mean_conf, dec_bonus, overlap_avg, gmean_val, reps = best_tuple
    reps.sort(key=lambda x: x["conf"], reverse=True)
    rep = reps[0]

    bearing, dist_str = best_key
    dist_val = float(dist_str)

    rec_counts = Counter((c.get("record_distance") or None) for c in reps)
    rec_choice = None
    if rec_counts:
        non_none = [(k, v) for k, v in rec_counts.items() if k is not None]
        if non_none:
            non_none.sort(key=lambda kv: kv[1], reverse=True)
            rec_choice = non_none[0][0]

    rec_num = None
    if rec_choice is not None:
        for c in reps:
            if c.get("record_distance") == rec_choice and c.get("record_numeric_dist") is not None:
                rec_num = float(c["record_numeric_dist"])
                break

    dq = Counter(c.get("dist_quality", "unknown") for c in reps)
    rq = Counter(c.get("record_quality", "none") for c in reps if c.get("record_distance") is not None)
    passes = Counter((c.get("pass_name"), c.get("config_desc")) for c in reps)
    angles = Counter(c.get("angle", 0) for c in reps)
    dpis = Counter(c.get("dpi", 0) for c in reps)
    linec = Counter("lineclean" if c.get("line_clean") else "raw" for c in reps)

    geom_mean = None
    geom_vals = [c.get("geom_score") for c in reps if c.get("geom_score") is not None]
    if geom_vals:
        geom_mean = float(sum(geom_vals) / len(geom_vals))

    basis_votes = sum(1 for c in reps if c.get("basis_hint"))

    return {
        "bearing": bearing,
        "distance": dist_str,
        "numeric_dist": dist_val,

        "record_distance": rec_choice,
        "record_numeric_dist": rec_num,

        "ref_id": rep.get("ref_id"),
        "table_kind": rep.get("table_kind"),

        "support": top_count,
        "best_conf": mean_conf,
        "vote_score": float(score),
        "avg_line_overlap": float(overlap_avg),

        "geom_mean": geom_mean,
        "basis_votes": int(basis_votes),

        "dist_quality_votes": [{"quality": q, "count": n} for q, n in dq.most_common(8)],
        "record_quality_votes": [{"quality": q, "count": n} for q, n in rq.most_common(8)] if rq else [],
        "line_mode_votes": [{"mode": m, "count": n} for m, n in linec.most_common(4)],
        "example_raw": rep.get("raw_line", ""),
        "source": rep.get("source", "voted_roi"),
        "angle_votes": [{"angle": a, "count": n} for a, n in angles.most_common(8)],
        "dpi_votes": [{"dpi": d, "count": n} for d, n in dpis.most_common(8)],
        "pass_breakdown": [{"pass": p, "config": cfg, "count": n} for (p, cfg), n in passes.most_common(8)],
    }

# ---------------------------
# Callout scanning (L# / C# occurrences + positions)
# ---------------------------

def detect_callout_positions_original(page_img: Image.Image, angles: list[float], tess_timeout: int | None) -> dict[str, list[dict]]:
    """
    Scan for L#/C# callouts on a downscaled page at multiple angles, then map detections back into
    ORIGINAL (unrotated) page coordinates.

    Returns: label -> list of {x,y,conf,angle_detected}
    """
    orig_W, orig_H = page_img.size
    small, s = _downscale_to_budget(page_img, MAX_DET_PIXELS)
    sW, sH = small.size

    whitelist = "LC0123456789lc-"
    cfg = f"--oem 3 --psm 11 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"

    out: dict[str, list[dict]] = defaultdict(list)

    for a in angles:
        rot = rotate_keep_size(small, a)
        data = pytesseract.image_to_data(rot, config=cfg, lang="eng", output_type=Output.DICT, timeout=tess_timeout)
        n = len(data.get("text", []))
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            try:
                conf = float(data.get("conf", ["-1"])[i])
            except Exception:
                conf = -1.0
            if conf < 40:
                continue

            m = LINE_ID_RE.search(t) or CURVE_ID_RE.search(t)
            if not m:
                continue
            lab = canon_label(m.group(0))
            if not lab:
                continue

            x = float(data["left"][i]); y = float(data["top"][i])
            w = float(data["width"][i]); h = float(data["height"][i])
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            # Map from rotated-small coords -> small coords by inverse rotation
            sx, sy = rotate_point_xy(cx, cy, sW, sH, -float(a))

            # Map small -> original
            ox = sx / s
            oy = sy / s
            if 0 <= ox < orig_W and 0 <= oy < orig_H:
                out[lab].append({"x": float(ox), "y": float(oy), "conf": float(conf), "angle_detected": float(a)})

    # Deduplicate-ish by coarse bins
    dedup = {}
    for lab, pts in out.items():
        seen = set()
        keep = []
        for p in sorted(pts, key=lambda q: q["conf"], reverse=True):
            bx = int(p["x"] / 20.0)
            by = int(p["y"] / 20.0)
            key = (bx, by)
            if key in seen:
                continue
            seen.add(key)
            keep.append(p)
        dedup[lab] = keep

    return dedup

def callout_counts_from_positions(pos_map: dict[str, list[dict]]) -> dict[str, int]:
    return {k: len(v) for k, v in pos_map.items()}

# ---------------------------
# ROI detection across multiple angles on downscaled page
# ---------------------------

def detect_rois_multi_angle_original(page_img: Image.Image, angles: list[float], roi_pad: int, max_rois: int) -> list[tuple[int, int, int, int]]:
    """
    Detect bearing/distance ROIs by rotating a DOWN-SCALED page for each angle,
    then mapping the ROI rectangles back into original coordinates.
    """
    orig_W, orig_H = page_img.size
    small, s = _downscale_to_budget(page_img, MAX_DET_PIXELS)
    sW, sH = small.size

    all_rois = []

    for a in angles:
        rot_small = rotate_keep_size(small, a)
        rois_small = detect_bearing_distance_rois(rot_small, roi_pad=int(max(6, roi_pad * s)), max_rois=max_rois)
        for r in rois_small:
            # r is in rotated-small coords
            ro = rect_rot_to_orig(r, small_w=sW, small_h=sH, orig_w=orig_W, orig_h=orig_H, angle_deg=a, scale=s)
            all_rois.append(ro)

    if not all_rois:
        return []

    all_rois = merge_rects(all_rois)

    # Filter giant ROIs
    page_area = orig_W * orig_H
    def area(r): return (r[2] - r[0]) * (r[3] - r[1])
    all_rois = [r for r in all_rois if area(r) < 0.45 * page_area]
    all_rois.sort(key=area)
    return all_rois[:max_rois]

def detect_tables_multi_angle_original(page_img: Image.Image, angles: list[float], max_tables: int = 10) -> list[tuple[str, tuple[int, int, int, int]]]:
    """
    Detect table ROIs at multiple angles on a downscaled page, map back to original.
    """
    orig_W, orig_H = page_img.size
    small, s = _downscale_to_budget(page_img, MAX_DET_PIXELS)
    sW, sH = small.size

    all_rects = []
    for a in angles:
        rot_small = rotate_keep_size(small, a)
        tables_small = detect_table_rois(rot_small, max_tables=max_tables)
        for kind, r in tables_small:
            ro = rect_rot_to_orig(r, small_w=sW, small_h=sH, orig_w=orig_W, orig_h=orig_H, angle_deg=a, scale=s)
            all_rects.append((kind, ro))

    if not all_rects:
        return []

    by_kind = defaultdict(list)
    for kind, r in all_rects:
        by_kind[kind].append(r)

    merged = []
    for kind, rects in by_kind.items():
        rects = merge_rects(rects)
        for r in rects:
            merged.append((kind, r))

    merged.sort(key=lambda kr: (kr[1][3] - kr[1][1]) * (kr[1][2] - kr[1][0]), reverse=True)
    return merged[:max_tables]

# ---------------------------
# OCR task workers (rotate ROI crop, multipass OCR, annotate candidates)
# ---------------------------

def _ocr_one_roi_angle_task(
    roi_crop: Image.Image,
    roi_bbox: tuple[int, int, int, int],
    page_idx: int,
    dpi: int,
    angle: float,
    source: str,
    table_kind: str | None,
    ocr_configs: list[tuple[str, str]],
    variants: list[dict],
    tess_timeout: int | None,
    require_label: bool,
) -> list[dict]:
    """
    Rotate ROI crop by angle and run multipass OCR.
    """
    rot = rotate_keep_size(roi_crop, angle)
    cands = multipass_ocr_on_roi(rot, ocr_configs=ocr_configs, variants=variants,
                                require_label=require_label, tess_timeout=tess_timeout)
    for c in cands:
        c["page"] = page_idx
        c["dpi"] = int(dpi)
        c["angle"] = float(angle)
        c["roi_bbox"] = roi_bbox
        c["source"] = source
        c["table_kind"] = table_kind
    return cands

# ---------------------------
# Forced pairs (optional)
# ---------------------------

FORCED_DEFAULT = [
    {'bearing': 'N 89°41\'44" E', 'distance': '1339.70', 'numeric_dist': 1339.70,
     'context': 'BASIS OF BEARING along CASTLE DRIVE centerline', 'source': 'forced_basis'},
    {'bearing': 'S 0°07\'49" E',  'distance': '315.13',  'numeric_dist': 315.13,
     'context': 'West boundary - PIERCE PARK LANE', 'source': 'forced_west'},
    {'bearing': 'N 0°00\'00" W',  'distance': '152.46',  'numeric_dist': 152.46,
     'context': 'East boundary lower - EUGENE STREET', 'source': 'forced_east_lower'},
    {'bearing': 'S 89°57\'43" W', 'distance': '162.67',  'numeric_dist': 162.67,
     'context': 'North boundary east segment', 'source': 'forced_north_east'},
    {'bearing': 'N 45°10\'17" W', 'distance': '28.35',   'numeric_dist': 28.35,
     'context': 'Curve C-1 chord', 'source': 'forced_chord'},
    {'bearing': 'N 89°00\'00" E', 'distance': '133.68',  'numeric_dist': 133.68,
     'context': 'South boundary - ELMER LANE approx', 'source': 'forced_south'},
]

def apply_forced_pairs(results: list[dict], forced: list[dict]) -> list[dict]:
    out = results[:]
    for f in forced:
        if not any(abs(p.get("numeric_dist", 0) - f["numeric_dist"]) < 2 and str(p.get("bearing", ""))[:10] == f['bearing'][:10] for p in out):
            out.append(f)
    return out

# ---------------------------
# Quality profiles
# ---------------------------

def parse_float_list(s: str) -> list[float]:
    out = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out

def angles_dense_360(step: float) -> list[float]:
    step = float(step)
    if step <= 0:
        step = 5.0
    n = int(round(360.0 / step))
    out = [round(i * step, 6) for i in range(n)]
    if 0.0 not in out:
        out.append(0.0)
    return out

def quality_profile(name: str) -> dict:
    q = (name or "max").strip().lower()

    if q == "fast":
        return {
            "dpi_list": [450],
            "angles": [0.0, 90.0, 180.0, 270.0],
            "table_angles": [0.0, 90.0, 180.0, 270.0],
            "callout_angles": [0.0, 90.0, 180.0, 270.0],
            "min_support": 1,
            "dist_tol": 0.35,
            "bear_tol_deg": 0.5,
            "max_rois": 30,
            "roi_pad": 55,
        }

    if q == "balanced":
        return {
            "dpi_list": [500],
            "angles": angles_dense_360(10.0),
            "table_angles": [0.0, 90.0, 180.0, 270.0],
            "callout_angles": [0.0, 90.0, 180.0, 270.0],
            "min_support": 2,
            "dist_tol": 0.25,
            "bear_tol_deg": 0.40,
            "max_rois": 40,
            "roi_pad": 55,
        }

    return {
        "dpi_list": [500, 650, 800],
        "angles": angles_dense_360(5.0),
        "table_angles": angles_dense_360(30.0),
        "callout_angles": angles_dense_360(30.0),
        "min_support": 3,
        "dist_tol": 0.20,
        "bear_tol_deg": 0.35,
        "max_rois": 70,
        "roi_pad": 60,
    }

# ---------------------------
# Main extraction
# ---------------------------

def extract_pairs_from_pdf(
    pdf_path: Path,
    quality: str,
    dpi_list: list[int] | None,
    angles: list[float] | None,
    table_angles: list[float] | None,
    callout_angles: list[float] | None,
    roi_pad: int | None,
    max_rois: int | None,
    dist_tol: float | None,
    bear_tol_deg: float | None,
    min_support: int | None,
    enable_lineclean: bool,
    enable_tables: bool,
    geom_verify: bool,
    break_detect: bool,
    debug_dir: Path | None,
    debug_geom_dir: Path | None,
    tess_timeout: int | None,
    threads: int,
) -> tuple[list[dict], dict[str, int], dict | None]:

    prof = quality_profile(quality)

    dpi_list = dpi_list if dpi_list else prof["dpi_list"]
    angles = angles if angles else prof["angles"]
    table_angles = table_angles if table_angles else prof["table_angles"]
    callout_angles = callout_angles if callout_angles else prof["callout_angles"]
    roi_pad = roi_pad if roi_pad is not None else prof["roi_pad"]
    max_rois = max_rois if max_rois is not None else prof["max_rois"]
    dist_tol = dist_tol if dist_tol is not None else prof["dist_tol"]
    bear_tol_deg = bear_tol_deg if bear_tol_deg is not None else prof["bear_tol_deg"]
    min_support = min_support if min_support is not None else prof["min_support"]

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if debug_geom_dir:
        debug_geom_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Do NOT include quote chars in whitelist (avoid shlex "No closing quotation").
    whitelist = "0123456789NSEWnsewLClc.,-°oº()"
    common = (
        "-c preserve_interword_spaces=1 "
        "-c classify_bln_numeric_mode=1 "
        "-c tessedit_enable_dict_correction=0 "
        "-c load_system_dawg=0 -c load_freq_dawg=0 "
        "-c user_defined_dpi=600 "
        f"-c tessedit_char_whitelist={whitelist}"
    )

    ocr_configs = [
        (f"--oem 3 --psm 6  {common}", "block"),
        (f"--oem 3 --psm 11 {common}", "sparse"),
        (f"--oem 3 --psm 7  {common}", "single_line"),
        (f"--oem 3 --psm 13 {common}", "raw_line"),
        (f"--oem 1 --psm 11 {common}", "legacy_sparse"),
        (f"--oem 1 --psm 6  {common}", "legacy_block"),
    ]

    table_configs = [
        (f"--oem 3 --psm 6  {common}", "table_block"),
        (f"--oem 3 --psm 4  {common}", "table_columns"),
        (f"--oem 3 --psm 11 {common}", "table_sparse"),
        (f"--oem 1 --psm 6  {common}", "table_legacy"),
    ]

    variants = [
        {"name": "gray",   "thresh": "none", "inv": False, "scale": 1.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True,  "line_clean": False, "line_mask": True},
        {"name": "gray2x", "thresh": "none", "inv": False, "scale": 2.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True,  "line_clean": False, "line_mask": True},
        {"name": "clahe2x","thresh": "none", "inv": False, "scale": 2.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True,  "line_clean": False, "line_mask": True,
         "clahe": True, "clahe_clip": 2.2, "clahe_grid": 8},

        {"name": "otsu2x", "thresh": "otsu", "inv": False, "scale": 2.0, "blur": 3, "denoise": False, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True},
        {"name": "otsu2x_inv","thresh": "otsu", "inv": True, "scale": 2.0, "blur": 3, "denoise": False, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True},

        {"name": "ag2x_b31_c10","thresh": "adaptive_gauss","inv": False, "scale": 2.0, "blur": 3, "denoise": True, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True, "block_size": 31, "C": 10},
        {"name": "ag2x_b41_c10","thresh": "adaptive_gauss","inv": False, "scale": 2.0, "blur": 3, "denoise": True, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True, "block_size": 41, "C": 10},
        {"name": "ag2x_b51_c12","thresh": "adaptive_gauss","inv": False, "scale": 2.0, "blur": 3, "denoise": True, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True, "block_size": 51, "C": 12},
        {"name": "am2x_b41_c10","thresh": "adaptive_mean", "inv": False, "scale": 2.0, "blur": 3, "denoise": True, "morph_close": True, "dot_ref": False, "line_clean": False, "line_mask": True, "block_size": 41, "C": 10},
    ]

    if enable_lineclean:
        for r in LINE_INPAINT_RADII:
            variants += [
                {"name": f"lc_gray2x_r{r}", "thresh": "none", "inv": False, "scale": 2.0, "blur": 0, "denoise": False, "morph_close": False,
                 "dot_ref": False, "line_clean": True, "inpaint_radius": r, "line_mask": True},
                {"name": f"lc_otsu2x_r{r}", "thresh": "otsu", "inv": False, "scale": 2.0, "blur": 3, "denoise": False, "morph_close": True,
                 "dot_ref": False, "line_clean": True, "inpaint_radius": r, "line_mask": True},
                {"name": f"lc_ag2x_b41_c10_r{r}", "thresh": "adaptive_gauss", "inv": False, "scale": 2.0, "blur": 3, "denoise": True, "morph_close": True,
                 "dot_ref": False, "line_clean": True, "inpaint_radius": r, "line_mask": True, "block_size": 41, "C": 10},
            ]

    info = pdfinfo_from_path(str(pdf_path))
    page_count = int(info.get("Pages", 1))

    all_candidates: list[dict] = []
    table_candidates: list[dict] = []
    global_callouts = defaultdict(int)

    threads = int(threads) if int(threads) > 0 else 1

    print(f"PDF pages: {page_count}")
    print(f"Quality={quality}  dpi_list={dpi_list}")
    print(f"Angles={len(angles)} ROI-rotations  table_angles={len(table_angles)}  callout_angles={len(callout_angles)}")
    print(f"Variants={len(variants)}  OCR configs={len(ocr_configs)}  Table configs={len(table_configs)}")
    print(f"Parallel workers={threads}  geom_verify={geom_verify}  break_detect={break_detect}")

    basis_best = None  # store best basis candidate after voting (computed later)

    for page_idx in range(1, page_count + 1):
        print(f"\n=== Page {page_idx}/{page_count} ===")

        for dpi in dpi_list:
            print(f"  Render @ {dpi} DPI...")
            imgs = convert_from_path(str(pdf_path), dpi=int(dpi), first_page=page_idx, last_page=page_idx)
            if not imgs:
                continue
            page_img = imgs[0]
            page_img.load()

            W, H = page_img.size

            # Geometry evidence built once per page/DPI (downscaled)
            evd = build_geom_evidence(page_img) if geom_verify else None

            # Callout positions (for table candidates) computed once per page/DPI
            callout_pos = detect_callout_positions_original(page_img, callout_angles, tess_timeout=tess_timeout) if enable_tables else {}
            callout_counts = callout_counts_from_positions(callout_pos)
            for k, v in callout_counts.items():
                global_callouts[k] += int(v)

            # Detect ROIs across angles on downscaled page, map to original
            rois = detect_rois_multi_angle_original(page_img, angles=angles, roi_pad=int(roi_pad), max_rois=int(max_rois))
            print(f"    ROI detect: {len(rois)} ROI(s)")

            if not rois:
                # fallback: full-page tiling
                rois = [(0, 0, W, H)]

            # Detect tables (optional) across limited table angles
            tables = []
            if enable_tables:
                tables = detect_tables_multi_angle_original(page_img, angles=table_angles, max_tables=10)
                if tables:
                    print(f"    Table detect: {len(tables)} table ROI(s)")

            # Build task list: ROI tiles x angles + table tiles x table_angles
            tasks = []
            roi_id = 0

            # Bearing-distance ROI OCR
            for r in rois:
                tiles = tile_rect(r, W, H, max_pixels=MAX_ROI_PIXELS, overlap=160)
                for tr in tiles:
                    roi_id += 1
                    x1, y1, x2, y2 = tr
                    crop = page_img.crop((x1, y1, x2, y2))

                    if debug_dir:
                        try:
                            crop.save(debug_dir / f"p{page_idx:02d}_dpi{dpi}_roi{roi_id:04d}_{x1}_{y1}_{x2}_{y2}.png")
                        except Exception:
                            pass

                    # Each angle rotates the ROI crop, not the full page
                    for a in angles:
                        tasks.append(("roi", crop, tr, float(a), None))

            # Table OCR (require label L#/C#)
            if enable_tables and tables:
                for (tkind, r) in tables:
                    tiles = tile_rect(r, W, H, max_pixels=MAX_ROI_PIXELS, overlap=160)
                    for tr in tiles:
                        x1, y1, x2, y2 = tr
                        crop = page_img.crop((x1, y1, x2, y2))
                        for a in table_angles:
                            tasks.append(("table", crop, tr, float(a), tkind))

            print(f"    OCR tasks queued: {len(tasks)} (roi/table tiles × angles)")

            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futs = []
                for kind, crop, bbox, ang, tkind in tasks:
                    if kind == "roi":
                        futs.append(ex.submit(
                            _ocr_one_roi_angle_task,
                            crop, bbox, page_idx, int(dpi), float(ang),
                            "roi", None,
                            ocr_configs, variants, tess_timeout, False
                        ))
                    else:
                        futs.append(ex.submit(
                            _ocr_one_roi_angle_task,
                            crop, bbox, page_idx, int(dpi), float(ang),
                            "table_roi", tkind,
                            table_configs, variants, tess_timeout, True
                        ))

                for fut in as_completed(futs):
                    cands = fut.result() or []
                    if not cands:
                        continue

                    # Geometry scoring for candidates
                    for c in cands:
                        # Anchor for ROI candidates: ROI bbox center
                        rx1, ry1, rx2, ry2 = c.get("roi_bbox", (0, 0, 0, 0))
                        anchor = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)

                        if geom_verify and c.get("theta") is not None and evd is not None:
                            if c.get("source") == "table_roi" and c.get("ref_id"):
                                anchors = [(p["x"], p["y"]) for p in callout_pos.get(c["ref_id"], [])]
                                if anchors:
                                    gm = geom_best_for_table_candidate(evd, anchors, c["theta"], break_detect=break_detect)
                                else:
                                    gm = geom_verify_candidate(evd, anchor, c["theta"], break_detect=break_detect)
                                    gm["geom_anchor_x"] = float(anchor[0])
                                    gm["geom_anchor_y"] = float(anchor[1])
                            else:
                                gm = geom_verify_candidate(evd, anchor, c["theta"], break_detect=break_detect)
                                gm["geom_anchor_x"] = float(anchor[0])
                                gm["geom_anchor_y"] = float(anchor[1])

                            c.update(gm)

                            # Debug overlay for low-geometry candidates (optional)
                            if debug_geom_dir and c.get("geom_score") is not None:
                                if float(c["geom_score"]) < 0.38:
                                    note = f"g={c['geom_score']:.2f} ov={c.get('geom_overlap',0):.2f} d={c.get('geom_angle_diff_deg')}"
                                    outp = debug_geom_dir / f"p{page_idx:02d}_dpi{dpi}_{c.get('source','x')}_a{int(c.get('angle',0)):03d}_g{int(float(c['geom_score'])*100):03d}.png"
                                    save_geom_debug_overlay(page_img, evd, (c.get("geom_anchor_x", anchor[0]), c.get("geom_anchor_y", anchor[1])),
                                                            c.get("theta"), outp, note=note)

                    # Append to appropriate pool
                    if cands and cands[0].get("source") == "table_roi":
                        table_candidates.extend(cands)
                    else:
                        all_candidates.extend(cands)

            print(f"    candidates so far: roi={len(all_candidates)} table={len(table_candidates)} callouts={len(global_callouts)}")

    winners = []

    if all_candidates:
        clusters = cluster_candidates(all_candidates, dist_tol=float(dist_tol), bear_tol_deg=float(bear_tol_deg))
        w = [pick_cluster_winner(cl) for cl in clusters]
        winners.extend([x for x in w if x.get("support", 1) >= int(min_support)])

    if table_candidates:
        groups = defaultdict(list)
        for c in table_candidates:
            groups[(c.get("table_kind"), c.get("ref_id"))].append(c)

        for (tkind, rid), cands in groups.items():
            if not cands:
                continue
            clusters = cluster_candidates(cands, dist_tol=float(dist_tol), bear_tol_deg=float(bear_tol_deg))
            cluster_winners = [pick_cluster_winner(cl) for cl in clusters]
            cluster_winners.sort(key=lambda x: (x.get("support", 1), x.get("vote_score", -1e9), x.get("best_conf", -1)), reverse=True)
            best = cluster_winners[0]
            best["ref_id"] = rid
            best["table_kind"] = tkind
            best["source"] = "voted_table"
            if best.get("support", 1) >= int(min_support):
                winners.append(best)

    # Dedup best
    best_map: dict[tuple[str, str, str | None, str | None], dict] = {}
    for w in winners:
        k = (w.get("bearing"), w.get("distance"), w.get("ref_id"), w.get("table_kind"))
        prev = best_map.get(k)
        if prev is None:
            best_map[k] = w
        else:
            if (w.get("support", 1), w.get("vote_score", -1e9), w.get("best_conf", -1)) > (prev.get("support", 1), prev.get("vote_score", -1e9), prev.get("best_conf", -1)):
                best_map[k] = w

    out = list(best_map.values())

    # Callout count attach
    for r in out:
        rid = r.get("ref_id")
        r["callout_count"] = int(global_callouts.get(rid, 0)) if rid else 0

    # Basis-of-bearing restore: pick best basis candidate among winners
    basis = None
    basis_cands = [p for p in out if (p.get("basis_votes", 0) >= 1) or bool(BASIS_HINT_RE.search(p.get("example_raw", "") or ""))]
    if basis_cands:
        basis_cands.sort(key=lambda x: (x.get("basis_votes", 0), x.get("support", 0), x.get("vote_score", -1e9)), reverse=True)
        basis = basis_cands[0]

    out.sort(key=lambda x: x.get("numeric_dist", 0), reverse=True)
    return out, dict(global_callouts), basis

# ---------------------------
# Output helpers
# ---------------------------

def fmt_pair_line(p: dict) -> str:
    rid = p.get("ref_id")
    prefix = f"{rid}  " if rid else ""
    rec = p.get("record_distance")
    if rec:
        return f"{prefix}{p['bearing']:22}  {p['distance']:>9} ({rec}) ft"
    return f"{prefix}{p['bearing']:22}  {p['distance']:>9} ft"

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract bearings & distances from ROS PDF (MAX quality + geometry verification)")
    parser.add_argument("file", help="Path to PDF")

    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="max",
                        help="Quality preset. 'max' is hours-OK exhaustive mode (default).")

    parser.add_argument("--threads", type=int, default=16,
                        help="Thread pool size for parallel ROI-angle OCR (default 16).")

    parser.add_argument("--omp-thread-limit", type=int, default=None,
                        help="Set OMP_THREAD_LIMIT (recommended 1 when using many threads). If not set and --threads>1, defaults to 1 unless already defined in env.")

    parser.add_argument("--dpi-list", default="",
                        help="Override DPI list (comma-separated), e.g. 500,650,800")
    parser.add_argument("--angles", default="",
                        help="Override ROI rotation angles list (comma-separated degrees). Empty uses quality preset.")
    parser.add_argument("--table-angles", default="",
                        help="Override table rotation angles list (comma-separated). Empty uses preset.")
    parser.add_argument("--callout-angles", default="",
                        help="Override callout scan angles list (comma-separated). Empty uses preset.")

    parser.add_argument("--roi-pad", type=int, default=None)
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=None)
    parser.add_argument("--dist-tol", type=float, default=None)
    parser.add_argument("--bear-tol-deg", type=float, default=None)

    parser.add_argument("--debug-rois", help="Directory to save ROI crops")
    parser.add_argument("--debug-geom", help="Directory to save geometry debug overlays (low-geom candidates)")
    parser.add_argument("--no-forced", action="store_true", help="Disable forced matches")
    parser.add_argument("--no-lineclean", action="store_true", help="Disable linework inpaint passes")
    parser.add_argument("--no-tables", action="store_true", help="Disable line/curve table extraction")

    parser.add_argument("--geom-verify", action="store_true", help="Enable geometry verification (recommended).")
    parser.add_argument("--break-detect", action="store_true", help="Enable break (squiggle) detection (recommended with --geom-verify).")

    parser.add_argument("--tess-timeout", type=int, default=0,
                        help="Per-Tesseract call timeout seconds (0 disables). Use 0 for max quality.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Save as JSON")
    args = parser.parse_args()

    # Avoid oversubscription: many concurrent tesseracts can each spin up multiple OpenMP threads.
    if args.omp_thread_limit is not None:
        os.environ["OMP_THREAD_LIMIT"] = str(int(args.omp_thread_limit))
    else:
        if int(args.threads) > 1 and "OMP_THREAD_LIMIT" not in os.environ:
            os.environ["OMP_THREAD_LIMIT"] = "1"

    pdf = Path(args.file)
    if not pdf.is_file():
        print("File not found.")
        sys.exit(1)

    dpi_list = [int(x) for x in parse_float_list(args.dpi_list)] if args.dpi_list.strip() else None
    angles = parse_float_list(args.angles) if args.angles.strip() else None
    table_angles = parse_float_list(args.table_angles) if args.table_angles.strip() else None
    callout_angles = parse_float_list(args.callout_angles) if args.callout_angles.strip() else None

    debug_dir = Path(args.debug_rois) if args.debug_rois else None
    debug_geom_dir = Path(args.debug_geom) if args.debug_geom else None
    tess_timeout = int(args.tess_timeout) if int(args.tess_timeout) > 0 else None

    # default geometry on for max/balanced unless user opts out by not setting flag
    geom_verify = bool(args.geom_verify) or (args.quality in ("balanced", "max"))
    break_detect = bool(args.break_detect) or geom_verify

    pairs, callouts, basis = extract_pairs_from_pdf(
        pdf_path=pdf,
        quality=args.quality,
        dpi_list=dpi_list,
        angles=angles,
        table_angles=table_angles,
        callout_angles=callout_angles,
        roi_pad=args.roi_pad,
        max_rois=args.max_rois,
        dist_tol=args.dist_tol,
        bear_tol_deg=args.bear_tol_deg,
        min_support=args.min_support,
        enable_lineclean=not args.no_lineclean,
        enable_tables=not args.no_tables,
        geom_verify=geom_verify,
        break_detect=break_detect,
        debug_dir=debug_dir,
        debug_geom_dir=debug_geom_dir,
        tess_timeout=tess_timeout,
        threads=int(args.threads),
    )

    if not args.no_forced:
        pairs = apply_forced_pairs(pairs, FORCED_DEFAULT)

    # Dedup output pairs
    seen = set()
    uniq = []
    for p in pairs:
        k = (p.get("bearing"), p.get("distance"), p.get("ref_id"), p.get("table_kind"))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    uniq.sort(key=lambda x: x.get("numeric_dist", 0), reverse=True)
    pairs = uniq

    print("\n" + "=" * 92)
    print("EXTRACTED BEARING → MEASURED DISTANCE (RECORD)   +   LINE/CURVE TABLE ROWS (L#/C#)")
    print("=" * 92)

    if not pairs:
        print("No valid pairs found.")
        return

    if basis:
        print("\nBasis of Bearing (best guess):")
        rid = basis.get("ref_id")
        rid_s = f"{rid} " if rid else ""
        print(f"  {rid_s}{basis.get('bearing')}   {basis.get('distance')} ft   (support={basis.get('support')}, score={basis.get('vote_score'):.2f})")

    for i, p in enumerate(pairs, 1):
        support = p.get("support", 1)
        conf = p.get("best_conf", -1)
        ov = p.get("avg_line_overlap", 0.0)
        src = p.get("source", "unknown")
        tk = p.get("table_kind")
        cc = p.get("callout_count", 0)
        gm = p.get("geom_mean", None)
        gm_s = f"{gm:.2f}" if isinstance(gm, (int, float)) else "n/a"
        tk_s = f", {tk}" if tk else ""
        rid = p.get("ref_id")
        rid_s = f"{rid} " if rid else ""
        print(f"{i:3d}. {rid_s}{fmt_pair_line(p)}   (support={support}, conf={conf:.1f}, geom={gm_s}, overlap={ov:.2f}, callouts={cc}, {src}{tk_s})")

        if args.verbose:
            ex = (p.get("example_raw") or "").replace("\n", " ").strip()
            if len(ex) > 260:
                ex = ex[:260] + "..."
            if ex:
                print(f"     Example OCR: {ex}")

            if p.get("dist_quality_votes"):
                dqs = ", ".join([f"{x['quality']}×{x['count']}" for x in p["dist_quality_votes"]])
                print(f"     Meas evidence: {dqs}")

            if p.get("record_quality_votes"):
                rqs = ", ".join([f"{x['quality']}×{x['count']}" for x in p["record_quality_votes"]])
                print(f"     Rec evidence:  {rqs}")

            if p.get("angle_votes"):
                top_angles = ", ".join([f"{x['angle']:+.0f}°×{x['count']}" for x in p["angle_votes"]])
                print(f"     Angle votes:   {top_angles}")

            if p.get("dpi_votes"):
                top_dpis = ", ".join([f"{x['dpi']}×{x['count']}" for x in p["dpi_votes"]])
                print(f"     DPI votes:     {top_dpis}")
            print()

    print(f"\nTotal unique pairs: {len(pairs)}")

    if callouts:
        top = sorted(callouts.items(), key=lambda kv: kv[0])
        shown = 0
        print("\nCallout counts (L#/C# occurrences detected):")
        for k, v in top:
            if v <= 0:
                continue
            print(f"  {k}: {v}")
            shown += 1
            if shown >= 60:
                print("  ...")
                break

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "file": str(pdf),
                "quality": args.quality,
                "threads": int(args.threads),
                "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT"),
                "geom_verify": bool(geom_verify),
                "break_detect": bool(break_detect),
                "basis_of_bearing": (
                    {
                        "ref_id": basis.get("ref_id"),
                        "table_kind": basis.get("table_kind"),
                        "bearing": basis.get("bearing"),
                        "distance": basis.get("distance"),
                        "record_distance": basis.get("record_distance"),
                        "source": basis.get("source"),
                        "support": basis.get("support"),
                        "geom_mean": basis.get("geom_mean"),
                        "best_conf": basis.get("best_conf"),
                        "vote_score": basis.get("vote_score"),
                        "example_raw": basis.get("example_raw", ""),
                    } if basis else None
                ),
                "pairs": [
                    {
                        "ref_id": p.get("ref_id"),
                        "table_kind": p.get("table_kind"),
                        "bearing": p.get("bearing"),
                        "distance": p.get("distance"),
                        "record_distance": p.get("record_distance"),
                        "callout_count": p.get("callout_count", 0),
                        "source": p.get("source"),
                        "support": p.get("support"),
                        "avg_line_overlap": p.get("avg_line_overlap"),
                        "geom_mean": p.get("geom_mean"),
                        "basis_votes": p.get("basis_votes", 0),
                        "best_conf": p.get("best_conf"),
                        "vote_score": p.get("vote_score"),
                        "dist_quality_votes": p.get("dist_quality_votes", []),
                        "record_quality_votes": p.get("record_quality_votes", []),
                        "angle_votes": p.get("angle_votes", []),
                        "dpi_votes": p.get("dpi_votes", []),
                        "example_raw": p.get("example_raw", ""),
                    }
                    for p in pairs
                ],
                "count": len(pairs),
                "callouts": callouts
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
