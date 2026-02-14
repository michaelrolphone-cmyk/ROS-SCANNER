#!/usr/bin/env python3
"""
ros_ultra_ocr.py — Ultra / Max-quality OCR for Record of Survey (scanned-image PDFs)

Primary goal:
  Extract bearings + measured distances (+ optional record distances in parentheses)
  from noisy ROS scans, including:
    - Callout text near linework (bearings/distances at arbitrary angles)
    - Line tables / curve tables (L1/L2..., C1/C2...)
    - Basis of Bearing (restore detection)
    - Heavy multipass + multi-angle + multi-DPI voting (hours OK)

This file is a full replacement that:
  ✅ Fixes PaddleOCR v3.x crash: removes show_log, supports use_textline_orientation, and normalizes outputs
  ✅ Keeps your max-quality rotation/ROI/multipass approach
  ✅ Uses ThreadPoolExecutor for per-angle work (default 16 threads)
  ✅ Avoids Tesseract shlex quote errors by ensuring configs contain NO quotes
  ✅ Adds optional DocTR + optional PaddleOCR engines (used if installed; fails soft if not)
  ✅ Retains pixel-dot decimal recovery (no digit-length guessing)
  ✅ Retains linework mask + inpaint variants, overlap penalties, clutter filters
  ✅ Restores Basis of Bearing detection (keyword-driven + voting)

Install minimum:
  pip install pytesseract pdf2image Pillow opencv-python-headless numpy

Optional (recommended) engines:
  pip install paddleocr
  pip install python-doctr[torch]   # DocTR

Usage (max quality, threaded):
  python ros_ultra_ocr.py file.pdf --quality max --threads 16 --output out.json --verbose

Disable optional engines:
  python ros_ultra_ocr.py file.pdf --no-paddle --no-doctr
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

# ---------------------------
# Core deps (required)
# ---------------------------
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
# Optional deps (best-effort)
# ---------------------------
_DOCTR_OK = False
_PADDLE_OK = False

try:
    # DocTR (optional)
    from doctr.models import ocr_predictor  # type: ignore
    from doctr.io import DocumentFile       # type: ignore
    _DOCTR_OK = True
except Exception:
    _DOCTR_OK = False

try:
    # PaddleOCR (optional)
    from paddleocr import PaddleOCR         # type: ignore
    _PADDLE_OK = True
except Exception:
    _PADDLE_OK = False

# ---------------------------
# PIL / large image safety
# ---------------------------
Image.MAX_IMAGE_PIXELS = 800_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Pixel budget controls (keep you from OOM when DPI is huge)
MAX_DET_PIXELS = 12_000_000         # low-res detection stage budget
MAX_ROI_PIXELS = 18_000_000         # per ROI crop budget (tiling splits if larger)
MAX_VARIANT_PIXELS = 18_000_000     # per preprocess variant scaling cap

# ---------------------------
# Linework filtering controls
# ---------------------------
LINE_OVERLAP_HARD_REJECT = 0.60   # if line mask covers >60% of token bbox, skip candidate
LINE_OVERLAP_PENALTY_WT  = 3.0    # score penalty weight during voting
LINE_INPAINT_RADII       = [2, 3, 4]  # try multiple line-inpaint radii in max mode

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

# Basis of bearing cues
BASIS_CUE_RE = re.compile(r"(?i)\b(basis\s+of\s+bearing|basis\s+of\s+bearings|b\.?o\.?b\.?)\b")

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

    # common OCR: degree sometimes 100+ or 200+ or 300+ from garbage prefix
    if deg > 90:
        if 100 <= deg < 190:
            deg -= 100
        elif 200 <= deg < 290:
            deg -= 200
        elif 300 <= deg < 390:
            deg -= 300

    # common OCR: minute/sec sometimes offset by 60
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

# ---------------------------
# Linework detection + cleanup
# ---------------------------
def _adaptive_inv(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

def build_line_mask(gray: np.ndarray) -> np.ndarray:
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
    Returns N = number of digit-components to the RIGHT of a dot blob, if a dot blob is detected.
    Used to reconstruct missing decimal points from pixel evidence (not digit-length guessing).
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

    # measured distance token may include "(124)" glued; strip it
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    if not s:
        return None, "invalid"

    # comma-as-decimal fix ONLY when clearly a decimal comma (NOT length based)
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

    # explicit decimal wins
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

    # pixel-dot evidence insertion
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

    # integer fallback
    try:
        v = float(digits)
        if min_ft <= v <= max_ft:
            return v, "integer"
    except ValueError:
        pass

    return None, "invalid"

# ---------------------------
# Rotation helpers
# ---------------------------
def rotate_keep_size(pil_img: Image.Image, angle_deg: float) -> Image.Image:
    a = float(angle_deg)
    if abs(a) < 1e-9:
        return pil_img

    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]

    # exact multiples fast path
    am = (a % 360.0)
    if abs(am - 90.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if (out.shape[1], out.shape[0]) != (w, h):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(out)
    if abs(am - 270.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        if (out.shape[1], out.shape[0]) != (w, h):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
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
# OCR: Tesseract (word bboxes)
# ---------------------------
def ocr_image_to_lines_with_words_tesseract(pil_img: Image.Image, config: str, timeout: int | None = None) -> list[dict]:
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
# OCR: PaddleOCR wrapper (FIXED for v3.x)
# ---------------------------
class PaddleOcrEngine:
    """
    PaddleOCR wrapper compatible with PaddleOCR 2.x and 3.x:
      - NO show_log arg (removed in 3.x)
      - uses use_textline_orientation when available; fallback to use_angle_cls
      - normalizes output to list-of-lines: [[quad, (text, score)], ...]
    """
    def __init__(self, lang="en"):
        # Reduce logs (best-effort)
        os.environ.setdefault("FLAGS_minlog_level", "3")
        os.environ.setdefault("GLOG_minloglevel", "3")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        if not _PADDLE_OK:
            raise RuntimeError("PaddleOCR not installed")

        self.ocr = None
        last_err = None

        # Try PaddleOCR 3.x preferred arg, then PaddleOCR 2.x arg, then none
        for extra in (
            {"use_textline_orientation": True},
            {"use_angle_cls": True},
            {},
        ):
            try:
                self.ocr = PaddleOCR(lang=lang, **extra)
                self._init_extra = extra
                break
            except Exception as e:
                last_err = e

        if self.ocr is None:
            raise last_err

    def ocr_image(self, pil_img: Image.Image, timeout: int | None = None):
        import numpy as np

        img = np.array(pil_img.convert("RGB"))
        try:
            raw = self.ocr.ocr(img, cls=False)
        except TypeError:
            raw = self.ocr.ocr(img)

        # PaddleOCR 3.x returns OCRResult objects with .json payload
        if raw and hasattr(raw[0], "json"):
            page_lines = []
            for res_obj in raw:
                payload = res_obj.json
                if isinstance(payload, dict) and "res" in payload:
                    payload = payload["res"]

                texts = payload.get("rec_texts", []) or []
                scores = payload.get("rec_scores", []) or []
                polys = payload.get("rec_polys", None)

                try:
                    scores = scores.tolist()
                except Exception:
                    pass
                if polys is not None:
                    try:
                        polys = polys.tolist()
                    except Exception:
                        pass

                if polys is None:
                    polys = payload.get("dt_polys", None)
                    if polys is not None:
                        try:
                            polys = polys.tolist()
                        except Exception:
                            pass

                n = min(len(texts), len(scores), len(polys) if polys is not None else len(texts))
                for i in range(n):
                    quad = polys[i] if polys is not None else [[0, 0], [0, 0], [0, 0], [0, 0]]
                    txt = texts[i]
                    sc = float(scores[i]) if i < len(scores) else 0.0
                    page_lines.append([quad, (txt, sc)])

            return [page_lines]

        # PaddleOCR 2.x returns list-of-lines already
        return raw if raw is not None else [[]]

# ---------------------------
# OCR: DocTR wrapper (optional, uses torch device if available)
# ---------------------------
class DoctrOcrEngine:
    """
    DocTR OCR wrapper.
    Produces list-of-lines similar to Paddle:
      [[quad, (text, score)], ...]
    """
    def __init__(self):
        if not _DOCTR_OK:
            raise RuntimeError("DocTR not installed")

        # choose device: MPS > CUDA > CPU (best-effort)
        self.device = "cpu"
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            self.device = "cpu"

        # high-quality default predictor
        self.predictor = ocr_predictor(pretrained=True).to(self.device)

    def ocr_image(self, pil_img: Image.Image):
        # DocumentFile wants list of images; easiest: convert PIL->numpy and use from_images
        arr = np.array(pil_img.convert("RGB"))
        doc = DocumentFile.from_images([arr])
        res = self.predictor(doc)
        exported = res.export()

        # exported structure:
        # {'pages':[{'blocks':[{'lines':[{'words':[{'value':...,'geometry':((x1,y1),(x2,y2))}, ...]}]}]}]}]}
        page = exported["pages"][0]
        H, W = pil_img.size[1], pil_img.size[0]

        out_lines = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", [])
                if not words:
                    continue
                text = " ".join(w.get("value", "") for w in words).strip()
                if not text:
                    continue

                # geometry is relative (0..1); convert to px bbox
                xs = []
                ys = []
                confs = []
                for w in words:
                    geom = w.get("geometry", None)
                    conf = w.get("confidence", None)
                    if isinstance(conf, (int, float)):
                        confs.append(float(conf))
                    if geom and isinstance(geom, (list, tuple)) and len(geom) == 2:
                        (x1, y1), (x2, y2) = geom
                        xs += [x1, x2]
                        ys += [y1, y2]
                if not xs or not ys:
                    continue

                x1 = int(min(xs) * W)
                y1 = int(min(ys) * H)
                x2 = int(max(xs) * W)
                y2 = int(max(ys) * H)

                quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                score = float(np.mean(confs)) if confs else 0.5
                out_lines.append([quad, (text, score)])

        return [out_lines]

# ---------------------------
# Geometry helpers
# ---------------------------
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
# ROI detection: bearing/distance callouts (tesseract on downscaled binarized page)
# ---------------------------
def detect_bearing_distance_rois(page_img: Image.Image, roi_pad: int = 55, max_rois: int = 60, tess_timeout: int | None = None) -> list[tuple[int, int, int, int]]:
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

    # IMPORTANT: no quotes anywhere in config (prevents shlex "No closing quotation")
    det_cfg = "--oem 3 --psm 11 -c preserve_interword_spaces=1"
    lines = ocr_image_to_lines_with_words_tesseract(det_img, det_cfg, timeout=tess_timeout)

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

        x1 -= pad_s; y1 -= pad_s; x2 += pad_s; y2 += pad_s
        x1, y1, x2, y2 = clamp_rect((x1, y1, x2, y2), sW, sH)

        if scale != 1.0:
            ox1 = int(x1 / scale); oy1 = int(y1 / scale)
            ox2 = int(x2 / scale); oy2 = int(y2 / scale)
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
# ROI detection: table regions
# ---------------------------
TABLE_KEYWORDS = {"table", "line", "curve", "bearing", "distance", "chord", "delta", "radius", "length", "arc"}

def detect_table_rois(page_img: Image.Image, max_tables: int = 10, tess_timeout: int | None = None) -> list[tuple[str, tuple[int, int, int, int]]]:
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

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()°.,-"
    cfg = f"--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(pil, config=cfg, lang="eng", output_type=Output.DICT, timeout=tess_timeout)

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
            ox1 = int(x1 / scale); oy1 = int(y1 / scale)
            ox2 = int(x2 / scale); oy2 = int(y2 / scale)
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

def parse_record_distance_from_following(words: list[dict], dist_idx: int, dot_ref_img: Image.Image | None) -> tuple[float | None, str | None, str | None]:
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
# Extract label L#/C# from line words
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

# ---------------------------
# Extract pairs from a tesseract line using word bboxes + overlap + clutter rules
# ---------------------------
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

            # find the next plausible distance token
            dist_idx = None
            for k in range(j, min(len(texts), j + 14)):
                tok = (texts[k] or "").strip()
                if not tok:
                    continue
                if re.search(r"[A-Za-z]", tok):
                    continue

                # allow "(124)" etc but we want measured token first
                if "(" in tok or ")" in tok:
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

            # reject acreage/area contexts
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
                "basis_hint": bool(BASIS_CUE_RE.search(line_text or "")),
            })
            break

    return out

# ---------------------------
# Parse pairs from a simple text line (Paddle/DocTR lines lack word boxes)
# ---------------------------
def extract_pairs_from_textline_simple(line_text: str, require_label: bool = False) -> list[dict]:
    """
    Lower-fidelity parser used for Paddle/DocTR lines:
      - no per-word dot evidence
      - weaker record-distance extraction
      - still useful for recall on skewed/rotated annotations
    """
    out = []
    if not line_text:
        return out
    if is_metadata_line(line_text):
        return out

    # optional label requirement for tables
    ref_id = None
    m = LINE_ID_RE.search(line_text) or CURVE_ID_RE.search(line_text)
    if m:
        ref_id = canon_label(m.group(0))
    if require_label and not ref_id:
        return out

    # find bearing
    bm = BEARING_RE.search(line_text)
    if not bm:
        return out

    b_canon = parse_bearing_to_canon(bm.group(0))
    if not b_canon:
        return out

    # find distance after bearing
    tail = line_text[bm.end():]
    # capture something like 123.45 or 12345 or 123.45(124)
    dm = re.search(r"([0-9][0-9,]{0,12}(?:[.·•∙⋅][0-9]{1,4})?)", tail)
    if not dm:
        return out

    d_raw = dm.group(1)
    meas_val, meas_q = parse_distance_with_quality(d_raw, dot_digits_right=None, min_ft=1.0)
    if meas_val is None:
        return out

    # record distance: (124) later in tail
    rec_val = None
    rec_str = None
    rec_q = None
    pm = re.search(r"\(([^)]+)\)", tail)
    if pm:
        inside2 = re.sub(r"[^0-9.,·•∙⋅]", "", pm.group(1) or "")
        if inside2:
            rv, rq = parse_distance_with_quality(inside2, dot_digits_right=None, min_ft=0.0)
            if rv is not None:
                rec_val = float(rv)
                rec_q = rq
                rec_str = f"{int(rv)}" if abs(rv - round(rv)) < 1e-9 else f"{rv:.2f}".rstrip("0").rstrip(".")

    out.append({
        "canon_bearing": b_canon,
        "numeric_dist": float(meas_val),
        "distance": f"{meas_val:.2f}",
        "dist_quality": meas_q,
        "dist_raw": d_raw,
        "dist_bbox": None,
        "line_overlap": 0.0,

        "record_numeric_dist": rec_val,
        "record_distance": rec_str,
        "record_quality": rec_q,

        "ref_id": ref_id,
        "basis_hint": bool(BASIS_CUE_RE.search(line_text or "")),
    })
    return out

# ---------------------------
# Multipass OCR on ROI (multi-engine)
# ---------------------------
def multipass_ocr_on_roi(
    roi_img: Image.Image,
    ocr_configs: list[tuple[str, str]],
    table_configs: list[tuple[str, str]],
    variants: list[dict],
    require_label: bool,
    tess_timeout: int | None,
    engines: dict,
) -> list[dict]:
    """
    Returns list of candidate dicts from:
      - Tesseract (full word boxes + dot evidence + overlap penalties)
      - optional PaddleOCR (simple parsing)
      - optional DocTR (simple parsing)
    """
    candidates = []

    # --- Tesseract passes (highest-fidelity for decimal dots / overlap)
    for v in variants:
        proc, aux = preprocess_variant(roi_img, v)
        dot_ref_img = proc if aux.get("dot_ref", False) else None
        line_mask = aux.get("line_mask", None)

        for cfg, desc in ocr_configs:
            lines = ocr_image_to_lines_with_words_tesseract(proc, cfg, timeout=tess_timeout)
            for ln in lines:
                for c in extract_pairs_from_line_words(
                    line_text=ln["text"],
                    line_words=ln["words"],
                    dot_ref_img=dot_ref_img,
                    line_mask=line_mask,
                    min_ft=1.0,
                    require_label=require_label,
                ):
                    theta = bearing_theta_deg(c["canon_bearing"])
                    candidates.append({
                        "canon_bearing": c["canon_bearing"],
                        "distance": c["distance"],
                        "numeric_dist": c["numeric_dist"],
                        "theta": theta,
                        "conf": float(ln["conf"]) if ln["conf"] is not None else -1.0,

                        "engine": "tesseract",
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

    # --- PaddleOCR (optional)
    paddle = engines.get("paddle")
    if paddle is not None:
        try:
            out = paddle.ocr_image(roi_img, timeout=tess_timeout)
            lines = out[0] if out else []
            for quad, (text, score) in lines:
                text = (text or "").strip()
                if not text:
                    continue
                for c in extract_pairs_from_textline_simple(text, require_label=require_label):
                    candidates.append({
                        "canon_bearing": c["canon_bearing"],
                        "distance": c["distance"],
                        "numeric_dist": c["numeric_dist"],
                        "theta": bearing_theta_deg(c["canon_bearing"]),
                        "conf": float(score) * 100.0,

                        "engine": "paddle",
                        "pass_name": "paddle",
                        "config_desc": "paddle",
                        "raw_line": text,

                        "dist_quality": c.get("dist_quality"),
                        "dist_raw": c.get("dist_raw"),
                        "dist_bbox": None,
                        "line_overlap": 0.0,
                        "line_clean": False,

                        "record_numeric_dist": c.get("record_numeric_dist"),
                        "record_distance": c.get("record_distance"),
                        "record_quality": c.get("record_quality"),

                        "ref_id": c.get("ref_id"),
                        "basis_hint": bool(c.get("basis_hint", False)),
                    })
        except Exception:
            pass

    # --- DocTR (optional)
    doctr = engines.get("doctr")
    if doctr is not None:
        try:
            out = doctr.ocr_image(roi_img)
            lines = out[0] if out else []
            for quad, (text, score) in lines:
                text = (text or "").strip()
                if not text:
                    continue
                for c in extract_pairs_from_textline_simple(text, require_label=require_label):
                    candidates.append({
                        "canon_bearing": c["canon_bearing"],
                        "distance": c["distance"],
                        "numeric_dist": c["numeric_dist"],
                        "theta": bearing_theta_deg(c["canon_bearing"]),
                        "conf": float(score) * 100.0,

                        "engine": "doctr",
                        "pass_name": "doctr",
                        "config_desc": "doctr",
                        "raw_line": text,

                        "dist_quality": c.get("dist_quality"),
                        "dist_raw": c.get("dist_raw"),
                        "dist_bbox": None,
                        "line_overlap": 0.0,
                        "line_clean": False,

                        "record_numeric_dist": c.get("record_numeric_dist"),
                        "record_distance": c.get("record_distance"),
                        "record_quality": c.get("record_quality"),

                        "ref_id": c.get("ref_id"),
                        "basis_hint": bool(c.get("basis_hint", False)),
                    })
        except Exception:
            pass

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

        dpis = {c.get("dpi") for c in reps}
        angs = {c.get("angle") for c in reps}
        engines = {c.get("engine") for c in reps}
        if len(dpis) >= 2:
            score += 0.6
        if len(angs) >= 3:
            score += 0.5
        if len(engines) >= 2:
            score += 0.5

        # basis hint bonus if many reps had basis cue
        bh = sum(1 for c in reps if c.get("basis_hint"))
        if bh >= 2:
            score += 0.8

        return score, mean_conf, dec_bonus, overlap_avg, reps

    best_key = None
    best_tuple = (-1e18, -1e18, -1e18, 1e18, [])
    for k in top_keys:
        sc, mc, db, ov, reps = score_key(k)
        if (sc, mc, db, -ov) > (best_tuple[0], best_tuple[1], best_tuple[2], -best_tuple[3]):
            best_tuple = (sc, mc, db, ov, reps)
            best_key = k

    score, mean_conf, dec_bonus, overlap_avg, reps = best_tuple
    reps.sort(key=lambda x: x["conf"], reverse=True)
    rep = reps[0]

    bearing, dist_str = best_key
    dist_val = float(dist_str)

    # record distance voting
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
    passes = Counter((c.get("engine"), c.get("pass_name"), c.get("config_desc")) for c in reps)
    angles = Counter(c.get("angle", 0) for c in reps)
    dpis = Counter(c.get("dpi", 0) for c in reps)
    linec = Counter("lineclean" if c.get("line_clean") else "raw" for c in reps)

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

        "basis_votes": int(basis_votes),

        "dist_quality_votes": [{"quality": q, "count": n} for q, n in dq.most_common(8)],
        "record_quality_votes": [{"quality": q, "count": n} for q, n in rq.most_common(8)] if rq else [],
        "line_mode_votes": [{"mode": m, "count": n} for m, n in linec.most_common(4)],
        "example_raw": rep.get("raw_line", ""),
        "source": rep.get("source", "voted_roi"),
        "angle_votes": [{"angle": a, "count": n} for a, n in angles.most_common(8)],
        "dpi_votes": [{"dpi": d, "count": n} for d, n in dpis.most_common(8)],
        "pass_breakdown": [{"engine": e, "pass": p, "config": cfg, "count": n} for (e, p, cfg), n in passes.most_common(12)],
    }

# ---------------------------
# Callout scanning (L# / C# occurrences detected) — tesseract
# ---------------------------
def scan_label_callouts(page_img: Image.Image, angles: list[float], tess_timeout: int | None) -> dict[str, int]:
    counts = defaultdict(set)

    whitelist = "LC0123456789lc-"
    cfg = f"--oem 3 --psm 11 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}"
    for a in angles:
        rot = rotate_keep_size(page_img, a)
        W, H = rot.size

        scale = min(1.0, (MAX_DET_PIXELS / (W * H)) ** 0.5) if W * H > 0 else 1.0
        if scale < 1.0:
            small = rot.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BICUBIC)
        else:
            small = rot

        data = pytesseract.image_to_data(small, config=cfg, lang="eng", output_type=Output.DICT, timeout=tess_timeout)
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

            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            bx = int(x / 20); by = int(y / 20)
            counts[lab].add((a, bx, by, int(w / 10), int(h / 10)))

    return {lab: len(s) for lab, s in counts.items()}

# ---------------------------
# Basis of Bearing detection (restored)
# ---------------------------
def detect_basis_candidates_on_page(page_img: Image.Image, angles: list[float], tess_timeout: int | None) -> list[dict]:
    """
    Lightweight pass across a few angles to find 'BASIS OF BEARING' lines and extract the nearby bearing/distance.
    Returns list of candidate dicts (same schema as normal candidates but marked basis_candidate=True).
    """
    cands = []
    cfg = "--oem 3 --psm 11 -c preserve_interword_spaces=1"

    for a in angles:
        rot = rotate_keep_size(page_img, a)
        W, H = rot.size

        # downscale for speed
        scale = min(1.0, (MAX_DET_PIXELS / (W * H)) ** 0.5) if W * H > 0 else 1.0
        if scale < 1.0:
            small = rot.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BICUBIC)
        else:
            small = rot

        # binarize for better keyword OCR
        arr = np.array(small.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 11)
        pil = Image.fromarray(bw)

        lines = ocr_image_to_lines_with_words_tesseract(pil, cfg, timeout=tess_timeout)
        for ln in lines:
            t = (ln["text"] or "").strip()
            if not t:
                continue
            if not BASIS_CUE_RE.search(t):
                continue

            # search within this line for bearing/distance; else look at next few lines is expensive; keep local
            bm = BEARING_RE.search(t)
            if not bm:
                continue
            b = parse_bearing_to_canon(bm.group(0))
            if not b:
                continue
            tail = t[bm.end():]
            dm = re.search(r"([0-9][0-9,]{0,12}(?:[.·•∙⋅][0-9]{1,4})?)", tail)
            if not dm:
                continue
            d_raw = dm.group(1)
            dv, dq = parse_distance_with_quality(d_raw, dot_digits_right=None, min_ft=1.0)
            if dv is None:
                continue

            cands.append({
                "canon_bearing": b,
                "distance": f"{dv:.2f}",
                "numeric_dist": float(dv),
                "theta": bearing_theta_deg(b),
                "conf": float(ln["conf"]) if ln["conf"] is not None else -1.0,
                "engine": "tesseract",
                "pass_name": "basis_scan",
                "config_desc": "basis_scan",
                "raw_line": t,
                "dist_quality": dq,
                "dist_raw": d_raw,
                "dist_bbox": None,
                "line_overlap": 0.0,
                "line_clean": False,
                "record_numeric_dist": None,
                "record_distance": None,
                "record_quality": None,
                "ref_id": None,
                "basis_hint": True,
                "basis_candidate": True,
                "angle": float(a),
            })

    return cands

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
            "fallback_tile_angles": {0.0, 90.0, 180.0, 270.0},
            "callout_angles": [0.0, 90.0, 180.0, 270.0],
            "basis_angles": [0.0, 90.0, 180.0, 270.0],
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
            "fallback_tile_angles": set(angles_dense_360(30.0)),
            "callout_angles": [0.0, 90.0, 180.0, 270.0],
            "basis_angles": [0.0, 90.0, 180.0, 270.0],
            "min_support": 2,
            "dist_tol": 0.25,
            "bear_tol_deg": 0.40,
            "max_rois": 40,
            "roi_pad": 55,
        }

    # max / exhaustive
    return {
        "dpi_list": [500, 650, 800],
        "angles": angles_dense_360(5.0),
        "fallback_tile_angles": set(angles_dense_360(15.0)),
        "callout_angles": angles_dense_360(30.0),
        "basis_angles": [0.0, 90.0, 180.0, 270.0],
        "min_support": 3,
        "dist_tol": 0.20,
        "bear_tol_deg": 0.35,
        "max_rois": 70,
        "roi_pad": 60,
    }

# ---------------------------
# Per-angle worker
# ---------------------------
def _process_one_angle(
    page_img: Image.Image,
    page_idx: int,
    dpi: int,
    angle: float,
    fallback_tile_angles: set[float],
    roi_pad: int,
    max_rois: int,
    enable_tables: bool,
    debug_dir: Path | None,
    ocr_configs: list[tuple[str, str]],
    table_configs: list[tuple[str, str]],
    variants: list[dict],
    tess_timeout: int | None,
    engines: dict,
) -> tuple[list[dict], list[dict]]:
    rot_img = rotate_keep_size(page_img, angle)
    W, H = rot_img.size

    rois = detect_bearing_distance_rois(rot_img, roi_pad=int(roi_pad), max_rois=int(max_rois), tess_timeout=tess_timeout)
    tiled_mode = False
    if not rois:
        if float(angle) in fallback_tile_angles:
            tiled_mode = True
            rois = tile_rect((0, 0, W, H), W, H, max_pixels=MAX_ROI_PIXELS, overlap=170)
        else:
            rois = []

    roi_candidates: list[dict] = []
    table_candidates: list[dict] = []

    roi_counter = 0
    for r in rois:
        tiles = tile_rect(r, W, H, max_pixels=MAX_ROI_PIXELS, overlap=160)
        for tr in tiles:
            roi_counter += 1
            x1, y1, x2, y2 = tr
            crop = rot_img.crop((x1, y1, x2, y2))

            if debug_dir:
                crop_path = debug_dir / f"p{page_idx:02d}_dpi{dpi}_a{int(angle):03d}_roi{roi_counter:04d}_{x1}_{y1}_{x2}_{y2}.png"
                try:
                    crop.save(crop_path)
                except Exception:
                    pass

            cands = multipass_ocr_on_roi(
                crop,
                ocr_configs=ocr_configs,
                table_configs=table_configs,
                variants=variants,
                require_label=False,
                tess_timeout=tess_timeout,
                engines=engines,
            )

            for c in cands:
                c["page"] = page_idx
                c["dpi"] = int(dpi)
                c["angle"] = float(angle)
                c["roi"] = roi_counter
                c["roi_bbox"] = (x1, y1, x2, y2)
                c["table_kind"] = None
                c["source"] = "roi"

                if tiled_mode:
                    roi_cy = (y1 + y2) / 2.0
                    c["_rej_hf"] = bool(roi_cy <= HEADER_FOOTER_STRIP_FRAC * H or roi_cy >= (1.0 - HEADER_FOOTER_STRIP_FRAC) * H)
                else:
                    c["_rej_hf"] = False

            cands = [c for c in cands if not c.get("_rej_hf")]
            roi_candidates.extend(cands)

    if enable_tables:
        table_rois = detect_table_rois(rot_img, max_tables=10, tess_timeout=tess_timeout)
        for tkind, tr in table_rois:
            tx1, ty1, tx2, ty2 = tr
            tcrop = rot_img.crop((tx1, ty1, tx2, ty2))

            cands = multipass_ocr_on_roi(
                tcrop,
                ocr_configs=table_configs,
                table_configs=table_configs,
                variants=variants,
                require_label=True,
                tess_timeout=tess_timeout,
                engines=engines,
            )

            for c in cands:
                rid = c.get("ref_id")
                if not rid:
                    continue
                if tkind == "line_table" and not rid.startswith("L"):
                    continue
                if tkind == "curve_table" and not rid.startswith("C"):
                    continue

                raw = (c.get("raw_line") or "")
                c["curve_hint"] = bool(CURVE_HINT_RE.search(raw))

                c["page"] = page_idx
                c["dpi"] = int(dpi)
                c["angle"] = float(angle)
                c["roi"] = "table"
                c["roi_bbox"] = (tx1, ty1, tx2, ty2)
                c["table_kind"] = tkind
                c["source"] = "table_roi"

                table_candidates.append(c)

    return roi_candidates, table_candidates

# ---------------------------
# Extraction: PDF
# ---------------------------
def extract_pairs_from_pdf(
    pdf_path: Path,
    quality: str,
    dpi_list: list[int] | None,
    angles: list[float] | None,
    fallback_tile_angles: set[float] | None,
    callout_angles: list[float] | None,
    basis_angles: list[float] | None,
    roi_pad: int | None,
    max_rois: int | None,
    dist_tol: float | None,
    bear_tol_deg: float | None,
    min_support: int | None,
    enable_lineclean: bool,
    enable_tables: bool,
    debug_dir: Path | None,
    tess_timeout: int | None,
    threads: int,
    engines: dict,
) -> tuple[list[dict], dict[str, int], dict | None]:

    prof = quality_profile(quality)

    dpi_list = dpi_list if dpi_list else prof["dpi_list"]
    angles = angles if angles else prof["angles"]
    fallback_tile_angles = fallback_tile_angles if fallback_tile_angles else prof["fallback_tile_angles"]
    callout_angles = callout_angles if callout_angles else prof["callout_angles"]
    basis_angles = basis_angles if basis_angles else prof["basis_angles"]
    roi_pad = roi_pad if roi_pad is not None else prof["roi_pad"]
    max_rois = max_rois if max_rois is not None else prof["max_rois"]
    dist_tol = dist_tol if dist_tol is not None else prof["dist_tol"]
    bear_tol_deg = bear_tol_deg if bear_tol_deg is not None else prof["bear_tol_deg"]
    min_support = min_support if min_support is not None else prof["min_support"]

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: no quotes in whitelist; avoid shlex failures
    whitelist = "0123456789NSEWnsewLC.,-()°oº"
    common = (
        f"-c preserve_interword_spaces=1 "
        f"-c classify_bln_numeric_mode=1 "
        f"-c tessedit_enable_dict_correction=0 "
        f"-c load_system_dawg=0 -c load_freq_dawg=0 "
        f"-c user_defined_dpi=600 "
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
    print(f"Angles={len(angles)} rotations  fallback_tile_angles={len(fallback_tile_angles)}  callout_angles={len(callout_angles)}")
    print(f"Variants={len(variants)}  OCR configs={len(ocr_configs)}  Table configs={len(table_configs)}")
    print(f"Parallel angle threads={threads}")

    basis_cands_all = []

    for page_idx in range(1, page_count + 1):
        print(f"\n=== Page {page_idx}/{page_count} ===")

        for dpi in dpi_list:
            print(f"  Render @ {dpi} DPI...")
            imgs = convert_from_path(str(pdf_path), dpi=int(dpi), first_page=page_idx, last_page=page_idx)
            if not imgs:
                continue
            page_img = imgs[0]

            # callouts for line/curve references
            callouts = scan_label_callouts(page_img, callout_angles, tess_timeout=tess_timeout)
            for k, v in callouts.items():
                global_callouts[k] += v

            # basis scan (lightweight)
            basis_cands = detect_basis_candidates_on_page(page_img, basis_angles, tess_timeout=tess_timeout)
            for bc in basis_cands:
                bc["page"] = page_idx
                bc["dpi"] = int(dpi)
            basis_cands_all.extend(basis_cands)

            futures = []
            with ThreadPoolExecutor(max_workers=threads) as ex:
                for angle in angles:
                    futures.append(ex.submit(
                        _process_one_angle,
                        page_img, page_idx, int(dpi), float(angle),
                        fallback_tile_angles, int(roi_pad), int(max_rois),
                        bool(enable_tables), debug_dir,
                        ocr_configs, table_configs, variants, tess_timeout, engines
                    ))

                for fut in as_completed(futures):
                    roi_cands, tbl_cands = fut.result()
                    all_candidates.extend(roi_cands)
                    table_candidates.extend(tbl_cands)

            print(f"  candidates so far: roi={len(all_candidates)} table={len(table_candidates)} callouts={len(global_callouts)} basis={len(basis_cands_all)}")

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

    # basis winner (separate)
    basis_winner = None
    if basis_cands_all:
        # cluster/vote basis candidates too
        bclusters = cluster_candidates(basis_cands_all, dist_tol=0.50, bear_tol_deg=0.6)
        bw = [pick_cluster_winner(cl) for cl in bclusters]
        bw.sort(key=lambda x: (x.get("basis_votes", 0), x.get("support", 1), x.get("vote_score", -1e9), x.get("best_conf", -1)), reverse=True)
        basis_winner = bw[0]
        basis_winner["source"] = "basis_of_bearing"
        basis_winner["table_kind"] = None
        basis_winner["ref_id"] = None

    # dedupe bests
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

    # attach callout counts
    for r in out:
        rid = r.get("ref_id")
        r["callout_count"] = int(global_callouts.get(rid, 0)) if rid else 0

    out.sort(key=lambda x: x.get("numeric_dist", 0), reverse=True)
    return out, dict(global_callouts), basis_winner

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
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract bearings & distances from ROS scanned PDFs (ultra/max quality)")
    parser.add_argument("file", help="Path to PDF")

    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="max",
                        help="Quality preset. 'max' is hours-OK exhaustive mode (default).")

    parser.add_argument("--threads", type=int, default=16,
                        help="Thread pool size for parallel angle processing (default 16).")

    parser.add_argument("--omp-thread-limit", type=int, default=None,
                        help="Set OMP_THREAD_LIMIT (recommended 1 when using many threads). If not set and --threads>1, defaults to 1 unless already defined in env.")

    parser.add_argument("--dpi-list", default="", help="Override DPI list (comma-separated), e.g. 500,650,800")
    parser.add_argument("--angles", default="", help="Override angles list (comma-separated degrees). Empty uses quality preset.")
    parser.add_argument("--fallback-tile-angles", default="", help="Override fallback tile angles list (comma-separated). Empty uses preset.")
    parser.add_argument("--callout-angles", default="", help="Override callout angles list (comma-separated). Empty uses preset.")
    parser.add_argument("--basis-angles", default="", help="Override basis scan angles (comma-separated). Empty uses preset.")

    parser.add_argument("--roi-pad", type=int, default=None)
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--min-support", type=int, default=None)
    parser.add_argument("--dist-tol", type=float, default=None)
    parser.add_argument("--bear-tol-deg", type=float, default=None)

    parser.add_argument("--debug-rois", help="Directory to save ROI crops")
    parser.add_argument("--no-lineclean", action="store_true", help="Disable linework inpaint passes")
    parser.add_argument("--no-tables", action="store_true", help="Disable line/curve table extraction")

    parser.add_argument("--no-paddle", action="store_true", help="Disable PaddleOCR engine even if installed")
    parser.add_argument("--no-doctr", action="store_true", help="Disable DocTR engine even if installed")

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
    fallback_tile_angles = set(parse_float_list(args.fallback_tile_angles)) if args.fallback_tile_angles.strip() else None
    callout_angles = parse_float_list(args.callout_angles) if args.callout_angles.strip() else None
    basis_angles = parse_float_list(args.basis_angles) if args.basis_angles.strip() else None

    debug_dir = Path(args.debug_rois) if args.debug_rois else None
    tess_timeout = int(args.tess_timeout) if int(args.tess_timeout) > 0 else None

    # init optional engines
    engines = {"paddle": None, "doctr": None}
    if not args.no_paddle and _PADDLE_OK:
        try:
            engines["paddle"] = PaddleOcrEngine(lang="en")
            print("[engine] PaddleOCR enabled")
        except Exception as e:
            print(f"[engine] PaddleOCR init failed; continuing without it: {e}")
            engines["paddle"] = None
    else:
        if args.no_paddle:
            print("[engine] PaddleOCR disabled by flag")
        elif not _PADDLE_OK:
            print("[engine] PaddleOCR not installed")

    if not args.no_doctr and _DOCTR_OK:
        try:
            engines["doctr"] = DoctrOcrEngine()
            print(f"[engine] DocTR enabled (device={engines['doctr'].device})")
        except Exception as e:
            print(f"[engine] DocTR init failed; continuing without it: {e}")
            engines["doctr"] = None
    else:
        if args.no_doctr:
            print("[engine] DocTR disabled by flag")
        elif not _DOCTR_OK:
            print("[engine] DocTR not installed")

    pairs, callouts, basis = extract_pairs_from_pdf(
        pdf_path=pdf,
        quality=args.quality,
        dpi_list=dpi_list,
        angles=angles,
        fallback_tile_angles=fallback_tile_angles,
        callout_angles=callout_angles,
        basis_angles=basis_angles,
        roi_pad=args.roi_pad,
        max_rois=args.max_rois,
        dist_tol=args.dist_tol,
        bear_tol_deg=args.bear_tol_deg,
        min_support=args.min_support,
        enable_lineclean=not args.no_lineclean,
        enable_tables=not args.no_tables,
        debug_dir=debug_dir,
        tess_timeout=tess_timeout,
        threads=int(args.threads),
        engines=engines,
    )

    # Dedup
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

    if basis:
        print("\nBASIS OF BEARING (best guess):")
        print(f"  {basis.get('bearing')}   {basis.get('distance')} ft   (support={basis.get('support')}, conf={basis.get('best_conf', -1):.1f})")
        if args.verbose:
            ex = (basis.get("example_raw") or "").replace("\n", " ").strip()
            if len(ex) > 260:
                ex = ex[:260] + "..."
            if ex:
                print(f"    Example OCR: {ex}")

    if not pairs:
        print("\nNo valid pairs found.")
        return

    for i, p in enumerate(pairs, 1):
        support = p.get("support", 1)
        conf = p.get("best_conf", -1)
        ov = p.get("avg_line_overlap", 0.0)
        src = p.get("source", "unknown")
        tk = p.get("table_kind")
        cc = p.get("callout_count", 0)
        tk_s = f", {tk}" if tk else ""
        rid = p.get("ref_id")
        rid_s = f"{rid} " if rid else ""
        print(f"{i:3d}. {rid_s}{fmt_pair_line(p)}   (support={support}, conf={conf:.1f}, overlap={ov:.2f}, callouts={cc}, {src}{tk_s})")

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

            if p.get("pass_breakdown"):
                pb = ", ".join([f"{x['engine']}:{x['pass']}×{x['count']}" for x in p["pass_breakdown"][:6]])
                print(f"     Pass votes:    {pb}")

            if p.get("angle_votes"):
                top_angles = ", ".join([f"{x['angle']:+.0f}°×{x['count']}" for x in p["angle_votes"][:6]])
                print(f"     Angle votes:   {top_angles}")

            if p.get("dpi_votes"):
                top_dpis = ", ".join([f"{x['dpi']}×{x['count']}" for x in p["dpi_votes"][:6]])
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
                "basis_of_bearing": {
                    "bearing": basis.get("bearing"),
                    "distance": basis.get("distance"),
                    "support": basis.get("support"),
                    "best_conf": basis.get("best_conf"),
                    "example_raw": basis.get("example_raw"),
                } if basis else None,
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
                        "best_conf": p.get("best_conf"),
                        "vote_score": p.get("vote_score"),
                        "basis_votes": p.get("basis_votes", 0),
                        "dist_quality_votes": p.get("dist_quality_votes", []),
                        "record_quality_votes": p.get("record_quality_votes", []),
                        "angle_votes": p.get("angle_votes", []),
                        "dpi_votes": p.get("dpi_votes", []),
                        "pass_breakdown": p.get("pass_breakdown", []),
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
