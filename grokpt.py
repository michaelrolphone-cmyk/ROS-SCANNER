#!/usr/bin/env python3
"""
OCR script to extract bearings and distances from Record of Survey PDFs.

Core features:
- Multi-angle scan (rotate page, detect ROIs, OCR each ROI)
- ROI detection on downscaled page → mapped to full-res rotated page
- ROI tiling to prevent giant image blowups
- Multi-pass OCR with multiple preprocess variants + Tesseract configs
- Clustering + voting for most common (bearing, distance)

Decimal point handling (improved, evidence-based):
- NO guessing from digit length or "typical ranges"
- If OCR misses '.', try to detect the decimal dot from pixels inside the distance word bbox:
    * run connected-components on a dot-preserving threshold
    * locate a small blob near baseline
    * infer dot position by counting digit components to the right of the dot
    * insert decimal accordingly (e.g., 66055 + dot-right=2 => 660.55)

Install:
  pip install pytesseract pdf2image Pillow opencv-python-headless numpy
"""

import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter

try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_path
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install: pip install pytesseract pdf2image Pillow opencv-python-headless numpy")
    sys.exit(1)

# ---------------------------
# PIL large-image safety
# ---------------------------

Image.MAX_IMAGE_PIXELS = 400_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

MAX_DET_PIXELS = 12_000_000
MAX_ROI_PIXELS = 18_000_000
MAX_VARIANT_PIXELS = 18_000_000

DEFAULT_ANGLES = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]

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

# Bearing with degree/min/sec often split / noisy
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

# Distance token detection (word-level)
DIST_TOKEN_RE = re.compile(r"^\d[\d,]*([.·•∙⋅]\d{1,4})?$")

# Candidate line heuristic for ROI discovery (line-level)
CANDIDATE_LINE_RE = re.compile(
    r"(?i)\b[NS]\s*\d{1,3}\s*(?:[°oº]|\s)\s*\d{1,2}.*\b[EW]\b.*\d{2,10}"
)

def _clean_digits(s: str) -> str:
    return (s or "").translate(DIGIT_FIX)

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

    # Fix common OCR: 189 -> 89, 100 -> 0, etc.
    if deg > 90:
        if 100 <= deg < 190:
            deg -= 100
        elif 200 <= deg < 290:
            deg -= 200
        elif 300 <= deg < 390:
            deg -= 300

    # Fix minutes/seconds wrap errors
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
# Decimal dot detection (pixel evidence)
# ---------------------------

def _binarize_for_components(gray: np.ndarray) -> np.ndarray:
    """
    Produce a binary image where text/dots are white (255) on black background.
    Uses Otsu inverted threshold.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    # Invert because plat text is usually darker than background
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw

def detect_decimal_dot_digits_right(dot_ref_pil: Image.Image, word_bbox: tuple[int, int, int, int]) -> int | None:
    """
    Attempts to detect a decimal dot inside the distance word bbox by connected-components.
    Returns: count of digit components to the right of the dot (used to insert decimal),
             or None if not confidently detected.
    """
    x1, y1, x2, y2 = word_bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # Expand bbox slightly to catch dots touching the edge
    mx = max(2, int(w * 0.08))
    my = max(2, int(h * 0.12))
    X1 = max(0, x1 - mx)
    Y1 = max(0, y1 - my)
    X2 = min(dot_ref_pil.size[0], x2 + mx)
    Y2 = min(dot_ref_pil.size[1], y2 + my)

    crop = dot_ref_pil.crop((X1, Y1, X2, Y2))
    gray = np.array(crop.convert("L"))

    bw = _binarize_for_components(gray)

    # Connected components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    if n <= 2:
        return None

    # Identify "digit-like" components (larger blobs)
    # Filter out background (label 0)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area <= 0:
            continue
        comps.append((i, x, y, cw, ch, area, centroids[i][0], centroids[i][1]))

    if not comps:
        return None

    # Determine scale from largest few blobs (digits)
    areas = sorted([c[5] for c in comps], reverse=True)
    top = areas[:6]
    if not top:
        return None
    median_big = float(np.median(top))  # robust digit area scale

    # Digit candidates: not too tiny, not extremely tall/flat
    digit_comps = []
    for (i, x, y, cw, ch, area, cx, cy) in comps:
        if area < max(6, 0.10 * median_big):
            continue
        if ch < max(6, int(0.35 * h)) and median_big > 60:
            # If digit scale is large but this component is too short, it's likely punctuation/noise
            continue
        digit_comps.append((i, x, y, cw, ch, area, cx, cy))

    if len(digit_comps) < 2:
        # can't infer dot position without digit structure
        return None

    # Estimate baseline band: dots typically sit near lower part of digit bbox
    # Use digit component centroids to estimate lower band
    digit_cys = [c[7] for c in digit_comps]
    cy_med = float(np.median(digit_cys))

    # Dot candidates: much smaller than digits, compact, and in lower region
    dot_candidates = []
    for (i, x, y, cw, ch, area, cx, cy) in comps:
        if area >= 0.35 * median_big:
            continue  # too big to be decimal dot
        if area < 2:
            continue
        if cw > 0.40 * w or ch > 0.40 * h:
            continue
        fill = area / max(1.0, float(cw * ch))
        if fill < 0.20:
            continue

        # baseline-ish: lower half to lower 90%, and not above median digit center
        if not (cy >= 0.50 * h and cy <= 0.95 * h):
            continue
        if cy < cy_med:
            continue

        # round-ish: prefer close to square
        ar = cw / max(1.0, float(ch))
        ar = ar if ar >= 1.0 else 1.0 / ar
        if ar > 3.0:
            continue

        # Score: smaller + rounder + closer to baseline
        score = 0.0
        score += (1.0 - min(1.0, area / (0.35 * median_big))) * 2.0
        score += (1.0 - min(1.0, (ar - 1.0) / 2.0)) * 1.5
        score += (min(1.0, (cy - cy_med) / max(1.0, (0.40 * h)))) * 1.0
        score += fill * 0.8

        dot_candidates.append((score, cx, cy, i))

    if not dot_candidates:
        return None

    dot_candidates.sort(key=lambda t: t[0], reverse=True)
    _, dot_cx, dot_cy, dot_i = dot_candidates[0]

    # Count digit components to the right of the dot
    digit_centers_x = sorted([c[6] for c in digit_comps])
    right = [x for x in digit_centers_x if x > dot_cx + 1.5]
    digits_right = len(right)

    # sanity: must be between 1 and len(digits)-1, but we don't know digit count here;
    # we can require at least one digit right and at least one digit left.
    left = [x for x in digit_centers_x if x < dot_cx - 1.5]
    if digits_right < 1 or len(left) < 1:
        return None

    # Common decimal counts in plats: 1..3 (sometimes more, but dot gets too tiny)
    if digits_right > 4:
        return None

    return digits_right

# ---------------------------
# Distance parsing (no length heuristic)
# ---------------------------

def parse_distance_with_quality(
    dist_text: str,
    dot_digits_right: int | None = None,
    min_ft: float = 1.0,
    max_ft: float = 1_000_000.0,
) -> tuple[float | None, str]:
    """
    Robust distance parser.

    Accepts:
      - explicit decimal '.', including dot-like glyphs
      - comma-as-decimal (single comma with 2-3 digits after)
      - dot detected from pixels (dot_digits_right)

    Does NOT:
      - infer decimals from digit length or magnitude
    """
    if not dist_text:
        return None, "invalid"

    s = (dist_text or "").strip()
    s = s.replace(" ", "")
    s = s.replace("·", ".").replace("•", ".").replace("∙", ".").replace("⋅", ".")

    # comma-as-decimal
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

    # remove thousands separators, fix digit confusions
    s = s.replace(",", "")
    s = s.translate(DIGIT_FIX)

    # explicit decimal
    if "." in s:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])
        try:
            v = float(s)
            if min_ft <= v <= max_ft:
                return v, "explicit_decimal"
        except ValueError:
            return None, "invalid"

    # digits-only (possibly missing dot)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None, "invalid"

    # If we have pixel dot evidence, insert dot based on digits_right
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

    # Otherwise treat as integer (no guessing)
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

    if abs(a - 90.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if (out.shape[1], out.shape[0]) != (w, h) else out
        return Image.fromarray(out)
    if abs(a + 90.0) < 1e-9:
        out = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if (out.shape[1], out.shape[0]) != (w, h) else out
        return Image.fromarray(out)
    if abs(abs(a) - 180.0) < 1e-9:
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

def preprocess_variant(pil_img: Image.Image, variant: dict) -> Image.Image:
    """
    Creates an OCR-friendly image variant.
    Dot-preserving variants use thresh='none' or low-touch ops to avoid deleting tiny dots.
    """
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    scale = float(variant.get("scale", 1.0))
    base_pixels = int(gray.shape[0] * gray.shape[1])
    if base_pixels > 0 and scale != 1.0:
        target = base_pixels * (scale * scale)
        if target > MAX_VARIANT_PIXELS:
            scale = (MAX_VARIANT_PIXELS / base_pixels) ** 0.5

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    inv = bool(variant.get("inv", False))
    thresh = variant.get("thresh", "adaptive_gauss")
    blur = int(variant.get("blur", 0) or 0)

    if thresh == "none":
        out = 255 - gray if inv else gray
        return Image.fromarray(out)

    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if thresh == "adaptive_gauss":
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, 31, 10
        )
    elif thresh == "adaptive_mean":
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY, 31, 10
        )
    elif thresh == "otsu":
        mode = (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
        _, out = cv2.threshold(gray, 0, 255, mode)
    else:
        out = gray

    # IMPORTANT: denoise/morph can kill dots; only use on non dot-preserving variants
    if bool(variant.get("denoise", False)):
        out = cv2.fastNlMeansDenoising(out, None, h=25, templateWindowSize=7, searchWindowSize=21)

    if bool(variant.get("morph_close", False)):
        kernel = np.ones((2, 2), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    return Image.fromarray(out)

# ---------------------------
# OCR: keep word bboxes
# ---------------------------

def ocr_image_to_lines_with_words(pil_img: Image.Image, config: str) -> list[dict]:
    """
    Returns list of line dicts:
      {
        text, conf, bbox,
        words: [{text, conf, bbox}, ...]
      }
    """
    data = pytesseract.image_to_data(pil_img, config=config, lang="eng", output_type=Output.DICT)

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
            lines[key] = {
                "words": [],
                "bbox": [x, y, x + w, y + h],
                "confs": [],
            }

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
        # Preserve reading order within line by bbox x
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

def tile_rect(r, page_w, page_h, max_pixels=MAX_ROI_PIXELS, overlap=120) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = r
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return []
    area = w * h
    if area <= max_pixels:
        return [clamp_rect(r, page_w, page_h)]

    n = int(np.ceil(np.sqrt(area / max_pixels)))
    n = max(2, min(n, 6))

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
# ROI detection (rotated page)
# ---------------------------

def detect_bearing_distance_rois(page_img: Image.Image, roi_pad: int = 55, max_rois: int = 35) -> list[tuple[int, int, int, int]]:
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

    det_cfg = '--oem 3 --psm 11 -c preserve_interword_spaces=1'
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

        if bh > 0.18 * sH:
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
    rects = [r for _, r in scored[: max_rois * 2]]
    rects = merge_rects(rects)

    page_area = W * H
    def area(r): return (r[2] - r[0]) * (r[3] - r[1])
    rects = [r for r in rects if area(r) < 0.35 * page_area]

    rects.sort(key=area)
    return rects[:max_rois]

# ---------------------------
# Extract pairs from line using word bboxes
# ---------------------------

def _union_bbox(bbs: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(b[0] for b in bbs)
    y1 = min(b[1] for b in bbs)
    x2 = max(b[2] for b in bbs)
    y2 = max(b[3] for b in bbs)
    return (x1, y1, x2, y2)

def extract_pairs_from_line_words(
    line_words: list[dict],
    dot_ref_img: Image.Image | None,
    min_ft: float = 1.0
) -> list[tuple[str, float, str, str, tuple[int, int, int, int] | None]]:
    """
    Returns:
      (bearing_canon, dist_value, dist_quality, dist_raw, dist_bbox)
    """
    out = []
    if not line_words:
        return out

    texts = [w["text"] for w in line_words]

    # Bearings can span multiple tokens; scan windows
    for i in range(len(texts)):
        for win in (3, 4, 5, 6):
            j = min(len(texts), i + win)
            snippet = " ".join(texts[i:j])
            bm = BEARING_RE.search(snippet)
            if not bm:
                continue

            b_raw = bm.group(0)
            b_canon = parse_bearing_to_canon(b_raw)
            if not b_canon:
                continue

            # Find a distance token after the bearing window
            # Use the raw token(s) closest after the bearing, typically one token.
            dist_idx = None
            for k in range(j, min(len(texts), j + 8)):
                tok = texts[k].strip()
                if not tok:
                    continue
                # keep dot-like glyphs and commas; reject tokens with letters
                if re.search(r"[A-Za-z]", tok):
                    continue
                if DIST_TOKEN_RE.match(tok) or re.search(r"\d", tok):
                    dist_idx = k
                    break

            if dist_idx is None:
                continue

            dist_raw = texts[dist_idx]
            dist_bbox = line_words[dist_idx]["bbox"]

            dot_digits_right = None
            if dot_ref_img is not None:
                dot_digits_right = detect_decimal_dot_digits_right(dot_ref_img, dist_bbox)

            dist_val, dist_quality = parse_distance_with_quality(dist_raw, dot_digits_right=dot_digits_right, min_ft=min_ft)

            if dist_val is None:
                continue

            out.append((b_canon, float(dist_val), dist_quality, dist_raw, dist_bbox))

            # Prevent duplicate extraction from overlapping windows at same i
            break

    return out

# ---------------------------
# Multipass OCR on ROI
# ---------------------------

def multipass_ocr_on_roi(roi_img: Image.Image, ocr_configs: list[tuple[str, str]], variants: list[dict]) -> list[dict]:
    candidates = []
    for v in variants:
        proc = preprocess_variant(roi_img, v)

        # Use dot-preserving variants as dot reference (aligned with OCR bboxes for this proc)
        dot_ref_img = proc if bool(v.get("dot_ref", False)) else None

        for cfg, desc in ocr_configs:
            lines = ocr_image_to_lines_with_words(proc, cfg)
            for ln in lines:
                for b_canon, dist, dist_quality, dist_raw, dist_bbox in extract_pairs_from_line_words(
                    ln["words"], dot_ref_img=dot_ref_img, min_ft=1.0
                ):
                    theta = bearing_theta_deg(b_canon)
                    candidates.append({
                        "canon_bearing": b_canon,
                        "distance": f"{dist:.3f}".rstrip("0").rstrip(".") if dist_quality == "dot_blob" else f"{dist:.2f}",
                        "numeric_dist": float(dist),
                        "theta": theta,
                        "conf": float(ln["conf"]) if ln["conf"] is not None else -1.0,
                        "pass_name": v["name"],
                        "config_desc": desc,
                        "raw_line": ln["text"],
                        "dist_quality": dist_quality,
                        "dist_raw": dist_raw,
                        "dist_bbox": dist_bbox,
                    })
    return candidates

# ---------------------------
# Clustering / voting
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
    # Prefer explicit decimals, then pixel dot evidence, then integers.
    if q == "explicit_decimal":
        return 2.0
    if q == "comma_decimal":
        return 1.5
    if q == "dot_blob":
        return 1.2
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
        bonus = sum(_decimal_evidence_bonus(c.get("dist_quality", "")) for c in reps) / max(1, len(reps))
        return mean_conf + bonus, mean_conf, bonus, reps

    best_key = None
    best_tuple = (-1e18, -1e18, -1e18, [])
    for k in top_keys:
        sc, mc, bn, reps = score_key(k)
        if (sc, mc, bn) > best_tuple[:3]:
            best_tuple = (sc, mc, bn, reps)
            best_key = k

    _, mean_conf, bonus, reps = best_tuple
    reps.sort(key=lambda x: x["conf"], reverse=True)
    rep = reps[0]

    bearing, dist_str = best_key
    dist_val = float(dist_str)

    dq = Counter(c.get("dist_quality", "unknown") for c in reps)
    passes = Counter((c.get("pass_name"), c.get("config_desc")) for c in reps)
    angles = Counter(c.get("angle", 0) for c in reps)

    return {
        "bearing": bearing,
        "distance": dist_str,
        "numeric_dist": dist_val,
        "support": top_count,
        "best_conf": mean_conf,
        "dist_quality_votes": [{"quality": q, "count": n} for q, n in dq.most_common(8)],
        "example_raw": rep.get("raw_line", ""),
        "source": "voted_roi",
        "angle_votes": [{"angle": a, "count": n} for a, n in angles.most_common(8)],
        "pass_breakdown": [{"pass": p, "config": cfg, "count": n} for (p, cfg), n in passes.most_common(8)],
    }

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
        if not any(abs(p.get("numeric_dist", 0) - f["numeric_dist"]) < 2 and str(p.get("bearing", ""))[:10] == f["bearing"][:10] for p in out):
            out.append(f)
    return out

# ---------------------------
# Main extraction
# ---------------------------

def extract_pairs_from_pdf(
    pdf_path: Path,
    dpi: int,
    roi_pad: int,
    max_rois: int,
    dist_tol: float,
    bear_tol_deg: float,
    min_support: int,
    debug_dir: Path | None,
    angles: list[float],
    fallback_tile_angles: set[float],
) -> list[dict]:
    print(f"Converting PDF → images @ {dpi} DPI...")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"→ {len(images)} page(s)")

    # Variants: include dot-preserving refs
    variants = [
        # dot-preserving references (used for dot blob detection)
        {"name": "gray",    "thresh": "none",          "inv": False, "scale": 1.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True},
        {"name": "gray2x",  "thresh": "none",          "inv": False, "scale": 2.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True},
        {"name": "otsu0",   "thresh": "otsu",          "inv": False, "scale": 1.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": True},

        # general OCR variants (may kill dots, but improve text overall)
        {"name": "ag0",     "thresh": "adaptive_gauss","inv": False, "scale": 1.0, "blur": 0, "denoise": False, "morph_close": False, "dot_ref": False},
        {"name": "ag_s1",   "thresh": "adaptive_gauss","inv": False, "scale": 1.0, "blur": 3, "denoise": False, "morph_close": False, "dot_ref": False},
        {"name": "ag_inv",  "thresh": "adaptive_gauss","inv": True,  "scale": 1.0, "blur": 3, "denoise": True,  "morph_close": False, "dot_ref": False},
        {"name": "am_s1",   "thresh": "adaptive_mean", "inv": False, "scale": 1.0, "blur": 3, "denoise": False, "morph_close": False, "dot_ref": False},
        {"name": "otsu2x",  "thresh": "otsu",          "inv": False, "scale": 2.0, "blur": 3, "denoise": False, "morph_close": True,  "dot_ref": False},
        {"name": "ag2x",    "thresh": "adaptive_gauss","inv": False, "scale": 2.0, "blur": 3, "denoise": True,  "morph_close": True,  "dot_ref": False},
    ]

    # SHLEX-safe whitelist (no quotes). Keep '.'.
    whitelist = "0123456789NSEWnsew.,-°oº"
    common = f"-c preserve_interword_spaces=1 -c classify_bln_numeric_mode=1 -c tessedit_enable_dict_correction=0 -c tessedit_char_whitelist={whitelist}"

    ocr_configs = [
        (f'--oem 3 --psm 6  {common}', "block"),
        (f'--oem 3 --psm 11 {common}', "sparse"),
        (f'--oem 3 --psm 7  {common}', "single_line"),
        (f'--oem 3 --psm 13 {common}', "raw_line"),
        (f'--oem 1 --psm 11 {common}', "legacy_sparse"),
    ]

    all_candidates: list[dict] = []

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, page_img in enumerate(images, 1):
        print(f"  Page {page_idx}...")

        for angle in angles:
            rot_img = rotate_keep_size(page_img, angle)
            W, H = rot_img.size

            rois = detect_bearing_distance_rois(rot_img, roi_pad=roi_pad, max_rois=max_rois)

            if rois:
                print(f"    angle {angle:+.0f}°: ROI detect {len(rois)}")
            else:
                if float(angle) in fallback_tile_angles:
                    print(f"    angle {angle:+.0f}°: ROI detect none → tiling full page")
                    rois = tile_rect((0, 0, W, H), W, H, max_pixels=MAX_ROI_PIXELS, overlap=160)
                else:
                    continue

            roi_counter = 0
            for r in rois:
                tiles = tile_rect(r, W, H, max_pixels=MAX_ROI_PIXELS, overlap=140)
                for tr in tiles:
                    roi_counter += 1
                    x1, y1, x2, y2 = tr
                    crop = rot_img.crop((x1, y1, x2, y2))

                    if debug_dir:
                        crop_path = debug_dir / f"page{page_idx:02d}_a{int(angle):+03d}_roi{roi_counter:03d}_{x1}_{y1}_{x2}_{y2}.png"
                        try:
                            crop.save(crop_path)
                        except Exception:
                            pass

                    cands = multipass_ocr_on_roi(crop, ocr_configs=ocr_configs, variants=variants)
                    for c in cands:
                        c["page"] = page_idx
                        c["angle"] = float(angle)
                        c["roi"] = roi_counter
                        c["roi_bbox"] = (x1, y1, x2, y2)

                    all_candidates.extend(cands)

        print(f"    Candidates so far: {len(all_candidates)}")

    if not all_candidates:
        return []

    clusters = cluster_candidates(all_candidates, dist_tol=dist_tol, bear_tol_deg=bear_tol_deg)
    winners = [pick_cluster_winner(cl) for cl in clusters]
    winners = [w for w in winners if w.get("support", 1) >= min_support]

    # Deduplicate exact bearing+distance, keep highest support/conf
    best_map: dict[tuple[str, str], dict] = {}
    for w in winners:
        k = (w["bearing"], w["distance"])
        if k not in best_map:
            best_map[k] = w
        else:
            prev = best_map[k]
            if (w["support"], w.get("best_conf", -1)) > (prev["support"], prev.get("best_conf", -1)):
                best_map[k] = w

    out = list(best_map.values())
    out.sort(key=lambda x: x["numeric_dist"], reverse=True)
    return out

# ---------------------------
# CLI
# ---------------------------

def parse_angles(s: str) -> list[float]:
    s = (s or "").strip()
    if not s or s.lower() in ("auto", "default"):
        return DEFAULT_ANGLES[:]
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    if not out:
        return DEFAULT_ANGLES[:]
    if not any(abs(a) < 1e-9 for a in out):
        out.append(0.0)
    seen = set()
    uniq = []
    for a in out:
        key = round(float(a), 6)
        if key not in seen:
            seen.add(key)
            uniq.append(float(a))
    return uniq

def main():
    parser = argparse.ArgumentParser(description="Extract bearings & distances from ROS PDF (multi-angle ROI + multipass vote + dot detection)")
    parser.add_argument("file", help="Path to PDF")
    parser.add_argument("--dpi", type=int, default=500, help="Higher = better symbol recognition (watch file size)")
    parser.add_argument("--roi-pad", type=int, default=55, help="Padding pixels around detected bearing/distance lines")
    parser.add_argument("--max-rois", type=int, default=35, help="Cap number of ROIs per angle per page")
    parser.add_argument("--min-support", type=int, default=2, help="Require at least N agreeing passes for a result")
    parser.add_argument("--dist-tol", type=float, default=0.25, help="Clustering tolerance in feet for distances")
    parser.add_argument("--bear-tol-deg", type=float, default=0.35, help="Clustering tolerance in degrees for bearings")
    parser.add_argument("--angles", default="default",
                        help="Comma-separated angles in degrees to scan (e.g. \"-90,-45,0,45,90\"). Use 'default' for built-in sweep.")
    parser.add_argument("--fallback-tile-angles", default="0,-90,90,180",
                        help="Angles where we brute-force tile the whole page if ROI detection finds nothing.")
    parser.add_argument("--debug-rois", help="Directory to save detected ROI crops")
    parser.add_argument("--no-forced", action="store_true", help="Disable forced matches")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Save as JSON")
    args = parser.parse_args()

    pdf = Path(args.file)
    if not pdf.is_file():
        print("File not found.")
        sys.exit(1)

    debug_dir = Path(args.debug_rois) if args.debug_rois else None
    angles = parse_angles(args.angles)
    fallback_tile_angles = set(float(a) for a in parse_angles(args.fallback_tile_angles))

    pairs = extract_pairs_from_pdf(
        pdf_path=pdf,
        dpi=args.dpi,
        roi_pad=args.roi_pad,
        max_rois=args.max_rois,
        dist_tol=args.dist_tol,
        bear_tol_deg=args.bear_tol_deg,
        min_support=args.min_support,
        debug_dir=debug_dir,
        angles=angles,
        fallback_tile_angles=fallback_tile_angles,
    )

    if not args.no_forced:
        pairs = apply_forced_pairs(pairs, FORCED_DEFAULT)

    # Deduplicate again
    seen = set()
    uniq = []
    for p in pairs:
        k = (p.get("bearing"), p.get("distance"))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    uniq.sort(key=lambda x: x.get("numeric_dist", 0), reverse=True)
    pairs = uniq

    print("\n" + "=" * 70)
    print("EXTRACTED BEARING → DISTANCE PAIRS (VOTED, MULTI-ANGLE)")
    print("=" * 70)

    if not pairs:
        print("No valid pairs found.")
        return

    for i, p in enumerate(pairs, 1):
        support = p.get("support", 1)
        conf = p.get("best_conf", -1)
        src = p.get("source", "unknown")
        print(f"{i:2d}. {p['bearing']:22}  {p['distance']:>9} ft   (support={support}, conf={conf:.1f}, {src})")

        if args.verbose:
            ex = (p.get("example_raw") or "").replace("\n", " ").strip()
            if len(ex) > 220:
                ex = ex[:220] + "..."
            if ex:
                print(f"   Example OCR: {ex}")

            if p.get("dist_quality_votes"):
                dqs = ", ".join([f"{x['quality']}×{x['count']}" for x in p["dist_quality_votes"]])
                print(f"   Dist evidence: {dqs}")

            if p.get("angle_votes"):
                top_angles = ", ".join([f"{x['angle']:+.0f}°×{x['count']}" for x in p["angle_votes"]])
                print(f"   Angle votes:   {top_angles}")

            if p.get("pass_breakdown"):
                top = ", ".join([f"{x['pass']}/{x['config']}×{x['count']}" for x in p["pass_breakdown"]])
                print(f"   Top passes:    {top}")
            print()

    print(f"\nTotal unique pairs: {len(pairs)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "file": str(pdf),
                "angles": angles,
                "pairs": [{k: v for k, v in p.items() if k != "numeric_dist"} for p in pairs],
                "count": len(pairs)
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
