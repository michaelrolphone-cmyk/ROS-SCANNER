#!/usr/bin/env python3
"""
survey_recon.py — General Record of Survey reconstruction CLI (evidence-first, plan-driven)

WHAT THIS TOOL DOES (GENERAL, NOT SURVEY-SPECIFIC)
- Extract/confirm record calls from plat text (OCR + user confirmation), with snippet evidence.
- Detect drawn line segments in the plat image (OpenCV Hough) and associate callouts to segments.
- Auto-order a set of record line calls into a traverse that minimizes misclosure (beam search),
  then assign sequence numbers and generate a plan step.

DEFENSIBILITY RULES ENFORCED
- No invented numbers: record values must come from OCR/user-confirmed text snippets.
- Derived geometry is allowed only if explicitly marked computed with derivation notes.
- Auto steps only PROPOSE order; the tool records the scoring + misclosure and you can reject/adjust.
- Gates stop the run unless --force is used (which marks outputs "NON-DEFENSIBLE / REVIEW").

WORKFLOW (typical)
  python survey_recon.py init plat.png --out out/myros
  python survey_recon.py detect out/myros/project.json          # OCR candidate scan
  python survey_recon.py review out/myros/project.json          # confirm calls + optionally compose line calls
  python survey_recon.py lines-detect out/myros/project.json    # detect drawn segments
  python survey_recon.py associate out/myros/project.json       # link calls to segments
  python survey_recon.py autotrav out/myros/project.json --layer outer --set-seq --suggest-plan
  python survey_recon.py plan-new out/myros/project.json --out-plan out/myros/plan.json
  python survey_recon.py solve out/myros/project.json --plan out/myros/plan.json
  python survey_recon.py render out/myros/project.json --svg --dxf
  python survey_recon.py report out/myros/project.json

NOTE
- This is NOT boundary determination. It reconstructs record geometry + proves closure layers.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------
# Precision / policy defaults
# -----------------------------
PROJECT_VERSION = "1.1.0-skeleton"

DIST_REPORT_DECIMALS = 2          # ft
MISCLS_REPORT_DECIMALS = 3        # ft
BEARING_REPORT_SECONDS = 1        # 1"
EPS = 1e-10


# -----------------------------
# Filesystem / formatting
# -----------------------------
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def fmt_ft(x: float, d: int = DIST_REPORT_DECIMALS) -> str:
    return f"{x:.{d}f}"

def fmt_ft_mis(x: float) -> str:
    return f"{x:.{MISCLS_REPORT_DECIMALS}f}"

def ft_to_inches(ft: float) -> float:
    return ft * 12.0

def stable_id(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


# -----------------------------
# Audit logging (append-only)
# -----------------------------
class AuditLogger:
    def __init__(self, ndjson_path: Path):
        self.path = ndjson_path
        ensure_dir(self.path.parent)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": now_iso(), "event": event, "payload": payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# -----------------------------
# Geometry primitives
# -----------------------------
@dataclass(frozen=True)
class Pt:
    e: float
    n: float

    def __add__(self, v: "Vec") -> "Pt":
        return Pt(self.e + v.de, self.n + v.dn)

    def __sub__(self, other: Union["Pt", "Vec"]) -> Union["Vec", "Pt"]:
        if isinstance(other, Pt):
            return Vec(self.e - other.e, self.n - other.n)
        return Pt(self.e - other.de, self.n - other.dn)

@dataclass(frozen=True)
class Vec:
    de: float
    dn: float

    def mag(self) -> float:
        return math.hypot(self.de, self.dn)

    def scale(self, s: float) -> "Vec":
        return Vec(self.de * s, self.dn * s)

    def unit(self) -> "Vec":
        m = self.mag()
        if m < EPS:
            raise ValueError("Zero-length vector")
        return self.scale(1.0 / m)

def cross(a: Vec, b: Vec) -> float:
    return a.de * b.dn - a.dn * b.de

def az_deg_from_vec(v: Vec) -> float:
    # azimuth clockwise from North: atan2(E, N)
    ang = math.degrees(math.atan2(v.de, v.dn))
    if ang < 0:
        ang += 360.0
    return ang

def vec_from_az_deg(az_deg: float) -> Vec:
    az = math.radians(az_deg)
    return Vec(math.sin(az), math.cos(az))  # de, dn

def dist(a: Pt, b: Pt) -> float:
    return (a - b).mag()

def normalize_az(az: float) -> float:
    az %= 360.0
    if az < 0:
        az += 360.0
    return az

def interp_along(a: Pt, b: Pt, d_from_a: float) -> Pt:
    ab = (b - a)
    L = ab.mag()
    if L < EPS:
        raise ValueError("Cannot interpolate on zero-length segment")
    t = d_from_a / L
    return Pt(a.e + ab.de * t, a.n + ab.dn * t)

def line_intersection(p: Pt, r: Vec, q: Pt, s: Vec) -> Optional[Pt]:
    rxs = cross(r, s)
    if abs(rxs) < EPS:
        return None
    qmp = q - p
    t = cross(qmp, s) / rxs
    return Pt(p.e + t * r.de, p.n + t * r.dn)

def close_ring(points: List[Pt]) -> List[Pt]:
    if not points:
        return points
    if dist(points[0], points[-1]) > 1e-6:
        return points + [points[0]]
    return points


# -----------------------------
# Bearings (Quadrant DMS)
# -----------------------------
@dataclass(frozen=True)
class DMS:
    deg: int
    minutes: int
    seconds: float

    def to_degrees(self) -> float:
        return self.deg + self.minutes / 60.0 + self.seconds / 3600.0

    @staticmethod
    def from_degrees(x: float) -> "DMS":
        deg = int(x)
        rem = (x - deg) * 60.0
        minutes = int(rem)
        seconds = (rem - minutes) * 60.0
        seconds = round(seconds / BEARING_REPORT_SECONDS) * BEARING_REPORT_SECONDS
        if seconds >= 60.0:
            seconds -= 60.0
            minutes += 1
        if minutes >= 60:
            minutes -= 60
            deg += 1
        return DMS(deg, minutes, seconds)

@dataclass(frozen=True)
class QuadrantBearing:
    ns: str
    angle: DMS
    ew: str

    def to_azimuth_deg(self) -> float:
        theta = self.angle.to_degrees()
        ns = self.ns.upper()
        ew = self.ew.upper()
        if ns == "N" and ew == "E":
            az = theta
        elif ns == "S" and ew == "E":
            az = 180.0 - theta
        elif ns == "S" and ew == "W":
            az = 180.0 + theta
        elif ns == "N" and ew == "W":
            az = 360.0 - theta
        else:
            raise ValueError(f"Invalid quadrant bearing: {self.ns} {self.ew}")
        return normalize_az(az)

    @staticmethod
    def from_azimuth_deg(az: float) -> "QuadrantBearing":
        az = normalize_az(az)
        if 0 <= az < 90:
            ns, ew = "N", "E"
            theta = az
        elif 90 <= az < 180:
            ns, ew = "S", "E"
            theta = 180.0 - az
        elif 180 <= az < 270:
            ns, ew = "S", "W"
            theta = az - 180.0
        else:
            ns, ew = "N", "W"
            theta = 360.0 - az
        return QuadrantBearing(ns, DMS.from_degrees(theta), ew)

    def format(self) -> str:
        a = self.angle
        sec = int(round(a.seconds))
        return f"{self.ns} {a.deg:02d}°{a.minutes:02d}'{sec:02d}\" {self.ew}"

BEARING_RE = re.compile(
    r"""
    (?P<ns>[NS])\s*
    (?P<deg>\d{1,3})\s*(?:°|º|d|deg)\s*
    (?P<min>\d{1,2})\s*(?:'|m|min)\s*
    (?P<sec>\d{1,2}(?:\.\d+)?)\s*(?:"|s|sec)\s*
    (?P<ew>[EW])
    """,
    re.IGNORECASE | re.VERBOSE,
)

DIST_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)")

def parse_quadrant_bearing(text: str) -> QuadrantBearing:
    t = text.strip().replace("º", "°")
    m = BEARING_RE.search(t)
    if not m:
        raise ValueError(f"Could not parse bearing: {text!r}")
    ns = m.group("ns").upper()
    ew = m.group("ew").upper()
    deg = int(m.group("deg"))
    minutes = int(m.group("min"))
    seconds = float(m.group("sec"))
    if not (0 <= deg <= 90):
        raise ValueError(f"Quadrant bearing degrees must be 0..90; got {deg}")
    if not (0 <= minutes < 60) or not (0 <= seconds < 60):
        raise ValueError(f"Bearing minutes/seconds out of range: {minutes} {seconds}")
    return QuadrantBearing(ns, DMS(deg, minutes, seconds), ew)

def parse_distance_ft(text: str) -> float:
    m = DIST_RE.search(text)
    if not m:
        raise ValueError(f"Could not parse distance: {text!r}")
    return float(m.group("val"))


# -----------------------------
# Evidence + Calls
# -----------------------------
@dataclass
class Evidence:
    type: str  # 'ocr', 'user_confirmed', 'computed'
    bbox: Optional[List[int]] = None  # [x,y,w,h] pixels in image
    snip_path: Optional[str] = None
    ocr_text: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class CallCandidate:
    id: str
    kind_guess: str  # 'bearing'|'distance'|'unknown'
    text: str
    bbox: List[int]          # [x,y,w,h]
    confidence: float = 0.0

@dataclass
class RecordCall:
    id: str
    kind: str            # 'bearing'|'distance'|'line'|'curve'|'text'
    layer: str = "unassigned"
    bearing_text: Optional[str] = None
    distance_text: Optional[str] = None
    bearing: Optional[QuadrantBearing] = None
    distance_ft: Optional[float] = None
    tags: List[str] = field(default_factory=list)  # includes 'record' or 'computed'
    derivation: Optional[str] = None
    evidence: Evidence = field(default_factory=lambda: Evidence(type="ocr"))
    seq: Optional[int] = None  # ordering inside a traverse, if applicable

    # segment association fields (image space)
    assoc_segment_id: Optional[str] = None
    assoc_segment_dist_px: Optional[float] = None

    def is_record(self) -> bool:
        return "record" in self.tags

    def is_computed(self) -> bool:
        return "computed" in self.tags


# -----------------------------
# Image line segments + association
# -----------------------------
@dataclass
class LineSegment:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float
    angle_deg: float  # 0..180 measured from +x axis for convenience
    source: str = "hough"

def seg_mid(seg: LineSegment) -> Tuple[float, float]:
    return ((seg.x1 + seg.x2) / 2.0, (seg.y1 + seg.y2) / 2.0)

def point_to_segment_distance_px(px: float, py: float, seg: LineSegment) -> float:
    # distance from point to line segment in pixel coords
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    L2 = vx * vx + vy * vy
    if L2 <= EPS:
        return math.hypot(px - x1, py - y1)
    t = (wx * vx + wy * vy) / L2
    t = max(0.0, min(1.0, t))
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


# -----------------------------
# Closures / Results
# -----------------------------
@dataclass
class Closure:
    sum_dn: float
    sum_de: float
    misclosure_ft: float
    misclosure_bearing: QuadrantBearing
    total_length_ft: float
    closure_ratio: float

@dataclass
class LayerResult:
    id: str
    build_type: str
    calls_used: List[str]
    points: List[Pt]
    closure: Closure
    gate_passed: bool
    gate_notes: str

@dataclass
class LotResult:
    id: str
    polygon: List[Pt]
    perimeter_ft: float
    area_sqft: float
    closure: Closure


# -----------------------------
# Project model
# -----------------------------
@dataclass
class Project:
    version: str
    created_at: str
    image_path: str
    out_dir: str

    candidates: List[CallCandidate] = field(default_factory=list)
    calls: Dict[str, RecordCall] = field(default_factory=dict)

    segments: List[LineSegment] = field(default_factory=list)

    layers: Dict[str, LayerResult] = field(default_factory=dict)
    lots: Dict[str, LotResult] = field(default_factory=dict)

    flags: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        def qb_to_dict(q: Optional[QuadrantBearing]) -> Any:
            if q is None:
                return None
            return {
                "ns": q.ns, "ew": q.ew,
                "deg": q.angle.deg, "min": q.angle.minutes, "sec": q.angle.seconds
            }

        def pt_to_list(p: Pt) -> List[float]:
            return [p.e, p.n]

        def closure_to_dict(c: Closure) -> Dict[str, Any]:
            return {
                "sum_dn": c.sum_dn,
                "sum_de": c.sum_de,
                "misclosure_ft": c.misclosure_ft,
                "misclosure_bearing": qb_to_dict(c.misclosure_bearing),
                "total_length_ft": c.total_length_ft,
                "closure_ratio": c.closure_ratio,
            }

        return {
            "version": self.version,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "out_dir": self.out_dir,
            "flags": self.flags,
            "diagnostics": self.diagnostics,
            "candidates": [dataclasses.asdict(c) for c in self.candidates],
            "segments": [dataclasses.asdict(s) for s in self.segments],
            "calls": {
                k: {
                    **dataclasses.asdict(v),
                    "bearing": qb_to_dict(v.bearing),
                }
                for k, v in self.calls.items()
            },
            "layers": {
                k: {
                    "id": v.id,
                    "build_type": v.build_type,
                    "calls_used": v.calls_used,
                    "points": [pt_to_list(p) for p in v.points],
                    "closure": closure_to_dict(v.closure),
                    "gate_passed": v.gate_passed,
                    "gate_notes": v.gate_notes,
                }
                for k, v in self.layers.items()
            },
            "lots": {
                k: {
                    "id": v.id,
                    "polygon": [pt_to_list(p) for p in v.polygon],
                    "perimeter_ft": v.perimeter_ft,
                    "area_sqft": v.area_sqft,
                    "closure": closure_to_dict(v.closure),
                }
                for k, v in self.lots.items()
            },
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Project":
        def dict_to_qb(d: Any) -> Optional[QuadrantBearing]:
            if d is None:
                return None
            return QuadrantBearing(
                d["ns"],
                DMS(int(d["deg"]), int(d["min"]), float(d["sec"])),
                d["ew"],
            )

        proj = Project(
            version=obj["version"],
            created_at=obj["created_at"],
            image_path=obj["image_path"],
            out_dir=obj["out_dir"],
        )
        proj.flags = obj.get("flags", {})
        proj.diagnostics = obj.get("diagnostics", {})
        proj.candidates = [CallCandidate(**c) for c in (obj.get("candidates") or [])]
        proj.segments = [LineSegment(**s) for s in (obj.get("segments") or [])]

        calls = {}
        for k, v in (obj.get("calls") or {}).items():
            ev = v.get("evidence") or {}
            rc = RecordCall(
                id=v["id"],
                kind=v["kind"],
                layer=v.get("layer", "unassigned"),
                bearing_text=v.get("bearing_text"),
                distance_text=v.get("distance_text"),
                bearing=dict_to_qb(v.get("bearing")),
                distance_ft=v.get("distance_ft"),
                tags=v.get("tags", []),
                derivation=v.get("derivation"),
                evidence=Evidence(**ev),
                seq=v.get("seq"),
                assoc_segment_id=v.get("assoc_segment_id"),
                assoc_segment_dist_px=v.get("assoc_segment_dist_px"),
            )
            calls[k] = rc
        proj.calls = calls

        def list_to_pt(a: List[float]) -> Pt:
            return Pt(float(a[0]), float(a[1]))

        def dict_to_closure(c: Dict[str, Any]) -> Closure:
            mb = dict_to_qb(c["misclosure_bearing"]) or QuadrantBearing("N", DMS(0, 0, 0), "E")
            return Closure(
                sum_dn=float(c["sum_dn"]),
                sum_de=float(c["sum_de"]),
                misclosure_ft=float(c["misclosure_ft"]),
                misclosure_bearing=mb,
                total_length_ft=float(c["total_length_ft"]),
                closure_ratio=float(c["closure_ratio"]),
            )

        layers = {}
        for k, v in (obj.get("layers") or {}).items():
            layers[k] = LayerResult(
                id=v["id"],
                build_type=v["build_type"],
                calls_used=list(v["calls_used"]),
                points=[list_to_pt(p) for p in v["points"]],
                closure=dict_to_closure(v["closure"]),
                gate_passed=bool(v["gate_passed"]),
                gate_notes=str(v["gate_notes"]),
            )
        proj.layers = layers

        lots = {}
        for k, v in (obj.get("lots") or {}).items():
            lots[k] = LotResult(
                id=v["id"],
                polygon=[list_to_pt(p) for p in v["polygon"]],
                perimeter_ft=float(v["perimeter_ft"]),
                area_sqft=float(v["area_sqft"]),
                closure=dict_to_closure(v["closure"]),
            )
        proj.lots = lots

        return proj

    def save(self, path: Union[str, Path]) -> None:
        write_json(path, self.to_json())

    @staticmethod
    def load(path: Union[str, Path]) -> "Project":
        return Project.from_json(read_json(path))


# -----------------------------
# OCR helpers (optional)
# -----------------------------
def try_import_ocr():
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
        return Image, pytesseract
    except Exception:
        return None, None

def crop_save_snip(image_path: Path, bbox: List[int], out_path: Path, pad: int = 6) -> None:
    Image, _ = try_import_ocr()
    if Image is None:
        raise RuntimeError("Pillow not installed (pip install pillow).")
    x, y, w, h = bbox
    img = Image.open(image_path)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.size[0], x + w + pad)
    y1 = min(img.size[1], y + h + pad)
    snip = img.crop((x0, y0, x1, y1))
    ensure_dir(out_path.parent)
    snip.save(out_path)

def ocr_scan_candidates(image_path: Path) -> List[CallCandidate]:
    """
    Heuristic detection:
    - pytesseract image_to_data gives words + bboxes
    - group by approximate text lines
    - detect bearing patterns or distance-ish patterns
    Output must be confirmed in review.
    """
    Image, pytesseract = try_import_ocr()
    if Image is None or pytesseract is None:
        raise RuntimeError("OCR deps missing. Install: pip install pillow pytesseract (and system tesseract).")

    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append((y, x, w, h, txt, conf))
    if not words:
        return []

    words.sort(key=lambda t: (t[0], t[1]))
    lines: List[List[Tuple[int,int,int,int,str,float]]] = []
    y_tol = 8
    for w in words:
        y = w[0]
        if not lines:
            lines.append([w])
        else:
            last_y = lines[-1][0][0]
            if abs(y - last_y) <= y_tol:
                lines[-1].append(w)
            else:
                lines.append([w])

    candidates: List[CallCandidate] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda t: t[1])
        text = " ".join(t[4] for t in ln_sorted)
        x0 = min(t[1] for t in ln_sorted)
        y0 = min(t[0] for t in ln_sorted)
        x1 = max(t[1] + t[2] for t in ln_sorted)
        y1 = max(t[0] + t[3] for t in ln_sorted)
        bbox = [x0, y0, x1 - x0, y1 - y0]

        kind_guess = "unknown"
        if BEARING_RE.search(text):
            kind_guess = "bearing"
        elif re.search(r"\d+\.\d+", text) and (("'" in text) or ("FT" in text.upper()) or ("FEET" in text.upper())):
            kind_guess = "distance"

        if kind_guess != "unknown":
            confs = [t[5] for t in ln_sorted if t[5] >= 0]
            conf = float(sum(confs) / len(confs)) if confs else 0.0
            cid = stable_id("cand", kind_guess, text, bbox)
            candidates.append(CallCandidate(id=cid, kind_guess=kind_guess, text=text, bbox=bbox, confidence=conf))

    return candidates


# -----------------------------
# OpenCV line detection (optional)
# -----------------------------
def try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def detect_line_segments_hough(
    image_path: Path,
    canny1: int = 50,
    canny2: int = 150,
    hough_thresh: int = 60,
    min_len: int = 60,
    max_gap: int = 10,
    max_segments: int = 400,
) -> List[LineSegment]:
    cv2 = try_import_cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. Install: pip install opencv-python")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, canny1, canny2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, threshold=hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)

    segs: List[LineSegment] = []
    if lines is None:
        return segs

    # LinesP returns shape (N,1,4)
    for L in lines:
        x1, y1, x2, y2 = [int(v) for v in L[0]]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        ang = abs(ang)  # normalize to 0..180-ish
        if ang > 180:
            ang = 360 - ang
        sid = stable_id("seg", x1, y1, x2, y2)
        segs.append(LineSegment(id=sid, x1=x1, y1=y1, x2=x2, y2=y2, length_px=length, angle_deg=ang))

    # Keep longest segments (helps reduce noise)
    segs.sort(key=lambda s: s.length_px, reverse=True)
    return segs[:max_segments]


# -----------------------------
# Traverse + closure math
# -----------------------------
def lat_dep_from_call(bearing: QuadrantBearing, distance_ft: float) -> Tuple[float, float]:
    az = math.radians(bearing.to_azimuth_deg())
    dn = distance_ft * math.cos(az)
    de = distance_ft * math.sin(az)
    return dn, de

def compute_closure_from_line_calls(calls: List[RecordCall]) -> Closure:
    sum_dn = 0.0
    sum_de = 0.0
    total_len = 0.0
    for c in calls:
        if c.bearing is None or c.distance_ft is None:
            raise ValueError(f"Call {c.id} missing bearing/distance")
        dn, de = lat_dep_from_call(c.bearing, float(c.distance_ft))
        sum_dn += dn
        sum_de += de
        total_len += float(c.distance_ft)

    mis = math.hypot(sum_dn, sum_de)
    mis_b = QuadrantBearing.from_azimuth_deg(az_deg_from_vec(Vec(sum_de, sum_dn)))
    ratio = (total_len / mis) if mis > EPS else float("inf")

    return Closure(
        sum_dn=sum_dn,
        sum_de=sum_de,
        misclosure_ft=mis,
        misclosure_bearing=mis_b,
        total_length_ft=total_len,
        closure_ratio=ratio,
    )

def build_traverse(start: Pt, calls: List[RecordCall]) -> List[Pt]:
    pts = [start]
    cur = start
    for c in calls:
        if c.bearing is None or c.distance_ft is None:
            raise ValueError(f"Call {c.id} missing bearing/distance")
        v = vec_from_az_deg(c.bearing.to_azimuth_deg()).unit().scale(float(c.distance_ft))
        cur = cur + v
        pts.append(cur)
    return pts


# -----------------------------
# Gates
# -----------------------------
@dataclass
class GateResult:
    passed: bool
    notes: str
    details: Dict[str, Any] = field(default_factory=dict)

def gate_closure(layer_id: str, closure: Closure, spec: Dict[str, Any]) -> GateResult:
    max_mis = float(spec.get("max_mis_ft", 0.05))
    min_ratio = float(spec.get("min_ratio", 50000))
    if closure.misclosure_ft > max_mis or closure.closure_ratio < min_ratio:
        return GateResult(
            False,
            f"{layer_id}: Misclosure {closure.misclosure_ft:.4f} ft (max {max_mis}) "
            f"ratio 1:{closure.closure_ratio:.0f} (min 1:{min_ratio:.0f}).",
            {"misclosure_ft": closure.misclosure_ft, "ratio": closure.closure_ratio},
        )
    return GateResult(True, f"{layer_id}: closure gate passed.")


# -----------------------------
# Plan (declarative solver)
# -----------------------------
@dataclass
class PlanStep:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    gates: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Plan:
    id: str
    description: str = ""
    steps: List[PlanStep] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "steps": [
                {"id": s.id, "type": s.type, "params": s.params, "gates": s.gates}
                for s in self.steps
            ],
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Plan":
        steps = [PlanStep(id=s["id"], type=s["type"], params=s.get("params", {}), gates=s.get("gates", [])) for s in obj.get("steps", [])]
        return Plan(id=obj.get("id", "plan"), description=obj.get("description", ""), steps=steps)

    @staticmethod
    def load(path: Union[str, Path]) -> "Plan":
        return Plan.from_json(read_json(path))

    def save(self, path: Union[str, Path]) -> None:
        write_json(path, self.to_json())


# -----------------------------
# Plan Solver (generic step engine)
# -----------------------------
class PlanSolver:
    def __init__(self, project: Project, audit: AuditLogger, force: bool = False):
        self.project = project
        self.audit = audit
        self.force = force

    def solve(self, plan: Plan) -> None:
        self.audit.log("solve_begin", {"plan_id": plan.id, "steps": [s.id for s in plan.steps], "force": self.force})

        for step in plan.steps:
            self.audit.log("step_begin", {"step_id": step.id, "type": step.type})
            if step.type == "traverse":
                self._step_traverse(step)
            else:
                raise SystemExit(f"Unknown step type: {step.type}")
            self.audit.log("step_end", {"step_id": step.id})

        self.audit.log("solve_complete", {"layers": list(self.project.layers.keys()), "lots": list(self.project.lots.keys())})

    def _step_traverse(self, step: PlanStep) -> None:
        layer_id = step.params.get("layer_id", step.id)
        call_ids = step.params.get("call_ids", [])
        start = step.params.get("start", {"e": 0.0, "n": 0.0})
        start_pt = Pt(float(start["e"]), float(start["n"]))

        calls = []
        for cid in call_ids:
            if cid not in self.project.calls:
                raise SystemExit(f"{layer_id}: Missing call {cid}")
            c = self.project.calls[cid]
            if c.kind != "line":
                raise SystemExit(f"{layer_id}: Call {cid} must be kind='line'")
            if c.bearing is None or c.distance_ft is None:
                raise SystemExit(f"{layer_id}: Call {cid} missing bearing/distance")
            if not (c.is_record() or c.is_computed()):
                raise SystemExit(f"{layer_id}: Call {cid} must be tagged 'record' or 'computed'")
            calls.append(c)

        pts = close_ring(build_traverse(start_pt, calls))
        cl = compute_closure_from_line_calls(calls)

        gate_pass = True
        notes = "OK"
        for g in step.gates:
            if g.get("type") == "closure":
                gr = gate_closure(layer_id, cl, g)
                self.audit.log("gate", {"step_id": step.id, "gate": g, "passed": gr.passed, "notes": gr.notes, "details": gr.details})
                if not gr.passed:
                    gate_pass = False
                    notes = gr.notes
                    if not self.force:
                        self.project.diagnostics[f"gate_{step.id}_closure"] = dataclasses.asdict(gr)
                        raise SystemExit(f"FAIL: {gr.notes}")

        self.project.layers[layer_id] = LayerResult(
            id=layer_id,
            build_type="traverse",
            calls_used=call_ids,
            points=pts,
            closure=cl,
            gate_passed=gate_pass,
            gate_notes=notes,
        )


# -----------------------------
# Loop finding: beam search ordering by closure feasibility
# -----------------------------
@dataclass
class LoopSolveResult:
    ordered_call_ids: List[str]
    misclosure_ft: float
    closure_ratio: float
    sum_dn: float
    sum_de: float

def _call_vector(c: RecordCall) -> Tuple[float, float, float]:
    """
    Returns (dn, de, length).
    """
    if c.bearing is None or c.distance_ft is None:
        raise ValueError(f"Call {c.id} missing bearing/distance")
    dn, de = lat_dep_from_call(c.bearing, float(c.distance_ft))
    return dn, de, float(c.distance_ft)

def find_best_loop_order_beam(
    calls: List[RecordCall],
    beam_width: int = 800,
) -> LoopSolveResult:
    """
    Beam search over permutations to minimize final misclosure.

    Defensible framing:
    - This is a numeric proposal engine. It does NOT change any bearings/distances.
    - It only suggests an order that yields the smallest closure error.
    """
    n = len(calls)
    if n < 3:
        raise ValueError("Need at least 3 calls for a loop.")

    vecs = [(_call_vector(c)) for c in calls]
    ids = [c.id for c in calls]
    lengths = [v[2] for v in vecs]
    total_len = sum(lengths)

    # Precompute remaining length sums quickly
    # We'll use a mask-based remaining length sum with a cached list of lengths.
    # For n > 60 you'd need a different representation; typical plats are far below that.
    if n > 60:
        raise ValueError("Too many calls for bitmask beam search (n>60). Split into subsets.")

    # State: (score_bound, sum_dn, sum_de, used_mask, order_list)
    # Bound is optimistic lower bound on achievable misclosure:
    #   lb = max(0, |sum_vec| - remaining_total_length)
    init = (0.0, 0.0, 0.0, 0, [])
    beam = [init]

    # Precompute length by index
    len_by_i = lengths

    for depth in range(n):
        next_beam = []
        for _, sdn, sde, mask, order in beam:
            # remaining length total
            rem_len = 0.0
            for i in range(n):
                if not (mask >> i) & 1:
                    rem_len += len_by_i[i]

            for i in range(n):
                if (mask >> i) & 1:
                    continue
                dn, de, L = vecs[i]
                nsdn = sdn + dn
                nsde = sde + de
                nmask = mask | (1 << i)
                norder = order + [ids[i]]

                # optimistic bound on final misclosure after adding this vector
                # remaining after selecting i:
                rem2 = rem_len - L
                cur_mag = math.hypot(nsdn, nsde)
                lb = max(0.0, cur_mag - rem2)

                next_beam.append((lb, nsdn, nsde, nmask, norder))

        # Keep best beam_width by lb, tie-break by shorter partial |sum|
        next_beam.sort(key=lambda t: (t[0], math.hypot(t[1], t[2])))
        beam = next_beam[:beam_width]

    # Evaluate final states
    best = None
    for _, sdn, sde, mask, order in beam:
        if mask != (1 << n) - 1:
            continue
        mis = math.hypot(sdn, sde)
        ratio = (total_len / mis) if mis > EPS else float("inf")
        cand = (mis, -ratio, sdn, sde, order)  # minimize mis
        if best is None or cand < best:
            best = cand

    if best is None:
        raise RuntimeError("Beam search failed to produce a complete permutation.")

    mis, _, sdn, sde, order = best
    ratio = (total_len / mis) if mis > EPS else float("inf")

    return LoopSolveResult(
        ordered_call_ids=order,
        misclosure_ft=mis,
        closure_ratio=ratio,
        sum_dn=sdn,
        sum_de=sde,
    )


# -----------------------------
# Rendering (SVG + minimal DXF)
# -----------------------------
def escape_xml(s: str) -> str:
    return (s.replace("&", "&amp;")
              .replace("<", "&lt;")
              .replace(">", "&gt;")
              .replace('"', "&quot;")
              .replace("'", "&apos;"))

def centroid(poly: List[Pt]) -> Tuple[float, float]:
    ring = close_ring(poly)
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i].e, ring[i].n
        x2, y2 = ring[i + 1].e, ring[i + 1].n
        cr = x1 * y2 - x2 * y1
        A += cr
        Cx += (x1 + x2) * cr
        Cy += (y1 + y2) * cr
    if abs(A) < EPS:
        xs = [p.e for p in poly]
        ys = [p.n for p in poly]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    A *= 0.5
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)

def render_svg(project: Project, out_path: Path) -> None:
    all_pts: List[Pt] = []
    for lr in project.layers.values():
        all_pts.extend(lr.points)
    for lot in project.lots.values():
        all_pts.extend(lot.polygon)
    if not all_pts:
        raise RuntimeError("No geometry to render (run solve first).")

    min_e = min(p.e for p in all_pts)
    max_e = max(p.e for p in all_pts)
    min_n = min(p.n for p in all_pts)
    max_n = max(p.n for p in all_pts)

    width = max_e - min_e
    height = max_n - min_n
    pad = max(width, height) * 0.05 + 10.0

    vb_min_e = min_e - pad
    vb_min_n = min_n - pad
    vb_w = width + 2 * pad
    vb_h = height + 2 * pad

    def svg_xy(p: Pt) -> Tuple[float, float]:
        x = p.e
        y = (vb_min_n + vb_h) - p.n
        return x, y

    def poly_path(points: List[Pt]) -> str:
        pts = close_ring(points)
        parts = []
        for i, p in enumerate(pts):
            x, y = svg_xy(p)
            parts.append(("M" if i == 0 else "L") + f" {x:.3f} {y:.3f}")
        parts.append("Z")
        return " ".join(parts)

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_min_e:.3f} {((vb_min_n+vb_h)-vb_h):.3f} {vb_w:.3f} {vb_h:.3f}">')
    svg.append('<style>')
    svg.append('.layer{fill:none;stroke:#e9eef6;stroke-width:0.6;}')
    svg.append('.lot{fill:none;stroke:#9aa7ba;stroke-width:0.35;}')
    svg.append('.label{fill:#e9eef6;font-family:Arial, sans-serif;font-size:6px;}')
    svg.append('</style>')

    for lid, lr in project.layers.items():
        if not lr.points:
            continue
        svg.append(f'<path class="layer" d="{poly_path(lr.points)}"/>')
        cx, cy = centroid(lr.points)
        x, y = svg_xy(Pt(cx, cy))
        svg.append(f'<text class="label" x="{x:.3f}" y="{y:.3f}">{escape_xml(lid)}</text>')

    for lot in project.lots.values():
        svg.append(f'<path class="lot" d="{poly_path(lot.polygon)}"/>')
        cx, cy = centroid(lot.polygon)
        x, y = svg_xy(Pt(cx, cy))
        svg.append(f'<text class="label" x="{x:.3f}" y="{y:.3f}">{escape_xml(lot.id)}</text>')

    svg.append("</svg>")
    write_text(out_path, "\n".join(svg))

def render_dxf_minimal(project: Project, out_path: Path) -> None:
    entities: List[str] = []

    def add_line(a: Pt, b: Pt, layer: str):
        entities.extend([
            "0", "LINE",
            "8", layer,
            "10", f"{a.e:.6f}",
            "20", f"{a.n:.6f}",
            "11", f"{b.e:.6f}",
            "21", f"{b.n:.6f}",
        ])

    def add_text(p: Pt, text: str, height: float, layer: str):
        entities.extend([
            "0", "TEXT",
            "8", layer,
            "10", f"{p.e:.6f}",
            "20", f"{p.n:.6f}",
            "40", f"{height:.6f}",
            "1", text,
        ])

    for lid, lr in project.layers.items():
        pts = close_ring(lr.points)
        for i in range(len(pts) - 1):
            add_line(pts[i], pts[i + 1], lid.upper())

    for lot in project.lots.values():
        pts = close_ring(lot.polygon)
        for i in range(len(pts) - 1):
            add_line(pts[i], pts[i + 1], "LOTS")
        cx, cy = centroid(lot.polygon)
        add_text(Pt(cx, cy), lot.id, height=1.5, layer="LOT_LABELS")

    dxf = []
    dxf.extend(["0", "SECTION", "2", "HEADER"])
    dxf.extend(["9", "$ACADVER", "1", "AC1015"])  # AutoCAD 2000
    dxf.extend(["0", "ENDSEC"])
    dxf.extend(["0", "SECTION", "2", "ENTITIES"])
    dxf.extend(entities)
    dxf.extend(["0", "ENDSEC", "0", "EOF"])
    write_text(out_path, "\n".join(dxf))


# -----------------------------
# Reporting (Markdown)
# -----------------------------
def render_table(rows: List[List[str]]) -> str:
    try:
        from tabulate import tabulate  # type: ignore
        headers = rows[0]
        body = rows[1:]
        return tabulate(body, headers=headers, tablefmt="github")
    except Exception:
        widths = [0] * len(rows[0])
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(str(cell)))
        out = []
        for idx, r in enumerate(rows):
            out.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))) + " |")
            if idx == 0:
                out.append("| " + " | ".join("-" * widths[i] for i in range(len(r))) + " |")
        return "\n".join(out)

def report_markdown(project: Project) -> str:
    lines: List[str] = []
    lines.append("# Survey Record Reconstruction Report")
    lines.append("")
    lines.append(f"- Project version: {project.version}")
    lines.append(f"- Created: {project.created_at}")
    lines.append(f"- Image: `{project.image_path}`")
    lines.append("")

    lines.append("## A) EXTRACTED RECORD CALLS")
    lines.append("")
    rows = [["id", "kind", "layer", "seq", "bearing", "distance_ft", "tags", "evidence", "seg", "seg_dist_px"]]
    for cid, c in sorted(project.calls.items(), key=lambda kv: kv[0]):
        b = c.bearing.format() if c.bearing else (c.bearing_text or "")
        d = fmt_ft(float(c.distance_ft)) if c.distance_ft is not None else (c.distance_text or "")
        rows.append([
            cid, c.kind, c.layer, str(c.seq) if c.seq is not None else "",
            b, d, ",".join(c.tags), c.evidence.type,
            c.assoc_segment_id or "", f"{c.assoc_segment_dist_px:.1f}" if c.assoc_segment_dist_px is not None else ""
        ])
    lines.append(render_table(rows))
    lines.append("")

    lines.append("## B) DETECTED LINE SEGMENTS (IMAGE SPACE)")
    lines.append("")
    rows = [["id", "len_px", "angle_deg", "x1", "y1", "x2", "y2"]]
    for s in project.segments[:80]:
        rows.append([s.id, f"{s.length_px:.1f}", f"{s.angle_deg:.1f}", str(s.x1), str(s.y1), str(s.x2), str(s.y2)])
    if len(project.segments) > 80:
        rows.append(["…", f"(+{len(project.segments)-80} more)", "", "", "", "", ""])
    lines.append(render_table(rows))
    lines.append("")

    lines.append("## C) LAYER CLOSURES")
    lines.append("")
    rows = [["layer", "build", "ΣΔN (ft)", "ΣΔE (ft)", "miscl (ft)", "miscl (in)", "miscl bearing", "total (ft)", "ratio", "gate"]]
    for lid, lr in project.layers.items():
        cl = lr.closure
        rows.append([
            lid,
            lr.build_type,
            fmt_ft_mis(cl.sum_dn),
            fmt_ft_mis(cl.sum_de),
            fmt_ft_mis(cl.misclosure_ft),
            fmt_ft_mis(ft_to_inches(cl.misclosure_ft)),
            cl.misclosure_bearing.format(),
            fmt_ft(cl.total_length_ft),
            "∞" if not math.isfinite(cl.closure_ratio) else f"1:{cl.closure_ratio:.0f}",
            "PASS" if lr.gate_passed else "FAIL",
        ])
    lines.append(render_table(rows))
    lines.append("")

    if project.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(project.diagnostics, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# Interactive review utilities
# -----------------------------
def prompt_yes_no(msg: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        v = input(msg + suffix).strip().lower()
        if not v:
            return default
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False

def prompt_str(msg: str, default: str = "") -> str:
    v = input(f"{msg}{' ['+default+']' if default else ''}: ").strip()
    return v if v else default

def prompt_int(msg: str, default: Optional[int] = None) -> Optional[int]:
    v = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default

def prompt_choice(msg: str, choices: List[str], default: Optional[str] = None) -> str:
    ch = ", ".join(choices)
    while True:
        v = input(f"{msg} ({ch}){(' ['+default+']') if default else ''}: ").strip()
        if not v and default:
            return default
        if v in choices:
            return v
        print("Invalid choice.")


# -----------------------------
# Plan generation helper
# -----------------------------
def generate_plan_from_classified_traverses(project: Project) -> Plan:
    """
    Generates a starter plan.json based on calls that have:
      - kind='line'
      - layer != 'unassigned'
      - seq set
    Creates one traverse step per layer.
    """
    layer_map: Dict[str, List[RecordCall]] = {}
    for c in project.calls.values():
        if c.kind != "line":
            continue
        if c.layer == "unassigned":
            continue
        if c.seq is None:
            continue
        layer_map.setdefault(c.layer, []).append(c)

    steps: List[PlanStep] = []
    for layer_id, calls in sorted(layer_map.items(), key=lambda kv: kv[0]):
        ordered = sorted(calls, key=lambda c: c.seq if c.seq is not None else 0)
        call_ids = [c.id for c in ordered]
        gates = [{"type": "closure",
                  "max_mis_ft": project.flags.get("default_max_mis_ft", 0.05),
                  "min_ratio": project.flags.get("default_min_ratio", 50000)}]
        steps.append(PlanStep(
            id=f"traverse_{layer_id}",
            type="traverse",
            params={"layer_id": layer_id, "start": {"e": 0.0, "n": 0.0}, "call_ids": call_ids},
            gates=gates,
        ))

    return Plan(
        id=project.flags.get("plan_id", "plan"),
        description="Starter plan generated from classified traverse calls. Add interior steps as needed.",
        steps=steps,
    )


# -----------------------------
# CLI commands
# -----------------------------
def cmd_init(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    proj = Project(
        version=PROJECT_VERSION,
        created_at=now_iso(),
        image_path=str(Path(args.image).resolve()),
        out_dir=str(out_dir),
    )

    proj.flags.setdefault("default_max_mis_ft", 0.05)
    proj.flags.setdefault("default_min_ratio", 50000)
    proj.flags.setdefault("plan_id", "plan")
    proj.flags.setdefault("force", False)

    proj_path = out_dir / "project.json"
    proj.save(proj_path)
    audit.log("init", {"project": str(proj_path), "image": proj.image_path, "out": str(out_dir)})
    print(f"Created project: {proj_path}")

def cmd_detect(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    cands = ocr_scan_candidates(Path(proj.image_path))
    existing = {c.id for c in proj.candidates}
    added = 0
    for c in cands:
        if c.id not in existing:
            proj.candidates.append(c)
            added += 1

    proj.save(proj_path)
    audit.log("detect", {"added": added, "total": len(proj.candidates)})
    print(f"Detected {added} new candidates (total {len(proj.candidates)}).")

def cmd_review(args: argparse.Namespace) -> None:
    """
    Interactive evidence confirmation:
    - Accept OCR candidates into record calls with snippet evidence
    - Optionally compose bearing+distance into line calls
    """
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")
    snips_dir = ensure_dir(out_dir / "snips")
    img_path = Path(proj.image_path)

    for cand in sorted(proj.candidates, key=lambda c: (-c.confidence, c.id)):
        print("\n------------------------------------------------------------")
        print(f"Candidate: {cand.id}  guess={cand.kind_guess}  conf~{cand.confidence:.1f}")
        print(f"Text: {cand.text!r}")
        print(f"BBox: {cand.bbox}")

        take = prompt_yes_no("Accept this candidate as a call?", default=(cand.confidence >= 70))
        if not take:
            audit.log("review_skip", {"cand": cand.id})
            continue

        snip_path = snips_dir / f"{cand.id}.png"
        try:
            crop_save_snip(img_path, cand.bbox, snip_path)
        except Exception as e:
            print(f"[WARN] Could not save snippet: {e}")
            snip_path = None  # type: ignore

        kind_default = cand.kind_guess if cand.kind_guess in ("bearing", "distance") else "text"
        kind = prompt_choice("Call kind", ["bearing", "distance", "text"], default=kind_default)
        call_id = prompt_str("Call id (unique)", default=f"{kind}_{cand.id}")

        rc = RecordCall(
            id=call_id,
            kind=kind,
            layer=prompt_str("Layer name (e.g., outer, block, etc.)", default="unassigned"),
            seq=prompt_int("Sequence number in traverse (optional)", default=None),
            tags=["record"],
            evidence=Evidence(
                type="user_confirmed",
                bbox=cand.bbox,
                snip_path=str(snip_path) if snip_path else None,
                ocr_text=cand.text,
                confidence=cand.confidence,
            ),
        )

        raw = prompt_str("Final printed text (edit if OCR wrong)", default=cand.text)

        try:
            if kind == "bearing":
                rc.bearing_text = raw
                rc.bearing = parse_quadrant_bearing(raw)
            elif kind == "distance":
                rc.distance_text = raw
                rc.distance_ft = parse_distance_ft(raw)
            else:
                rc.distance_text = raw
        except Exception as e:
            raise SystemExit(f"Parse error: {e}")

        proj.calls[call_id] = rc
        audit.log("review_accept", {"cand": cand.id, "call_id": call_id, "kind": kind, "layer": rc.layer, "seq": rc.seq})

    while prompt_yes_no("Create a LINE call from existing bearing+distance calls?", default=False):
        line_id = prompt_str("Line call id", default=f"line_{stable_id(now_iso())}")
        layer = prompt_str("Layer for this line", default="unassigned")
        seq = prompt_int("Sequence number in traverse (optional)", default=None)

        b_id = prompt_str("Bearing call id")
        d_id = prompt_str("Distance call id")
        if b_id not in proj.calls or d_id not in proj.calls:
            print("Missing bearing or distance call id.")
            continue
        b = proj.calls[b_id]
        d = proj.calls[d_id]
        if b.bearing is None:
            print("Bearing call missing parsed bearing.")
            continue
        if d.distance_ft is None:
            print("Distance call missing parsed distance.")
            continue

        rc = RecordCall(
            id=line_id,
            kind="line",
            layer=layer,
            seq=seq,
            bearing=b.bearing,
            distance_ft=float(d.distance_ft),
            bearing_text=b.bearing_text,
            distance_text=d.distance_text,
            tags=["record"],
            evidence=Evidence(type="computed", notes=f"composed from {b_id}+{d_id}"),
        )
        proj.calls[line_id] = rc
        audit.log("line_compose", {"line_id": line_id, "from_bearing": b_id, "from_distance": d_id, "layer": layer, "seq": seq})
        print(f"Created line call {line_id}")

    proj.save(proj_path)
    print(f"Updated project: {proj_path}")

def cmd_lines_detect(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    segs = detect_line_segments_hough(
        Path(proj.image_path),
        canny1=args.canny1,
        canny2=args.canny2,
        hough_thresh=args.hough,
        min_len=args.min_len,
        max_gap=args.max_gap,
        max_segments=args.max_segments,
    )
    proj.segments = segs
    proj.save(proj_path)
    audit.log("lines_detect", {"count": len(segs), "params": {
        "canny1": args.canny1, "canny2": args.canny2, "hough": args.hough,
        "min_len": args.min_len, "max_gap": args.max_gap, "max_segments": args.max_segments
    }})
    print(f"Detected {len(segs)} line segments and stored in project.")

def cmd_associate(args: argparse.Namespace) -> None:
    """
    Associates each record call having an evidence bbox to the nearest detected line segment.
    This is a PROPOSAL linkage: it helps you group calls by the edge they belong to.
    """
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    if not proj.segments:
        raise SystemExit("No segments found. Run lines-detect first.")

    updated = 0
    for c in proj.calls.values():
        if c.evidence.bbox is None:
            continue
        x, y, w, h = c.evidence.bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        best = None
        for seg in proj.segments:
            dpx = point_to_segment_distance_px(cx, cy, seg)
            if best is None or dpx < best[0]:
                best = (dpx, seg.id)

        if best is None:
            continue
        dpx, sid = best
        if args.max_dist_px is not None and dpx > args.max_dist_px:
            continue

        c.assoc_segment_id = sid
        c.assoc_segment_dist_px = float(dpx)
        updated += 1

    proj.save(proj_path)
    audit.log("associate", {"updated": updated, "max_dist_px": args.max_dist_px})
    print(f"Associated {updated} calls to nearest segments.")

def cmd_autotrav(args: argparse.Namespace) -> None:
    """
    Auto-order line calls into a closure-minimizing loop (beam search).
    - Select calls by --layer, or explicit --call-ids
    - Writes seq numbers if --set-seq
    - Optionally suggests a traverse plan step in project.flags if --suggest-plan
    """
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    call_ids: List[str] = []
    if args.call_ids:
        call_ids = [s.strip() for s in args.call_ids.split(",") if s.strip()]
    else:
        # all line calls in layer
        for cid, c in proj.calls.items():
            if c.kind == "line" and c.layer == args.layer:
                call_ids.append(cid)

    if not call_ids:
        raise SystemExit("No calls selected for autotrav (check --layer or --call-ids).")

    calls = []
    for cid in call_ids:
        if cid not in proj.calls:
            raise SystemExit(f"Missing call id: {cid}")
        c = proj.calls[cid]
        if c.kind != "line":
            raise SystemExit(f"{cid} is not kind='line'")
        if c.bearing is None or c.distance_ft is None:
            raise SystemExit(f"{cid} missing bearing/distance")
        if not c.is_record():
            raise SystemExit(f"{cid} must be tagged 'record' (auto-order does not accept untagged calls)")
        calls.append(c)

    res = find_best_loop_order_beam(calls, beam_width=args.beam)
    audit.log("autotrav_result", {"layer": args.layer, "beam": args.beam, "result": dataclasses.asdict(res)})

    print("\n=== Auto Traverse Proposal ===")
    print(f"Calls: {len(calls)}")
    print(f"Misclosure: {res.misclosure_ft:.4f} ft ({ft_to_inches(res.misclosure_ft):.2f} in)")
    if math.isfinite(res.closure_ratio):
        print(f"Closure ratio: 1:{res.closure_ratio:.0f}")
    else:
        print("Closure ratio: ∞")
    print("Order:")
    for i, cid in enumerate(res.ordered_call_ids, start=1):
        print(f"  {i:02d}. {cid}")

    if args.set_seq:
        for i, cid in enumerate(res.ordered_call_ids, start=1):
            proj.calls[cid].seq = i
        proj.save(proj_path)
        audit.log("autotrav_set_seq", {"count": len(res.ordered_call_ids), "layer": args.layer})
        print(f"\nWrote seq=1..{len(res.ordered_call_ids)} into calls (layer={args.layer}).")

    # Suggest a plan step (does not overwrite plan.json; it writes a proposal in project.flags)
    if args.suggest_plan:
        step = {
            "id": f"traverse_{args.layer}",
            "type": "traverse",
            "params": {
                "layer_id": args.layer,
                "start": {"e": args.start_e, "n": args.start_n},
                "call_ids": res.ordered_call_ids,
            },
            "gates": [{
                "type": "closure",
                "max_mis_ft": proj.flags.get("default_max_mis_ft", 0.05),
                "min_ratio": proj.flags.get("default_min_ratio", 50000),
            }]
        }
        proj.flags.setdefault("plan_step_proposals", [])
        proj.flags["plan_step_proposals"].append(step)
        proj.save(proj_path)
        audit.log("autotrav_suggest_plan", {"step": step})
        print("\nStored plan step proposal in project.flags['plan_step_proposals'].")

def cmd_plan_new(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    plan = generate_plan_from_classified_traverses(proj)

    # If proposals exist, append them (in order added)
    proposals = proj.flags.get("plan_step_proposals", [])
    for p in proposals:
        plan.steps.append(PlanStep(id=p["id"], type=p["type"], params=p.get("params", {}), gates=p.get("gates", [])))

    out_plan = Path(args.out_plan).resolve() if args.out_plan else (out_dir / "plan.json")
    plan.save(out_plan)
    audit.log("plan_new", {"out_plan": str(out_plan), "steps": [s.id for s in plan.steps]})
    print(f"Wrote plan: {out_plan}")

def cmd_solve(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    plan = Plan.load(Path(args.plan).resolve())
    force = bool(args.force)
    proj.flags["force"] = force

    PlanSolver(proj, audit, force=force).solve(plan)
    proj.save(proj_path)
    print(f"Solved and updated project: {proj_path}")

def cmd_render(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    if args.svg:
        svg_path = out_dir / "drawing.svg"
        render_svg(proj, svg_path)
        audit.log("render_svg", {"path": str(svg_path)})
        print(f"Wrote {svg_path}")

    if args.dxf:
        dxf_path = out_dir / "drawing.dxf"
        render_dxf_minimal(proj, dxf_path)
        audit.log("render_dxf", {"path": str(dxf_path)})
        print(f"Wrote {dxf_path}")

def cmd_report(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    md = report_markdown(proj)
    report_path = out_dir / "report.md"
    write_text(report_path, md)
    audit.log("report", {"path": str(report_path)})
    print(f"Wrote {report_path}")


# -----------------------------
# CLI plumbing
# -----------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="survey-recon", description="General survey record reconstruction CLI (evidence-first).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="Initialize a project.")
    s.add_argument("image", help="Plat image path (PNG/JPG).")
    s.add_argument("--out", required=True, help="Output directory for the project.")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("detect", help="OCR scan to propose call candidates.")
    s.add_argument("project", help="Project JSON path.")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("review", help="Interactive review of candidates into confirmed record calls.")
    s.add_argument("project", help="Project JSON path.")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("lines-detect", help="Detect drawn line segments in plat image (OpenCV Hough).")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--canny1", type=int, default=50)
    s.add_argument("--canny2", type=int, default=150)
    s.add_argument("--hough", type=int, default=60)
    s.add_argument("--min-len", type=int, default=60)
    s.add_argument("--max-gap", type=int, default=10)
    s.add_argument("--max-segments", type=int, default=400)
    s.set_defaults(func=cmd_lines_detect)

    s = sub.add_parser("associate", help="Associate confirmed calls (with bbox) to nearest detected line segment.")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--max-dist-px", type=float, default=None, help="Ignore associations beyond this pixel distance.")
    s.set_defaults(func=cmd_associate)

    s = sub.add_parser("autotrav", help="Auto-order selected line calls into a loop (beam search) and optionally set seq.")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--layer", default="outer", help="Layer to select calls from if --call-ids not provided.")
    s.add_argument("--call-ids", default="", help="Comma-separated call ids to order (overrides --layer).")
    s.add_argument("--beam", type=int, default=800, help="Beam width (higher = more robust, more CPU).")
    s.add_argument("--set-seq", action="store_true", help="Write seq numbers into calls per proposed order.")
    s.add_argument("--suggest-plan", action="store_true", help="Store a traverse plan step proposal in project.flags.")
    s.add_argument("--start-e", type=float, default=0.0)
    s.add_argument("--start-n", type=float, default=0.0)
    s.set_defaults(func=cmd_autotrav)

    s = sub.add_parser("plan-new", help="Generate plan.json from calls (layer + seq). Also appends plan_step_proposals if present.")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--out-plan", default="", help="Output plan path (default: <out_dir>/plan.json)")
    s.set_defaults(func=cmd_plan_new)

    s = sub.add_parser("solve", help="Solve geometry using a plan.json.")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--plan", required=True, help="Plan JSON path.")
    s.add_argument("--force", action="store_true", help="Bypass gate failures (marks outputs non-defensible).")
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("render", help="Render outputs (SVG/DXF).")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--svg", action="store_true")
    s.add_argument("--dxf", action="store_true")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("report", help="Write report.md")
    s.add_argument("project", help="Project JSON path.")
    s.set_defaults(func=cmd_report)

    return p

def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_cli()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
