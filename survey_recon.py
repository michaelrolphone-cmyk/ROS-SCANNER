#!/usr/bin/env python3
"""
survey_recon.py — General Record of Survey reconstruction CLI (evidence-first, plan-driven)
v1.2.0-skeleton  (adds: subset selection + curves)

ADDED IN THIS VERSION
1) SUBSET SELECTION ("autoloops"):
   - Given many record line/curve calls, automatically proposes a *subset* that best closes
     (min misclosure) AND looks like an outer boundary (max area/perimeter, low self-intersections).
   - Then finds the best *ordering* of that subset (beam search) and can apply layer+seq.

2) CURVES:
   - Supports record curve calls (kind="curve") with radius + chord bearing + chord length + direction.
   - Traverse builder supports curve segments, closure uses chord vector, perimeter uses arc length.
   - DXF writer outputs LWPOLYLINE with bulges (works for left and right curves).
   - SVG writer outputs true arcs (A commands) based on computed center/radius.

DEFENSIBILITY RULES ENFORCED
- "record" values must come from OCR/user-confirmed snippets (or user entry during review).
- "computed" segments must carry derivation notes.
- Auto steps only PROPOSE; application (layer/seq) is explicit and logged.
- Gates stop the run unless --force, which stamps diagnostics.

WORKFLOW (typical)
  python survey_recon.py init plat.png --out out/myros
  python survey_recon.py detect out/myros/project.json
  python survey_recon.py review out/myros/project.json     # includes line-compose + curve-compose
  python survey_recon.py lines-detect out/myros/project.json
  python survey_recon.py associate out/myros/project.json

  # Propose outer loop subset + order:
  python survey_recon.py autoloops out/myros/project.json --pool-layer any --min-sides 4 --max-sides 12 --top 5
  python survey_recon.py autoloops out/myros/project.json --pool-layer any --apply 1 --set-layer outer --set-seq --suggest-plan

  python survey_recon.py plan-new out/myros/project.json --out-plan out/myros/plan.json
  python survey_recon.py solve out/myros/project.json --plan out/myros/plan.json
  python survey_recon.py render out/myros/project.json --svg --dxf
  python survey_recon.py report out/myros/project.json

OPTIONAL DEPS
  pip install pillow pytesseract
  pip install opencv-python
  pip install tabulate
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
PROJECT_VERSION = "1.2.0-skeleton"

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

def dot(a: Vec, b: Vec) -> float:
    return a.de * b.de + a.dn * b.dn

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

def polygon_area(points: List[Pt]) -> float:
    ring = close_ring(points)
    a = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i].e, ring[i].n
        x2, y2 = ring[i + 1].e, ring[i + 1].n
        a += x1 * y2 - x2 * y1
    return 0.5 * a  # signed

def polygon_perimeter(points: List[Pt]) -> float:
    ring = close_ring(points)
    return sum(dist(ring[i], ring[i + 1]) for i in range(len(ring) - 1))


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

ANGLE_DMS_RE = re.compile(
    r"""
    (?P<deg>\d{1,3})\s*(?:°|º|d|deg)\s*
    (?P<min>\d{1,2})\s*(?:'|m|min)\s*
    (?P<sec>\d{1,2}(?:\.\d+)?)\s*(?:"|s|sec)
    """,
    re.IGNORECASE | re.VERBOSE,
)

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

def parse_angle_deg(text: str) -> float:
    """
    Accepts:
      - DMS like 10°20'30"
      - or decimal like 10.345
    """
    t = text.strip().replace("º", "°")
    m = ANGLE_DMS_RE.search(t)
    if m:
        deg = int(m.group("deg"))
        minutes = int(m.group("min"))
        seconds = float(m.group("sec"))
        if not (0 <= minutes < 60) or not (0 <= seconds < 60):
            raise ValueError(f"Angle minutes/seconds out of range: {minutes} {seconds}")
        return deg + minutes / 60.0 + seconds / 3600.0
    # decimal fallback
    try:
        return float(re.findall(r"[-+]?\d+(?:\.\d+)?", t)[0])
    except Exception as e:
        raise ValueError(f"Could not parse angle: {text!r}") from e


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

    # Curve fields (for kind="curve")
    curve_dir: Optional[str] = None        # 'L' or 'R' (CCW or CW in model space)
    curve_radius_ft: Optional[float] = None
    curve_delta_deg: Optional[float] = None
    curve_arc_len_ft: Optional[float] = None
    curve_chord_bearing_text: Optional[str] = None
    curve_chord_bearing: Optional[QuadrantBearing] = None
    curve_chord_len_ft: Optional[float] = None

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
# Image line segments + association (optional)
# -----------------------------
@dataclass
class LineSegment:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float
    angle_deg: float  # 0..180 measured from +x axis
    source: str = "hough"

def point_to_segment_distance_px(px: float, py: float, seg: LineSegment) -> float:
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
# Traverse primitives (for render)
# -----------------------------
@dataclass
class Primitive:
    type: str                # 'line' or 'arc'
    start: Pt
    end: Pt
    bulge: float = 0.0       # for DXF LWPOLYLINE
    radius_ft: Optional[float] = None
    center: Optional[Pt] = None
    dir: Optional[str] = None        # 'L'/'R'
    delta_deg: Optional[float] = None


# -----------------------------
# Closures / Results
# -----------------------------
@dataclass
class Closure:
    sum_dn: float
    sum_de: float
    misclosure_ft: float
    total_length_ft: float
    closure_ratio: float

@dataclass
class LayerResult:
    id: str
    build_type: str
    calls_used: List[str]
    vertices: List[Pt]             # without duplicated start
    closed: bool
    primitives: List[Primitive]    # segment-by-segment
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
            return {"ns": q.ns, "ew": q.ew, "deg": q.angle.deg, "min": q.angle.minutes, "sec": q.angle.seconds}

        def pt_to_list(p: Pt) -> List[float]:
            return [p.e, p.n]

        def primitive_to_dict(p: Primitive) -> Dict[str, Any]:
            return {
                "type": p.type,
                "start": pt_to_list(p.start),
                "end": pt_to_list(p.end),
                "bulge": p.bulge,
                "radius_ft": p.radius_ft,
                "center": (None if p.center is None else pt_to_list(p.center)),
                "dir": p.dir,
                "delta_deg": p.delta_deg,
            }

        def closure_to_dict(c: Closure) -> Dict[str, Any]:
            return {
                "sum_dn": c.sum_dn,
                "sum_de": c.sum_de,
                "misclosure_ft": c.misclosure_ft,
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
                    "curve_chord_bearing": qb_to_dict(v.curve_chord_bearing),
                }
                for k, v in self.calls.items()
            },
            "layers": {
                k: {
                    "id": v.id,
                    "build_type": v.build_type,
                    "calls_used": v.calls_used,
                    "vertices": [pt_to_list(p) for p in v.vertices],
                    "closed": v.closed,
                    "primitives": [primitive_to_dict(p) for p in v.primitives],
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
            return QuadrantBearing(d["ns"], DMS(int(d["deg"]), int(d["min"]), float(d["sec"])), d["ew"])

        def list_to_pt(a: List[float]) -> Pt:
            return Pt(float(a[0]), float(a[1]))

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

                curve_dir=v.get("curve_dir"),
                curve_radius_ft=v.get("curve_radius_ft"),
                curve_delta_deg=v.get("curve_delta_deg"),
                curve_arc_len_ft=v.get("curve_arc_len_ft"),
                curve_chord_bearing_text=v.get("curve_chord_bearing_text"),
                curve_chord_bearing=dict_to_qb(v.get("curve_chord_bearing")),
                curve_chord_len_ft=v.get("curve_chord_len_ft"),

                tags=v.get("tags", []),
                derivation=v.get("derivation"),
                evidence=Evidence(**ev),
                seq=v.get("seq"),
                assoc_segment_id=v.get("assoc_segment_id"),
                assoc_segment_dist_px=v.get("assoc_segment_dist_px"),
            )
            calls[k] = rc
        proj.calls = calls

        layers = {}
        for k, v in (obj.get("layers") or {}).items():
            prims = []
            for pd in (v.get("primitives") or []):
                prims.append(Primitive(
                    type=pd["type"],
                    start=list_to_pt(pd["start"]),
                    end=list_to_pt(pd["end"]),
                    bulge=float(pd.get("bulge", 0.0)),
                    radius_ft=pd.get("radius_ft"),
                    center=(None if pd.get("center") is None else list_to_pt(pd["center"])),
                    dir=pd.get("dir"),
                    delta_deg=pd.get("delta_deg"),
                ))
            cld = v["closure"]
            layers[k] = LayerResult(
                id=v["id"],
                build_type=v["build_type"],
                calls_used=list(v["calls_used"]),
                vertices=[list_to_pt(p) for p in v["vertices"]],
                closed=bool(v.get("closed", False)),
                primitives=prims,
                closure=Closure(
                    sum_dn=float(cld["sum_dn"]),
                    sum_de=float(cld["sum_de"]),
                    misclosure_ft=float(cld["misclosure_ft"]),
                    total_length_ft=float(cld["total_length_ft"]),
                    closure_ratio=float(cld["closure_ratio"]),
                ),
                gate_passed=bool(v["gate_passed"]),
                gate_notes=str(v["gate_notes"]),
            )
        proj.layers = layers

        lots = {}
        for k, v in (obj.get("lots") or {}).items():
            cld = v["closure"]
            lots[k] = LotResult(
                id=v["id"],
                polygon=[list_to_pt(p) for p in v["polygon"]],
                perimeter_ft=float(v["perimeter_ft"]),
                area_sqft=float(v["area_sqft"]),
                closure=Closure(
                    sum_dn=float(cld["sum_dn"]),
                    sum_de=float(cld["sum_de"]),
                    misclosure_ft=float(cld["misclosure_ft"]),
                    total_length_ft=float(cld["total_length_ft"]),
                    closure_ratio=float(cld["closure_ratio"]),
                ),
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

    for L in lines:
        x1, y1, x2, y2 = [int(v) for v in L[0]]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        ang = abs(ang)
        if ang > 180:
            ang = 360 - ang
        sid = stable_id("seg", x1, y1, x2, y2)
        segs.append(LineSegment(id=sid, x1=x1, y1=y1, x2=x2, y2=y2, length_px=length, angle_deg=ang))

    segs.sort(key=lambda s: s.length_px, reverse=True)
    return segs[:max_segments]


# -----------------------------
# Curve parsing helpers
# -----------------------------
CURVE_R_RE = re.compile(r"(?:\bR\b|R=|RADIUS)\s*=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
CURVE_L_RE = re.compile(r"(?:\bL\b|LEN|ARC)\s*=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
CURVE_CH_RE = re.compile(r"(?:\bCH\b|CHD|CHORD)\s*=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
CURVE_DIR_RE = re.compile(r"\b(LT|LEFT|RT|RIGHT|L|R)\b", re.IGNORECASE)

def parse_curve_text(raw: str) -> Dict[str, Any]:
    """
    Very tolerant curve parser. Intended for user-confirmed curve table lines.
    Extracts: radius, arc length, chord length, chord bearing, delta, direction.
    """
    t = raw.strip().replace("º", "°")
    out: Dict[str, Any] = {}

    # direction
    md = CURVE_DIR_RE.search(t)
    if md:
        d = md.group(1).upper()
        if d in ("LT", "LEFT", "L"):
            out["dir"] = "L"
        elif d in ("RT", "RIGHT", "R"):
            out["dir"] = "R"

    mr = CURVE_R_RE.search(t)
    if mr:
        out["radius_ft"] = float(mr.group(1))

    ml = CURVE_L_RE.search(t)
    if ml:
        out["arc_len_ft"] = float(ml.group(1))

    mch = CURVE_CH_RE.search(t)
    if mch:
        out["chord_len_ft"] = float(mch.group(1))

    # chord bearing: find a quadrant bearing anywhere
    mb = BEARING_RE.search(t)
    if mb:
        out["chord_bearing_text"] = mb.group(0)
        out["chord_bearing"] = parse_quadrant_bearing(mb.group(0))

    # delta: look for Δ or DELTA with an angle pattern
    mdel = re.search(r"(?:Δ|DELTA)\s*=?\s*([0-9°'\"\s\.]+)", t, re.IGNORECASE)
    if mdel:
        out["delta_deg"] = parse_angle_deg(mdel.group(1))

    return out


# -----------------------------
# Traverse math (supports lines + curves)
# -----------------------------
def _curve_requirements(c: RecordCall) -> None:
    if c.curve_dir not in ("L", "R"):
        raise ValueError(f"Curve {c.id} missing curve_dir (L/R)")
    if c.curve_radius_ft is None or c.curve_radius_ft <= 0:
        raise ValueError(f"Curve {c.id} missing radius")
    # chord bearing/len is the primary supported representation
    if c.curve_chord_bearing is None or c.curve_chord_len_ft is None:
        raise ValueError(f"Curve {c.id} requires chord bearing + chord length (for v1.2)")
    if c.curve_chord_len_ft <= 0:
        raise ValueError(f"Curve {c.id} chord length invalid")

def _curve_delta_rad(c: RecordCall) -> float:
    # derive delta from chord length + radius if missing
    R = float(c.curve_radius_ft)
    C = float(c.curve_chord_len_ft)
    x = min(1.0, max(-1.0, C / (2.0 * R)))
    theta = 2.0 * math.asin(x)  # radians, magnitude
    if c.curve_delta_deg is None:
        return theta
    # trust record delta if provided, but sanity check
    rec = math.radians(float(c.curve_delta_deg))
    # if wildly inconsistent, keep record but caller should gate in practice
    return rec

def _curve_arc_len(c: RecordCall) -> float:
    if c.curve_arc_len_ft is not None and c.curve_arc_len_ft > 0:
        return float(c.curve_arc_len_ft)
    R = float(c.curve_radius_ft)
    theta = _curve_delta_rad(c)
    return R * theta

def _curve_endpoint_vec(c: RecordCall) -> Vec:
    # chord bearing + chord length defines end point relative to start
    az = c.curve_chord_bearing.to_azimuth_deg()
    u = vec_from_az_deg(az).unit()
    C = float(c.curve_chord_len_ft)
    return u.scale(C)

def _curve_center_from_chord(start: Pt, end: Pt, R: float, dir_lr: str) -> Pt:
    v = end - start
    C = v.mag()
    if C < EPS:
        raise ValueError("Curve chord is zero")
    if R + 1e-9 < C / 2.0:
        raise ValueError(f"Radius {R} too small for chord {C} (R < C/2)")

    u = v.unit()
    # left perpendicular (CCW rotation) in (E,N): (-dn, de)
    left = Vec(-u.dn, u.de)
    right = Vec(u.dn, -u.de)

    h = math.sqrt(max(0.0, R * R - (C * 0.5) * (C * 0.5)))
    mid = Pt((start.e + end.e) * 0.5, (start.n + end.n) * 0.5)
    if dir_lr == "L":
        return mid + left.scale(h)
    return mid + right.scale(h)

def _bulge_from_delta(delta_rad_signed: float) -> float:
    return math.tan(delta_rad_signed / 4.0)

def call_endpoint_and_primitive(cur: Pt, c: RecordCall) -> Tuple[Pt, float, Primitive]:
    """
    Returns:
      next_point, length_for_total, primitive (line/arc)
    For closure vector sums, we always use endpoint vector (next - cur).
    For total length, lines use distance_ft, curves use arc length (or computed).
    """
    if c.kind == "line":
        if c.bearing is None or c.distance_ft is None:
            raise ValueError(f"Line {c.id} missing bearing/distance")
        u = vec_from_az_deg(c.bearing.to_azimuth_deg()).unit()
        L = float(c.distance_ft)
        nxt = cur + u.scale(L)
        prim = Primitive(type="line", start=cur, end=nxt, bulge=0.0)
        return nxt, L, prim

    if c.kind == "curve":
        _curve_requirements(c)
        dv = _curve_endpoint_vec(c)
        nxt = cur + dv
        R = float(c.curve_radius_ft)
        theta = _curve_delta_rad(c)
        # signed theta: L is +CCW, R is -CW
        signed = theta if c.curve_dir == "L" else -theta
        bulge = _bulge_from_delta(signed)
        center = _curve_center_from_chord(cur, nxt, R, c.curve_dir)
        prim = Primitive(
            type="arc",
            start=cur,
            end=nxt,
            bulge=bulge,
            radius_ft=R,
            center=center,
            dir=c.curve_dir,
            delta_deg=math.degrees(theta),
        )
        return nxt, _curve_arc_len(c), prim

    raise ValueError(f"Unsupported call kind in traverse: {c.kind}")

def closure_from_calls(calls: List[RecordCall]) -> Closure:
    sum_dn = 0.0
    sum_de = 0.0
    total_len = 0.0
    cur = Pt(0.0, 0.0)
    for c in calls:
        nxt, seg_len, _ = call_endpoint_and_primitive(cur, c)
        dv = nxt - cur
        sum_de += dv.de
        sum_dn += dv.dn
        total_len += seg_len
        cur = nxt

    mis = math.hypot(sum_dn, sum_de)
    ratio = (total_len / mis) if mis > EPS else float("inf")
    return Closure(sum_dn=sum_dn, sum_de=sum_de, misclosure_ft=mis, total_length_ft=total_len, closure_ratio=ratio)

def build_traverse_vertices_and_prims(start: Pt, calls: List[RecordCall], close_tol: float = 0.10) -> Tuple[List[Pt], bool, List[Primitive]]:
    verts: List[Pt] = [start]
    prims: List[Primitive] = []
    cur = start
    for c in calls:
        nxt, _, prim = call_endpoint_and_primitive(cur, c)
        prims.append(prim)
        verts.append(nxt)
        cur = nxt

    # verts currently includes final endpoint
    closed = dist(verts[0], verts[-1]) <= close_tol
    if closed:
        verts = verts[:-1]  # drop duplicate
    return verts, closed, prims


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
# Plan Solver (supports traverse lines+curves)
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
            if c.kind not in ("line", "curve"):
                raise SystemExit(f"{layer_id}: Call {cid} must be kind='line' or 'curve'")
            if not (c.is_record() or c.is_computed()):
                raise SystemExit(f"{layer_id}: Call {cid} must be tagged 'record' or 'computed'")
            calls.append(c)

        cl = closure_from_calls(calls)
        verts, closed, prims = build_traverse_vertices_and_prims(start_pt, calls, close_tol=float(step.params.get("close_tol_ft", 0.10)))

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
            vertices=verts,
            closed=closed,
            primitives=prims,
            closure=cl,
            gate_passed=gate_pass,
            gate_notes=notes,
        )


# -----------------------------
# Loop finding (ordering) + subset selection (outer loop proposals)
# -----------------------------
@dataclass
class LoopSolveResult:
    ordered_call_ids: List[str]
    misclosure_ft: float
    closure_ratio: float
    sum_dn: float
    sum_de: float
    total_len_ft: float

def _call_vector_and_len(c: RecordCall) -> Tuple[float, float, float]:
    """
    Returns (dn, de, total_len_contrib).
    For lines: dn,de from bearing/distance and total length=distance.
    For curves: dn,de from chord and total length=arc length.
    """
    if c.kind == "line":
        if c.bearing is None or c.distance_ft is None:
            raise ValueError(f"Line {c.id} missing bearing/distance")
        u = vec_from_az_deg(c.bearing.to_azimuth_deg()).unit()
        L = float(c.distance_ft)
        return (u.dn * L, u.de * L, L)

    if c.kind == "curve":
        _curve_requirements(c)
        dv = _curve_endpoint_vec(c)
        # dv = (de,dn)
        return (dv.dn, dv.de, _curve_arc_len(c))

    raise ValueError(f"Unsupported call kind: {c.kind}")

def find_best_loop_order_beam(
    calls: List[RecordCall],
    beam_width: int = 800,
) -> LoopSolveResult:
    """
    Beam search over permutations to minimize final misclosure.
    Note: vector sum is order-invariant, but ordering impacts polygon self-intersection/area.
    This routine optimizes misclosure only; caller may apply post-scoring.
    """
    n = len(calls)
    if n < 3:
        raise ValueError("Need at least 3 calls for a loop.")
    if n > 60:
        raise ValueError("Too many calls for bitmask ordering search (n>60).")

    vecs = [(_call_vector_and_len(c)) for c in calls]
    ids = [c.id for c in calls]
    total_len = sum(v[2] for v in vecs)

    # state: (lb, sum_dn, sum_de, used_mask, order)
    beam = [(0.0, 0.0, 0.0, 0, [])]

    len_by_i = [v[2] for v in vecs]

    for _depth in range(n):
        next_beam = []
        for _lb, sdn, sde, mask, order in beam:
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

                rem2 = rem_len - L
                cur_mag = math.hypot(nsdn, nsde)
                lb2 = max(0.0, cur_mag - rem2)
                next_beam.append((lb2, nsdn, nsde, nmask, norder))

        next_beam.sort(key=lambda t: (t[0], math.hypot(t[1], t[2])))
        beam = next_beam[:beam_width]

    best = None
    for _lb, sdn, sde, mask, order in beam:
        if mask != (1 << n) - 1:
            continue
        mis = math.hypot(sdn, sde)
        ratio = (total_len / mis) if mis > EPS else float("inf")
        cand = (mis, -ratio, order, sdn, sde)
        if best is None or cand < best:
            best = cand

    if best is None:
        raise RuntimeError("Ordering beam search failed.")
    mis, neg_ratio, order, sdn, sde = best
    ratio = -neg_ratio
    return LoopSolveResult(order, mis, ratio, sdn, sde, total_len)

def traverse_points_for_order(calls_by_id: Dict[str, RecordCall], ordered_ids: List[str], start: Pt = Pt(0.0, 0.0)) -> List[Pt]:
    pts = [start]
    cur = start
    for cid in ordered_ids:
        c = calls_by_id[cid]
        nxt, _, _ = call_endpoint_and_primitive(cur, c)
        pts.append(nxt)
        cur = nxt
    return pts

def _seg_intersect_strict(a1: Pt, a2: Pt, b1: Pt, b2: Pt) -> bool:
    """
    Proper intersection test excluding shared endpoints.
    """
    def orient(p: Pt, q: Pt, r: Pt) -> float:
        return (q.e - p.e) * (r.n - p.n) - (q.n - p.n) * (r.e - p.e)

    def on_segment(p: Pt, q: Pt, r: Pt) -> bool:
        return (min(p.e, r.e) - 1e-12 <= q.e <= max(p.e, r.e) + 1e-12 and
                min(p.n, r.n) - 1e-12 <= q.n <= max(p.n, r.n) + 1e-12)

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    # general
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    # collinear special cases
    if abs(o1) < 1e-12 and on_segment(a1, b1, a2): return True
    if abs(o2) < 1e-12 and on_segment(a1, b2, a2): return True
    if abs(o3) < 1e-12 and on_segment(b1, a1, b2): return True
    if abs(o4) < 1e-12 and on_segment(b1, a2, b2): return True

    return False

def count_self_intersections(polyline_pts: List[Pt]) -> int:
    """
    Counts segment intersections for a closed ring polyline (using successive points).
    Excludes adjacent edges.
    """
    pts = polyline_pts
    if len(pts) < 4:
        return 0
    # close ring if not closed
    if dist(pts[0], pts[-1]) > 1e-6:
        pts = pts + [pts[0]]
    m = len(pts) - 1
    edges = [(pts[i], pts[i+1]) for i in range(m)]
    cnt = 0
    for i in range(m):
        for j in range(i+1, m):
            # skip adjacent and shared endpoints
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == m - 1:
                continue
            a1, a2 = edges[i]
            b1, b2 = edges[j]
            if _seg_intersect_strict(a1, a2, b1, b2):
                cnt += 1
    return cnt

@dataclass
class LoopProposal:
    subset_ids: List[str]
    ordered_ids: List[str]
    misclosure_ft: float
    ratio: float
    perimeter_ft: float
    area_sqft: float
    intersections: int

def select_best_subsets_beam(
    calls: List[RecordCall],
    k: int,
    beam_width: int = 800,
    top_n: int = 20,
) -> List[Tuple[List[int], float, float]]:
    """
    Beam search in subset space (combinations) minimizing |sum vector|.
    Returns list of (indices, misclosure, perimeter).
    """
    n = len(calls)
    vecs = [(_call_vector_and_len(c)) for c in calls]

    # states: (score, last_index, sum_dn, sum_de, perim, picked_indices)
    # score uses |sum|, tie-break -perim (prefer longer)
    beam = [(0.0, -1, 0.0, 0.0, 0.0, [])]

    for depth in range(k):
        next_beam = []
        for _score, last_i, sdn, sde, perim, picked in beam:
            for i in range(last_i + 1, n):
                dn, de, L = vecs[i]
                nsdn = sdn + dn
                nsde = sde + de
                nper = perim + L
                np = picked + [i]
                mag = math.hypot(nsdn, nsde)
                next_beam.append((mag, i, nsdn, nsde, nper, np))

        # prune
        next_beam.sort(key=lambda t: (t[0], -t[4]))
        beam = next_beam[:beam_width]

    finals = []
    for mag, last_i, sdn, sde, perim, idxs in beam:
        finals.append((idxs, mag, perim))
    finals.sort(key=lambda t: (t[1], -t[2]))
    return finals[:top_n]

def propose_loops_from_pool(
    calls_by_id: Dict[str, RecordCall],
    pool_ids: List[str],
    min_sides: int,
    max_sides: int,
    top: int,
    beam_subset: int,
    beam_order: int,
) -> List[LoopProposal]:
    # Build call list
    pool_calls = [calls_by_id[cid] for cid in pool_ids]
    # Pre-filter: ensure vectors exist
    filtered: List[RecordCall] = []
    for c in pool_calls:
        if c.kind not in ("line", "curve"):
            continue
        if not c.is_record():
            continue
        try:
            _call_vector_and_len(c)
        except Exception:
            continue
        filtered.append(c)

    # Hard cap to keep subset beam tractable (favor longer perimeter contributions)
    filtered.sort(key=lambda c: _call_vector_and_len(c)[2], reverse=True)
    # default cap if not supplied via flags: 40
    filtered = filtered[: min(len(filtered), 40)]
    if len(filtered) < min_sides:
        raise ValueError(f"Not enough usable calls in pool after filtering (have {len(filtered)}).")

    proposals: List[LoopProposal] = []

    for k in range(min_sides, max_sides + 1):
        subset_candidates = select_best_subsets_beam(filtered, k=k, beam_width=beam_subset, top_n=max(10, top * 4))
        for idxs, mis, perim in subset_candidates:
            subset = [filtered[i] for i in idxs]
            # order for area/intersection scoring
            order_res = find_best_loop_order_beam(subset, beam_width=beam_order)
            pts = traverse_points_for_order(calls_by_id, order_res.ordered_call_ids, start=Pt(0.0, 0.0))
            # If misclosure is large, area is meaningless; still report but will rank low.
            area = abs(polygon_area(pts))
            inter = count_self_intersections(pts)
            proposals.append(LoopProposal(
                subset_ids=[c.id for c in subset],
                ordered_ids=order_res.ordered_call_ids,
                misclosure_ft=order_res.misclosure_ft,
                ratio=order_res.closure_ratio,
                perimeter_ft=order_res.total_len_ft,
                area_sqft=area,
                intersections=inter,
            ))

    # Rank: prioritize (low misclosure), then (few intersections), then (max area), then (max perimeter), then (ratio)
    def score(p: LoopProposal) -> Tuple[float, int, float, float, float]:
        return (p.misclosure_ft, p.intersections, -p.area_sqft, -p.perimeter_ft, -p.ratio)

    proposals.sort(key=score)
    # keep unique by subset signature to avoid duplicates
    seen = set()
    uniq: List[LoopProposal] = []
    for p in proposals:
        sig = tuple(sorted(p.subset_ids))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(p)
        if len(uniq) >= top:
            break
    return uniq


# -----------------------------
# Rendering (SVG with arcs, DXF LWPOLYLINE with bulges)
# -----------------------------
def escape_xml(s: str) -> str:
    return (s.replace("&", "&amp;")
              .replace("<", "&lt;")
              .replace(">", "&gt;")
              .replace('"', "&quot;")
              .replace("'", "&apos;"))

def _svg_viewbox_from_layers(project: Project) -> Tuple[float,float,float,float]:
    pts: List[Pt] = []
    for lr in project.layers.values():
        pts.extend(lr.vertices)
    for lot in project.lots.values():
        pts.extend(lot.polygon)
    if not pts:
        raise RuntimeError("No geometry to render.")
    min_e = min(p.e for p in pts)
    max_e = max(p.e for p in pts)
    min_n = min(p.n for p in pts)
    max_n = max(p.n for p in pts)
    w = max_e - min_e
    h = max_n - min_n
    pad = max(w, h) * 0.05 + 10.0
    vb_min_e = min_e - pad
    vb_min_n = min_n - pad
    vb_w = w + 2 * pad
    vb_h = h + 2 * pad
    return vb_min_e, vb_min_n, vb_w, vb_h

def render_svg(project: Project, out_path: Path) -> None:
    vb_min_e, vb_min_n, vb_w, vb_h = _svg_viewbox_from_layers(project)

    def svg_xy(p: Pt) -> Tuple[float, float]:
        x = p.e
        # invert y
        y = (vb_min_n + vb_h) - p.n
        return x, y

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_min_e:.3f} {((vb_min_n+vb_h)-vb_h):.3f} {vb_w:.3f} {vb_h:.3f}">')
    svg.append('<style>')
    svg.append('.layer{fill:none;stroke:#e9eef6;stroke-width:0.6;}')
    svg.append('.lot{fill:none;stroke:#9aa7ba;stroke-width:0.35;}')
    svg.append('.label{fill:#e9eef6;font-family:Arial, sans-serif;font-size:6px;}')
    svg.append('</style>')

    # draw traverse layers using primitives
    for lid, lr in project.layers.items():
        if not lr.primitives:
            continue
        dparts = []
        # move to start
        x0, y0 = svg_xy(lr.primitives[0].start)
        dparts.append(f"M {x0:.3f} {y0:.3f}")
        for prim in lr.primitives:
            if prim.type == "line":
                x, y = svg_xy(prim.end)
                dparts.append(f"L {x:.3f} {y:.3f}")
            elif prim.type == "arc":
                if prim.radius_ft is None or prim.dir is None or prim.delta_deg is None:
                    # fallback to straight
                    x, y = svg_xy(prim.end)
                    dparts.append(f"L {x:.3f} {y:.3f}")
                else:
                    # SVG flags: coordinate y inverted, so sweep flips.
                    # Model 'L' (CCW) becomes clockwise in SVG => sweep=1; 'R' => sweep=0
                    sweep = 1 if prim.dir == "L" else 0
                    large_arc = 1 if float(prim.delta_deg) > 180.0 else 0
                    r = float(prim.radius_ft)
                    x, y = svg_xy(prim.end)
                    dparts.append(f"A {r:.3f} {r:.3f} 0 {large_arc} {sweep} {x:.3f} {y:.3f}")
            else:
                x, y = svg_xy(prim.end)
                dparts.append(f"L {x:.3f} {y:.3f}")

        if lr.closed:
            dparts.append("Z")

        svg.append(f'<path class="layer" d="{" ".join(dparts)}"/>')
        # label near centroid of vertices (rough)
        vx = [p.e for p in lr.vertices]
        vy = [p.n for p in lr.vertices]
        cx = sum(vx) / len(vx)
        cy = sum(vy) / len(vy)
        lx, ly = svg_xy(Pt(cx, cy))
        svg.append(f'<text class="label" x="{lx:.3f}" y="{ly:.3f}">{escape_xml(lid)}</text>')

    # lots (polyline)
    for lot in project.lots.values():
        pts = close_ring(lot.polygon)
        dparts = []
        for i, p in enumerate(pts):
            x, y = svg_xy(p)
            dparts.append(("M" if i == 0 else "L") + f" {x:.3f} {y:.3f}")
        dparts.append("Z")
        svg.append(f'<path class="lot" d="{" ".join(dparts)}"/>')

    svg.append("</svg>")
    write_text(out_path, "\n".join(svg))

def render_dxf_minimal(project: Project, out_path: Path) -> None:
    """
    Outputs:
      - LWPOLYLINE per traverse layer, with bulges for arcs (supports L/R)
      - LWPOLYLINE for lots as straight segments
    """
    ents: List[str] = []

    def add_lwpolyline(vertices: List[Pt], bulges: List[float], closed: bool, layer: str):
        # LWPOLYLINE
        n = len(vertices)
        if n < 2:
            return
        if len(bulges) != n:
            raise ValueError("bulges length must equal vertices length")
        ents.extend([
            "0", "LWPOLYLINE",
            "8", layer,
            "90", str(n),
            "70", "1" if closed else "0",
        ])
        for i, p in enumerate(vertices):
            ents.extend(["10", f"{p.e:.6f}", "20", f"{p.n:.6f}"])
            b = float(bulges[i])
            if abs(b) > 1e-12:
                ents.extend(["42", f"{b:.12f}"])

    def add_text(p: Pt, text: str, height: float, layer: str):
        ents.extend([
            "0", "TEXT",
            "8", layer,
            "10", f"{p.e:.6f}",
            "20", f"{p.n:.6f}",
            "40", f"{height:.6f}",
            "1", text,
        ])

    # layers
    for lid, lr in project.layers.items():
        if not lr.primitives:
            continue
        # vertices are stored without duplicate start
        verts = lr.vertices
        if len(verts) < 2:
            continue
        # bulges aligned with vertices: bulge at vertex i for segment from i to i+1 (wrap if closed)
        bulges = [0.0] * len(verts)
        # primitives length should equal number of segments; if closed, segments == len(verts)
        for i, prim in enumerate(lr.primitives):
            if i >= len(verts):
                break
            bulges[i] = float(prim.bulge) if prim.type == "arc" else 0.0
        add_lwpolyline(verts, bulges, lr.closed, lid.upper())
        # label
        cx = sum(p.e for p in verts) / len(verts)
        cy = sum(p.n for p in verts) / len(verts)
        add_text(Pt(cx, cy), lid, height=1.5, layer="LAYER_LABELS")

    # lots (straight, no bulges)
    for lot in project.lots.values():
        poly = close_ring(lot.polygon)
        verts = poly[:-1]
        bulges = [0.0] * len(verts)
        add_lwpolyline(verts, bulges, True, "LOTS")
        cx = sum(p.e for p in verts) / len(verts)
        cy = sum(p.n for p in verts) / len(verts)
        add_text(Pt(cx, cy), lot.id, height=1.5, layer="LOT_LABELS")

    dxf = []
    dxf.extend(["0", "SECTION", "2", "HEADER"])
    dxf.extend(["9", "$ACADVER", "1", "AC1015"])  # AutoCAD 2000
    dxf.extend(["0", "ENDSEC"])
    dxf.extend(["0", "SECTION", "2", "ENTITIES"])
    dxf.extend(ents)
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
    rows = [["id","kind","layer","seq","bearing/chord","dist/chord","R","Δ","arc","dir","tags","evidence","seg","seg_dist_px"]]
    for cid, c in sorted(project.calls.items(), key=lambda kv: kv[0]):
        if c.kind == "line":
            b = c.bearing.format() if c.bearing else (c.bearing_text or "")
            d = fmt_ft(float(c.distance_ft)) if c.distance_ft is not None else (c.distance_text or "")
            rows.append([cid,c.kind,c.layer,str(c.seq or ""),b,d,"","","","",",".join(c.tags),c.evidence.type,c.assoc_segment_id or "", f"{c.assoc_segment_dist_px:.1f}" if c.assoc_segment_dist_px is not None else ""])
        elif c.kind == "curve":
            cb = c.curve_chord_bearing.format() if c.curve_chord_bearing else (c.curve_chord_bearing_text or "")
            ch = fmt_ft(float(c.curve_chord_len_ft)) if c.curve_chord_len_ft is not None else ""
            rows.append([cid,c.kind,c.layer,str(c.seq or ""),cb,ch,
                         fmt_ft(float(c.curve_radius_ft)) if c.curve_radius_ft is not None else "",
                         f"{float(c.curve_delta_deg):.4f}" if c.curve_delta_deg is not None else "",
                         fmt_ft(float(c.curve_arc_len_ft)) if c.curve_arc_len_ft is not None else "",
                         c.curve_dir or "",
                         ",".join(c.tags),c.evidence.type,c.assoc_segment_id or "", f"{c.assoc_segment_dist_px:.1f}" if c.assoc_segment_dist_px is not None else ""])
        else:
            rows.append([cid,c.kind,c.layer,str(c.seq or ""),c.bearing_text or "",c.distance_text or "","","","","",",".join(c.tags),c.evidence.type,c.assoc_segment_id or "", f"{c.assoc_segment_dist_px:.1f}" if c.assoc_segment_dist_px is not None else ""])
    lines.append(render_table(rows))
    lines.append("")

    lines.append("## B) LAYER CLOSURES")
    lines.append("")
    rows = [["layer","build","ΣΔN (ft)","ΣΔE (ft)","miscl (ft)","miscl (in)","total (ft)","ratio","gate"]]
    for lid, lr in project.layers.items():
        cl = lr.closure
        rows.append([
            lid, lr.build_type,
            fmt_ft_mis(cl.sum_dn), fmt_ft_mis(cl.sum_de),
            fmt_ft_mis(cl.misclosure_ft), fmt_ft_mis(ft_to_inches(cl.misclosure_ft)),
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
# Interactive prompts
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

def prompt_float(msg: str, default: Optional[float] = None) -> Optional[float]:
    v = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
    if not v:
        return default
    try:
        return float(v)
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
    layer_map: Dict[str, List[RecordCall]] = {}
    for c in project.calls.values():
        if c.kind not in ("line", "curve"):
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
        gates = [{
            "type": "closure",
            "max_mis_ft": project.flags.get("default_max_mis_ft", 0.05),
            "min_ratio": project.flags.get("default_min_ratio", 50000),
        }]
        steps.append(PlanStep(
            id=f"traverse_{layer_id}",
            type="traverse",
            params={"layer_id": layer_id, "start": {"e": 0.0, "n": 0.0}, "call_ids": call_ids, "close_tol_ft": project.flags.get("close_tol_ft", 0.10)},
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
    proj.flags.setdefault("close_tol_ft", 0.10)

    proj_path = out_dir / "project.json"
    proj.save(proj_path)
    audit.log("init", {"project": str(proj_path), "image": proj.image_path, "out": str(out_dir)})
    print(f"Created project: {proj_path}")

def cmd_detect(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    audit = AuditLogger(Path(proj.out_dir) / "audit.ndjson")

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
    - Compose LINE calls from bearing+distance
    - Compose CURVE calls from curve table text or manual entry
    """
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")
    snips_dir = ensure_dir(out_dir / "snips")
    img_path = Path(proj.image_path)

    # Candidate review
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

    # Compose line calls
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

    # Compose curve calls
    while prompt_yes_no("Create a CURVE call (record curve table entry)?", default=False):
        curve_id = prompt_str("Curve call id", default=f"curve_{stable_id(now_iso())}")
        layer = prompt_str("Layer for this curve", default="unassigned")
        seq = prompt_int("Sequence number in traverse (optional)", default=None)

        mode = prompt_choice("Curve input mode", ["paste", "manual"], default="paste")
        rc = RecordCall(
            id=curve_id,
            kind="curve",
            layer=layer,
            seq=seq,
            tags=["record"],
            evidence=Evidence(type="user_confirmed", notes="curve entry"),
        )

        if mode == "paste":
            raw = prompt_str("Paste curve table text line (R/Δ/L/CH/CB, LT/RT etc.)")
            parsed = parse_curve_text(raw)
            # if missing, ask
            rc.curve_dir = parsed.get("dir") or prompt_choice("Direction", ["L", "R"], default="L")
            rc.curve_radius_ft = parsed.get("radius_ft") or prompt_float("Radius (ft)")  # type: ignore
            cb = parsed.get("chord_bearing")
            if cb is None:
                cbtxt = prompt_str("Chord bearing (quadrant DMS, e.g., N 12°34'56\" E)")
                rc.curve_chord_bearing_text = cbtxt
                rc.curve_chord_bearing = parse_quadrant_bearing(cbtxt)
            else:
                rc.curve_chord_bearing = cb
                rc.curve_chord_bearing_text = parsed.get("chord_bearing_text")

            rc.curve_chord_len_ft = parsed.get("chord_len_ft") or prompt_float("Chord length (ft)")  # type: ignore
            if "delta_deg" in parsed:
                rc.curve_delta_deg = float(parsed["delta_deg"])
            if "arc_len_ft" in parsed:
                rc.curve_arc_len_ft = float(parsed["arc_len_ft"])

        else:
            rc.curve_dir = prompt_choice("Direction", ["L", "R"], default="L")
            rc.curve_radius_ft = prompt_float("Radius (ft)")  # type: ignore
            cbtxt = prompt_str("Chord bearing (quadrant DMS)")
            rc.curve_chord_bearing_text = cbtxt
            rc.curve_chord_bearing = parse_quadrant_bearing(cbtxt)
            rc.curve_chord_len_ft = prompt_float("Chord length (ft)")  # type: ignore
            delta_txt = prompt_str("Delta (DMS or decimal) (optional)", default="")
            if delta_txt:
                rc.curve_delta_deg = parse_angle_deg(delta_txt)
            arc_txt = prompt_str("Arc length (ft) (optional)", default="")
            if arc_txt:
                rc.curve_arc_len_ft = float(arc_txt)

        # compute missing delta/arc if possible (still marked record, but derived from record fields)
        try:
            _curve_requirements(rc)
            if rc.curve_delta_deg is None:
                rc.curve_delta_deg = math.degrees(_curve_delta_rad(rc))
            if rc.curve_arc_len_ft is None:
                rc.curve_arc_len_ft = _curve_arc_len(rc)
        except Exception as e:
            print(f"[FAIL] Curve incomplete: {e}")
            continue

        proj.calls[curve_id] = rc
        audit.log("curve_compose", {"curve_id": curve_id, "layer": layer, "seq": seq,
                                   "dir": rc.curve_dir, "R": rc.curve_radius_ft, "delta": rc.curve_delta_deg,
                                   "arc": rc.curve_arc_len_ft, "cb": rc.curve_chord_bearing_text, "ch": rc.curve_chord_len_ft})
        print(f"Created curve call {curve_id}")

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

def cmd_autoloops(args: argparse.Namespace) -> None:
    """
    Subset selection + ordering:
      - Select pool of calls (record line/curve)
      - Propose top-N loop subsets + orders
      - Optionally apply one: set layer + seq, and suggest a plan step.
    """
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    # pool ids
    pool_layer = args.pool_layer
    pool_ids: List[str] = []
    for cid, c in proj.calls.items():
        if c.kind not in ("line", "curve"):
            continue
        if not c.is_record():
            continue
        if pool_layer != "any" and c.layer != pool_layer:
            continue
        if args.require_assoc and not c.assoc_segment_id:
            continue
        pool_ids.append(cid)

    if not pool_ids:
        raise SystemExit("No calls in pool (check --pool-layer and tagging).")

    props = propose_loops_from_pool(
        proj.calls,
        pool_ids=pool_ids,
        min_sides=args.min_sides,
        max_sides=args.max_sides,
        top=args.top,
        beam_subset=args.beam_subset,
        beam_order=args.beam_order,
    )

    audit.log("autoloops_propose", {"pool_layer": pool_layer, "count_pool": len(pool_ids), "proposals": [dataclasses.asdict(p) for p in props]})

    print("\n=== Loop Proposals (subset selection + ordering) ===")
    for i, p in enumerate(props, start=1):
        print(f"\n[{i}] sides={len(p.subset_ids)} mis={p.misclosure_ft:.4f} ft ({ft_to_inches(p.misclosure_ft):.2f} in) "
              f"ratio={'∞' if not math.isfinite(p.ratio) else f'1:{p.ratio:.0f}'} "
              f"perim={p.perimeter_ft:.2f} ft area={p.area_sqft:.1f} sqft inter={p.intersections}")
        print("  ordered:")
        for j, cid in enumerate(p.ordered_ids, start=1):
            print(f"    {j:02d}. {cid}")

    if args.apply is None:
        return

    k = int(args.apply)
    if k < 1 or k > len(props):
        raise SystemExit("--apply index out of range")
    chosen = props[k - 1]

    set_layer = args.set_layer
    if set_layer:
        for cid in chosen.ordered_ids:
            proj.calls[cid].layer = set_layer

    if args.set_seq:
        for i, cid in enumerate(chosen.ordered_ids, start=1):
            proj.calls[cid].seq = i

    if args.suggest_plan:
        step = {
            "id": f"traverse_{set_layer or 'loop'}",
            "type": "traverse",
            "params": {
                "layer_id": set_layer or "loop",
                "start": {"e": args.start_e, "n": args.start_n},
                "call_ids": chosen.ordered_ids,
                "close_tol_ft": proj.flags.get("close_tol_ft", 0.10),
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
    audit.log("autoloops_apply", {"apply": k, "set_layer": set_layer, "set_seq": args.set_seq, "suggest_plan": args.suggest_plan})
    print("\nApplied selected proposal to project (layer/seq updated; plan step stored if requested).")

def cmd_plan_new(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    plan = generate_plan_from_classified_traverses(proj)
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
    p = argparse.ArgumentParser(prog="survey-recon", description="General survey record reconstruction CLI (subset selection + curves).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="Initialize a project.")
    s.add_argument("image", help="Plat image path (PNG/JPG).")
    s.add_argument("--out", required=True, help="Output directory for the project.")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("detect", help="OCR scan to propose call candidates.")
    s.add_argument("project", help="Project JSON path.")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("review", help="Interactive review of candidates into confirmed record calls (includes line/curve compose).")
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

    s = sub.add_parser("autoloops", help="Subset selection + ordering to propose outer loop(s). Optionally apply layer/seq and suggest plan step.")
    s.add_argument("project", help="Project JSON path.")
    s.add_argument("--pool-layer", default="any", help="Select calls from this layer (or 'any').")
    s.add_argument("--require-assoc", action="store_true", help="Only use calls that are associated to a detected segment.")
    s.add_argument("--min-sides", type=int, default=4)
    s.add_argument("--max-sides", type=int, default=12)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--beam-subset", type=int, default=800)
    s.add_argument("--beam-order", type=int, default=800)
    s.add_argument("--apply", type=int, default=None, help="Apply proposal index (1..top) to project.")
    s.add_argument("--set-layer", default="", help="When applying, set calls' layer to this.")
    s.add_argument("--set-seq", action="store_true", help="When applying, write seq numbers 1..N per proposed order.")
    s.add_argument("--suggest-plan", action="store_true", help="When applying, store traverse step proposal in project.flags.")
    s.add_argument("--start-e", type=float, default=0.0)
    s.add_argument("--start-n", type=float, default=0.0)
    s.set_defaults(func=cmd_autoloops)

    s = sub.add_parser("plan-new", help="Generate plan.json from calls (layer + seq). Appends plan_step_proposals if present.")
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
