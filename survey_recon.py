#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survey_recon.py — init / run geometry reconstruction (minimal operator flow)

Flow:
  init <image> --out <dir> [--page N]
  run <out/project.json>

Dependencies:
  pip install pillow pytesseract
System:
  tesseract-ocr on PATH
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

EPS = 1e-10
PROJECT_VERSION = "1.3.0-dev"

# ----------------------------
# Utils
# ----------------------------
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")

def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def stable_id(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ft_to_inches(ft: float) -> float:
    return ft * 12.0

def token_type_hist(tokens: List[Dict[str, Any]]) -> Dict[str, int]:
    h: Dict[str, int] = {}
    for t in tokens:
        tt = str(t.get("token_type") or "unknown")
        h[tt] = h.get(tt, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))

def bbox_center(b: List[int]) -> Tuple[float, float]:
    x, y, w, h = [float(v) for v in b]
    return (x + w * 0.5, y + h * 0.5)

def bbox_union(a: List[int], b: List[int]) -> List[int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

# ----------------------------
# Audit log
# ----------------------------
class AuditLogger:
    def __init__(self, ndjson_path: Path):
        self.path = ndjson_path
        ensure_dir(self.path.parent)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": now_iso(), "event": event, "payload": payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

# ----------------------------
# Geometry
# ----------------------------
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

def dist(a: Pt, b: Pt) -> float:
    return (a - b).mag()

def vec_from_az_deg(az_deg: float) -> Vec:
    az = math.radians(az_deg)
    return Vec(math.sin(az), math.cos(az))

def normalize_az(az: float) -> float:
    az %= 360.0
    if az < 0:
        az += 360.0
    return az

# ----------------------------
# Bearings + distances (robust)
# ----------------------------
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

KEYWORD_BLOCK = (
    "TOWNSHIP", "RANGE", "SECTION", "SHEET", "PLAT", "SCALE", "B.M", " BM", "LOT "
)

def _normalize_text_for_parse(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("º", "°")
    t = t.replace("′", "'").replace("’", "'").replace("`", "'")
    t = t.replace("″", '"').replace("“", '"').replace("”", '"')
    t = t.replace('\\"', '"')
    t = re.sub(r"\s+", " ", t)
    return t

@dataclass(frozen=True)
class DMS:
    deg: int
    minutes: int
    seconds: float

    def to_degrees(self) -> float:
        return self.deg + self.minutes / 60.0 + self.seconds / 3600.0

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
            raise ValueError("Invalid quadrant bearing")
        return normalize_az(az)

    def format(self) -> str:
        sec = int(round(self.angle.seconds))
        return f'{self.ns} {self.angle.deg:02d}°{self.angle.minutes:02d}\'{sec:02d}" {self.ew}'

def parse_quadrant_bearing(text: str) -> QuadrantBearing:
    t = _normalize_text_for_parse(text)
    u = t.upper()

    if any(k in u for k in KEYWORD_BLOCK):
        if ("°" not in u) and ("'" not in u) and ('"' not in u) and ("DEG" not in u):
            raise ValueError("Blocked non-bearing context")

    ns_matches = list(re.finditer(r"(?<![A-Z])([NS])(?![A-Z])", u))
    ew_matches = list(re.finditer(r"(?<![A-Z])([EW])(?![A-Z])", u))
    if not ns_matches or not ew_matches:
        raise ValueError("No quadrant letters")

    best = None
    for nm in ns_matches:
        for em in ew_matches:
            if em.start() <= nm.start():
                continue
            span = u[nm.start():em.start()+1]
            span_len = len(span)
            if span_len > 70 and ("°" not in span) and ("'" not in span) and ('"' not in span):
                continue
            nums = [float(x) for x in NUM_RE.findall(span)]
            if not nums:
                continue

            ns = nm.group(1)
            ew = em.group(1)

            if len(nums) == 1:
                theta = float(nums[0])
                if theta < 0 or theta > 90:
                    continue
                deg = int(theta)
                rem = (theta - deg) * 60.0
                minutes = int(rem)
                seconds = (rem - minutes) * 60.0
            elif len(nums) == 2:
                deg = int(nums[0]); minutes = int(nums[1]); seconds = 0.0
            else:
                deg = int(nums[0]); minutes = int(nums[1]); seconds = float(nums[2])

            if not (0 <= deg <= 90 and 0 <= minutes < 60 and 0 <= seconds < 60.0):
                continue

            cand = QuadrantBearing(ns, DMS(deg, minutes, seconds), ew)
            score = (span_len, -len(nums))
            if best is None or score < best[0]:
                best = (score, cand)

    if best is None:
        raise ValueError("Could not parse bearing")
    return best[1]

def try_parse_distance_ft(text: str) -> Optional[float]:
    t = _normalize_text_for_parse(text)
    u = t.upper()

    has_unit = ("'" in u) or (" FT" in u) or ("FEET" in u) or u.endswith("FT") or u.endswith("FEET")

    u2 = re.sub(r"\bFEET\b", "", u)
    u2 = re.sub(r"\bFT\b", "", u2)
    u2 = u2.replace("'", " ").replace('"', " ")
    u2 = re.sub(r"\s+", " ", u2).strip()

    if re.search(r"[A-Z]", u2):
        return None

    nums = [float(x) for x in NUM_RE.findall(u2)]
    if len(nums) != 1:
        return None

    v = float(nums[0])
    if not (0.01 <= v <= 200000.0):
        return None

    if not has_unit and any(k in u for k in KEYWORD_BLOCK):
        return None

    return v

def try_parse_offset_ft(text: str) -> Optional[float]:
    t = _normalize_text_for_parse(text)
    u = t.upper()
    if "'" not in u and "FT" not in u and "FEET" not in u:
        return None
    v = try_parse_distance_ft(t)
    if v is not None:
        return v
    nums = [float(x) for x in NUM_RE.findall(u)]
    if len(nums) == 1 and 0.01 <= nums[0] <= 200000.0:
        return float(nums[0])
    return None

def distance_after_ew_if_bearing(text: str) -> Optional[float]:
    t = _normalize_text_for_parse(text)
    u = t.upper()

    ns_i = None
    ew_i = None
    for m in re.finditer(r"(?<![A-Z])([NS])(?![A-Z])", u):
        ns_i = m.start()
        m2 = re.search(r"(?<![A-Z])([EW])(?![A-Z])", u[m.end():])
        if not m2:
            continue
        ew_i = m.end() + m2.start()
        break

    if ns_i is None or ew_i is None:
        return None

    post = u[ew_i + 1:]
    nums = [float(x) for x in NUM_RE.findall(post)]
    if not nums:
        return None
    v = float(nums[0])
    if not (0.01 <= v <= 200000.0):
        return None
    return v

# ----------------------------
# OCR + image helpers
# ----------------------------
# IMPORTANT: do NOT include a literal double-quote character in this config string.
# pytesseract uses shlex.split(config) and will throw "No closing quotation".
OCR_CONFIG_CALLS = (
    "--psm 6 "
    "-c preserve_interword_spaces=1 "
    "-c tessedit_char_whitelist=0123456789NSEWnsew°'.-"
)
OCR_CONFIG_FALLBACK = "--psm 6 -c preserve_interword_spaces=1"
MINERU_TEXT_EXTENSIONS = (".md", ".txt")
MINERU_JSON_EXTENSIONS = (".json",)

def try_import_pillow():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception:
        return None

def open_pillow_image(path: Path, page: int = 0):
    Image = try_import_pillow()
    if Image is None:
        raise RuntimeError("Pillow required (pip install pillow)")
    img = Image.open(path)
    try:
        if getattr(img, "n_frames", 1) > 1:
            img.seek(int(page))
    except Exception:
        pass
    return img

def preprocess_pil(img, mode: str = "gray", threshold: int = 180, scale: float = 1.0):
    Image = try_import_pillow()
    if Image is None:
        raise RuntimeError("Pillow required.")
    if scale and abs(scale - 1.0) > 1e-6:
        w, h = img.size
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), resample=Image.Resampling.BILINEAR)
    if mode == "gray":
        return img.convert("L")
    if mode == "binary":
        g = img.convert("L")
        return g.point(lambda p: 255 if p >= int(threshold) else 0, mode="1").convert("L")
    return img.convert("L")

def clamp_bbox(b: List[int], W: int, H: int) -> List[int]:
    x, y, w, h = b
    x = max(0, x); y = max(0, y)
    w = max(1, w); h = max(1, h)
    if x + w > W: w = max(1, W - x)
    if y + h > H: h = max(1, H - y)
    return [x, y, w, h]

def try_import_ocr():
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return Image, pytesseract
    except Exception:
        return None, None

def detect_default_ocr_engine() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "mineru"
    return "tesseract"

def resolve_ocr_engine() -> str:
    val = (os.environ.get("SURVEY_RECON_OCR_ENGINE", "") or "").strip().lower()
    if val in {"tesseract", "mineru"}:
        return val
    return detect_default_ocr_engine()

def _mineru_command_candidates(image_path: Path, out_dir: Path) -> List[List[str]]:
    return [
        ["mineru", "-i", str(image_path), "-o", str(out_dir)],
        ["mineru", "--input", str(image_path), "--output", str(out_dir)],
        [sys.executable, "-m", "mineru", "-i", str(image_path), "-o", str(out_dir)],
    ]

def _mineru_extract_lines(out_dir: Path) -> List[str]:
    lines: List[str] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in MINERU_TEXT_EXTENSIONS:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if s:
                    lines.append(s)
            continue
        if path.suffix.lower() in MINERU_JSON_EXTENSIONS:
            try:
                obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, str):
                    s = cur.strip()
                    if s:
                        lines.append(s)
                elif isinstance(cur, dict):
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
    return lines

def _mineru_image_to_lines(image_path: Path) -> List[str]:
    if shutil.which("mineru") is None:
        try:
            import mineru  # noqa: F401 # type: ignore
        except Exception as exc:
            raise RuntimeError(f"mineru_not_installed: {exc}")

    with tempfile.TemporaryDirectory(prefix="survey-recon-mineru-") as tmp:
        out_dir = Path(tmp) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        last_err = "mineru invocation failed"
        for cmd in _mineru_command_candidates(image_path, out_dir):
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if proc.stderr.strip():
                    last_err = proc.stderr.strip()
                break
            except Exception as exc:
                last_err = str(exc)
        else:
            raise RuntimeError(f"mineru_failed: {last_err}")

        lines = _mineru_extract_lines(out_dir)
        if not lines:
            raise RuntimeError("mineru_no_text_output")
        return lines

def ocr_image_to_string(img, image_path: Optional[Path], config: str = OCR_CONFIG_CALLS) -> str:
    engine = resolve_ocr_engine()
    if engine == "mineru":
        if image_path is None:
            raise RuntimeError("mineru_requires_image_path")
        return "\n".join(_mineru_image_to_lines(Path(image_path)))

    Image, pytesseract = try_import_ocr()
    if Image is None or pytesseract is None:
        raise RuntimeError("OCR deps missing.")
    return _tess_image_to_string_safe(pytesseract, img, config)

def check_ocr_ready() -> Tuple[bool, str]:
    engine = resolve_ocr_engine()
    if engine == "mineru":
        if shutil.which("mineru") is not None:
            return True, "ok:mineru"
        try:
            import mineru  # noqa: F401 # type: ignore
            return True, "ok:mineru"
        except Exception as e:
            return False, f"mineru_missing: {e}"

    Image, pytesseract = try_import_ocr()
    if Image is None or pytesseract is None:
        return False, "pytesseract_or_pillow_missing"
    try:
        _ = pytesseract.get_tesseract_version()
        return True, "ok"
    except Exception as e:
        return False, f"tesseract_missing_or_not_on_path: {e}"

@dataclass
class CallCandidate:
    id: str
    text: str
    bbox: List[int]
    confidence: float = 0.0

def _tess_image_to_data_safe(pytesseract, img, config: str):
    try:
        return pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    except ValueError as e:
        # the exact crash you hit
        if "No closing quotation" in str(e):
            return pytesseract.image_to_data(img, config=OCR_CONFIG_FALLBACK, output_type=pytesseract.Output.DICT)
        raise

def _tess_image_to_string_safe(pytesseract, img, config: str) -> str:
    try:
        return pytesseract.image_to_string(img, config=config)
    except ValueError as e:
        if "No closing quotation" in str(e):
            return pytesseract.image_to_string(img, config=OCR_CONFIG_FALLBACK)
        raise

def ocr_scan_lines(img, config: str = OCR_CONFIG_CALLS, image_path: Optional[Path] = None) -> List[CallCandidate]:
    engine = resolve_ocr_engine()
    if engine == "mineru":
        if image_path is None:
            raise RuntimeError("mineru_requires_image_path")
        lines = _mineru_image_to_lines(Path(image_path))
        w, h = img.size
        return [
            CallCandidate(
                id=stable_id("cand", text, idx),
                text=text,
                bbox=[0, min(h - 1, idx * 20), max(1, w), min(20, h)],
                confidence=95.0,
            )
            for idx, text in enumerate(lines)
        ]

    Image, pytesseract = try_import_ocr()
    if Image is None or pytesseract is None:
        raise RuntimeError("OCR deps missing.")
    data = _tess_image_to_data_safe(pytesseract, img, config=config)

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
    y_tol = 10
    for w in words:
        if not lines:
            lines.append([w]); continue
        if abs(w[0] - lines[-1][0][0]) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])

    out: List[CallCandidate] = []
    for ln in lines:
        ln = sorted(ln, key=lambda t: t[1])
        text = " ".join(t[4] for t in ln)
        x0 = min(t[1] for t in ln); y0 = min(t[0] for t in ln)
        x1 = max(t[1] + t[2] for t in ln); y1 = max(t[0] + t[3] for t in ln)
        bbox = [x0, y0, x1 - x0, y1 - y0]
        confs = [t[5] for t in ln if t[5] >= 0]
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        out.append(CallCandidate(id=stable_id("cand", text, bbox), text=text, bbox=bbox, confidence=conf))
    return out

def quick_bearing_likeness(text: str) -> float:
    u = _normalize_text_for_parse(text).upper()
    if not u:
        return 0.0
    s = 0.0
    s += 1.5 * len(re.findall(r"(?<![A-Z])[NS](?![A-Z])", u))
    s += 1.5 * len(re.findall(r"(?<![A-Z])[EW](?![A-Z])", u))
    s += 1.0 * u.count("°")
    s += 0.5 * u.count("'")
    s += 0.25 * u.count('"')  # might still appear even if not whitelisted
    if re.search(r"(?<![A-Z])[NS](?![A-Z]).{0,40}\d.{0,40}(?<![A-Z])[EW](?![A-Z])", u):
        s += 4.0
    if any(k in u for k in KEYWORD_BLOCK):
        s *= 0.35
    return s

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Evidence:
    type: str
    bbox_base: Optional[List[int]] = None
    aoi_id: Optional[str] = None
    page: Optional[int] = None
    ocr_text: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class RecordCall:
    id: str
    kind: str
    layer: str = "unassigned"
    seq: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    evidence: Evidence = field(default_factory=lambda: Evidence(type="ocr"))

    bearing_text: Optional[str] = None
    distance_text: Optional[str] = None
    bearing: Optional[QuadrantBearing] = None
    distance_ft: Optional[float] = None

    def is_record(self) -> bool:
        return "record" in self.tags

@dataclass
class Token:
    id: str
    token_type: str
    fields: Dict[str, Any]
    confidence: float = 0.0
    aoi_id: Optional[str] = None
    bbox_base_px: Optional[List[int]] = None
    page: Optional[int] = None
    raw_text: Optional[str] = None

@dataclass
class Primitive:
    type: str
    start: Pt
    end: Pt
    bulge: float = 0.0

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
    calls_used: List[str]
    vertices: List[Pt]
    closed: bool
    primitives: List[Primitive]
    closure: Closure
    gate_passed: bool
    gate_notes: str

@dataclass
class Project:
    version: str
    created_at: str
    image_path: str
    out_dir: str
    flags: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    aois: Dict[str, Any] = field(default_factory=dict)
    calls: Dict[str, RecordCall] = field(default_factory=dict)
    layers: Dict[str, LayerResult] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        def qb_to_dict(q: Optional[QuadrantBearing]) -> Any:
            if q is None:
                return None
            return {"ns": q.ns, "ew": q.ew, "deg": q.angle.deg, "min": q.angle.minutes, "sec": q.angle.seconds}

        def pt_to_list(p: Pt) -> List[float]:
            return [p.e, p.n]

        obj = {
            "version": self.version,
            "created_at": self.created_at,
            "image_path": self.image_path,
            "out_dir": self.out_dir,
            "flags": self.flags,
            "diagnostics": self.diagnostics,
            "aois": self.aois,
            "calls": {
                cid: {
                    **dataclasses.asdict(rc),
                    "bearing": qb_to_dict(rc.bearing),
                }
                for cid, rc in self.calls.items()
            },
            "layers": {
                lid: {
                    "id": lr.id,
                    "calls_used": lr.calls_used,
                    "vertices": [pt_to_list(p) for p in lr.vertices],
                    "closed": lr.closed,
                    "primitives": [dataclasses.asdict(p) for p in lr.primitives],
                    "closure": dataclasses.asdict(lr.closure),
                    "gate_passed": lr.gate_passed,
                    "gate_notes": lr.gate_notes,
                }
                for lid, lr in self.layers.items()
            },
        }
        write_json(path, obj)

    @staticmethod
    def load(path: Union[str, Path]) -> "Project":
        obj = read_json(path)

        def dict_to_qb(d: Any) -> Optional[QuadrantBearing]:
            if d is None:
                return None
            return QuadrantBearing(d["ns"], DMS(int(d["deg"]), int(d["min"]), float(d["sec"])), d["ew"])

        proj = Project(
            version=obj["version"],
            created_at=obj["created_at"],
            image_path=obj["image_path"],
            out_dir=obj["out_dir"],
            flags=obj.get("flags", {}),
            diagnostics=obj.get("diagnostics", {}),
        )
        proj.aois = obj.get("aois", {}) or {}

        calls: Dict[str, RecordCall] = {}
        for cid, v in (obj.get("calls", {}) or {}).items():
            ev = v.get("evidence") or {}
            rc = RecordCall(
                id=v["id"], kind=v["kind"], layer=v.get("layer", "unassigned"), seq=v.get("seq"),
                tags=v.get("tags", []), evidence=Evidence(**ev),
                bearing_text=v.get("bearing_text"), distance_text=v.get("distance_text"),
                bearing=dict_to_qb(v.get("bearing")), distance_ft=v.get("distance_ft"),
            )
            calls[cid] = rc
        proj.calls = calls

        layers: Dict[str, LayerResult] = {}
        for lid, v in (obj.get("layers", {}) or {}).items():
            prims = [Primitive(**p) for p in (v.get("primitives", []) or [])]
            cl = v.get("closure") or {}
            layers[lid] = LayerResult(
                id=v["id"],
                calls_used=v.get("calls_used", []),
                vertices=[Pt(*p) for p in (v.get("vertices", []) or [])],
                closed=bool(v.get("closed", False)),
                primitives=prims,
                closure=Closure(
                    sum_dn=float(cl.get("sum_dn", 0.0)),
                    sum_de=float(cl.get("sum_de", 0.0)),
                    misclosure_ft=float(cl.get("misclosure_ft", 0.0)),
                    total_length_ft=float(cl.get("total_length_ft", 0.0)),
                    closure_ratio=float(cl.get("closure_ratio", float("inf"))),
                ),
                gate_passed=bool(v.get("gate_passed", False)),
                gate_notes=str(v.get("gate_notes", "")),
            )
        proj.layers = layers
        return proj

# ----------------------------
# AOIs
# ----------------------------
def aoi_root(project: Project) -> Path:
    return ensure_dir(Path(project.out_dir) / "aoi")

def next_aoi_id(project: Project) -> str:
    existing = set(project.aois.keys())
    i = 1
    while True:
        aid = f"aoi-{i:04d}"
        if aid not in existing:
            return aid
        i += 1

def aoi_hash(source_sha: str, page: int, bbox_px: List[int], preprocess: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update((source_sha + "|" + str(page) + "|" + json.dumps(bbox_px) + "|" + json.dumps(preprocess, sort_keys=True) + "|" + PROJECT_VERSION).encode("utf-8"))
    return h.hexdigest()

def create_aoi(project: Project, bbox_px: List[int], purpose: str, page: int,
               aoi_type: str = "boundary_callouts", mode: str = "binary", threshold: int = 170, scale: float = 3.0,
               aoi_id: Optional[str] = None, force: bool = False) -> str:
    base = open_pillow_image(Path(project.image_path), page=page)
    bbox_px = clamp_bbox([int(v) for v in bbox_px], base.size[0], base.size[1])

    aoi_id = aoi_id or next_aoi_id(project)
    d = ensure_dir(aoi_root(project) / aoi_id)

    source_sha = str(project.flags.get("source_sha256") or sha256_file(Path(project.image_path)))
    project.flags["source_sha256"] = source_sha

    preprocess = {"mode": mode, "threshold": int(threshold), "scale": float(scale)}
    hsh = aoi_hash(source_sha, page, bbox_px, preprocess)

    meta_path = d / "meta.json"
    if meta_path.exists() and not force:
        try:
            old = read_json(meta_path)
            if old.get("hash") == hsh:
                project.aois[aoi_id] = old
                return aoi_id
        except Exception:
            pass

    x, y, w, h = bbox_px
    raw = base.convert("RGB").crop((x, y, x + w, y + h))
    raw_path = d / "image.png"
    raw.save(raw_path)

    pre = preprocess_pil(raw, mode=mode, threshold=threshold, scale=scale)
    pre_path = d / "preprocess.png"
    pre.save(pre_path)

    meta = {
        "aoi_id": aoi_id,
        "type": aoi_type,
        "purpose": purpose,
        "source": {"image_path": str(Path(project.image_path).resolve()), "page": page},
        "bbox_px": bbox_px,
        "created_at": now_iso(),
        "hash": hsh,
        "preprocess": preprocess,
        "selected": False,
        "score": 0.0,
        "paths": {"dir": str(d), "raw_image": str(raw_path), "preprocess_image": str(pre_path), "meta": str(meta_path)},
    }
    write_json(meta_path, meta)
    project.aois[aoi_id] = meta
    return aoi_id

def load_aoi_meta(project: Project, aoi_id: str) -> Dict[str, Any]:
    if aoi_id in project.aois and isinstance(project.aois[aoi_id], dict):
        meta = project.aois[aoi_id]
        mp = Path(meta.get("paths", {}).get("meta", ""))
        if mp.exists():
            return meta
    mp = aoi_root(project) / aoi_id / "meta.json"
    meta = read_json(mp)
    project.aois[aoi_id] = meta
    return meta

def aoi_propose_linework(project: Project, audit: AuditLogger, page: int,
                         k: int = 18, refine_top_m: int = 40,
                         tile: int = 520, stride: int = 320,
                         pad: int = 80) -> List[List[int]]:
    from PIL import ImageFilter  # type: ignore

    ok, _ = check_ocr_ready()

    base = open_pillow_image(Path(project.image_path), page=page).convert("L")
    W, H = base.size

    max_dim = 2200
    ds = 1.0
    if max(W, H) > max_dim:
        ds = max_dim / float(max(W, H))
        nw = int(round(W * ds))
        nh = int(round(H * ds))
        small = base.resize((nw, nh))
    else:
        small = base
    wS, hS = small.size

    ex_x0 = int(wS * 0.62)
    ex_y0 = int(hS * 0.70)

    edges = small.filter(ImageFilter.FIND_EDGES)

    tiles: List[Tuple[float, int, int, int, int]] = []
    for y in range(0, hS, stride):
        for x in range(0, wS, stride):
            bw = min(tile, wS - x)
            bh = min(tile, hS - y)
            if bw < 180 or bh < 180:
                continue
            cx = x + bw // 2
            cy = y + bh // 2
            if cx >= ex_x0 and cy >= ex_y0:
                continue

            crop = edges.crop((x, y, x + bw, y + bh))
            hist = crop.histogram()
            edge_cnt = sum(hist[40:])
            density = edge_cnt / float(bw * bh)
            tiles.append((density, x, y, bw, bh))

    tiles.sort(key=lambda t: t[0], reverse=True)
    tiles = tiles[:max(10, refine_top_m)]

    ranked: List[Tuple[float, List[int]]] = []
    Image, pytesseract = try_import_ocr()
    for density, xS, yS, bwS, bhS in tiles:
        x0 = int(round(xS / ds))
        y0 = int(round(yS / ds))
        w0 = int(round(bwS / ds))
        h0 = int(round(bhS / ds))
        bbox = clamp_bbox([x0 - pad, y0 - pad, w0 + 2 * pad, h0 + 2 * pad], W, H)

        score = density * 10.0
        if ok and pytesseract is not None:
            rx, ry, rw, rh = bbox
            probe = open_pillow_image(Path(project.image_path), page=page).convert("RGB").crop((rx, ry, rx + rw, ry + rh))
            probe_pre = preprocess_pil(probe, mode="binary", threshold=170, scale=2.0)
            try:
                s = ocr_image_to_string(probe_pre, image_path=None, config=OCR_CONFIG_CALLS)
                score += quick_bearing_likeness(s)
            except Exception as e:
                audit.log("aoi_probe_ocr_fail", {"err": str(e)})

        ranked.append((score, bbox))

    ranked.sort(key=lambda t: t[0], reverse=True)

    out: List[List[int]] = []
    used: List[Tuple[int, int]] = []
    for score, bb in ranked:
        if len(out) >= k:
            break
        cx = bb[0] + bb[2] // 2
        cy = bb[1] + bb[3] // 2
        ok2 = True
        for ux, uy in used:
            if abs(cx - ux) < bb[2] * 0.45 and abs(cy - uy) < bb[3] * 0.45:
                ok2 = False
                break
        if not ok2:
            continue
        used.append((cx, cy))
        out.append(bb)

    audit.log("aoi_propose_linework", {"count": len(out)})
    return out

# ----------------------------
# Tokenization
# ----------------------------
def _new_token_id(aoi_id: str, token_type: str, text: str, bbox_base: Optional[List[int]]) -> str:
    return stable_id("tok", aoi_id, token_type, text, bbox_base)

def aoi_run_tokens(project: Project, aoi_id: str, force: bool = False) -> Dict[str, Any]:
    meta = load_aoi_meta(project, aoi_id)
    d = Path(meta["paths"]["dir"])
    out_path = d / "tokens.json"
    if out_path.exists() and not force:
        try:
            cur = read_json(out_path)
            if cur.get("aoi_hash") == meta.get("hash"):
                return cur
        except Exception:
            pass

    ok, msg = check_ocr_ready()
    if not ok:
        payload = {"aoi_id": aoi_id, "aoi_hash": meta.get("hash"), "created_at": now_iso(),
                   "token_count": 0, "tokens": [], "error": msg}
        write_json(out_path, payload)
        return payload

    img = open_pillow_image(Path(meta["paths"]["preprocess_image"]))
    cands = ocr_scan_lines(img, config=OCR_CONFIG_CALLS, image_path=Path(meta["paths"]["preprocess_image"]))

    scale = float(meta.get("preprocess", {}).get("scale", 1.0) or 1.0)
    bx, by, _, _ = [int(v) for v in meta["bbox_px"]]
    page = int(meta.get("source", {}).get("page", 0))

    toks: List[Dict[str, Any]] = []
    for c in cands:
        txt = _normalize_text_for_parse(c.text)
        if not txt:
            continue

        ax, ay, aw, ah = [int(v) for v in c.bbox]
        ux = int(round(ax / scale))
        uy = int(round(ay / scale))
        uw = max(1, int(round(aw / scale)))
        uh = max(1, int(round(ah / scale)))
        bbox_base = [bx + ux, by + uy, uw, uh]

        qb = None
        try:
            qb = parse_quadrant_bearing(txt)
        except Exception:
            qb = None

        off = try_parse_offset_ft(txt)
        dft = try_parse_distance_ft(txt)

        dft2 = None
        if qb is not None:
            dft2 = distance_after_ew_if_bearing(txt)

        if off is not None:
            tid = _new_token_id(aoi_id, "offset", str(off), bbox_base)
            toks.append(dataclasses.asdict(Token(
                id=tid, token_type="offset", fields={"value_ft": float(off)},
                confidence=float(c.confidence), aoi_id=aoi_id, bbox_base_px=bbox_base, page=page, raw_text=txt
            )))

        if qb is not None:
            tid = _new_token_id(aoi_id, "bearing", qb.format(), bbox_base)
            toks.append(dataclasses.asdict(Token(
                id=tid, token_type="bearing", fields={"bearing": qb.format()},
                confidence=float(c.confidence), aoi_id=aoi_id, bbox_base_px=bbox_base, page=page, raw_text=txt
            )))

        if dft2 is not None:
            tid = _new_token_id(aoi_id, "distance", str(dft2), bbox_base)
            toks.append(dataclasses.asdict(Token(
                id=tid, token_type="distance", fields={"distance_ft": float(dft2)},
                confidence=float(c.confidence), aoi_id=aoi_id, bbox_base_px=bbox_base, page=page, raw_text=txt
            )))
            tid = _new_token_id(aoi_id, "bearing_distance", qb.format() + "|" + str(dft2), bbox_base)
            toks.append(dataclasses.asdict(Token(
                id=tid, token_type="bearing_distance", fields={"bearing": qb.format(), "distance_ft": float(dft2)},
                confidence=float(c.confidence), aoi_id=aoi_id, bbox_base_px=bbox_base, page=page, raw_text=txt
            )))
        elif dft is not None:
            tid = _new_token_id(aoi_id, "distance", str(dft), bbox_base)
            toks.append(dataclasses.asdict(Token(
                id=tid, token_type="distance", fields={"distance_ft": float(dft)},
                confidence=float(c.confidence), aoi_id=aoi_id, bbox_base_px=bbox_base, page=page, raw_text=txt
            )))

    payload = {"aoi_id": aoi_id, "aoi_hash": meta.get("hash"), "created_at": now_iso(),
               "token_count": len(toks), "tokens": toks}
    write_json(out_path, payload)
    return payload

# ----------------------------
# AOI scoring + selection
# ----------------------------
def score_tokens(tokens: List[Dict[str, Any]]) -> float:
    h = token_type_hist(tokens)
    b = float(h.get("bearing", 0))
    bd = float(h.get("bearing_distance", 0))
    d = float(h.get("distance", 0))
    off = float(h.get("offset", 0))
    score = 8.0 * bd + 3.0 * b + 0.2 * d + 0.05 * off
    if (b + bd) <= 0.0:
        score *= 0.05
    return score

def aoi_rank_and_select(project: Project, audit: AuditLogger, force_tokens: bool,
                        select_n: int = 10) -> Dict[str, Any]:
    scored: List[Tuple[float, str, Dict[str, int]]] = []
    total_bearings = 0

    for aid in sorted((project.aois or {}).keys()):
        payload = aoi_run_tokens(project, aid, force=force_tokens)
        toks = payload.get("tokens", []) or []
        h = token_type_hist(toks)
        total_bearings += int(h.get("bearing", 0) + h.get("bearing_distance", 0))
        sc = score_tokens(toks)
        scored.append((sc, aid, h))

    scored.sort(key=lambda x: x[0], reverse=True)

    eligible = [(sc, aid, h) for (sc, aid, h) in scored if (h.get("bearing", 0) + h.get("bearing_distance", 0)) > 0]
    chosen = eligible[:max(1, select_n)] if eligible else scored[:max(1, select_n)]
    selected = set(aid for _sc, aid, _h in chosen)

    for sc, aid, h in scored:
        meta = load_aoi_meta(project, aid)
        meta["selected"] = bool(aid in selected)
        meta["score"] = float(sc)
        meta["token_hist"] = h
        write_json(Path(meta["paths"]["meta"]), meta)
        project.aois[aid] = meta

    idx = {
        "created_at": now_iso(),
        "selected": sorted(selected),
        "scored": [{"aoi_id": aid, "score": float(sc), "hist": h} for sc, aid, h in scored],
        "total_bearings_across_all_aois": int(total_bearings),
    }
    idx_path = Path(project.out_dir) / "aoi_index.json"
    write_json(idx_path, idx)
    audit.log("aoi_rank_select", {"selected": sorted(selected), "idx": str(idx_path), "total_bearings": total_bearings})

    return {"selected": sorted(selected), "total_bearings": total_bearings, "idx_path": str(idx_path)}

# ----------------------------
# Calls from tokens
# ----------------------------
def calls_from_tokens(project: Project, tokens: List[Dict[str, Any]], layer: str) -> Dict[str, Any]:
    created_bd = 0
    created_pairs = 0

    for t in tokens:
        if t.get("token_type") != "bearing_distance":
            continue
        f = t.get("fields", {}) or {}
        btxt = f.get("bearing")
        dft = f.get("distance_ft")
        if not btxt or dft is None:
            continue
        try:
            qb = parse_quadrant_bearing(str(btxt))
            dist_ft = float(dft)
            if dist_ft <= 0:
                raise ValueError
        except Exception:
            continue

        call_id = f"line_tok_{t.get('id')}"
        if call_id in project.calls:
            continue

        ev = Evidence(
            type="ocr",
            aoi_id=t.get("aoi_id"),
            page=t.get("page"),
            bbox_base=t.get("bbox_base_px"),
            ocr_text=t.get("raw_text"),
            confidence=float(t.get("confidence", 0.0) or 0.0),
            notes="auto from bearing_distance token",
        )
        project.calls[call_id] = RecordCall(
            id=call_id, kind="line", layer=layer, seq=None, tags=["record"], evidence=ev,
            bearing_text=qb.format(), distance_text=str(dist_ft), bearing=qb, distance_ft=dist_ft
        )
        created_bd += 1

    by_aoi: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for t in tokens:
        tt = t.get("token_type")
        if tt not in ("bearing", "distance"):
            continue
        aid = str(t.get("aoi_id") or "none")
        by_aoi.setdefault(aid, {}).setdefault(tt, []).append(t)

    max_pair_dist_px = float(project.flags.get("pair_max_dist_px", 220.0))

    for aid, g in by_aoi.items():
        bearings = g.get("bearing", [])
        dists = g.get("distance", [])
        if not bearings or not dists:
            continue

        used_dist = set()
        for bt in bearings:
            bb = bt.get("bbox_base_px")
            if not bb:
                continue
            bcx, bcy = bbox_center(bb)

            best = None
            for dtok in dists:
                did = dtok.get("id")
                if did in used_dist:
                    continue
                db = dtok.get("bbox_base_px")
                if not db:
                    continue
                dcx, dcy = bbox_center(db)
                d = math.hypot(dcx - bcx, dcy - bcy)
                if d <= max_pair_dist_px and (best is None or d < best[0]):
                    best = (d, dtok)
            if best is None:
                continue

            _, dtok = best
            used_dist.add(dtok.get("id"))

            try:
                qb = parse_quadrant_bearing(str((bt.get("fields") or {}).get("bearing")))
                dist_ft = float((dtok.get("fields") or {}).get("distance_ft"))
                if dist_ft <= 0:
                    raise ValueError
            except Exception:
                continue

            call_id = f"line_pair_{bt.get('id')}_{dtok.get('id')}"
            if call_id in project.calls:
                continue

            ub = bbox_union(bb, dtok.get("bbox_base_px"))
            ev = Evidence(
                type="ocr",
                aoi_id=aid,
                page=bt.get("page") or dtok.get("page"),
                bbox_base=ub,
                ocr_text=f"{bt.get('raw_text','')} | {dtok.get('raw_text','')}",
                confidence=float(max(bt.get("confidence", 0.0) or 0.0, dtok.get("confidence", 0.0) or 0.0)),
                notes="auto paired bearing+distance",
            )
            project.calls[call_id] = RecordCall(
                id=call_id, kind="line", layer=layer, seq=None, tags=["record"], evidence=ev,
                bearing_text=qb.format(), distance_text=str(dist_ft), bearing=qb, distance_ft=dist_ft
            )
            created_pairs += 1

    return {"created_bearing_distance": created_bd, "created_pairs": created_pairs}

# ----------------------------
# Preview solve (>=1 call)
# ----------------------------
def closure_from_calls(calls: List[RecordCall]) -> Closure:
    sum_dn = 0.0
    sum_de = 0.0
    total = 0.0
    cur = Pt(0.0, 0.0)
    for c in calls:
        if c.kind != "line" or c.bearing is None or c.distance_ft is None:
            continue
        u = vec_from_az_deg(c.bearing.to_azimuth_deg()).unit()
        L = float(c.distance_ft)
        nxt = cur + u.scale(L)
        dv = nxt - cur
        sum_de += dv.de
        sum_dn += dv.dn
        total += L
        cur = nxt
    mis = math.hypot(sum_dn, sum_de)
    ratio = (total / mis) if mis > EPS else float("inf")
    return Closure(sum_dn=sum_dn, sum_de=sum_de, misclosure_ft=mis, total_length_ft=total, closure_ratio=ratio)

def build_vertices_and_prims(start: Pt, calls: List[RecordCall], close_tol_ft: float) -> Tuple[List[Pt], bool, List[Primitive]]:
    verts: List[Pt] = [start]
    prims: List[Primitive] = []
    cur = start
    for c in calls:
        if c.kind != "line" or c.bearing is None or c.distance_ft is None:
            continue
        u = vec_from_az_deg(c.bearing.to_azimuth_deg()).unit()
        L = float(c.distance_ft)
        nxt = cur + u.scale(L)
        prims.append(Primitive(type="line", start=cur, end=nxt, bulge=0.0))
        verts.append(nxt)
        cur = nxt
    closed = dist(verts[0], verts[-1]) <= close_tol_ft if len(verts) >= 2 else False
    if closed:
        verts = verts[:-1]
    return verts, closed, prims

def solve_preview(project: Project, layer_id: str, call_ids: List[str]) -> bool:
    calls: List[RecordCall] = []
    for cid in call_ids:
        c = project.calls.get(cid)
        if not c:
            continue
        if c.kind != "line" or c.bearing is None or c.distance_ft is None:
            continue
        calls.append(c)

    if len(calls) < 1:
        return False

    close_tol = float(project.flags.get("close_tol_ft", 0.10))
    cl = closure_from_calls(calls)
    verts, closed, prims = build_vertices_and_prims(Pt(0.0, 0.0), calls, close_tol_ft=close_tol)

    project.layers[layer_id] = LayerResult(
        id=layer_id,
        calls_used=[c.id for c in calls],
        vertices=verts,
        closed=closed,
        primitives=prims,
        closure=cl,
        gate_passed=False,
        gate_notes="PREVIEW_ONLY",
    )
    return True

# ----------------------------
# Rendering
# ----------------------------
def render_dxf(project: Project, out_path: Path) -> None:
    ents: List[str] = []

    def add_lwpoly(verts: List[Pt], closed: bool, layer: str):
        n = len(verts)
        if n < 2:
            return
        ents.extend(["0", "LWPOLYLINE", "8", layer, "90", str(n), "70", "1" if closed else "0"])
        for p in verts:
            ents.extend(["10", f"{p.e:.6f}", "20", f"{p.n:.6f}"])

    for lid, lr in project.layers.items():
        if len(lr.vertices) >= 2:
            add_lwpoly(lr.vertices, lr.closed, lid.upper())

    dxf = []
    dxf.extend(["0", "SECTION", "2", "HEADER"])
    dxf.extend(["9", "$ACADVER", "1", "AC1015"])
    dxf.extend(["0", "ENDSEC"])
    dxf.extend(["0", "SECTION", "2", "ENTITIES"])
    dxf.extend(ents)
    dxf.extend(["0", "ENDSEC", "0", "EOF"])
    write_text(out_path, "\n".join(dxf))

def report_md(project: Project) -> str:
    lines = []
    lines.append("# Recon Report")
    lines.append("")
    lines.append(f"- created_at: {project.created_at}")
    lines.append(f"- image: `{project.image_path}`")
    lines.append("")
    lines.append("## AOIs")
    sel = [aid for aid, m in (project.aois or {}).items() if m.get("selected")]
    lines.append(f"- selected: {len(sel)} of {len(project.aois or {})}")
    lines.append("")
    lines.append("## Layers")
    if not project.layers:
        lines.append("- (none)")
    for lid, lr in project.layers.items():
        c = lr.closure
        lines.append(f"- **{lid}**: mis {c.misclosure_ft:.4f} ft ({ft_to_inches(c.misclosure_ft):.2f} in) ratio 1:{c.closure_ratio:.0f} total {c.total_length_ft:.2f} ft — {lr.gate_notes}")
    lines.append("")
    return "\n".join(lines)

# ----------------------------
# Quarantine helper
# ----------------------------
@dataclass
class Verdict:
    status: str
    code: str
    msg: str
    details: Dict[str, Any] = field(default_factory=dict)

def quarantine_emit(project: Project, audit: AuditLogger, reason: Verdict, context: Dict[str, Any]) -> Path:
    qdir = ensure_dir(Path(project.out_dir) / "quarantine")
    qid = stable_id("quarantine", reason.code, now_iso())
    qp = qdir / f"{qid}.json"
    write_json(qp, {"created_at": now_iso(), "reason": dataclasses.asdict(reason), "context": context})
    audit.log("quarantine", {"path": str(qp), "reason": reason.code})
    return qp

# ----------------------------
# Commands
# ----------------------------
def cmd_init(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(Path(args.out).resolve())
    audit = AuditLogger(out_dir / "audit.ndjson")
    img_path = Path(args.image).resolve()

    proj = Project(version=PROJECT_VERSION, created_at=now_iso(), image_path=str(img_path), out_dir=str(out_dir))
    proj.flags.update({
        "raster_page": int(args.page),
        "source_sha256": sha256_file(img_path),
        "pair_max_dist_px": 220.0,
        "close_tol_ft": 0.10,
    })
    proj_path = out_dir / "project.json"
    proj.save(proj_path)
    audit.log("init", {"project": str(proj_path), "image": str(img_path), "page": int(args.page)})
    print(proj_path)

def cmd_run(args: argparse.Namespace) -> None:
    proj_path = Path(args.project).resolve()
    proj = Project.load(proj_path)
    out_dir = Path(proj.out_dir)
    audit = AuditLogger(out_dir / "audit.ndjson")

    if args.page is not None:
        proj.flags["raster_page"] = int(args.page)
    page = int(proj.flags.get("raster_page", 0))

    ocr_ok, ocr_msg = check_ocr_ready()
    audit.log("ocr_check", {"ok": ocr_ok, "msg": ocr_msg})

    # Always add a fresh batch of AOIs each run (improves convergence)
    bboxes = aoi_propose_linework(proj, audit, page=page, k=int(args.k_linework), refine_top_m=int(args.refine_top_m))
    created = []
    for i, bb in enumerate(bboxes, 1):
        created.append(create_aoi(proj, bb, purpose=f"auto:linework:{i}", page=page,
                                 mode="binary", threshold=int(args.aoi_threshold), scale=float(args.aoi_scale)))
    audit.log("aoi_created_linework", {"count": len(created)})

    sel_info = aoi_rank_and_select(proj, audit, force_tokens=True, select_n=int(args.select_n))
    selected_aois = sel_info.get("selected", []) or []

    tokens: List[Dict[str, Any]] = []
    for aid in selected_aois:
        payload = aoi_run_tokens(proj, aid, force=False)
        tokens.extend(payload.get("tokens", []) or [])

    th = token_type_hist(tokens)
    audit.log("tokens_selected", {"selected_aois": selected_aois, "count": len(tokens), "hist": th})

    call_stats = calls_from_tokens(proj, tokens, layer=args.pool_layer)
    audit.log("calls_from_tokens", call_stats)

    if len(proj.calls) == 0:
        qp = quarantine_emit(
            proj, audit,
            Verdict("FAIL", "no_calls_created", "Selected AOIs did not yield usable calls.", {}),
            {"selected_aois": selected_aois, "token_hist": th, "aoi_index": str(Path(proj.out_dir) / "aoi_index.json")}
        )
        proj.diagnostics.setdefault("quarantines", []).append(str(qp))

    pool_ids = [cid for cid, c in proj.calls.items() if c.kind == "line" and c.is_record() and c.layer == args.pool_layer]
    preview_ids = pool_ids[:max(1, min(int(args.preview_max_calls), len(pool_ids)))] if pool_ids else []
    preview_ok = solve_preview(proj, "preview", preview_ids)

    proj.save(proj_path)

    dxf_path = out_dir / "drawing.dxf"
    report_path = out_dir / "report.md"
    final_path = out_dir / "final.json"

    try:
        render_dxf(proj, dxf_path)
    except Exception as e:
        audit.log("render_dxf_fail", {"error": str(e)})

    try:
        write_text(report_path, report_md(proj))
    except Exception as e:
        audit.log("report_fail", {"error": str(e)})

    final_obj = {
        "created_at": now_iso(),
        "status": "UNPROVEN_PREVIEW" if preview_ok else "UNPROVEN",
        "note": "preview built" if preview_ok else "No calls available for preview",
        "project": str(proj_path),
        "outputs": {
            "dxf": str(dxf_path) if dxf_path.exists() else "",
            "report": str(report_path) if report_path.exists() else "",
            "quarantines": proj.diagnostics.get("quarantines", []) or [],
        },
        "stats": {
            "aois_total": len(proj.aois or {}),
            "aois_selected": selected_aois,
            "tokens_total_selected": len(tokens),
            "token_hist_selected": th,
            "calls_total": len(proj.calls),
            "pool_calls": len(pool_ids),
            "preview_calls": len(preview_ids),
            "layers": list(proj.layers.keys()),
            "aoi_index": str(Path(proj.out_dir) / "aoi_index.json"),
        },
        "ocr": {"ok": ocr_ok, "msg": ocr_msg},
    }
    write_json(final_path, final_obj)

    packet_dir = ensure_dir(out_dir / "artifacts")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    zip_path = packet_dir / f"evidence-packet-{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(proj_path, arcname="project.json")
        aud = out_dir / "audit.ndjson"
        if aud.exists():
            z.write(aud, arcname="audit.ndjson")
        for fp, an in [
            (final_path, "final.json"),
            (dxf_path, "drawing.dxf"),
            (report_path, "report.md"),
            (Path(proj.out_dir) / "aoi_index.json", "aoi_index.json"),
        ]:
            if fp.exists():
                z.write(fp, arcname=an)
        qdir = out_dir / "quarantine"
        if qdir.exists():
            for qp in sorted(qdir.glob("*.json")):
                z.write(qp, arcname=str(Path("quarantine") / qp.name))
        for aid in sorted((proj.aois or {}).keys()):
            try:
                meta = load_aoi_meta(proj, aid)
            except Exception:
                continue
            d = Path(meta["paths"]["dir"])
            for fn in ("meta.json", "tokens.json", "image.png", "preprocess.png"):
                fp = d / fn
                if fp.exists():
                    z.write(fp, arcname=str(Path("aoi") / aid / fn))

    audit.log("run_complete", {"final": str(final_path), "zip": str(zip_path)})
    print(str(final_path))
    print(str(zip_path))

# ----------------------------
# CLI
# ----------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="survey-recon", description="init + run (minimal).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("image")
    s.add_argument("--out", required=True)
    s.add_argument("--page", type=int, default=0)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("run")
    s.add_argument("project")
    s.add_argument("--page", type=int, default=None)

    s.add_argument("--k-linework", type=int, default=18)
    s.add_argument("--refine-top-m", type=int, default=40)
    s.add_argument("--select-n", type=int, default=10)
    s.add_argument("--aoi-scale", type=float, default=3.0)
    s.add_argument("--aoi-threshold", type=int, default=170)

    s.add_argument("--pool-layer", default="pool")
    s.add_argument("--preview-max-calls", type=int, default=20)
    s.add_argument("--ocr-engine", choices=["tesseract", "mineru"], default=None,
                   help="Override OCR engine selection. Default is auto (MinerU on Apple Silicon, Tesseract otherwise).")

    s.set_defaults(func=cmd_run)
    return p

def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_cli().parse_args(argv)
    if getattr(args, "ocr_engine", None):
        os.environ["SURVEY_RECON_OCR_ENGINE"] = str(args.ocr_engine)
    args.func(args)

if __name__ == "__main__":
    main()
