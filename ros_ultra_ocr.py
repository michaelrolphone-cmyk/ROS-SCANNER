#!/usr/bin/env python3
"""
ros_ultra_ocr.py
Ultra-quality OCR extractor for *scanned* Record of Survey / plat drawings (raster-only).

Features:
- Bearing + distance extraction from angled callouts (N/S .. E/W)
- Decimal recovery when OCR drops the dot (pixel dot evidence + number-only re-read)
- Linework-crossing suppression (line mask overlap + inpaint variants)
- Line/Curve table harvesting (best-effort)
- Basis of Bearing detection ("BASIS OF BEARING" + nearest pair)
- Historic/record distance capture: (123.45) and REC/RECORD/OLD forms.

Deps:
  pip install pytesseract pdf2image Pillow opencv-python-headless numpy

macOS:
  brew install tesseract poppler

Usage (quality-first; can take a long time):
  python ros_ultra_ocr.py file.pdf --dpis 450,650,900 --pages 1-3 --threads 12 --out out.json --annotate debug.png --verbose
"""

from __future__ import annotations
import argparse, json, math, os, re, sys, warnings
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

Image.MAX_IMAGE_PIXELS = 900_000_000
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# ----------------- parsing -----------------

DIGIT_FIX = str.maketrans({"O":"0","o":"0","D":"0","I":"1","l":"1","|":"1","S":"5","s":"5","B":"8","Z":"2"})
BEARING_RE = re.compile(r"(?P<ns>[NS])\s*(?P<deg>[0-9OolIDSBZ|]{1,3})\s*(?:[°oº]\s*)?(?P<min>[0-9OolIDSBZ|]{1,2})\s*(?:[\'’]\s*)?(?P<sec>[0-9OolIDSBZ|]{1,2})?\s*(?:[”″])?\s*(?P<ew>[EW])", re.I)
PAREN_RE = re.compile(r"\(([^)]+)\)")
REC_RE = re.compile(r"(?i)\b(?:REC|RECORD|OLD)\s*=?\s*([0-9OolIDSBZ|][0-9OolIDSBZ|,]*([.·•∙⋅][0-9OolIDSBZ|]{1,4})?)")
LINE_ID_RE = re.compile(r"(?i)\bL[\s\-]?\d{1,3}\b")
CURVE_ID_RE = re.compile(r"(?i)\bC[\s\-]?\d{1,3}\b")
BASIS_RE = re.compile(r"(?i)\bbasis\s+of\s+bearing(s)?\b")

def norm(s: str) -> str:
    return re.sub(r"\s+"," ",(s or "").strip())

def canon_label(tok: str) -> Optional[str]:
    if not tok: return None
    t = tok.strip().upper().replace(" ","").replace("-","")
    return t if re.fullmatch(r"[LC]\d{1,3}", t) else None

def parse_bearing(text: str) -> Optional[str]:
    m = BEARING_RE.search(text or "")
    if not m: return None
    ns, ew = m.group("ns").upper(), m.group("ew").upper()
    deg_s = (m.group("deg") or "").translate(DIGIT_FIX)
    min_s = (m.group("min") or "").translate(DIGIT_FIX)
    sec_s = (m.group("sec") or "0").translate(DIGIT_FIX)
    try:
        deg, minute, sec = int(deg_s), int(min_s), int(sec_s)
    except ValueError:
        return None
    if deg > 90:
        for k in (300,200,100):
            if k <= deg < k+90:
                deg -= k; break
    if 60 <= minute <= 99: minute -= 60
    if 60 <= sec <= 99: sec -= 60
    if not (0<=deg<=90 and 0<=minute<=59 and 0<=sec<=59): return None
    return f"{ns} {deg:02d}°{minute:02d}'{sec:02d}\" {ew}"

def bearing_theta(canon: str) -> Optional[float]:
    m = re.match(r"^([NS])\s+(\d{2})°(\d{2})'(\d{2})\"\s+([EW])$", canon or "")
    if not m: return None
    ns, d, mi, se, ew = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    ang = d + mi/60.0 + se/3600.0
    if ns=="N" and ew=="E": return ang
    if ns=="N" and ew=="W": return (360-ang)%360
    if ns=="S" and ew=="E": return (180-ang)%360
    if ns=="S" and ew=="W": return (180+ang)%360
    return None

def record_distance(line: str) -> Optional[str]:
    t = line or ""
    m = PAREN_RE.search(t)
    if m:
        inside = re.sub(r"[^0-9.,·•∙⋅]","", m.group(1) or "")
        inside = inside.replace("·",".").replace("•",".").replace("∙",".").replace("⋅",".").replace(",","").translate(DIGIT_FIX)
        if inside and re.search(r"\d", inside):
            try:
                v = float(inside)
                return str(int(round(v))) if abs(v-round(v)) < 1e-9 else f"{v:.3f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
    m = REC_RE.search(t)
    if m:
        s=(m.group(1) or "").replace(",","").replace("·",".").replace("•",".").replace("∙",".").replace("⋅",".").translate(DIGIT_FIX)
        s=re.sub(r"[^0-9.]", "", s)
        if s:
            try:
                v=float(s)
                return str(int(round(v))) if abs(v-round(v)) < 1e-9 else f"{v:.3f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
    return None

def find_ref_id(line: str) -> Optional[str]:
    m = LINE_ID_RE.search(line or "") or CURVE_ID_RE.search(line or "")
    return canon_label(m.group(0)) if m else None

CURVE_RADIUS_RE = re.compile(r"(?i)\b(?:R|RAD|RADIUS)\s*=?\s*([0-9OolIDSBZ|][0-9OolIDSBZ|,]*([.·•∙⋅][0-9OolIDSBZ|]{1,4})?)")
CURVE_ARC_RE    = re.compile(r"(?i)\b(?:ARC\s*L(?:EN(?:GTH)?)?|ARCLEN|ARCLENGTH|AL|L\s*=)\s*=?\s*([0-9OolIDSBZ|][0-9OolIDSBZ|,]*([.·•∙⋅][0-9OolIDSBZ|]{1,4})?)")
CURVE_CH_RE     = re.compile(r"(?i)\b(?:CH(?:ORD)?\s*L(?:EN(?:GTH)?)?|CHL|CL)\s*=?\s*([0-9OolIDSBZ|][0-9OolIDSBZ|,]*([.·•∙⋅][0-9OolIDSBZ|]{1,4})?)")
CURVE_T_RE      = re.compile(r"(?i)\b(?:TAN(?:GENT)?|T)\s*=?\s*([0-9OolIDSBZ|][0-9OolIDSBZ|,]*([.·•∙⋅][0-9OolIDSBZ|]{1,4})?)")
CURVE_DELTA_RE  = re.compile(r"(?i)\b(?:DELTA|Δ)\s*=?\s*([0-9OolIDSBZ|]{1,3})\s*(?:[°oº]\s*)?([0-9OolIDSBZ|]{1,2})?\s*(?:[\'’]\s*)?([0-9OolIDSBZ|]{1,2})?\s*(?:[”″])?")

def _num_from(m: re.Match) -> Optional[float]:
    s=(m.group(1) or "").replace(",","").replace("·",".").replace("•",".").replace("∙",".").replace("⋅",".").translate(DIGIT_FIX)
    s=re.sub(r"[^0-9.]", "", s)
    if not s: return None
    try: return float(s)
    except ValueError: return None

def extract_curve_fields(text: str) -> Optional[dict]:
    t = text or ""
    out={}
    m=CURVE_RADIUS_RE.search(t)
    if m:
        v=_num_from(m)
        if v is not None: out["radius"]=v
    m=CURVE_ARC_RE.search(t)
    if m:
        v=_num_from(m)
        if v is not None: out["arc_length"]=v
    m=CURVE_CH_RE.search(t)
    if m:
        v=_num_from(m)
        if v is not None: out["chord_length"]=v
    m=CURVE_T_RE.search(t)
    if m:
        v=_num_from(m)
        if v is not None: out["tangent"]=v
    m=CURVE_DELTA_RE.search(t)
    if m:
        deg=(m.group(1) or "").translate(DIGIT_FIX); mi=(m.group(2) or "0").translate(DIGIT_FIX); se=(m.group(3) or "0").translate(DIGIT_FIX)
        try:
            d=int(deg); m2=int(mi); s2=int(se)
            if d>360:
                for k in (300,200,100):
                    if k <= d < k+360: d-=k; break
            if 60<=m2<=99: m2-=60
            if 60<=s2<=99: s2-=60
            if 0<=d<=360 and 0<=m2<=59 and 0<=s2<=59:
                out["delta"]=f"{d:02d}°{m2:02d}'{s2:02d}\""
        except ValueError:
            pass
    cb = parse_bearing(t)
    if cb and (("CH" in t.upper()) or ("CB" in t.upper()) or ("CHORD" in t.upper())):
        out["chord_bearing"]=cb
    return out or None

# ----------------- imaging -----------------

def to_gray(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)

def downscale(pil: Image.Image, max_pixels: int) -> tuple[Image.Image, float]:
    W,H = pil.size
    pix = W*H
    if pix <= max_pixels: return pil, 1.0
    s = math.sqrt(max_pixels/float(pix))
    return pil.resize((max(1,int(W*s)), max(1,int(H*s))), Image.BICUBIC), s

def rotate_keep(pil: Image.Image, ang: float) -> Image.Image:
    a = float(ang)%360.0
    if abs(a) < 1e-9: return pil
    arr = np.array(pil.convert("RGB"))
    h,w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), a, 1.0)
    out = cv2.warpAffine(arr, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return Image.fromarray(out)

def inv_rotate_point(x: float, y: float, W: int, H: int, ang_deg: float) -> tuple[float,float]:
    a = math.radians(float(ang_deg)%360.0)
    cx, cy = W/2.0, H/2.0
    xr, yr = x-cx, y-cy
    ca, sa = math.cos(-a), math.sin(-a)
    xo = xr*ca - yr*sa + cx
    yo = xr*sa + yr*ca + cy
    return xo, yo

def quad_to_bbox_original(quad: np.ndarray, W: int, H: int, ang_deg: float) -> tuple[int,int,int,int]:
    pts = [inv_rotate_point(float(p[0]), float(p[1]), W, H, ang_deg) for p in quad]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1,y1,x2,y2 = int(max(0,min(xs))), int(max(0,min(ys))), int(min(W,max(xs))), int(min(H,max(ys)))
    return (x1,y1,x2,y2)

def order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1); d = np.diff(pts,axis=1).reshape(-1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def warp_quad(pil: Image.Image, quad: np.ndarray, pad: int=6) -> Image.Image:
    q = order_quad(quad.astype(np.float32))
    wA = np.linalg.norm(q[2]-q[3]); wB = np.linalg.norm(q[1]-q[0])
    hA = np.linalg.norm(q[1]-q[2]); hB = np.linalg.norm(q[0]-q[3])
    Wt = int(max(wA,wB))+2*pad; Ht = int(max(hA,hB))+2*pad
    Wt = max(16, min(Wt, 5000)); Ht = max(16, min(Ht, 2400))
    dst = np.array([[pad,pad],[Wt-pad,pad],[Wt-pad,Ht-pad],[pad,Ht-pad]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    arr = np.array(pil.convert("RGB"))
    out = cv2.warpPerspective(arr, M, (Wt,Ht), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return Image.fromarray(out)

def adaptive_inv(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)

def build_line_mask(gray: np.ndarray) -> np.ndarray:
    h,w = gray.shape[:2]
    bw = adaptive_inv(gray)
    hk = max(25, w//18); vk = max(25, h//18)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(hk,1)), iterations=1)
    vert  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,vk)), iterations=1)
    mask = cv2.bitwise_or(horiz, vert)
    mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=1)
    return mask

def overlap(mask: np.ndarray, bb: tuple[int,int,int,int]) -> float:
    x1,y1,x2,y2 = bb
    x1=max(0,x1); y1=max(0,y1); x2=min(mask.shape[1],x2); y2=min(mask.shape[0],y2)
    if x2<=x1 or y2<=y1: return 0.0
    roi = mask[y1:y2,x1:x2]
    return float(np.count_nonzero(roi))/float(roi.size)

def inpaint(gray: np.ndarray, radius: int=3) -> tuple[np.ndarray,np.ndarray]:
    m = build_line_mask(gray)
    return cv2.inpaint(gray, m, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA), m

# ----------------- dot evidence + number reread -----------------

def dot_digits_right(dot_ref_gray: np.ndarray, bb: tuple[int,int,int,int]) -> Optional[int]:
    x1,y1,x2,y2 = bb
    w=max(1,x2-x1); h=max(1,y2-y1)
    mx=max(2,int(w*0.08)); my=max(2,int(h*0.12))
    X1=max(0,x1-mx); Y1=max(0,y1-my); X2=min(dot_ref_gray.shape[1],x2+mx); Y2=min(dot_ref_gray.shape[0],y2+my)
    crop = dot_ref_gray[Y1:Y2,X1:X2]
    if crop.size==0: return None
    _, bw = cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    n,_,stats,cent = cv2.connectedComponentsWithStats(bw,8)
    if n<=2: return None
    comps=[]
    for i in range(1,n):
        x,y,cw,ch,area = stats[i]
        if area<2: continue
        cx,cy = cent[i]
        comps.append((cw,ch,area,cx,cy))
    if not comps: return None
    areas=sorted([c[2] for c in comps], reverse=True)[:6]
    med = float(np.median(areas)) if areas else 0.0
    if med<=0: return None
    digits=[c for c in comps if c[2] >= max(6,0.10*med)]
    if len(digits)<2: return None
    cy_med = float(np.median([c[4] for c in digits]))
    dots=[]
    for (cw,ch,area,cx,cy) in comps:
        if area >= 0.35*med or area < 2: continue
        if cw>0.40*w or ch>0.40*h: continue
        fill = area/max(1.0,float(cw*ch))
        if fill<0.20: continue
        if not (cy>=0.50*h and cy<=0.95*h): continue
        if cy < cy_med: continue
        ar = cw/max(1.0,float(ch)); ar = ar if ar>=1 else 1/ar
        if ar>3.0: continue
        score = (1.0-min(1.0,area/(0.35*med)))*2.0 + (1.0-min(1.0,(ar-1.0)/2.0))*1.5 + fill
        dots.append((score,cx))
    if not dots: return None
    dots.sort(reverse=True)
    dot_cx = dots[0][1]
    dx = sorted([c[3] for c in digits])
    right=[x for x in dx if x>dot_cx+1.5]; left=[x for x in dx if x<dot_cx-1.5]
    if len(left)<1 or len(right)<1 or len(right)>4: return None
    return len(right)

def parse_dist(raw: str, dot_r: Optional[int], min_ft: float=0.0, max_ft: float=50_000_000.0) -> tuple[Optional[float], str]:
    s=(raw or "").strip().replace(" ","")
    s=s.replace("·",".").replace("•",".").replace("∙",".").replace("⋅",".")
    if "(" in s: s=s.split("(",1)[0]
    s=s.replace(",","").translate(DIGIT_FIX)
    if not s or not re.search(r"\d", s): return None,"invalid"
    if "." in s:
        try:
            v=float(s)
            return (v,"explicit_decimal") if min_ft<=v<=max_ft else (None,"invalid")
        except ValueError:
            pass
    digits=re.sub(r"\D","",s)
    if not digits: return None,"invalid"
    if dot_r is not None and 1<=dot_r<len(digits):
        ins=len(digits)-dot_r
        try:
            v=float(digits[:ins]+"."+digits[ins:])
            if min_ft<=v<=max_ft: return v,"dot_blob"
        except ValueError:
            pass
    try:
        v=float(digits)
        return (v,"integer") if min_ft<=v<=max_ft else (None,"invalid")
    except ValueError:
        return None,"invalid"

def number_reread(pil: Image.Image, timeout: Optional[int]) -> str:
    img=pil
    W,H=img.size
    if W*H>2_000_000:
        s=math.sqrt(2_000_000/(W*H)); img=img.resize((max(1,int(W*s)),max(1,int(H*s))), Image.BICUBIC)
    if max(W,H)<160: img=img.resize((int(img.size[0]*2.5), int(img.size[1]*2.5)), Image.BICUBIC)
    g=to_gray(img)
    outs=[]
    for inv in (False, True):
        mode=(cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)|cv2.THRESH_OTSU
        _,bw=cv2.threshold(g,0,255,mode)
        pilb=Image.fromarray(bw)
        cfg="--oem 3 --psm 8 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789."
        try:
            d=pytesseract.image_to_data(pilb,config=cfg,lang="eng",output_type=Output.DICT,timeout=timeout)
        except TypeError:
            d=pytesseract.image_to_data(pilb,config=cfg,lang="eng",output_type=Output.DICT)
        txt="".join((t or "") for t in d.get("text",[]) if t and re.search(r"\d",t))
        confs=[]
        for c in d.get("conf",[]):
            try:
                c=float(c)
                if c>=0: confs.append(c)
            except Exception:
                pass
        outs.append((norm(txt), (sum(confs)/len(confs)) if confs else -1.0))
    outs.sort(key=lambda x:x[1], reverse=True)
    return outs[0][0] if outs else ""

# ----------------- OCR line extraction -----------------

def curve_params_from_line(line: dict) -> Optional[dict]:
    txt = norm(line.get("text",""))
    rid = find_ref_id(txt)
    if not rid or not rid.startswith("C"):
        return None
    curve = extract_curve_fields(txt)
    if not curve:
        return None
    return {"ref_id": rid, "curve": curve, "conf": float(line.get("conf",-1.0)), "text": txt, "line_box": line.get("bbox")}

def ocr_lines(pil: Image.Image, cfg: str, timeout: Optional[int]) -> list[dict]:
    try:
        d=pytesseract.image_to_data(pil,config=cfg,lang="eng",output_type=Output.DICT,timeout=timeout)
    except TypeError:
        d=pytesseract.image_to_data(pil,config=cfg,lang="eng",output_type=Output.DICT)
    n=len(d.get("text",[]))
    lines={}
    for i in range(n):
        t=(d["text"][i] or "").strip()
        if not t: continue
        try: conf=float(d.get("conf",["-1"])[i])
        except Exception: conf=-1.0
        key=(int(d.get("block_num",[0])[i]), int(d.get("par_num",[0])[i]), int(d.get("line_num",[0])[i]))
        x,y,w,h = int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i])
        bb=(x,y,x+w,y+h)
        rec=lines.setdefault(key, {"words":[], "bbox":[x,y,x+w,y+h], "confs":[]})
        rec["words"].append({"text":t,"conf":conf,"bbox":bb})
        B=rec["bbox"]; B[0]=min(B[0],x); B[1]=min(B[1],y); B[2]=max(B[2],x+w); B[3]=max(B[3],y+h)
        if conf>=0: rec["confs"].append(conf)
    out=[]
    for rec in lines.values():
        rec["words"].sort(key=lambda w:w["bbox"][0])
        text=" ".join(w["text"] for w in rec["words"])
        conf=(sum(rec["confs"])/len(rec["confs"])) if rec["confs"] else -1.0
        out.append({"text":text, "conf":conf, "bbox":tuple(rec["bbox"]), "words":rec["words"]})
    out.sort(key=lambda r:(r["bbox"][1],r["bbox"][0]))
    return out

def preprocess_variants(crop: Image.Image, do_lineclean: bool) -> list[dict]:
    base=crop
    W,H=base.size
    if W*H>18_000_000:
        s=math.sqrt(18_000_000/(W*H)); base=base.resize((max(1,int(W*s)),max(1,int(H*s))), Image.BICUBIC)
    g=to_gray(base)
    g2=cv2.resize(g,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
    cla=cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8)).apply(g2)
    lm = build_line_mask(g2)
    lmc=build_line_mask(cla)
    out=[{"name":"gray2x","pil":Image.fromarray(g2),"dot":g2,"mask":lm},
         {"name":"clahe2x","pil":Image.fromarray(cla),"dot":cla,"mask":lmc}]
    if do_lineclean:
        clean,_=inpaint(g2, radius=3)
        out.append({"name":"inpaint2x","pil":Image.fromarray(clean),"dot":clean,"mask":lm})
    _,bw=cv2.threshold(cv2.GaussianBlur(cla,(3,3),0),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    out.append({"name":"otsu_clahe2x","pil":Image.fromarray(bw),"dot":None,"mask":lmc})
    return out

def extract_pairs_from_line(line: dict, dot_ref: Optional[np.ndarray], line_mask: Optional[np.ndarray], num_timeout: Optional[int]) -> list[dict]:
    txt=norm(line["text"])
    if not txt or len(txt)<4: return []
    bm=BEARING_RE.search(txt)
    if not bm: return []
    bcanon=parse_bearing(bm.group(0))
    if not bcanon: return []
    theta=bearing_theta(bcanon)
    rec=record_distance(txt)
    ref=find_ref_id(txt)
    curve=extract_curve_fields(txt)

    d_cands=[]
    words=line.get("words") or []
    b_end=-1
    joined=[w["text"] for w in words]
    for i in range(len(joined)):
        for win in (3,4,5,6):
            j=min(len(joined), i+win)
            if BEARING_RE.search(" ".join(joined[i:j])):
                b_end=j; break
        if b_end!=-1: break
    if b_end!=-1:
        for k in range(b_end, min(len(words), b_end+14)):
            t=(words[k]["text"] or "").strip()
            if not t or re.search(r"[A-Za-z]",t) or not re.search(r"\d",t): continue
            d_cands.append((t, words[k]["bbox"]))
    if not d_cands:
        tail=txt[bm.end():]
        m=re.search(r"\d[\d,]*([.·•∙⋅]\d{1,4})?", tail)
        if m: d_cands=[(m.group(0), None)]

    out=[]
    for raw, bb in d_cands[:5]:
        ov=overlap(line_mask, bb) if (line_mask is not None and bb is not None) else 0.0
        if ov>=0.65: continue
        dot_r = dot_digits_right(dot_ref, bb) if (dot_ref is not None and bb is not None) else None
        dv,q = parse_dist(raw, dot_r)
        if (dv is None or q in ("integer","invalid")) and (bb is not None) and (dot_ref is not None):
            x1,y1,x2,y2=bb; pad=max(3,int(0.25*(y2-y1)))
            x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(dot_ref.shape[1],x2+pad); y2=min(dot_ref.shape[0],y2+pad)
            rr=number_reread(Image.fromarray(dot_ref[y1:y2,x1:x2]), timeout=num_timeout).replace("..",".").translate(DIGIT_FIX)
            rr=re.sub(r"[^0-9.]", "", rr)
            if rr:
                dv2,q2 = parse_dist(rr, None)
                if dv2 is not None and (dv is None or (q in ("integer","invalid") and q2=="explicit_decimal")):
                    dv,q,raw = dv2,q2,rr
        if dv is None: continue
        dist_str=f"{dv:.3f}".rstrip("0").rstrip(".") if q=="dot_blob" else f"{dv:.2f}".rstrip("0").rstrip(".")
        out.append({"bearing":bcanon,"theta":theta,"distance":dist_str,"numeric_dist":float(dv),
                    "dist_quality":q,"dist_raw":raw,"record_distance":rec,"ref_id":ref,"curve":curve,
                    "conf":float(line.get("conf",-1.0)),"line_overlap":float(ov),"line_box":line["bbox"]})
        break
    return out

# ----------------- text detection (CV) -----------------

def detect_text_quads(page: Image.Image) -> list[np.ndarray]:
    small, sc = downscale(page, max_pixels=12_000_000)
    g=to_gray(small)
    g=cv2.GaussianBlur(g,(3,3),0)
    ink=adaptive_inv(g)
    lm=build_line_mask(g)
    text=cv2.bitwise_and(ink, cv2.bitwise_not(lm))
    text=cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    text=cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    text=cv2.medianBlur(text,3)
    cnts,_=cv2.findContours(text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads=[]
    Hs,Ws=text.shape[:2]
    for c in cnts:
        area=cv2.contourArea(c)
        if area<120: continue
        rect=cv2.minAreaRect(c)
        (cx,cy),(rw,rh),ang=rect
        if rw<18 or rh<10: continue
        if rw*rh > 0.25*(Ws*Hs): continue
        rect2=((cx,cy),(rw*1.10,rh*1.18),ang)
        box=cv2.boxPoints(rect2).astype(np.float32)
        if sc!=1.0: box/=sc
        quads.append(box)
    def qarea(q):
        xs=q[:,0]; ys=q[:,1]
        return float((xs.max()-xs.min())*(ys.max()-ys.min()))
    quads.sort(key=qarea)
    return quads[:1400]

# ----------------- table detection / parse (best-effort) -----------------

def detect_table_rects(page: Image.Image) -> list[tuple[int,int,int,int]]:
    small, sc = downscale(page, max_pixels=12_000_000)
    g=to_gray(small); g=cv2.GaussianBlur(g,(3,3),0)
    bw=adaptive_inv(g)
    h,w=bw.shape[:2]
    hk=max(35,w//12); vk=max(35,h//12)
    horiz=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(hk,1)), iterations=1)
    vert =cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,vk)), iterations=1)
    grid=cv2.dilate(cv2.bitwise_or(horiz,vert), np.ones((3,3),np.uint8), iterations=2)
    cnts,_=cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for c in cnts:
        x,y,ww,hh=cv2.boundingRect(c); area=ww*hh
        if area<8000: continue
        if ww<0.12*w or hh<0.08*h: continue
        if area>0.55*w*h: continue
        pad_x=int(0.03*w); pad_y=int(0.03*h)
        x1=max(0,x-pad_x); y1=max(0,y-pad_y); x2=min(w,x+ww+pad_x); y2=min(h,y+hh+pad_y)
        if sc!=1.0:
            x1=int(x1/sc); y1=int(y1/sc); x2=int(x2/sc); y2=int(y2/sc)
        rects.append((x1,y1,x2,y2))
    rects.sort(key=lambda r:(r[2]-r[0])*(r[3]-r[1]), reverse=True)
    return rects[:10]

def parse_table_roi(roi: Image.Image, do_lineclean: bool, tess_timeout: Optional[int], num_timeout: Optional[int]) -> tuple[list[dict], list[dict]]:
    out=[]; curvep=[]
    vars=preprocess_variants(roi, do_lineclean=do_lineclean)
    cfg="--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_enable_dict_correction=0 -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_char_whitelist=0123456789NSEWnsewLC.,-°oº()' "
    for v in vars:
        lines=ocr_lines(v["pil"], cfg, timeout=tess_timeout)
        for ln in lines:
            t=norm(ln["text"])
            if not t: continue
            rid=find_ref_id(t)
            if not rid: continue
            if rid.startswith('C'):
                c=extract_curve_fields(t)
                if c:
                    curvep.append({"ref_id":rid,"curve":c,"conf":float(ln.get('conf',-1.0)),"text":t,"line_box":ln.get('bbox'),"table_kind":"curve_table"})
            pairs=extract_pairs_from_line(ln, dot_ref=v["dot"], line_mask=v["mask"], num_timeout=num_timeout)
            for p in pairs:
                p["ref_id"]=rid
                p["table_kind"]="line_table" if rid.startswith("L") else "curve_table"
                out.append(p)
    return out, curvep

# ----------------- clustering / voting -----------------

def similar(a: dict, b: dict, dist_tol: float, bear_tol: float) -> bool:
    if (a.get("ref_id") or None) != (b.get("ref_id") or None): return False
    if (a.get("table_kind") or None) != (b.get("table_kind") or None): return False
    qa=a["bearing"].split()[0]+a["bearing"].split()[-1]
    qb=b["bearing"].split()[0]+b["bearing"].split()[-1]
    if qa!=qb: return False
    if abs(float(a["numeric_dist"])-float(b["numeric_dist"]))>dist_tol: return False
    ta,tb=a.get("theta"),b.get("theta")
    if ta is None or tb is None: return False
    d=abs(float(ta)-float(tb)); d=min(d,360.0-d)
    return d<=bear_tol

def cluster(cands: list[dict], dist_tol: float, bear_tol: float) -> list[list[dict]]:
    cl=[]
    for c in cands:
        placed=False
        for g in cl:
            if similar(c, g[0], dist_tol, bear_tol):
                g.append(c); placed=True; break
        if not placed: cl.append([c])
    return cl

def score(c: dict) -> float:
    conf=float(c.get("conf",-1.0))
    ov=float(c.get("line_overlap",0.0))
    q=c.get("dist_quality","")
    bonus = 2.2 if q=="explicit_decimal" else (1.3 if q=="dot_blob" else 0.0)
    return conf + bonus - 3.5*ov

def winner(group: list[dict]) -> dict:
    cnt=Counter((c["bearing"],c["distance"],c.get("record_distance") or None) for c in group)
    bestk=max(cnt, key=lambda k: cnt[k])
    reps=[c for c in group if (c["bearing"],c["distance"],c.get("record_distance") or None)==bestk]
    reps.sort(key=score, reverse=True)
    out=dict(reps[0]); out["support"]=int(cnt[bestk]); out["vote_score"]=float(score(reps[0]))
    return out

# ----------------- basis of bearing -----------------

@dataclass
class TextLine:
    text: str
    conf: float
    bbox: tuple[int,int,int,int]
    page: int
    dpi: int
    rot: float

def basis_of_bearing(text_lines: list[TextLine], pairs: list[dict]) -> Optional[dict]:
    basis=[ln for ln in text_lines if BASIS_RE.search(ln.text or "")]
    if not basis: return None
    basis.sort(key=lambda ln: ln.conf, reverse=True)
    lab=basis[0]
    bx1,by1,bx2,by2=lab.bbox
    bcx,bcy=(bx1+bx2)/2.0,(by1+by2)/2.0
    best=None; bestd=1e18
    for p in pairs:
        if p.get("page")!=lab.page or p.get("dpi")!=lab.dpi or float(p.get("det_rotation",0.0))!=float(lab.rot):
            continue
        bb=p.get("bbox_page")
        if not bb: continue
        x1,y1,x2,y2=bb; ccx,ccy=(x1+x2)/2.0,(y1+y2)/2.0
        dy=ccy-bcy
        if dy<-120: continue
        d=(ccx-bcx)**2+(ccy-bcy)**2
        if d<bestd: bestd=d; best=p
    out={"basis_text":lab.text,"page":lab.page,"dpi":lab.dpi,"det_rotation":lab.rot,"label_bbox":lab.bbox}
    if best:
        out.update({"bearing":best["bearing"],"distance":best["distance"],"record_distance":best.get("record_distance")})
    return out

# ----------------- main extraction -----------------

def parse_pages(spec: str, max_pages: int) -> list[int]:
    if not spec: return list(range(1,max_pages+1))
    out=set()
    for part in spec.split(","):
        part=part.strip()
        if not part: continue
        if "-" in part:
            a,b=part.split("-",1)
            try:
                a=int(a); b=int(b)
                for p in range(max(1,a), min(max_pages,b)+1): out.add(p)
            except ValueError:
                pass
        else:
            try:
                p=int(part)
                if 1<=p<=max_pages: out.add(p)
            except ValueError:
                pass
    return sorted(out) if out else list(range(1,max_pages+1))

def process_page(page_img: Image.Image, page_num: int, dpi: int, rotations: list[float],
                 do_lineclean: bool, do_tables: bool, threads: int,
                 tess_timeout: Optional[int], num_timeout: Optional[int],
                 max_crops: int) -> tuple[list[dict], list[dict], list[TextLine]]:
    W,H=page_img.size
    all_pairs=[]; curve_params=[]; all_lines=[]

    def process_rotation(rot: float):
        rot_img=rotate_keep(page_img, rot)
        quads=detect_text_quads(rot_img)[:max_crops]
        page_gray=to_gray(rot_img)
        page_lm=build_line_mask(page_gray)

        pairs=[]; curvep=[]; lines=[]
        def crop_worker(quad: np.ndarray):
            crop=warp_quad(rot_img, quad, pad=6)
            vars=preprocess_variants(crop, do_lineclean=do_lineclean)
            cfgs=[
                ("--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_enable_dict_correction=0 -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_char_whitelist=0123456789NSEWnsewLC.,-°oº()' ","psm6"),
                ("--oem 3 --psm 11 -c preserve_interword_spaces=1 -c tessedit_enable_dict_correction=0 -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_char_whitelist=0123456789NSEWnsewLC.,-°oº()' ","psm11"),
            ]
            local_pairs=[]; local_curvep=[]; local_lines=[]
            for v in vars:
                for cfg,_ in cfgs:
                    for ln in ocr_lines(v["pil"], cfg, timeout=tess_timeout):
                        t=norm(ln["text"])
                        if not t: continue
                        if BASIS_RE.search(t):
                            bb_orig=quad_to_bbox_original(quad, W, H, rot)
                            local_lines.append(TextLine(text=t, conf=float(ln["conf"]), bbox=bb_orig, page=page_num, dpi=dpi, rot=rot))
                        cp = curve_params_from_line(ln)
                        if cp:
                            cp["page"]=page_num; cp["dpi"]=dpi; cp["det_rotation"]=rot
                            cp["bbox_page"]=quad_to_bbox_original(quad, W, H, rot)
                            cp["source"]="callout"
                            local_curvep.append(cp)
                        for p in extract_pairs_from_line(ln, dot_ref=v["dot"], line_mask=v["mask"], num_timeout=num_timeout):
                            if ln.get("bbox"):
                                ov=overlap(page_lm, ln["bbox"])
                                p["line_overlap"]=float(min(1.0, 0.7*float(p.get("line_overlap",0.0))+0.3*ov))
                            p["page"]=page_num; p["dpi"]=dpi; p["det_rotation"]=rot
                            p["bbox_page"]=quad_to_bbox_original(quad, W, H, rot)
                            p["source"]="callout"
                            local_pairs.append(p)
            return local_pairs, local_curvep, local_lines

        with ThreadPoolExecutor(max_workers=max(1,threads)) as ex:
            futs=[ex.submit(crop_worker, q) for q in quads]
            for fut in as_completed(futs):
                lp, lcp, ll = fut.result()
                pairs.extend(lp); curvep.extend(lcp); lines.extend(ll)

        if do_tables:
            rects=detect_table_rects(rot_img)
            for r in rects:
                x1,y1,x2,y2=r
                roi=rot_img.crop((x1,y1,x2,y2))
                tbl, tcurvep=parse_table_roi(roi, do_lineclean=do_lineclean, tess_timeout=tess_timeout, num_timeout=num_timeout)
                for cp in tcurvep:
                    cp["page"]=page_num; cp["dpi"]=dpi; cp["det_rotation"]=rot
                    q=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                    cp["bbox_page"]=quad_to_bbox_original(q, W, H, rot)
                    cp["source"]="table"
                    curvep.append(cp)
                for p in tbl:
                    p["page"]=page_num; p["dpi"]=dpi; p["det_rotation"]=rot
                    q=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                    p["bbox_page"]=quad_to_bbox_original(q, W, H, rot)
                    p["source"]="table"
                    pairs.append(p)

        return pairs, curvep, lines

    # rotations sequential to avoid oversubscription; per-rotation crops are parallel
    for r in rotations:
        pairs, cp, lines = process_rotation(float(r))
        all_pairs.extend(pairs)
        curve_params.extend(cp)
        all_lines.extend(lines)

    return all_pairs, curve_params, all_lines

def extract(pdf: Path, dpis: list[int], pages: list[int], rotations: list[float], threads: int,
            do_lineclean: bool, do_tables: bool,
            tess_timeout: Optional[int], num_timeout: Optional[int],
            dist_tol: float, bear_tol: float, min_support: int, max_crops: int) -> dict:
    out_pairs=[]; out_curve_params=[]; text_lines=[]

    for dpi in dpis:
        for p in pages:
            imgs=convert_from_path(str(pdf), dpi=int(dpi), first_page=p, last_page=p)
            if not imgs:
                continue
            img=imgs[0]
            pairs, curvep, lines = process_page(img, p, dpi, rotations, do_lineclean, do_tables, threads, tess_timeout, num_timeout, max_crops)
            out_pairs.extend(pairs); out_curve_params.extend(curvep); text_lines.extend(lines)

    clusters=cluster(out_pairs, dist_tol=dist_tol, bear_tol=bear_tol)
    winners=[winner(g) for g in clusters if len(g)>0]
    winners=[w for w in winners if int(w.get("support",1))>=min_support]
    winners.sort(key=lambda x:(x.get("ref_id") is not None, x.get("support",1), x.get("vote_score",-1e9)), reverse=True)

    counts=defaultdict(int)
    for w in winners:
        if w.get("ref_id"): counts[w["ref_id"]]+=1
    for w in winners:
        w["callout_count"]=int(counts.get(w.get("ref_id"),0))

    basis=basis_of_bearing(text_lines, winners)

    return {
        "file": str(pdf),
        "dpis": dpis,
        "pages": pages,
        "rotations": rotations,
        "threads": threads,
        "lineclean": bool(do_lineclean),
        "tables": bool(do_tables),
        "dist_tol_ft": dist_tol,
        "bear_tol_deg": bear_tol,
        "min_support": min_support,
        "basis_of_bearing": basis,
        "curve_params": [{
            "source": c.get("source"),
            "table_kind": c.get("table_kind"),
            "ref_id": c.get("ref_id"),
            "curve": c.get("curve"),
            "conf": float(c.get("conf",-1.0)),
            "page": int(c.get("page",0)),
            "det_rotation": float(c.get("det_rotation",0.0)),
            "bbox_page": c.get("bbox_page"),
            "text": c.get("text"),
        } for c in out_curve_params],
        "pairs": [{
            "source": w.get("source"),
            "table_kind": w.get("table_kind"),
            "ref_id": w.get("ref_id"),
            "bearing": w["bearing"],
            "distance": w["distance"],
            "record_distance": w.get("record_distance"),
            "curve": w.get("curve"),
            "callout_count": int(w.get("callout_count",0)),
            "support": int(w.get("support",1)),
            "vote_score": float(w.get("vote_score",0.0)),
            "conf": float(w.get("conf",-1.0)),
            "line_overlap": float(w.get("line_overlap",0.0)),
            "dist_quality": w.get("dist_quality"),
            "bbox_page": w.get("bbox_page"),
            "page": int(w.get("page",0)),
            "det_rotation": float(w.get("det_rotation",0.0)),
        } for w in winners],
        "count": len(winners),
    }

def draw_debug(pdf: Path, dpi: int, pages: list[int], result: dict, out_png: Path):
    if not pages: return
    p=pages[0]
    imgs=convert_from_path(str(pdf), dpi=int(dpi), first_page=p, last_page=p)
    if not imgs: return
    img=imgs[0].convert("RGB")
    arr=np.array(img)
    for w in result.get("pairs",[]):
        if int(w.get("page",0))!=p: continue
        bb=w.get("bbox_page")
        if not bb: continue
        x1,y1,x2,y2 = map(int, bb)
        cv2.rectangle(arr, (x1,y1), (x2,y2), (0,255,0), 2)
        lab = (w.get("ref_id") or "") + " " + (w.get("distance") or "")
        cv2.putText(arr, lab[:22], (x1, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
    Image.fromarray(arr).save(out_png)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=int, default=650)
    ap.add_argument("--dpis", default="", help="comma list, overrides --dpi")
    ap.add_argument("--pages", default="", help="e.g. 1-2,4")
    ap.add_argument("--rotations", default="0,90,180,270")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--omp-thread-limit", type=int, default=None)
    ap.add_argument("--no-lineclean", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--tess-timeout", type=int, default=0)
    ap.add_argument("--number-timeout", type=int, default=0)
    ap.add_argument("--dist-tol", type=float, default=0.20)
    ap.add_argument("--bear-tol", type=float, default=0.35)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--max-crops", type=int, default=1100)
    ap.add_argument("--out", default="out.json")
    ap.add_argument("--annotate", default="")
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args()

    if args.omp_thread_limit is not None:
        os.environ["OMP_THREAD_LIMIT"]=str(int(args.omp_thread_limit))
    else:
        if args.threads>1 and "OMP_THREAD_LIMIT" not in os.environ:
            os.environ["OMP_THREAD_LIMIT"]="1"

    pdf=Path(args.pdf)
    if not pdf.is_file():
        print("File not found"); sys.exit(1)

    info=pdfinfo_from_path(str(pdf))
    max_pages=int(info.get("Pages",1))
    pages=parse_pages(args.pages, max_pages=max_pages)
    rotations=[float(x.strip()) for x in args.rotations.split(",") if x.strip()]

    tess_timeout = int(args.tess_timeout) if args.tess_timeout>0 else None
    num_timeout  = int(args.number_timeout) if args.number_timeout>0 else None

    dpis = [int(x) for x in args.dpis.split(',') if x.strip().isdigit()] if args.dpis else [int(args.dpi)]

    res=extract(
        pdf=pdf, dpis=dpis, pages=pages, rotations=rotations,
        threads=max(1,int(args.threads)),
        do_lineclean=not args.no_lineclean,
        do_tables=not args.no_tables,
        tess_timeout=tess_timeout, num_timeout=num_timeout,
        dist_tol=float(args.dist_tol), bear_tol=float(args.bear_tol),
        min_support=int(args.min_support), max_crops=int(args.max_crops)
    )

    with open(args.out,"w") as f:
        json.dump(res,f,indent=2)
    print(f"Saved {args.out}  pairs={res.get('count')}")

    if args.verbose:
        for i,p in enumerate(res.get("pairs",[]),1):
            rid=p.get("ref_id") or ""
            tk=f"[{p.get('table_kind')}]" if p.get("table_kind") else ""
            rec=f" ({p.get('record_distance')})" if p.get("record_distance") else ""
            print(f"{i:3d}. {tk} {rid:5s} {p['bearing']:22s} {p['distance']:>10s}{rec} ft  support={p['support']} score={p['vote_score']:.2f} overlap={p['line_overlap']:.2f}")

        if res.get("basis_of_bearing"):
            b=res["basis_of_bearing"]
            print("\nBasis of Bearing:", b.get("basis_text"))
            if b.get("bearing"): print("  ->", b.get("bearing"), b.get("distance"), "ft")

        if res.get("curve_params"):
            print(f"\nCurve-only params captured: {len(res.get('curve_params',[]))}")

    if args.annotate:
        draw_debug(pdf, int(dpis[0]), pages, res, Path(args.annotate))
        print(f"Saved debug overlay {args.annotate}")

if __name__=="__main__":
    main()
