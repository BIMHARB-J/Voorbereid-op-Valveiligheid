"""
IFC 2D Vloeraanzicht per verdieping
====================================
Aanpak zoals IFCOpenShell / IFC-viewers: elke vloer (IFCSLAB + IFCROOF)
wordt omgezet naar driehoekjes (tessellation) in wereldcoördinaten.
Voor een 2D-bovenaanzicht projecteren we de driehoekjes op het XY-vlak en
tekenen ze gevuld — overlappende driehoekjes vormen samen het vloervlak.
Geen omtrek-reconstructie, geen hull-trucs.

Transformatie volgt de IFCLOCALPLACEMENT-keten als 4×4 matrices,
exact zoals een IFC-viewer dat doet.

Geometrietypes (allemaal → driehoekjes):
  - IFCEXTRUDEDAREASOLID   (profiel → polygoon → fan-triangulatie)
  - IFCPOLYGONALFACESET    (faces, incl. WITHVOIDS)
  - IFCFACETEDBREP         (IFCCLOSEDSHELL → faces)
  - IFCBOOLEANRESULT       (neemt eerste operand)

Profieltypes:
  - IFCRECTANGLEPROFILEDEF
  - IFCARBITRARYCLOSEDPROFILEDEF
  - IFCARBITRARYPROFILEDEFWITHVOIDS

Gebruik:
    python ifc_2d_aanzicht.py bestand.ifc

Vereisten: numpy
"""

import re
import sys
import os
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Vloervelden die binnen deze afstand (mm) van elkaar liggen, worden
# samengevoegd tot één vloerveld.
MERGE_GAP_MM = 350.0

# Depth/contrast map instellingen
DEPTH_CUE_MM = 2500.0   # max hoogteverschil dat als grijswaarde wordt getoond
OFFSET_MM    = 6000.0   # marge rond de vloeren in de scope-box (mm)
RENDER_SCALE = 0.05     # mm -> pixel (1m = 50px)

# Naden tussen aangrenzende slabs (op gelijke hoogte) die smaller zijn dan
# dit, worden dichtgesmolten (geen valgevaar). Echte sparingen uit de IFC
# (void-ringen) blijven altijd behouden, ook al zijn ze smaller dan dit.
# 150mm dekt de gemeten naden tussen kanaalplaten (tot ~126mm) ruim af,
# terwijl echte sparingen ná de sluiting weer worden afgetrokken.
SEAM_CLOSE_MM = 150.0

# Hoogteverschil (mm) waarbinnen twee slabs als 'gelijk niveau' tellen en
# dus samengevoegd mogen worden tot één doorlopend vloervlak.
SAME_LEVEL_MM = 50.0


# ═══════════════════════════════════════════════════════════════════════════════
#  IFC PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class IFC:
    def __init__(self, fp):
        self.ent = {}
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.match(r'#(\d+)=(.*?);?\s*$', line.strip())
                if m:
                    self.ent[m.group(1)] = m.group(2).rstrip(';')

    def get(self, idx):
        return self.ent.get(str(idx).lstrip('#'), '')

    def of_type(self, prefix):
        return [k for k, v in self.ent.items() if v.startswith(prefix)]

    def inner(self, s):
        a, b = s.find('('), s.rfind(')')
        return s[a+1:b] if a != -1 and b != -1 else s

    def params(self, s):
        out, cur, d, q = [], [], 0, False
        for ch in s:
            if ch == "'":
                q = not q; cur.append(ch)
            elif not q:
                if ch in '([':
                    d += 1; cur.append(ch)
                elif ch in ')]':
                    d -= 1; cur.append(ch)
                elif ch == ',' and d == 0:
                    out.append(''.join(cur).strip()); cur = []
                else:
                    cur.append(ch)
            else:
                cur.append(ch)
        if cur:
            out.append(''.join(cur).strip())
        return out

    def ref(self, p):
        m = re.search(r'#(\d+)', p)
        return m.group(1) if m else None

    def refs(self, s):
        return re.findall(r'#(\d+)', s)

    def num(self, s):
        try: return float(s)
        except: return None

    def pt(self, idx):
        return [float(n) for n in re.findall(r'[-\d.Ee+]+', self.inner(self.get(idx)))]

    def name_of(self, idx):
        p = self.params(self.inner(self.get(idx)))
        return p[2].strip("'") if len(p) > 2 and p[2] != '$' else ''


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMATIE (4×4 matrices, zoals IFC-viewers)
# ═══════════════════════════════════════════════════════════════════════════════

def axis2placement3d(ifc, idx):
    """IFCAXIS2PLACEMENT3D → 4×4 matrix."""
    if not idx:
        return np.eye(4)
    p = ifc.params(ifc.inner(ifc.get(idx)))
    loc = ifc.pt(ifc.ref(p[0])) if p else [0, 0, 0]
    z = ifc.pt(ifc.ref(p[1])) if len(p) > 1 and p[1].strip() != '$' else [0, 0, 1]
    x = ifc.pt(ifc.ref(p[2])) if len(p) > 2 and p[2].strip() != '$' else [1, 0, 0]
    z = np.array(z[:3], float)
    nz = np.linalg.norm(z)
    z = z/nz if nz > 1e-9 else np.array([0., 0., 1.])
    x = np.array(x[:3], float)
    x = x - np.dot(x, z) * z
    nx = np.linalg.norm(x)
    x = x/nx if nx > 1e-9 else np.array([1., 0., 0.])
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z
    M[:3, 3] = (loc + [0, 0, 0])[:3]
    return M


def axis2placement2d(ifc, idx):
    """IFCAXIS2PLACEMENT2D → 4×4 matrix (in XY-vlak)."""
    if not idx:
        return np.eye(4)
    p = ifc.params(ifc.inner(ifc.get(idx)))
    loc = ifc.pt(ifc.ref(p[0])) if p else [0, 0]
    refd = ifc.pt(ifc.ref(p[1])) if len(p) > 1 and p[1].strip() != '$' else [1, 0]
    x = np.array([refd[0], refd[1], 0.], float)
    nx = np.linalg.norm(x)
    x = x/nx if nx > 1e-9 else np.array([1., 0., 0.])
    y = np.array([-x[1], x[0], 0.])
    M = np.eye(4)
    M[:3, 0] = x; M[:3, 1] = y
    M[0, 3] = loc[0]; M[1, 3] = loc[1] if len(loc) > 1 else 0.
    return M


def placement_matrix(ifc, idx, depth=0):
    """IFCLOCALPLACEMENT → cumulatieve 4×4 matrix (parent × lokaal)."""
    if not idx or depth > 30:
        return np.eye(4)
    v = ifc.get(idx)
    if not v.startswith('IFCLOCALPLACEMENT'):
        return np.eye(4)
    p = ifc.params(ifc.inner(v))
    rel = ifc.ref(p[0]) if p and p[0].strip() != '$' else None
    axis = ifc.ref(p[1]) if len(p) > 1 else None
    M = axis2placement3d(ifc, axis) if axis else np.eye(4)
    return placement_matrix(ifc, rel, depth+1) @ M


# ═══════════════════════════════════════════════════════════════════════════════
#  TESSELLATIE — elke geometrie → lijst driehoekjes (in lokale coords)
# ═══════════════════════════════════════════════════════════════════════════════

def _fan(poly_idx):
    """Triangle-fan indices voor een polygoon met n hoekpunten."""
    return [(0, i, i+1) for i in range(1, len(poly_idx)-1)]


def _earclip(pts):
    """
    Ear-clipping triangulatie van een (mogelijk niet-convexe) 2D-polygoon.
    pts: lijst van (x,y). Retourneert lijst van index-tripels in pts.
    Dit is hoe IFC-viewers een profiel correct vullen, ook bij L/U-vormen.
    """
    n = len(pts)
    if n < 3:
        return []
    # Zorg voor CCW-oriëntatie
    def area2(poly):
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i+1) % len(poly)]
            s += x1*y2 - x2*y1
        return s
    idx = list(range(n))
    if area2(pts) < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def in_tri(p, a, b, c):
        d1 = cross(p, a, b); d2 = cross(p, b, c); d3 = cross(p, c, a)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        ear_found = False
        m = len(idx)
        for i in range(m):
            ia, ib, ic = idx[(i-1) % m], idx[i], idx[(i+1) % m]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if cross(a, b, c) <= 0:     # reflex of degenerate → geen ear
                continue
            # Geen ander punt binnen deze driehoek?
            ok = True
            for j in idx:
                if j in (ia, ib, ic):
                    continue
                if in_tri(pts[j], a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((ia, ib, ic))
                idx.pop(i)
                ear_found = True
                break
        if not ear_found:
            break   # niet-simpele polygoon; stop met wat we hebben
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _arc_points(ifc, circle_idx, t1_deg, t2_deg, sense_agreement, n_seg=24):
    """
    Benader een cirkelboog (IFCCIRCLE) tussen twee parameters met punten.
    De parameters zijn hoeken in graden t.o.v. de x-as van de placement.
    Dit is de exacte wiskundige definitie van de boog uit de IFC, alleen
    gediscretiseerd — geen aanname over de vorm.
    """
    cv = ifc.get(circle_idx)
    if not cv.startswith('IFCCIRCLE'):
        return []
    cp = ifc.params(ifc.inner(cv))
    pos = ifc.ref(cp[0]) if cp else None
    radius = ifc.num(cp[1]) if len(cp) > 1 else 0.0
    if not pos or radius <= 0:
        return []
    # Placement: centrum + x-as (refdirection)
    pv = ifc.params(ifc.inner(ifc.get(pos)))
    centre = ifc.pt(ifc.ref(pv[0])) if pv else [0., 0.]
    refd = ifc.pt(ifc.ref(pv[1])) if len(pv) > 1 and pv[1].strip() != '$' else [1., 0.]
    base_ang = math.atan2(refd[1], refd[0])

    a1 = math.radians(t1_deg)
    a2 = math.radians(t2_deg)
    # .T. = tegen de klok in (toenemende parameter); anders met de klok mee
    if sense_agreement:
        if a2 <= a1:
            a2 += 2 * math.pi
    else:
        if a2 >= a1:
            a2 -= 2 * math.pi

    pts = []
    for i in range(n_seg + 1):
        a = a1 + (a2 - a1) * i / n_seg
        ang = base_ang + a
        pts.append((centre[0] + radius * math.cos(ang),
                    centre[1] + radius * math.sin(ang)))
    return pts


def _trim_value(trim_param):
    """Haal de numerieke parameter (graden) uit een trim, bv IFCPARAMETERVALUE(321.6)."""
    m = re.search(r'IFCPARAMETERVALUE\(([-\d.Ee+]+)\)', trim_param)
    if m:
        return float(m.group(1))
    return None


def _segment_points(ifc, idx):
    """
    Eén curve-segment → [(x,y),...]. Ondersteunt IFCPOLYLINE, IFCLINE,
    IFCTRIMMEDCURVE (boog op een cirkel). Geeft de punten in voorwaartse
    richting; de aanroeper plakt segmenten aan elkaar.
    """
    v = ifc.get(idx)
    head = v.split('(')[0]

    if head == 'IFCPOLYLINE':
        pts = []
        for r in ifc.refs(ifc.inner(v)):
            pv = ifc.get(r)
            if pv.startswith('IFCCARTESIANPOINT'):
                n = re.findall(r'[-\d.Ee+]+', ifc.inner(pv))
                if len(n) >= 2:
                    pts.append((float(n[0]), float(n[1])))
        return pts

    if head == 'IFCTRIMMEDCURVE':
        p = ifc.params(ifc.inner(v))
        base = ifc.ref(p[0]) if p else None
        t1 = _trim_value(p[1]) if len(p) > 1 else None
        t2 = _trim_value(p[2]) if len(p) > 2 else None
        sense = (p[3].strip().upper() == '.T.') if len(p) > 3 else True
        if base and ifc.get(base).startswith('IFCCIRCLE') and t1 is not None and t2 is not None:
            return _arc_points(ifc, base, t1, t2, sense)
        # Andere basiscurves: val terug op de trim-cartesianpoints indien aanwezig
        cpts = []
        for grp in (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else ''):
            for r in re.findall(r'#(\d+)', grp):
                cv = ifc.get(r)
                if cv.startswith('IFCCARTESIANPOINT'):
                    n = re.findall(r'[-\d.Ee+]+', ifc.inner(cv))
                    if len(n) >= 2:
                        cpts.append((float(n[0]), float(n[1])))
        return cpts

    if head == 'IFCLINE':
        # IFCLINE(point, vector) — gebruik alleen het beginpunt; lengte zit in vector
        p = ifc.params(ifc.inner(v))
        pid = ifc.ref(p[0]) if p else None
        if pid:
            n = re.findall(r'[-\d.Ee+]+', ifc.inner(ifc.get(pid)))
            if len(n) >= 2:
                return [(float(n[0]), float(n[1]))]
    return []


def _curve_points(ifc, idx):
    """
    Curve → [(x,y),...]. Ondersteunt IFCPOLYLINE, IFCINDEXEDPOLYCURVE en
    IFCCOMPOSITECURVE (samengesteld uit lijn- en boogsegmenten). Alle
    geometrie komt rechtstreeks uit de IFC; bogen worden exact berekend.
    """
    v = ifc.get(idx)
    head = v.split('(')[0]

    if head == 'IFCPOLYLINE':
        pts = _segment_points(ifc, idx)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return pts

    if head == 'IFCINDEXEDPOLYCURVE':
        p = ifc.params(ifc.inner(v))
        ci = ifc.ref(p[0]) if p else None
        if ci:
            prs = re.findall(r'\(\s*([-\d.Ee+]+)\s*,\s*([-\d.Ee+]+)', ifc.inner(ifc.get(ci)))
            return [(float(x), float(y)) for x, y in prs]
        return []

    if head == 'IFCCOMPOSITECURVE':
        p = ifc.params(ifc.inner(v))
        pts = []
        seg_refs = ifc.refs(p[0]) if p else []
        for seg in seg_refs:
            sv = ifc.get(seg)
            if not sv.startswith('IFCCOMPOSITECURVESEGMENT'):
                continue
            segp = ifc.params(ifc.inner(sv))
            same_sense = (segp[1].strip().upper() == '.T.') if len(segp) > 1 else True
            parent = ifc.ref(segp[-1]) if segp else None
            if not parent:
                continue
            spts = _segment_points(ifc, parent)
            if not same_sense:
                spts = spts[::-1]
            # Aaneenrijgen zonder dubbele knooppunten
            if pts and spts and abs(pts[-1][0]-spts[0][0]) < 1e-6 and abs(pts[-1][1]-spts[0][1]) < 1e-6:
                pts.extend(spts[1:])
            else:
                pts.extend(spts)
        if len(pts) > 1 and abs(pts[0][0]-pts[-1][0]) < 1e-6 and abs(pts[0][1]-pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        return pts

    return []


def _profile_polys(ifc, prof_idx):
    """Profiel → (outer [(x,y)], voids [[(x,y)]]) in profielvlak."""
    v = ifc.get(prof_idx)
    head = v.split('(')[0]

    if head == 'IFCRECTANGLEPROFILEDEF':
        p = ifc.params(ifc.inner(v))
        pos = ifc.ref(p[2]) if len(p) > 2 and p[2].strip() != '$' else None
        xd = ifc.num(p[3]) if len(p) > 3 else 0.
        yd = ifc.num(p[4]) if len(p) > 4 else 0.
        hx, hy = xd/2, yd/2
        local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        M = axis2placement2d(ifc, pos) if pos else np.eye(4)
        outer = [(M[0, 0]*x + M[0, 1]*y + M[0, 3],
                  M[1, 0]*x + M[1, 1]*y + M[1, 3]) for x, y in local]
        return outer, []

    if head == 'IFCARBITRARYCLOSEDPROFILEDEF':
        p = ifc.params(ifc.inner(v))
        oi = ifc.ref(p[2]) if len(p) > 2 else None
        return (_curve_points(ifc, oi) if oi else []), []

    if head == 'IFCARBITRARYPROFILEDEFWITHVOIDS':
        p = ifc.params(ifc.inner(v))
        oi = ifc.ref(p[2]) if len(p) > 2 else None
        outer = _curve_points(ifc, oi) if oi else []
        voids = []
        if len(p) > 3:
            for vi in ifc.refs(p[3]):
                vp = _curve_points(ifc, vi)
                if vp:
                    voids.append(vp)
        return outer, voids

    return [], []


def _parse_face(ifc, fv):
    """IFCINDEXEDPOLYGONALFACE(WITHVOIDS) → (outer 0-based, [voids 0-based])."""
    head = fv.split('(')[0]
    body = ifc.inner(fv)
    if head == 'IFCINDEXEDPOLYGONALFACE':
        return [int(n)-1 for n in re.findall(r'\d+', body)], []
    if head == 'IFCINDEXEDPOLYGONALFACEWITHVOIDS':
        pp = ifc.params(body)
        outer = [int(n)-1 for n in re.findall(r'\d+', pp[0])] if pp else []
        voids = []
        if len(pp) > 1:
            for vg in ifc.params(ifc.inner(pp[1])):
                vi = [int(n)-1 for n in re.findall(r'\d+', vg)]
                if vi:
                    voids.append(vi)
        return outer, voids
    return [], []


def extract_rings(ifc, item_idx, depth=0):
    """
    Geef een lijst van gesloten ringen [(x,y,z),...] terug voor een
    representatie-item, in LOKALE coordinaten. SVG vult de ringen zelf
    (fill-rule), dus geen triangulatie nodig — dit is robuust voor elke
    niet-convexe vorm, precies wat een 2D-aanzicht nodig heeft.

    Elke ring is een tuple (points_3d, is_void).
    """
    # Ruime limiet: boolean-ketens (vloer met veel sparingen) kunnen tientallen
    # niveaus diep zijn; elke sparing voegt een laag toe. Te laag zou de
    # uiteindelijke vloer-outer afkappen en het hele veld laten verdwijnen.
    if depth > 200:
        return []
    v = ifc.get(item_idx)
    head = v.split('(')[0]

    # --- Boolean: operand1 is de vorm; bij .DIFFERENCE. is operand2 een
    #     uitsparing (sparing) die we als void-ring meenemen. Zo gaan
    #     sparingen die via een boolean-aftrek zijn gemaakt niet verloren. ---
    if head in ('IFCBOOLEANRESULT', 'IFCBOOLEANCLIPPINGRESULT'):
        p = ifc.params(ifc.inner(v))
        operator = p[0].strip().upper() if p else ''
        op1 = ifc.ref(p[1]) if len(p) > 1 else None
        op2 = ifc.ref(p[2]) if len(p) > 2 else None
        rings = extract_rings(ifc, op1, depth+1) if op1 else []
        if op2 and '.DIFFERENCE.' in operator:
            # De afgetrokken vorm levert de sparing. We nemen de outer-ringen
            # ervan als void. (extract_rings geeft de XY-omtrek van operand2.)
            for (ring3d, is_void) in extract_rings(ifc, op2, depth+1):
                if not is_void:
                    rings.append((ring3d, True))
        return rings

    # --- Extruded solid: profiel-ring (outer + voids) ---
    if head == 'IFCEXTRUDEDAREASOLID':
        p = ifc.params(ifc.inner(v))
        prof = ifc.ref(p[0])
        pos = ifc.ref(p[1]) if len(p) > 1 else None
        outer, voids = _profile_polys(ifc, prof)
        if not outer:
            return []
        Mp = axis2placement3d(ifc, pos) if pos else np.eye(4)
        rings = []
        ow = [tuple((Mp @ np.array([x, y, 0., 1.]))[:3]) for (x, y) in outer]
        rings.append((ow, False))
        for vd in voids:
            vw = [tuple((Mp @ np.array([x, y, 0., 1.]))[:3]) for (x, y) in vd]
            rings.append((vw, True))
        return rings

    # --- Polygonal faceset: horizontale faces als ringen ---
    if head == 'IFCPOLYGONALFACESET':
        p = ifc.params(ifc.inner(v))
        ci = ifc.ref(p[0])
        trip = re.findall(r'\(([-\d.Ee+]+),([-\d.Ee+]+),([-\d.Ee+]+)\)',
                          ifc.inner(ifc.get(ci)))
        verts = [(float(x), float(y), float(z)) for x, y, z in trip]
        rings = []
        if len(p) > 2:
            for fr in ifc.refs(p[2]):
                outer, voids = _parse_face(ifc, ifc.get(fr))
                if len(outer) < 3:
                    continue
                if not _is_horizontal([verts[i] for i in outer]):
                    continue
                rings.append(([verts[i] for i in outer], False))
                for vd in voids:
                    if len(vd) >= 3:
                        rings.append(([verts[i] for i in vd], True))
        return rings

    # --- Faceted BRep: horizontale faces als ringen ---
    if head == 'IFCFACETEDBREP':
        p = ifc.params(ifc.inner(v))
        shell = ifc.ref(p[0]) if p else None
        rings = []
        if shell:
            for fr in ifc.refs(ifc.get(shell)):
                fv = ifc.get(fr)
                if not fv.startswith('IFCFACE'):
                    continue
                for bound in ifc.refs(fv):
                    bv = ifc.get(bound)
                    if 'BOUND' not in bv:
                        continue
                    is_outer = bv.startswith('IFCFACEOUTERBOUND')
                    loop_pts = []
                    for lr in ifc.refs(bv):
                        lv = ifc.get(lr)
                        if not lv.startswith('IFCPOLYLOOP'):
                            continue
                        for cp in ifc.refs(lv):
                            c = ifc.pt(cp)
                            if len(c) >= 3:
                                loop_pts.append((c[0], c[1], c[2]))
                    if len(loop_pts) >= 3 and _is_horizontal(loop_pts):
                        rings.append((loop_pts, not is_outer))
        return rings

    return []


def _is_horizontal(pts3d, thresh=0.5):
    """True als de face (vrijwel) horizontaal ligt — normaal wijst omhoog/omlaag."""
    if len(pts3d) < 3:
        return False
    a = np.array(pts3d[0]); b = np.array(pts3d[1]); c = np.array(pts3d[2])
    nrm = np.cross(b - a, c - a)
    nl = np.linalg.norm(nrm)
    if nl < 1e-9:
        return False
    return abs(nrm[2] / nl) >= thresh


# ═══════════════════════════════════════════════════════════════════════════════
#  VLOEREN INLEZEN
# ═══════════════════════════════════════════════════════════════════════════════

class Floor:
    def __init__(self, idx, name, kind, placement, rep):
        self.idx = idx; self.name = name; self.kind = kind
        self.placement = placement; self.rep = rep
        self.storey = None
        self.rings = []        # lijst van (ring_xy [(x,y),...], is_void)
        self.zmin = self.zmax = 0.0


def read_storeys(ifc):
    st = {}
    for idx in ifc.of_type('IFCBUILDINGSTOREY'):
        p = ifc.params(ifc.inner(ifc.get(idx)))
        name = p[2].strip("'") if len(p) > 2 and p[2] != '$' else f'Storey {idx}'
        elev = ifc.num(p[-1]) or 0.0
        pl = ifc.ref(p[5]) if len(p) > 5 else None
        st[idx] = {'idx': idx, 'name': name, 'elev': elev, 'placement': pl}
    return st


def read_floors(ifc):
    floors = []
    for prefix, kind in [('IFCSLAB(', 'slab'), ('IFCROOF(', 'roof')]:
        for idx in ifc.of_type(prefix):
            p = ifc.params(ifc.inner(ifc.get(idx)))
            if len(p) < 7:
                continue
            floors.append(Floor(
                idx=idx, name=ifc.name_of(idx), kind=kind,
                placement=ifc.ref(p[5]), rep=ifc.ref(p[6]),
            ))
    return floors


def link_to_storeys(ifc, floors, storeys):
    pl2st = {s['placement']: s for s in storeys.values() if s['placement']}
    for fl in floors:
        if not fl.placement:
            continue
        m = re.search(r'IFCLOCALPLACEMENT\(#(\d+),', ifc.get(fl.placement))
        if m and m.group(1) in pl2st:
            fl.storey = pl2st[m.group(1)]


def build_floor_rings(ifc, fl):
    """Haal alle ringen op en transformeer naar wereld-XY via placement-matrix."""
    rep_v = ifc.get(fl.rep)
    if not rep_v.startswith('IFCPRODUCTDEFINITIONSHAPE'):
        return
    M = placement_matrix(ifc, fl.placement)

    all_rings = []
    zs = []
    for sr in ifc.refs(ifc.inner(rep_v)):
        sr_v = ifc.get(sr)
        if not sr_v.startswith('IFCSHAPEREPRESENTATION'):
            continue
        for item in ifc.refs(ifc.inner(sr_v)):
            iv = ifc.get(item)
            if iv.startswith('IFCGEOMETRIC') or 'CONTEXT' in iv:
                continue
            for (ring3d, is_void) in extract_rings(ifc, item):
                world_xy = []
                for (x, y, z) in ring3d:
                    q = M @ np.array([x, y, z, 1.0])
                    world_xy.append((q[0], q[1]))
                    zs.append(q[2])
                if len(world_xy) >= 3:
                    all_rings.append((world_xy, is_void))
    # Voids filteren: een void telt alleen als echte sparing als hij binnen
    # de buitenomtrek van de vloer valt. Een .DIFFERENCE.-operand die de rand
    # overlapt of erbuiten ligt, is een rand-afsnijding die de omtrek al vormt
    # (geen extra valrand). Dit onderscheid komt objectief uit de geometrie,
    # niet uit een aanname.
    outers = []
    for (ring, is_void) in all_rings:
        if not is_void and len(ring) >= 3:
            try:
                po = Polygon(ring)
                if not po.is_valid:
                    po = po.buffer(0)
                if po.area > 1:
                    outers.append(po)
            except Exception:
                pass

    filtered = []
    for (ring, is_void) in all_rings:
        if not is_void:
            filtered.append((ring, is_void))
            continue
        if len(ring) < 3:
            continue
        try:
            pv = Polygon(ring)
            if not pv.is_valid:
                pv = pv.buffer(0)
        except Exception:
            continue
        if pv.is_empty or pv.area <= 1:
            continue
        # Behoud de void alleen als hij (vrijwel) volledig binnen een outer ligt
        inside = False
        for po in outers:
            if not po.is_valid:
                continue
            inter = pv.intersection(po).area
            if inter >= 0.98 * pv.area:   # void zit binnen de vloer = echte sparing
                inside = True
                break
        if inside:
            filtered.append((ring, is_void))

    fl.rings = filtered
    if zs:
        fl.zmin, fl.zmax = min(zs), max(zs)


# ═══════════════════════════════════════════════════════════════════════════════
#  2D AANZICHT SVG PER VERDIEPING
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = ['#2176AE', '#1A7A3E', '#AE6B21', '#AE2176', '#7621AE', '#3A8C82']


def tris_bbox(floors):
    xs, ys = [], []
    for fl in floors:
        for (ring, is_void) in fl.rings:
            for (x, y) in ring:
                xs.append(x); ys.append(y)
    if not xs:
        return 0, 0, 1, 1
    return min(xs), min(ys), max(xs), max(ys)


def merge_floor_fields(floors, gap_mm=MERGE_GAP_MM):
    """
    Voeg alle vloer-outlines van een verdieping samen tot één of meer
    aaneengesloten vloervelden, en geef per veld alleen de buitenomtrek terug.

    Stappen (zoals een GIS-dissolve):
      1. Alle outer-ringen -> Shapely-polygonen
      2. unary_union -> losse velden samengevoegd waar ze elkaar raken
      3. buffer(+gap/2) -> buffer(-gap/2): velden binnen gap_mm smelten samen,
         daarna terug naar oorspronkelijke maat (morfologische sluiting)
      4. Per resulterend veld: alleen de exterior (buitenomtrek)

    Retourneert lijst van omtrekken: [[(x,y),...], ...]
    """
    polys = []
    for fl in floors:
        outers = [r for r, v in fl.rings if not v]
        for ring in outers:
            if len(ring) < 3:
                continue
            try:
                p = Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and p.area > 1000:
                    polys.append(p)
            except Exception:
                pass

    if not polys:
        return []

    merged = unary_union(polys)
    half = gap_mm / 2.0
    closed = merged.buffer(half, join_style=2).buffer(-half, join_style=2)

    geoms = closed.geoms if closed.geom_type == 'MultiPolygon' else [closed]
    outlines = []
    for g in geoms:
        if g.is_empty or g.area < 1000:
            continue
        outlines.append([(x, y) for x, y in g.exterior.coords])
    return outlines


# ═══════════════════════════════════════════════════════════════════════════════
#  ISOMETRIE — vloervelden correct boven elkaar (werkelijke BOK per verdieping)
# ═══════════════════════════════════════════════════════════════════════════════

_ISO_AZ = math.radians(225)
_ISO_EL = math.radians(30)
_CA, _SA = math.cos(_ISO_AZ), math.sin(_ISO_AZ)
_SE, _CE = math.sin(_ISO_EL), math.cos(_ISO_EL)


def _iso_xy(wx, wy, wz):
    """Projecteer wereld (x,y,z mm) naar 2D isometrisch scherm."""
    rx =  wx * _CA + wy * _SA
    ry = -wx * _SA + wy * _CA
    return rx, -ry * _SE - wz * _CE


def build_iso_svg(floors, storeys):
    """
    Axonometrische SVG: elke vloer op zijn werkelijke 3D-positie
    (XY-wereldcoordinaten, Z = BOK = fl.zmax), zodat de verdiepingen
    correct boven elkaar gestapeld liggen. Laag -> hoog getekend.
    """
    SCALE, PAD = 0.014, 80
    geo = [f for f in floors if f.rings]
    if not geo:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'

    storey_list = sorted(storeys.values(), key=lambda s: s['elev'])
    order = {s['idx']: i for i, s in enumerate(storey_list)}

    proj = [_iso_xy(x, y, f.zmax)
            for f in geo for (ring, v) in f.rings for (x, y) in ring]
    minx = min(p[0] for p in proj); maxx = max(p[0] for p in proj)
    miny = min(p[1] for p in proj); maxy = max(p[1] for p in proj)

    W = int((maxx - minx) * SCALE) + PAD * 2
    H = int((maxy - miny) * SCALE) + PAD * 2 + 40
    OX = PAD - minx * SCALE
    OY = PAD + 40 - miny * SCALE

    def sp(x, y, z):
        px, py = _iso_xy(x, y, z)
        return px * SCALE + OX, py * SCALE + OY

    def path(ring, z):
        return 'M ' + ' L '.join(f'{a:.1f},{b:.1f}'
                                 for (x, y) in ring for (a, b) in [sp(x, y, z)]) + ' Z'

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'style="background:#0f1923;font-family:sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#0f1923"/>',
         f'<text x="{W//2}" y="24" fill="#ddd" font-size="15" text-anchor="middle" '
         f'font-weight="bold">IFC Vloervelden — Axonometrisch (boven elkaar)</text>']

    # Laag -> hoog (painter's algorithm)
    for fl in sorted(geo, key=lambda f: f.zmax):
        ci = order.get(fl.storey['idx'] if fl.storey else None, 0)
        col = PALETTE[ci % len(PALETTE)]
        outers = [r for r, v in fl.rings if not v]
        voids  = [r for r, v in fl.rings if v]
        for ring in outers:
            d = path(ring, fl.zmax)
            for vr in voids:
                d += ' ' + path(vr, fl.zmax)
            L.append(f'<path d="{d}" fill="{col}" fill-rule="evenodd" '
                     f'fill-opacity="0.62" stroke="{col}" stroke-width="0.7"/>')

    # Legenda
    items = [(s['name'], PALETTE[order[s['idx']] % len(PALETTE)])
             for s in storey_list
             if any(f.storey and f.storey['idx'] == s['idx'] and f.rings for f in floors)]
    lx, ly = PAD, H - PAD - len(items) * 18
    if items:
        L.append(f'<rect x="{lx-8}" y="{ly-16}" width="230" '
                 f'height="{len(items)*18+12}" rx="5" fill="#000" opacity="0.45"/>')
        for i, (name, col) in enumerate(items):
            yy = ly + i * 18
            L.append(f'<rect x="{lx}" y="{yy-10}" width="15" height="10" fill="{col}" '
                     f'fill-opacity="0.62" stroke="{col}"/>')
            L.append(f'<text x="{lx+22}" y="{yy}" fill="#ccc" font-size="10">{name}</text>')

    L.append('</svg>')
    return '\n'.join(L)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCOPE BOXES + RASTERISATIE (depth map per verdieping)
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_floors_to_shapely(floors_subset):
    """
    Voeg een set vloeren (van één hoogteniveau) samen tot naad-loze geometrie.

    1. Alle buitenranden (outers) -> unary_union, met morfologische sluiting
       (buffer +/- SEAM_CLOSE_MM/2) zodat naden tussen aangrenzende slabs
       dichtsmelten.
    2. Sparingen (voids) worden afgetrokken, MAAR alleen waar ze een echt gat
       vormen. Onderscheid:
       - NAAST elkaar liggende vloeren vullen elkaars uitsparing op. Voorbeeld:
         een dunne randbalk met een grote uitsparing die door losse
         kanaalplaten wordt opgevuld -> geen gat.
       - GESTAPELDE lagen (twee vloeren die elkaar grotendeels overlappen, elk
         met een eigen sparing op een andere plek) -> elke sparing is een echt
         gat, ook al ligt de andere laag eroverheen.
       We trekken een void daarom alleen af voor zover hij niet wordt opgevuld
       door een vloer die NIET gestapeld op de void-eigenaar ligt.

    Retourneert een Shapely (Multi)Polygon of None.
    """
    # Verzamel per vloer de outer-polygoon, de NETTO vloer (outer minus eigen
    # voids) en de void-polygonen.
    per_floor = []   # (fl, outer_union, netto_floor, [void_polys])
    all_outers = []
    for fl in floors_subset:
        fouters, fvoids = [], []
        for ring, is_void in fl.rings:
            if len(ring) < 3:
                continue
            try:
                p = Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.area > 1:
                    (fvoids if is_void else fouters).append(p)
            except Exception:
                pass
        fo = unary_union(fouters) if fouters else None
        # netto vloer = outer minus eigen sparingen (het werkelijke vloervlak)
        netto = fo
        if fo is not None and fvoids:
            try:
                netto = fo.difference(unary_union(fvoids))
            except Exception:
                netto = fo
        all_outers.extend(fouters)
        per_floor.append((fl, fo, netto, fvoids))

    if not all_outers:
        return None

    solid = unary_union(all_outers)
    h = SEAM_CLOSE_MM / 2.0
    solid = solid.buffer(h, join_style=2).buffer(-h, join_style=2)

    # Bepaal de echte gaten: per void kijken of een NIET-gestapelde vloer hem
    # opvult. 'Gestapeld' = de andere vloer overlapt grotendeels het NETTO
    # vloervlak van de void-eigenaar (twee lagen boven elkaar). Een dunne
    # randbalk heeft een klein netto vloervlak, dus een buurvloer die de
    # uitsparing vult is NIET gestapeld en vult de void wél op.
    OVERLAP_STACK = 0.5
    real_holes = []
    for fl, fo, netto, fvoids in per_floor:
        if not fvoids or netto is None or netto.is_empty:
            continue
        for vd in fvoids:
            remaining = vd
            for fl2, fo2, netto2, _ in per_floor:
                if fl2.idx == fl.idx or netto2 is None or netto2.is_empty:
                    continue
                if not netto2.intersects(remaining):
                    continue
                # gestapeld? overlap van fl2's netto vloer met EIGEN netto vloer
                inter = netto.intersection(netto2).area
                ref_area = min(netto.area, netto2.area)
                stacked = ref_area > 0 and (inter / ref_area) > OVERLAP_STACK
                if stacked:
                    continue
                try:
                    remaining = remaining.difference(netto2)
                except Exception:
                    pass
                if remaining.is_empty:
                    break
            if not remaining.is_empty:
                real_holes.append(remaining)

    if real_holes:
        try:
            solid = solid.difference(unary_union(real_holes))
        except Exception:
            pass
    return solid


def _shapely_rings(geom):
    """(Multi)Polygon -> lijst van (ring [(x,y)], is_void)."""
    if geom is None or geom.is_empty:
        return []
    geoms = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    out = []
    for g in geoms:
        if g.is_empty or g.area < 100:
            continue
        out.append(([(x, y) for x, y in g.exterior.coords], False))
        for ring in g.interiors:
            out.append(([(x, y) for x, y in ring.coords], True))
    return out


def storey_scope_boxes(floors, storeys):
    """
    Bouw per verdieping (BEHALVE de laagste = begane grond) een scope-box.
    De begane grond wordt overgeslagen: daar is geen valgevaar naar beneden.

    Per box, naast de bbox/pixelmaat:
      - 'here'   : vloeren van deze verdieping
      - 'bok'    : hoofd-loopvlak (dominant niveau, alleen voor grijswaarde)
      - 'levels' : lijst (z, mask) van ALLE relevante vloer-niveaus binnen
                   de bbox, hoog -> laag gesorteerd. Elk mask is een geraster
                   boolean vlak (naden dicht, sparingen als gat). Hierop werkt
                   zowel de grijswaarde-weergave als de lokale valgevaar-check.
    """
    storey_list = sorted(storeys.values(), key=lambda s: s['elev'])
    per = {s['idx']: [] for s in storey_list}
    for fl in floors:
        if fl.rings and fl.storey and fl.storey['idx'] in per:
            per[fl.storey['idx']].append(fl)

    boxes = []
    # [1:] slaat de laagste verdieping (begane grond) over
    for s in storey_list[1:]:
        here = per[s['idx']]
        if not here:
            continue

        # Hoofd-loopvlak = dominant niveau (grootste oppervlak), voor grijswaarde
        area_per_level = {}
        for level_z, group in _group_by_level(here):
            area = 0.0
            for fl in group:
                for (ring, v) in fl.rings:
                    if not v and len(ring) >= 3:
                        try:
                            area += Polygon(ring).area
                        except Exception:
                            pass
            area_per_level[level_z] = area
        bok = max(area_per_level, key=area_per_level.get) if area_per_level \
            else max(f.zmax for f in here)

        # bbox op basis van de eigen verdieping
        xs, ys = [], []
        for f in here:
            for (ring, v) in f.rings:
                for (x, y) in ring:
                    xs.append(x); ys.append(y)
        x0, y0 = min(xs) - OFFSET_MM, min(ys) - OFFSET_MM
        x1, y1 = max(xs) + OFFSET_MM, max(ys) + OFFSET_MM

        box = {
            'storey': s, 'here': here, 'bok': bok,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'w': max(1, int((x1 - x0) * RENDER_SCALE)),
            'h': max(1, int((y1 - y0) * RENDER_SCALE)),
        }

        # Twee maskersets:
        # 1) here_levels = ALLEEN de eigen verdieping, per niveau. Bepaalt de
        #    WEERGAVE (grijswaarde + zwarte vide) en is identiek aan het 2D-
        #    aanzicht: de vide is hier een schoon zwart gat, want onderliggende
        #    vloeren vullen hem niet op.
        here_groups = sorted(_group_by_level(here), key=lambda t: -t[0])
        here_levels = []
        for z, group in here_groups:
            mask = _rasterize_level_mask(box, group)
            if mask.any():
                here_levels.append((z, mask))
        box['here_levels'] = here_levels

        # Masker van de ECHTE IFC-sparingen van de eigen verdieping. Deze
        # pixels worden NOOIT door naad-dichting opgevuld (sparing != naad).
        box['void_mask'] = _rasterize_void_mask(box, here)

        # 2) levels = eigen verdieping PLUS alles wat lager ligt (opvangvloeren).
        #    Alleen voor de valrand-detectie: een rand is geen valgevaar als er
        #    binnen 2,5m eronder een opvangvloer ligt.
        top_z = max(f.zmax for f in here)
        relevant = [f for f in floors if f.rings and f.zmax <= top_z + 1.0
                    and (top_z - f.zmax) <= (DEPTH_CUE_MM + 5000.0)
                    and _overlaps_bbox(f, x0, y0, x1, y1)]
        groups = sorted(_group_by_level(relevant), key=lambda t: -t[0])
        levels = []
        for z, group in groups:
            mask = _rasterize_level_mask(box, group)
            if mask.any():
                levels.append((z, mask))
        box['levels'] = levels
        boxes.append(box)
    return boxes


def _overlaps_bbox(fl, x0, y0, x1, y1):
    """Snelle bbox-overlap-test voor een vloer."""
    xs = [x for (ring, v) in fl.rings for (x, y) in ring]
    ys = [y for (ring, v) in fl.rings for (x, y) in ring]
    if not xs:
        return False
    return not (max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1)


def _grey(distance_mm):
    """Hoogteverschil -> grijswaarde (dichterbij = lichter)."""
    t = max(0.0, min(1.0, distance_mm / DEPTH_CUE_MM))
    return int(round(255 * (1.0 - t)))


def _group_by_level(floors_subset):
    """
    Groepeer vloeren op hoogteniveau: vloeren waarvan de BOK binnen
    SAME_LEVEL_MM van elkaar ligt, horen bij hetzelfde niveau en mogen
    samengevoegd worden (naden dichten). Retourneert lijst van
    (representatieve_bok, [vloeren]) gesorteerd van laag naar hoog.
    """
    floors_sorted = sorted(floors_subset, key=lambda f: f.zmax)
    groups = []
    for fl in floors_sorted:
        if groups and abs(fl.zmax - groups[-1][0]) <= SAME_LEVEL_MM:
            groups[-1][1].append(fl)
        else:
            groups.append((fl.zmax, [fl]))
    return groups


def _rasterize_level_mask(box, group):
    """Rasteriseer één hoogteniveau tot een boolean masker (vloer = True),
    met dezelfde samengevoegde geometrie als het 2D-aanzicht (naden dicht,
    echte voids/sparingen als gat)."""
    w, h = box['w'], box['h']
    x0, y1 = box['x0'], box['y1']
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)

    def to_px(ring):
        return [(int((x - x0) * RENDER_SCALE), int((y1 - y) * RENDER_SCALE))
                for (x, y) in ring]

    merged = _merge_floors_to_shapely(group)
    if merged is None or merged.is_empty:
        return np.zeros((h, w), dtype=bool)
    rings = _shapely_rings(merged)
    for (ring, is_void) in rings:
        if is_void:
            continue
        px = to_px(ring)
        if len(px) >= 3:
            draw.polygon(px, fill=1)
    for (ring, is_void) in rings:
        if not is_void:
            continue
        px = to_px(ring)
        if len(px) >= 3:
            draw.polygon(px, fill=0)
    return np.array(img, dtype=bool)


def _rasterize_void_mask(box, floors_subset):
    """Rasteriseer de ECHTE resterende sparingen tot een masker. Een pixel telt
    als echte sparing als hij (a) na samenvoeging per niveau een gat is EN
    (b) in de oorspronkelijke IFC-voids zat. Zo blijven echte sparingen
    beschermd tegen naad-dichting, terwijl naden tussen losse slabs (die nooit
    een IFC-void waren) wél gedicht mogen worden. Een opgevulde randbalk-void
    valt af bij (a); een slab-naad valt af bij (b)."""
    w, h = box['w'], box['h']
    x0, y1 = box['x0'], box['y1']

    def to_px(ring):
        return [(int((x - x0) * RENDER_SCALE), int((y1 - y) * RENDER_SCALE))
                for (x, y) in ring]

    # (a) gaten na samenvoeging
    merged_holes = Image.new('L', (w, h), 0)
    dm = ImageDraw.Draw(merged_holes)
    for level_z, group in _group_by_level(floors_subset):
        merged = _merge_floors_to_shapely(group)
        for (ring, is_void) in _shapely_rings(merged):
            if is_void and len(ring) >= 3:
                px = to_px(ring)
                if len(px) >= 3:
                    dm.polygon(px, fill=1)
    merged_holes = np.array(merged_holes, dtype=bool)

    # (b) ruwe IFC-voids
    raw_voids = Image.new('L', (w, h), 0)
    dr = ImageDraw.Draw(raw_voids)
    for fl in floors_subset:
        for ring, is_void in fl.rings:
            if is_void and len(ring) >= 3:
                px = to_px(ring)
                if len(px) >= 3:
                    dr.polygon(px, fill=1)
    raw_voids = np.array(raw_voids, dtype=bool)

    return merged_holes & raw_voids


def rasterize_depth(box):
    """
    Rasteriseer een verdieping tot een grijswaarde-depthmap.

    Werkt met ALLE vloer-niveaus die binnen het zicht vallen (de eigen
    verdieping plus alles wat eronder ligt). Per pixel houden we de hoogte
    van de bovenste vloer bij; de grijswaarde toont het hoogteverschil met
    het hoofd-loopvlak van deze verdieping (puur visueel). De valgevaar-
    detectie gebeurt los hiervan in detect_edges_from_levels, lokaal per rand.

    De vide-/sparing-vorm is identiek aan het 2D-aanzicht.
    """
    w, h = box['w'], box['h']
    # Referentievlak = het grootste vloerveld van de verdieping (dominant
    # niveau). Dat is wit; vloeren die lager liggen worden donkerder, en een
    # vloer die 2,5m lager ligt is zwart. Een klein verhoogd deel (bordes,
    # schachtdek) maakt zo niet de hele hoofdvloer zwart.
    ref_top = box['bok']

    # Voor de WEERGAVE gebruiken we ALLEEN de eigen verdieping (here_levels),
    # zodat de vide-/sparing-vorm exact die van het 2D-aanzicht is: een schoon
    # zwart gat dat NIET wordt opgevuld door onderliggende vloeren. (Die
    # onderliggende vloeren tellen alleen mee bij de valrand-detectie.)
    levels = box['here_levels']   # lijst (z, mask) gesorteerd hoog -> laag

    # Hoogtekaart: Z van de bovenste vloer per pixel (NaN = leeg)
    height = np.full((h, w), np.nan, dtype=float)
    for z, mask in levels:   # hoog naar laag; eerste die een pixel vult wint
        fill = mask & np.isnan(height)
        height[fill] = z

    # Naden DICHTEN tussen aangrenzende vloeren. Twee vloeren van dezelfde
    # verdieping die op (vrijwel) gelijke of nabije hoogte liggen, raken elkaar
    # in de praktijk niet perfect; er blijft een dunne zwarte strook tussen
    # staan. Die naad is geen vide en mag geen valrand worden.
    #
    # Een naad is een SMAL zwart gebied dat aan weerszijden vloer heeft. Een
    # echte vide/sparing of de buitenruimte is breed. We onderscheiden ze met
    # een morfologische "opening" op het zwart: smalle zwarte stroken (naden)
    # verdwijnen, brede zwarte gebieden (vides, buitenkant) blijven staan. Wat
    # verdwijnt is naad en wordt met de dichtstbijzijnde vloerhoogte gevuld.
    #
    # Vierkante structuur (8-connectiviteit) houdt rechte hoeken recht; een
    # diamant zou de scherpe hoeken van vides/sparingen afschuinen.
    sq = np.ones((3, 3), dtype=bool)
    floor_mask = ~np.isnan(height)
    black_mask = ~floor_mask
    # Naad-breedte die we dichten: ruim boven de gemeten naden (~100-360mm),
    # maar onder de kleinste echte sparing. ~500mm is een veilige grens.
    naad_px = max(2, int(round(500.0 * RENDER_SCALE / 2)))   # halve breedte in px
    # Morfologische opening: erodeer het zwart en groei het terug. Smalle
    # naden (< 2*naad_px breed) overleven de erosie niet en verdwijnen.
    black_open = binary_dilation(
        ~binary_dilation(floor_mask, structure=sq, iterations=naad_px),
        structure=sq, iterations=naad_px)
    seam = black_mask & ~black_open      # smalle zwarte stroken = naden
    # Echte IFC-sparingen NOOIT dichten: trek het void-masker eraf. Maar het
    # void-masker kan dunne naad-achtige uitlopers bevatten (bv. waar een
    # opgevulde randbalk-void smalle naden tussen kanaalplaten overlapt). Een
    # echte sparing is COMPACT (bij benadering even hoog als breed); een naad
    # is LANGWERPIG (lijnvormig). We beschermen daarom alleen void-gebieden die
    # niet lijnvormig zijn: per samenhangend void-gebied vergelijken we de
    # oppervlakte met de bounding box — een dun lijnstuk vult zijn bbox slecht.
    void_mask = box.get('void_mask')
    if void_mask is not None:
        from scipy.ndimage import label as _label
        protect = np.zeros_like(void_mask)
        vlbl, vn = _label(void_mask)
        for i in range(1, vn + 1):
            ys, xs = np.where(vlbl == i)
            if len(ys) == 0:
                continue
            bw = xs.max() - xs.min() + 1
            bh = ys.max() - ys.min() + 1
            short = min(bw, bh)
            long = max(bw, bh)
            # lijnvormig (naad) als één dimensie veel groter is dan de andere
            # én de korte zijde smal is; anders is het een compacte sparing.
            is_line = (short <= 2 * naad_px) and (long > 4 * short)
            if not is_line:
                protect[vlbl == i] = True
        seam = seam & ~binary_dilation(protect, structure=sq, iterations=1)
    if seam.any():
        # Vul elke naad-pixel met de hoogte van de dichtstbijzijnde vloer.
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~floor_mask, return_distances=False,
                                     return_indices=True)
        nearest = height[tuple(idx)]
        height[seam] = nearest[seam]

    # Grijswaarde met VASTE schaal t.o.v. het referentievlak (grootste
    # vloerveld). Elke 2,5m lager = één volledige zwart-trap, dus elke
    # grijs-stap komt overeen met een vast hoogteverschil (consistent met
    # het 2,5m-valcriterium). Vloeren hoger dan de referentie clampen naar
    # wit (een verhoogd plateau is geen valgevaar als je erop staat).
    # Leeg blijft zwart.
    grey = np.zeros((h, w), dtype=np.uint8)
    valid = ~np.isnan(height)
    if valid.any():
        diff = np.where(valid, ref_top - height, 0.0)   # >0 = lager dan referentie
        t = np.clip(diff / DEPTH_CUE_MM, 0.0, 1.0)
        g = np.round(255 * (1.0 - t)).astype(np.uint8)
        grey[valid] = g[valid]
    return grey


def detect_edges_from_levels(box, thickness=2):
    """
    Valgevaar-detectie op basis van de DEPTHMAP zelf.

    Het contrast wordt bepaald uit de grijswaarde-depthmap: een valrand is de
    grens tussen een vloer (grijs > 0) en zwart (= leegte of een vloer die
    >2,5m lager ligt). De depthmap gebruikt de eigen verdieping (here_levels),
    dus de vide-/sparingvorm is exact die van het 2D-aanzicht.

    Eén correctie: een rand naar een zwart gebied is GEEN valgevaar als daar
    een opvangvloer ligt binnen 2,5m onder het vloerniveau. Die randen worden
    onderdrukt met behulp van de volledige niveaustapel (incl. opvangvloeren).
    """
    w, h = box['w'], box['h']
    k = np.ones((3, 3), dtype=bool)
    NEG = -1e9

    # Depthmap-weergave (eigen verdieping): vloer = grijs>0, vide/leeg = zwart.
    grey = rasterize_depth(box)
    floor = grey > 0
    black = grey == 0

    # Hoogtekaart van de eigen verdieping (op de vloerpixels).
    here_top = np.full((h, w), NEG, dtype=float)
    for z, mask in box['here_levels']:
        fill = mask & (here_top == NEG)
        here_top[fill] = z

    # Opvang-hoogtekaart: per pixel de hoogste vloer (uit de VOLLEDIGE stapel)
    # die NIET hoger ligt dan de eigen verdieping op dat punt. Een vloer die
    # HOGER ligt dan waar je staat is geen opvang — je valt er niet op. Daarom
    # bouwen we voor elke zwarte buur de hoogste vloer op die <= het vloer-
    # niveau van de aangrenzende vloer ligt. We benaderen dit met een vaste
    # opvang-kaart: de hoogste vloer per pixel die niet boven het dominante
    # loopvlak (ref_top) uitsteekt.
    ref_top = box['bok']
    catch = np.full((h, w), NEG, dtype=float)
    for z, mask in box['levels']:
        if z > ref_top + SAME_LEVEL_MM:
            continue                       # hoger dan loopvlak = geen opvang
        fill = mask & (catch == NEG)
        catch[fill] = z

    # Een vloerpixel is een valrand als hij grenst aan een zwarte pixel waar
    # GEEN opvang binnen 2,5m onder het eigen vloerniveau ligt: de buur is leeg,
    # of de hoogste opvangvloer daar ligt >2,5m lager (of hoger = geen opvang).
    edge = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb_black = np.zeros((h, w), dtype=bool)
            nb_catch = np.full((h, w), NEG, dtype=float)
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            sy0, sy1 = max(0, -dy), h - max(0, dy)
            sx0, sx1 = max(0, -dx), w - max(0, dx)
            nb_black[ys0:ys1, xs0:xs1] = black[sy0:sy1, sx0:sx1]
            nb_catch[ys0:ys1, xs0:xs1] = catch[sy0:sy1, sx0:sx1]
            # opvang aanwezig = er ligt een vloer in de buur, hoogstens 2,5m
            # lager dan het eigen vloerniveau (en niet hoger dan dat niveau).
            opvang = (nb_catch > NEG) & ((here_top - nb_catch) <= DEPTH_CUE_MM) \
                     & ((here_top - nb_catch) >= -SAME_LEVEL_MM)
            geen_opvang = ~opvang
            edge |= floor & (here_top > NEG) & nb_black & geen_opvang

    if thickness > 1:
        edge = binary_dilation(edge, structure=k, iterations=thickness - 1)
    return edge


def detect_edges(grey, thickness=2):
    """Behouden voor compatibiliteit: rand tussen vloer (grijs>0) en zwart.
    De fysiek-correcte detectie zit in detect_edges_from_levels."""
    k = np.ones((3, 3), dtype=bool)
    floor = grey > 0
    deep = grey == 0
    edge = (binary_dilation(floor, structure=k) & deep) | \
           (binary_dilation(deep, structure=k) & floor)
    if thickness > 1:
        edge = binary_dilation(edge, structure=k, iterations=thickness - 1)
    return edge


def _panels_png(images, title, out_path):
    """Combineer per-verdieping afbeeldingen zij-aan-zij tot één PNG met labels."""
    if not images:
        return None
    GAP, LBL, TITLE, PAD = 20, 22, 32, 16
    imgs = [im for _, im in images]
    labels = [lb for lb, _ in images]
    tw = sum(im.width for im in imgs) + GAP * (len(imgs) - 1) + PAD * 2
    th = TITLE + LBL + max(im.height for im in imgs) + PAD * 2
    canvas = Image.new('RGB', (tw, th), (17, 17, 17))
    d = ImageDraw.Draw(canvas)
    d.text((tw // 2, PAD), title, fill=(200, 200, 200), anchor='mt')
    x = PAD
    y = TITLE + LBL + PAD
    for im, lb in zip(imgs, labels):
        d.text((x + im.width // 2, TITLE + PAD + 2), lb, fill=(200, 200, 200), anchor='mt')
        bar = int(10000 * RENDER_SCALE)
        by = y + im.height - 8
        d.rectangle([x + 6, by - 2, x + 6 + bar, by + 2], fill=(136, 136, 136))
        d.text((x + 6 + bar // 2, by - 13), '10 m', fill=(136, 136, 136), anchor='mt')
        canvas.paste(im, (x, y))
        x += im.width + GAP
    canvas.save(str(out_path), 'PNG')
    return out_path


def build_depthmap_png(boxes, out_path):
    """Depth map per verdieping: grijswaarden naar hoogteverschil."""
    images = []
    for box in boxes:
        grey = rasterize_depth(box)
        rgb = np.stack([grey, grey, grey], axis=-1).astype(np.uint8)
        lbl = f"{box['storey']['name']}  BOK={box['bok']/1000:.3f}m"
        images.append((lbl, Image.fromarray(rgb, 'RGB')))
    return _panels_png(images, 'Depth Map per verdieping — wit=eigen vloer, '
                               'grijs=onderliggend, zwart=leeg', out_path)


def build_contrastmap_png(boxes, out_path):
    """Contrast map per verdieping: depth map + rode arcering op valranden.
    De valranden komen uit de lokale, fysiek-correcte detectie: elke rand
    waar je >2,5m naar beneden kunt vallen (ook tussen vloeren van hetzelfde
    verdiepingslevel) wordt rood gemarkeerd."""
    images = []
    total_edge = 0
    for box in boxes:
        grey = rasterize_depth(box)
        edges = detect_edges_from_levels(box)
        total_edge += int(edges.sum())
        rgb = np.stack([grey, grey, grey], axis=-1).astype(np.uint8)
        rgb[edges] = (255, 0, 0)   # rode valgevaarlijke randen
        lbl = f"{box['storey']['name']}  loopvlak={box['bok']/1000:.3f}m"
        images.append((lbl, Image.fromarray(rgb, 'RGB')))
    res = _panels_png(images, 'Contrast Map per verdieping — rode arcering = '
                              'valrand (>2,5m val mogelijk)', out_path)
    return res, total_edge


def build_2d_svg(ifc, floors, storeys):
    """Eén SVG met een 2D-bovenaanzicht per verdieping, naast elkaar."""
    # Groepeer vloeren per storey (op elevatie gesorteerd)
    storey_list = sorted(storeys.values(), key=lambda s: s['elev'])
    per = {s['idx']: [] for s in storey_list}
    unlinked = []
    for fl in floors:
        if fl.storey and fl.storey['idx'] in per:
            per[fl.storey['idx']].append(fl)
        else:
            unlinked.append(fl)

    # Globale bbox (zodat alle panelen dezelfde schaal hebben)
    gx0, gy0, gx1, gy1 = tris_bbox(floors)
    gw, gh = (gx1-gx0) or 1, (gy1-gy0) or 1

    PANEL_W, PANEL_H = 520, 420
    PAD, LABEL_H, GAP, TITLE_H = 18, 30, 24, 40
    scale = min((PANEL_W-2*PAD)/gw, (PANEL_H-2*PAD)/gh) * 0.95
    panels = [s for s in storey_list if per[s['idx']]]
    n = len(panels)

    total_w = n * (PANEL_W + GAP) - GAP + 2*PAD
    total_h = TITLE_H + LABEL_H + PANEL_H + 2*PAD
    if total_w < 400:
        total_w = 400

    def tx(x, ox): return ox + PAD + (x - gx0) * scale + ((PANEL_W-2*PAD) - gw*scale)/2
    def ty(y):     return TITLE_H + LABEL_H + PAD + (gy1 - y) * scale + ((PANEL_H-2*PAD) - gh*scale)/2

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
         f'style="background:#0f1923;font-family:sans-serif">',
         f'<rect width="{total_w}" height="{total_h}" fill="#0f1923"/>',
         f'<text x="{total_w//2}" y="26" fill="#ddd" font-size="16" '
         f'text-anchor="middle" font-weight="bold">'
         f'2D Vloeromtrek per verdieping — velden &lt;{int(MERGE_GAP_MM)}mm samengevoegd</text>']

    for pi, s in enumerate(panels):
        ox = PAD + pi * (PANEL_W + GAP)
        col = PALETTE[pi % len(PALETTE)]
        fls = per[s['idx']]

        # Paneel-kader + label
        L.append(f'<rect x="{ox}" y="{TITLE_H+LABEL_H+PAD-4}" width="{PANEL_W}" '
                 f'height="{PANEL_H}" fill="#0a1420" stroke="{col}" '
                 f'stroke-width="1" stroke-opacity="0.4"/>')
        L.append(f'<text x="{ox+PANEL_W//2}" y="{TITLE_H+18}" fill="{col}" '
                 f'font-size="13" text-anchor="middle" font-weight="bold">{s["name"]}</text>')
        L.append(f'<text x="{ox+PANEL_W//2}" y="{TITLE_H+LABEL_H-2}" fill="#888" '
                 f'font-size="10" text-anchor="middle">'
                 f'+{s["elev"]/1000:.2f}m · {len(fls)} vloeren</text>')

        # Clip
        cid = f'clip{pi}'
        L.append(f'<defs><clipPath id="{cid}"><rect x="{ox}" y="{TITLE_H+LABEL_H+PAD-4}" '
                 f'width="{PANEL_W}" height="{PANEL_H}"/></clipPath></defs>')
        L.append(f'<g clip-path="url(#{cid})">')

        # Voeg de vloervelden per niveau samen met dezelfde logica als de
        # depthmap: naden tussen aangrenzende slabs dichten, maar echte
        # sparingen/vides behouden als gaten. Zo is het 2D-aanzicht consistent
        # met de valgevaar-analyse en toont de vide een schone, doorlopende rand.
        for level_z, group in _group_by_level(fls):
            merged = _merge_floors_to_shapely(group)
            for (ring, is_void) in _shapely_rings(merged):
                if is_void:
                    continue
                # buitenrand + bijbehorende gaten in één path (fill-rule evenodd)
                d = 'M ' + ' L '.join(f'{tx(x,ox):.1f},{ty(y):.1f}' for (x, y) in ring) + ' Z'
                # voeg de gaten toe die binnen deze outer vallen
                outer_poly = Polygon(ring)
                for (vring, vv) in _shapely_rings(merged):
                    if not vv:
                        continue
                    try:
                        if outer_poly.contains(Polygon(vring).representative_point()):
                            d += ' M ' + ' L '.join(f'{tx(x,ox):.1f},{ty(y):.1f}'
                                                    for (x, y) in vring) + ' Z'
                    except Exception:
                        pass
                L.append(f'<path d="{d}" fill="{col}" fill-opacity="0.30" '
                         f'fill-rule="evenodd" stroke="{col}" stroke-width="1.6" '
                         f'stroke-linejoin="round"/>')

        L.append('</g>')

        # Schaalbalk 5m
        bar = 5000 * scale
        bx, by = ox + 10, TITLE_H + LABEL_H + PANEL_H - 6
        L.append(f'<line x1="{bx}" y1="{by}" x2="{bx+bar}" y2="{by}" '
                 f'stroke="#888" stroke-width="1.5"/>')
        L.append(f'<text x="{bx+bar/2}" y="{by-4}" fill="#888" font-size="9" '
                 f'text-anchor="middle">5 m</text>')

    L.append('</svg>')
    return '\n'.join(L)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        fp = sys.argv[1]
    else:
        fp = input('Pad naar IFC: ').strip().strip('"').strip("'")
    if not os.path.isfile(fp):
        print(f'FOUT: niet gevonden: {fp}'); sys.exit(1)

    path = Path(fp)
    out_dir = path.parent if os.access(path.parent, os.W_OK) else Path(__file__).parent
    stem = path.stem

    print(f'\n[1/6] Laden: {path.name}')
    ifc = IFC(fp)
    print(f'      {len(ifc.ent)} entiteiten')

    print('[2/6] Vloeren lezen (IFCSLAB + IFCROOF)...')
    storeys = read_storeys(ifc)
    floors = read_floors(ifc)
    link_to_storeys(ifc, floors, storeys)
    n_slab = sum(1 for f in floors if f.kind == 'slab')
    n_roof = sum(1 for f in floors if f.kind == 'roof')
    print(f'      {n_slab} slabs + {n_roof} roofs = {len(floors)} vloeren')

    print('[3/6] Geometrie -> ringen per vloer...')
    ok = 0
    for fl in floors:
        build_floor_rings(ifc, fl)
        if fl.rings:
            ok += 1
    print(f'      {ok}/{len(floors)} vloeren met geometrie')

    print('[4/6] 2D-aanzicht + isometrie genereren...')
    svg2d = out_dir / f'{stem}_2d_aanzicht.svg'
    svg2d.write_text(build_2d_svg(ifc, floors, storeys), encoding='utf-8')
    print(f'      ✓ {svg2d.name}')
    iso = out_dir / f'{stem}_iso.svg'
    iso.write_text(build_iso_svg(floors, storeys), encoding='utf-8')
    print(f'      ✓ {iso.name}')

    print('[5/6] Depth map per verdieping...')
    boxes = storey_scope_boxes(floors, storeys)
    depth_png = out_dir / f'{stem}_depthmap.png'
    build_depthmap_png(boxes, depth_png)
    print(f'      ✓ {depth_png.name}  ({len(boxes)} verdiepingen)')

    print('[6/6] Contrast map (rode valranden)...')
    contrast_png = out_dir / f'{stem}_contrastmap.png'
    _, n_edge = build_contrastmap_png(boxes, contrast_png)
    print(f'      ✓ {contrast_png.name}  ({n_edge} randpixels)')

    print(f'\n✓  Klaar — 4 outputs in {out_dir}')
    print(f'   {stem}_2d_aanzicht.svg   (vloervelden per verdieping)')
    print(f'   {stem}_iso.svg           (axonometrisch, boven elkaar)')
    print(f'   {stem}_depthmap.png      (hoogteverschillen)')
    print(f'   {stem}_contrastmap.png   (valgevaarlijke randen rood)')


# ═══════════════════════════════════════════════════════════════════════════════
#  RANDBEVEILIGING SWEEPEN LANGS DE VALRANDEN (contrastmap)  →  IFC4
# ═══════════════════════════════════════════════════════════════════════════════
#  Toegevoegd ná het bovenstaande script. Hergebruikt de analyse hierboven
#  (detect_edges_from_levels = exact de rode contrastmap-lijnen) en sweept de
#  Valwijzer-doorsnede langs elke valrand. De doorsnede is over de verticale as
#  gespiegeld: rood vlak BUITEN het vloerveld, groen scherm op de rand. Plaatsing
#  volgt de rode lijnen met juiste oriëntatie en vloer-BOK als hoogte (Z).
#
#  Gebruik:
#     python ifc_2d_aanzicht.py model1.ifc [model2.ifc ...] [-o VLW_DO_Valgevaar.ifc] [--verify]
#
#  Extra vereisten t.o.v. het bovenstaande: scikit-image (skimage).

from skimage.morphology import skeletonize
import datetime

_NEG = -1e9
S = RENDER_SCALE  # mm -> px (zelfde schaal als de contrastmap)

# Gespiegelde dwarsdoorsnede (rood BUITEN het vloerveld).
# Lokaal profielvlak: X = 'across' (+ = vloer-binnenzijde), Y = 'up' (= +Z wereld).
#  - Groene randbeveiliging   : X 0..149,   Y 0..1000   (binnenzijde, staand scherm)
#  - Rode valgevaar-markering : X -900..0,  Y -150..0   (buitenzijde, plat op vloer)
PROFILE_GREEN = [(0.0, 0.0), (149.0, 0.0), (149.0, 1000.0), (0.0, 1000.0)]
PROFILE_RED   = [(-900.0, -150.0), (0.0, -150.0), (0.0, 0.0), (-900.0, 0.0)]


_NEG = -1e9



def _here_top_map(box):
    h, w = box['h'], box['w']
    ht = np.full((h, w), _NEG, float)
    for z, mask in box['here_levels']:
        fill = mask & (ht == _NEG)
        ht[fill] = z
    return ht


def px_to_world(x_px, y_px, box):
    X = box['x0'] + x_px / S
    Y = box['y1'] - y_px / S
    return X, Y


def world_to_px(X, Y, box):
    return int(round((X - box['x0']) * S)), int(round((box['y1'] - Y) * S))


def _trace_skeleton(skel):
    """Skeleton (bool) -> lijst van pixel-paden [[(x,y),...]] (x=col, y=row)."""
    h, w = skel.shape
    pts = set(zip(*np.where(skel)[::-1]))  # (x,y)
    nb = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

    def neigh(p):
        x, y = p
        return [(x+dx, y+dy) for dx, dy in nb if (x+dx, y+dy) in pts]

    deg = {p: len(neigh(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}
    paths = []
    used = set()  # ongerichte edges als frozenset

    def walk(start, second):
        path = [start, second]
        used.add(frozenset((start, second)))
        prev, cur = start, second
        while deg.get(cur, 0) == 2 and cur not in nodes:
            nxt = [q for q in neigh(cur) if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            used.add(frozenset((prev, cur)))
            if cur == start:
                break
        return path

    for n in nodes:
        for q in neigh(n):
            if frozenset((n, q)) not in used:
                paths.append(walk(n, q))
    # losse lussen zonder node
    for p in pts:
        for q in neigh(p):
            e = frozenset((p, q))
            if e not in used:
                paths.append(walk(p, q))
    return [pa for pa in paths if len(pa) >= 2]


def _simplify(path, tol=2.0):
    """Douglas-Peucker op pixelpad."""
    pts = np.array(path, float)
    if len(pts) <= 2:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts)-1)]
    while stack:
        i, j = stack.pop()
        if j <= i+1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        rel = pts[i+1:j] - a
        if L < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(ab[0]*rel[:, 1] - ab[1]*rel[:, 0]) / L
        k = np.argmax(d)
        if d[k] > tol:
            idx = i+1+k
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return pts[keep]


def _inside_normal(A_w, B_w, box, floor_mask):
    """Eenheidsnormaal die naar de vloer-binnenzijde wijst (wereld-XY)."""
    ax, ay = A_w; bx, by = B_w
    tx, ty = bx-ax, by-ay
    L = math.hypot(tx, ty)
    if L < 1e-6:
        return None
    tx, ty = tx/L, ty/L
    mx, my = (ax+bx)/2, (ay+by)/2
    cand = [(-ty, tx), (ty, -tx)]
    score = [0, 0]
    h, w = floor_mask.shape
    for i, (nx, ny) in enumerate(cand):
        for off in (80.0, 160.0, 240.0):
            px, py = world_to_px(mx+nx*off, my+ny*off, box)
            if 0 <= px < w and 0 <= py < h and floor_mask[py, px]:
                score[i] += 1
    if score[0] == score[1]:
        return None
    return cand[0] if score[0] > score[1] else cand[1]


def _segment_Z(Aw, Bw, n, box, ht):
    """Lokale vloer-BOK (Z) van een segment: mediaan van here_top bemonsterd
    LANGS het segment (op de lijn en iets naar de vloer-binnenzijde). Zo krijgt
    elk segment de hoogte van de vloer waar het werkelijk ligt, ook als binnen
    dezelfde verdieping meerdere niveaus voorkomen."""
    h, w = ht.shape
    nx, ny = n
    vals = []
    N = 12
    for i in range(N + 1):
        t = i / N
        bx = Aw[0] + t * (Bw[0] - Aw[0])
        by = Aw[1] + t * (Bw[1] - Aw[1])
        for off in (0.0, 60.0, 120.0):     # op de lijn en iets naar binnen
            px, py = world_to_px(bx + nx * off, by + ny * off, box)
            if 0 <= px < w and 0 <= py < h and ht[py, px] > _NEG:
                vals.append(ht[py, px])
    return float(np.median(vals)) if vals else None


def segments_for_box(box, min_len_mm=250.0):
    """Lijst van (Z0, location(X,Y), axis_dir(d), ref_dir(n), depth) per segment."""
    edge = detect_edges_from_levels(box, thickness=1)
    if not edge.any():
        return []
    skel = skeletonize(edge)
    ht = _here_top_map(box)
    floor_mask = ht > _NEG
    paths = _trace_skeleton(skel)
    out = []
    for path in paths:
        sp = _simplify(path, tol=2.0)
        world = [px_to_world(x, y, box) for (x, y) in sp]
        for k in range(len(world)-1):
            Aw, Bw = world[k], world[k+1]
            seglen = math.hypot(Bw[0]-Aw[0], Bw[1]-Aw[1])
            if seglen < min_len_mm:
                continue
            n = _inside_normal(Aw, Bw, box, floor_mask)
            if n is None:
                continue
            nx, ny = n
            # Z LOKAAL per segment (niet één mediaan over de hele polylijn)
            Z0 = _segment_Z(Aw, Bw, n, box, ht)
            if Z0 is None:
                continue
            d = (ny, -nx)  # run-richting (langs de rand), zodat d x n = +Z
            # start zo kiezen dat +d naar het andere eindpunt wijst
            if (Bw[0]-Aw[0])*d[0] + (Bw[1]-Aw[1])*d[1] >= 0:
                loc = Aw
            else:
                loc = Bw
            out.append({
                'Z0': Z0, 'loc': (loc[0], loc[1], Z0),
                'axis': (d[0], d[1], 0.0), 'ref': (nx, ny, 0.0),
                'depth': seglen, 'storey': box['storey']['name'],
            })
    return out


def process_model(fp):
    ifc = IFC(fp)
    storeys = read_storeys(ifc)
    floors = read_floors(ifc)
    link_to_storeys(ifc, floors, storeys)
    for fl in floors:
        build_floor_rings(ifc, fl)
    boxes = storey_scope_boxes(floors, storeys)
    segs = []
    for box in boxes:
        segs += segments_for_box(box)
    return segs


# ===========================================================================
#  IFC4-WRITER
# ===========================================================================

_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"

def guid():
    import uuid
    b = uuid.uuid4().bytes
    n = int.from_bytes(b, 'big')
    out = ''
    for _ in range(22):
        out = _B64[n & 63] + out
        n >>= 6
    return out[:22]


class IfcWriter:
    def __init__(self):
        self.lines = []
        self.id = 0
        self._dir_cache = {}

    def add(self, body):
        self.id += 1
        self.lines.append(f'#{self.id}={body};')
        return self.id

    def pt3(self, x, y, z):
        return self.add(f'IFCCARTESIANPOINT(({x:.6f},{y:.6f},{z:.6f}))')

    def pt2(self, x, y):
        return self.add(f'IFCCARTESIANPOINT(({x:.6f},{y:.6f}))')

    def dir3(self, x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in self._dir_cache:
            self._dir_cache[key] = self.add(f'IFCDIRECTION(({x:.8f},{y:.8f},{z:.8f}))')
        return self._dir_cache[key]

    def write(self, path, segments, project_name='Randbeveiliging'):
        L = self.lines
        # --- basis ---
        oo = self.pt3(0, 0, 0)
        zdir = self.dir3(0, 0, 1)
        xdir = self.dir3(1, 0, 0)
        world = self.add(f'IFCAXIS2PLACEMENT3D(#{oo},$,$)')
        ctx = self.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.0E-5,#{world},$)")
        body_ctx = self.add(f"IFCGEOMETRICREPRESENTATIONSUBCONTEXT('Body','Model',*,*,*,*,#{ctx},$,.MODEL_VIEW.,$)")
        u1 = self.add('IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.)')
        u2 = self.add('IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.)')
        u3 = self.add('IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.)')
        u4 = self.add('IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.)')
        units = self.add(f'IFCUNITASSIGNMENT((#{u1},#{u2},#{u3},#{u4}))')
        # owner history (minimaal)
        per = self.add("IFCPERSON($,'EDGEDETECT',$,$,$,$,$,$)")
        org = self.add("IFCORGANIZATION($,'EDGEDETECT',$,$,$)")
        po = self.add(f'IFCPERSONANDORGANIZATION(#{per},#{org},$)')
        app = self.add(f"IFCAPPLICATION(#{org},'1.0','EDGEDETECT sweep','EDGEDETECT')")
        ts = int(datetime.datetime.now().timestamp())
        oh = self.add(f'IFCOWNERHISTORY(#{po},#{app},$,.ADDED.,{ts},#{po},#{app},{ts})')

        proj = self.add(f"IFCPROJECT('{guid()}',#{oh},'{project_name}',$,$,$,$,(#{ctx}),#{units})")
        site_pl = self.add(f'IFCLOCALPLACEMENT($,#{world})')
        site = self.add(f"IFCSITE('{guid()}',#{oh},'Site',$,$,#{site_pl},$,$,.ELEMENT.,$,$,$,$,$)")
        bld_pl = self.add(f'IFCLOCALPLACEMENT(#{site_pl},#{world})')
        bld = self.add(f"IFCBUILDING('{guid()}',#{oh},'Building',$,$,#{bld_pl},$,$,.ELEMENT.,$,$,$)")

        # --- styles (groen scherm, rood vlak) ---
        col_g = self.add('IFCCOLOURRGB($,0.,1.,0.)')
        ren_g = self.add(f'IFCSURFACESTYLERENDERING(#{col_g},0.0,$,$,$,$,IFCNORMALISEDRATIOMEASURE(0.5),IFCSPECULAREXPONENT(64.),.NOTDEFINED.)')
        sty_g = self.add(f"IFCSURFACESTYLE('Randbeveiliging verticaal',.BOTH.,(#{ren_g}))")
        col_r = self.add('IFCCOLOURRGB($,1.,0.,0.)')
        ren_r = self.add(f'IFCSURFACESTYLERENDERING(#{col_r},0.3,$,$,$,$,IFCNORMALISEDRATIOMEASURE(0.5),IFCSPECULAREXPONENT(64.),.NOTDEFINED.)')
        sty_r = self.add(f"IFCSURFACESTYLE('Valgevaar horizontaal',.BOTH.,(#{ren_r}))")

        # --- gedeelde profielen ---
        def profile(name, pts):
            ptids = [self.pt2(x, y) for (x, y) in pts]
            ptids.append(ptids[0])  # sluiten
            poly = self.add('IFCPOLYLINE((' + ','.join(f'#{i}' for i in ptids) + '))')
            return self.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,'{name}',#{poly})")
        prof_g = profile('Randbeveiliging', PROFILE_GREEN)
        prof_r = profile('Valgevaar', PROFILE_RED)

        # --- groepeer segmenten per (model, storey) ---
        groups = {}
        for s in segments:
            key = (s.get('model', ''), s['storey'], round(s['Z0'], 1))
            groups.setdefault(key, []).append(s)

        storey_ids = []
        contained = []  # (storey_id, [proxy_ids])
        for (model, sname, z), segs in sorted(groups.items(), key=lambda kv: kv[0][2]):
            spl = self.add(f'IFCLOCALPLACEMENT(#{bld_pl},#{world})')
            disp = f'{model} {sname}'.strip()
            st = self.add(f"IFCBUILDINGSTOREY('{guid()}',#{oh},'{disp}',$,$,#{spl},$,$,.ELEMENT.,{z:.3f})")
            storey_ids.append(st)

            green_ex, red_ex = [], []
            for sg in segs:
                lx, ly, lz = sg['loc']
                loc = self.pt3(lx, ly, lz)
                ax = self.dir3(*sg['axis'])
                rf = self.dir3(*sg['ref'])
                pos = self.add(f'IFCAXIS2PLACEMENT3D(#{loc},#{ax},#{rf})')
                dep = sg['depth']
                eg = self.add(f'IFCEXTRUDEDAREASOLID(#{prof_g},#{pos},#{zdir},{dep:.6f})')
                er = self.add(f'IFCEXTRUDEDAREASOLID(#{prof_r},#{pos},#{zdir},{dep:.6f})')
                self.add(f'IFCSTYLEDITEM(#{eg},(#{sty_g}),$)')
                self.add(f'IFCSTYLEDITEM(#{er},(#{sty_r}),$)')
                green_ex.append(eg); red_ex.append(er)

            proxies = []
            for label, exids in [('Randbeveiliging', green_ex), ('Valgevaar-markering', red_ex)]:
                if not exids:
                    continue
                rep = self.add(f"IFCSHAPEREPRESENTATION(#{body_ctx},'Body','SweptSolid',("
                               + ','.join(f'#{i}' for i in exids) + '))')
                pds = self.add(f'IFCPRODUCTDEFINITIONSHAPE($,$,(#{rep}))')
                ppl = self.add(f'IFCLOCALPLACEMENT(#{spl},#{world})')
                px = self.add(f"IFCBUILDINGELEMENTPROXY('{guid()}',#{oh},'{label} {disp}',$,$,#{ppl},#{pds},$,.NOTDEFINED.)")
                proxies.append(px)
            contained.append((st, proxies))

        # --- aggregaties ---
        self.add(f"IFCRELAGGREGATES('{guid()}',#{oh},$,$,#{proj},(#{site}))")
        self.add(f"IFCRELAGGREGATES('{guid()}',#{oh},$,$,#{site},(#{bld}))")
        self.add(f"IFCRELAGGREGATES('{guid()}',#{oh},$,$,#{bld},("
                 + ','.join(f'#{i}' for i in storey_ids) + '))')
        for st, proxies in contained:
            if proxies:
                self.add(f"IFCRELCONTAINEDINSPATIALSTRUCTURE('{guid()}',#{oh},$,$,("
                         + ','.join(f'#{i}' for i in proxies) + f'),#{st})')

        # --- file ---
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        header = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('ViewDefinition [ReferenceView_V1.2]'),'2;1');\n"
            f"FILE_NAME('{os.path.basename(path)}','{now}',(''),(''),"
            "'EDGEDETECT','EDGEDETECT sweep','');\n"
            "FILE_SCHEMA(('IFC4'));\n"
            "ENDSEC;\n"
            "DATA;\n"
        )
        with open(path, 'w') as f:
            f.write(header)
            f.write('\n'.join(self.lines))
            f.write('\nENDSEC;\nEND-ISO-10303-21;\n')
        return self.id


def build(models, out_path):
    all_segs = []
    for label, fp in models:
        segs = process_model(fp)
        for s in segs:
            s['model'] = label
        all_segs += segs
        print(f'  {label}: {len(segs)} segmenten')
    w = IfcWriter()
    n = w.write(out_path, all_segs)
    print(f'  -> {out_path}  ({n} entiteiten, {len(all_segs)} segmenten)')
    return all_segs



# ===========================================================================
#  VERIFICATIE-RENDER (optioneel, --verify)
# ===========================================================================

def panel(box):
    ht=_here_top_map(box); fm=ht>_NEG; h,w=fm.shape
    img=Image.new('RGB',(w,h),(15,25,35))
    arr=np.array(img); arr[fm]=(70,80,92); img=Image.fromarray(arr)
    d=ImageDraw.Draw(img,'RGBA')
    for sg in segments_for_box(box):
        lx,ly,_=sg['loc']; ax,ay,_=sg['axis']; nx,ny,_=sg['ref']; L=sg['depth']
        ex,ey=lx+ax*L, ly+ay*L
        # rood vlak (buiten): loc, loc-900n, +run
        rb=[(lx,ly),(lx-nx*900,ly-ny*900),(ex-nx*900,ey-ny*900),(ex,ey)]
        d.polygon([world_to_px(x,y,box) for x,y in rb],fill=(230,60,60,150))
        # groen scherm (binnen, 149 dik) op de rand
        gb=[(lx,ly),(lx+nx*149,ly+ny*149),(ex+nx*149,ey+ny*149),(ex,ey)]
        d.polygon([world_to_px(x,y,box) for x,y in gb],fill=(40,210,90,220))
    return img

def panels(imgs,labels,title,out):
    GAP,LBL,TT,PAD=18,20,30,14
    sc=min(1.0, 1100/max(im.width for im in imgs))
    imgs=[im.resize((int(im.width*sc),int(im.height*sc))) for im in imgs]
    tw=sum(im.width for im in imgs)+GAP*(len(imgs)-1)+PAD*2
    th=TT+LBL+max(im.height for im in imgs)+PAD*2
    cv=Image.new('RGB',(tw,th),(17,17,17)); dr=ImageDraw.Draw(cv)
    dr.text((tw//2,PAD),title,fill=(210,210,210),anchor='mt')
    x=PAD; y=TT+LBL+PAD
    for im,lb in zip(imgs,labels):
        dr.text((x+im.width//2,TT+PAD),lb,fill=(200,200,200),anchor='mt')
        cv.paste(im,(x,y)); x+=im.width+GAP
    cv.save(out); return out



# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT (sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def render_verification(models, out_dir):
    for lbl, fp in models:
        ifc = IFC(fp); st = read_storeys(ifc); fl = read_floors(ifc); link_to_storeys(ifc, fl, st)
        for f in fl: build_floor_rings(ifc, f)
        boxes = storey_scope_boxes(fl, st)
        imgs = [panel(b) for b in boxes]; labels = [b['storey']['name'] for b in boxes]
        if imgs:
            p = os.path.join(out_dir, f'verify_{lbl}.png')
            panels(imgs, labels,
                   f'{lbl} - groen=randbeveiliging op rand, rood=valgevaar-markering buiten vloerveld', p)
            print(f'      verificatie -> {p}')


def sweep_main():
    import argparse
    ap = argparse.ArgumentParser(description='Randbeveiliging sweepen langs valranden -> IFC4')
    ap.add_argument('ifc', nargs='+', help='een of meer bron-IFC bestanden')
    ap.add_argument('-o', '--out', default='VLW_DO_Valgevaar.ifc', help='uitvoer IFC4')
    ap.add_argument('--verify', action='store_true', help='verificatie-PNG per model renderen')
    args = ap.parse_args()
    models = [(os.path.splitext(os.path.basename(p))[0].split('_')[0], p) for p in args.ifc]
    print(f'[1/2] Sweepen ({len(models)} model(len))...')
    build(models, args.out)
    if args.verify:
        print('[2/2] Verificatie renderen...')
        render_verification(models, os.path.dirname(os.path.abspath(args.out)) or '.')
    print('Klaar.')


if __name__ == '__main__':
    sweep_main()