#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  CV BRIDGE ANALYSIS GENERATOR · v2.0
  Dr. Ing. Luis Rojas Valdivia · GMT Group / Innovaminer · CODELCO
================================================================================

Generador de reportes interactivos HTML autocontenidos a partir de archivos
Excel de inspección estructural de puentes apiladores (formato RESUMEN_GRIETAS).

Pipeline completo:
  1) Parsing del Excel (detección de SECCION X y patrón ítem N.M)
  2) Feature engineering por regex (tipo_falla, componente, zona, recurrencia)
  3) PCA via SVD sobre matriz one-hot estandarizada
  4) NHPP Bayesiano por sección (conjugación Gamma-Poisson, α=2, β=2)
  5) Scoring multicriterio (default + juicio experto GMT) + buckets temporales
  6) CPM scheduling con asignación greedy a N cuadrillas
  7) Geometría procedural 3D para el visor Three.js
  8) Inyección en plantilla HTML (Three.js + jsPDF + I_RE)

DEPENDENCIAS:
    pip install pandas numpy openpyxl

USO:
    python cv_bridge_generator.py            # GUI
    python cv_bridge_generator.py archivo.xlsx [--out salida.html] [--crews 3]

================================================================================
"""

import sys
import os
import re
import json
import math
import gzip
import base64
import argparse
import threading
import traceback
import warnings
import webbrowser
from pathlib import Path
from datetime import datetime
from collections import Counter

def _configure_console_encoding():
    """Keep argparse/log output usable on Windows consoles with legacy code pages."""
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, 'reconfigure'):
            continue
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

_configure_console_encoding()

# Silence noisy warnings from pandas date parsing
warnings.filterwarnings('ignore', category=UserWarning, module='dateutil')
warnings.filterwarnings('ignore', message='.*dayfirst.*')
warnings.filterwarnings('ignore', message='.*Could not infer format.*')

# ---------------- DEPENDENCIES ----------------
_DEPS_OK = True
_DEPS_MSG = ""
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    _DEPS_OK = False
    _DEPS_MSG = f"Falta dependencia: {e.name}\n\nInstale con:\n  pip install pandas numpy openpyxl"

# tkinter (built-in)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    _TK_OK = True
except ImportError:
    _TK_OK = False


# ================================================================================
# FEA STRUCTURAL MODEL — embedded (gzip+base64 compressed)
# Real geometry from Autodesk Robot / RFEM CSV exports:
#   858 nodes · 1206 bars · 8 section types · Y-axis span 369.7 m
# Auto-injected into HTML at generation time as `const FEA_MODEL = ...`
# ================================================================================
FEA_MODEL_GZB64 = (
    "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"
)


def get_fea_model_json():
    """Return the decompressed FEA model as a JSON string (ready to inject)."""
    return gzip.decompress(base64.b64decode(FEA_MODEL_GZB64)).decode('utf-8')


# ==============================================================================
# HTML TEMPLATE — embedded (gzip+base64 compressed, ~33 KB)
# Decoded at runtime; placeholder __DATA_JSON__ is replaced with the dataset.
# ================================================================================
TEMPLATE_GZB64 = (
    "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"
    "Hf+idfXm0v8BPEZtF2GjAwA="
)


def get_template():
    """Return the decompressed HTML template string."""
    return gzip.decompress(base64.b64decode(TEMPLATE_GZB64)).decode('utf-8')


# ================================================================================
# REGEX PATTERNS — feature engineering
# ================================================================================
RE_SECTION = re.compile(r'^\s*SECCI[OÓ]N\s+(\d+)\s*$', re.IGNORECASE)
RE_ITEM    = re.compile(r'^\s*(\d+)\.(\d+)\s*$')

# Component patterns (priority-ordered: more specific first)
COMP_PATTERNS = [
    ('Tirante',             r'tirante'),
    ('Cuerda superior',     r'cuerda\s*sup'),
    ('Cuerda inferior',     r'cuerda\s*inf'),
    ('Viga long. sup.',     r'viga\s*long\w*\s*sup'),
    ('Viga long. inf.',     r'viga\s*long\w*\s*inf'),
    ('Susp. estructural',   r'susp(?:en|\.|\b)'),
    ('Tubular diagonal',    r'tubular\s*diag|diagonal\b'),
    ('Tubular transversal', r'tubular\s*transv|transvers'),
    ('Tubular horizontal',  r'tubular\s*horiz|horizont'),
    ('Perfil vertical',     r'perfil\s*vert|montante'),
    ('Corbata oruga',       r'corbata\s*oruga|corbata'),
    ('Arriostramiento',     r'arriostr|riostr'),
]

# Failure type patterns
FALLA_PATTERNS = [
    ('Grieta pasante',  r'grieta\s*pasante|pasante'),
    ('Falla soldadura', r'fall[ae]?\s*sold|defect\w*\s*sold|sold\w*\s*defect|cordon\s*sold'),
    ('Rotura',          r'rotur|fractur'),
    ('Grieta',          r'grieta'),
    ('Fisura',          r'fisura'),
]

# Zone patterns
ZONA_PATTERNS = [
    ('Cabezal cabeza', r'cabezal\s*cabeza'),
    ('Cabezal cola',   r'cabezal\s*cola'),
    ('Sobre oruga',    r'sobre\s*oruga|oruga'),
    ('Semi-sección',   r'semi\W*secci'),
]

# Recurrence
RE_RECURRENCIA = re.compile(
    r'reparaci[oó]n\s*anterior|se\s*repite|reaparec|nuevamente|otra\s*vez|recurr',
    re.IGNORECASE
)


# ================================================================================
# PARSING & FEATURE ENGINEERING
# ================================================================================
# Accept multiple weekly-sheet naming conventions:
#   SEM18 · SEM 18 · SEMANA18 · SEMANA 18 · S18 · W18 · SEM-18 · SEM_18
RE_WEEK_SHEET = re.compile(r'^\s*(?:SEM(?:ANA)?|S|W)\s*[-_ ]?\s*(\d+)\s*$', re.IGNORECASE)


def clean_zws(s):
    """Strip zero-width spaces (U+200B), NBSP (U+00A0), and trim. Idempotent on non-strings."""
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    # Remove zero-width spaces, zero-width non-joiners, BOM, NBSP, etc.
    for ch in ('\u200b', '\u200c', '\u200d', '\ufeff', '\u00a0'):
        s = s.replace(ch, ' ' if ch == '\u00a0' else '')
    return s.strip()


def detect_week_sheets(xl):
    """Return list of (sheet_name, week_number) sorted ascending by week.
    Sheet names are cleaned of invisible characters before matching, so
    polluted tabs (e.g. 'SEM19\u200b') are still detected."""
    found = []
    for sh in xl.sheet_names:
        clean_name = clean_zws(str(sh))
        m = RE_WEEK_SHEET.match(clean_name)
        if m:
            found.append((sh, int(m.group(1))))   # keep ORIGINAL name for pandas access
    return sorted(found, key=lambda x: x[1])


def parse_excel(filepath, log=print, week_sheet=None):
    """Parse a single sheet (or auto-detect). Returns the raw DataFrame.
    If `week_sheet` is given, that sheet is used; otherwise the first sheet."""
    log(f"📂 Leyendo: {filepath}")
    xl = pd.ExcelFile(filepath)
    sheet_name = week_sheet or xl.sheet_names[0]
    df_raw = xl.parse(sheet_name, header=None)
    log(f"   Hoja: '{sheet_name}' · {len(df_raw)} filas")

    # Detect header row (look for 'observ' + ('aviso'|'recomend'|'criticidad'))
    header_idx = None
    for i, row in df_raw.iterrows():
        cells = [clean_zws(str(c)).lower() for c in row if pd.notna(c)]
        joined = ' '.join(cells)
        if 'observ' in joined and any(k in joined for k in ('aviso', 'recomend', 'criticidad')):
            header_idx = i
            break
    if header_idx is None:
        log("   ⚠ No se detectó header — asumiendo fila 3")
        header_idx = 3
    log(f"   Header detectado en fila {header_idx}")

    df = xl.parse(sheet_name, header=header_idx)
    df.columns = [clean_zws(str(c)) for c in df.columns]
    log(f"   Columnas: {list(df.columns)[:8]}")
    return df


def parse_excel_multi(filepath, log=print):
    """Auto-detect SEM<n> sheets and parse each one. Returns dict {week_name: records}.
    If no SEM sheets are found, falls back to single-sheet mode using the first sheet."""
    xl = pd.ExcelFile(filepath)
    weeks = detect_week_sheets(xl)
    weekly_records = {}

    if not weeks:
        log(f"   ⓘ Sin pestañas SEM<n> · modo single-sheet")
        df = parse_excel(filepath, log=log)
        records = enrich_features(df, log=log)
        # Clean ZWS in records too (some sheets pollute cells)
        for r in records:
            for k, v in list(r.items()):
                r[k] = clean_zws(v) if isinstance(v, str) else v
        weekly_records['__single__'] = records
        return weekly_records

    log(f"📅 Pestañas semanales detectadas: {[clean_zws(str(s[0])) for s in weeks]}")
    for sh, wn in weeks:
        # Canonical week label: SEM<n> regardless of how the tab was named
        week_label = f"SEM{wn}"
        log(f"\n── Procesando '{clean_zws(str(sh))}' → {week_label} ──────────────────────────")
        df = parse_excel(filepath, log=log, week_sheet=sh)
        records = enrich_features(df, log=log)
        # Clean ZWS from all string fields in records (SEM19 has \u200b everywhere)
        for r in records:
            for k, v in list(r.items()):
                r[k] = clean_zws(v) if isinstance(v, str) else v
        weekly_records[week_label] = records
        log(f"   ✓ {week_label}: {len(records)} ítems")

    return weekly_records


def find_col(df, *keywords):
    """Find a column whose name contains any of the keywords (case-insensitive)."""
    for col in df.columns:
        cl = str(col).lower()
        for kw in keywords:
            if kw.lower() in cl:
                return col
    return None


def parse_estado(val):
    """Map 'Recomendación' raw value to canonical estado."""
    if val is None: return 'N/D'
    s = str(val).strip().lower()
    if not s or s in ('nan', 'none', '-'): return 'N/D'
    if s.startswith('repar') and any(s.endswith(suf) for suf in ('do', 'da', 'do ', 'da ', 'rado')):
        return 'Reparado'
    if 'reparado' in s or 'reparada' in s or 'repararado' in s:
        return 'Reparado'
    if s.startswith('repar'):     # Reparar / Repara / Reparar  → pending action
        return 'Pendiente'
    return s.capitalize()


def parse_criticidad(val):
    """Parse 'Nivel N' or numeric → int; '-' or empty → None."""
    if val is None: return None
    s = str(val).strip()
    if not s or s in ('nan', 'None', '-'): return None
    m = re.search(r'(\d+)', s)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def enrich_features(df, log=print):
    """Detect SECCION + N.M items, extract regex features. Returns list of dicts."""
    log("🔍 Engineering de variables (regex sobre observación)...")

    # The ITEM column contains both 'SECCION X' headers and 'N.M' item codes.
    # Find it by looking for any column whose name is 'ITEM' or whose values match the pattern.
    col_item = find_col(df, 'item') or df.columns[1] if len(df.columns) > 1 else df.columns[0]
    col_obs    = find_col(df, 'observ')
    col_aviso  = find_col(df, 'aviso')
    col_lado   = find_col(df, 'lado', 'cara')
    col_fecha  = find_col(df, 'fecha aviso', 'fecha de aviso', 'f. aviso', 'fecha')
    # 'Recomendación' or 'Estado' or 'Acción' — it's the workflow status
    col_estado = find_col(df, 'estado', 'recomend', 'acci', 'status')
    col_crit   = find_col(df, 'critic', 'nivel')
    col_repar  = find_col(df, 'reparac', 'fecha repar')

    log(f"   Cols: item={col_item} · obs={col_obs} · estado={col_estado} · crit={col_crit}")

    records = []
    seen_items = set()       # avoid duplicate item codes within one sheet (SEM19 has some dupes)
    current_section = None

    for idx, row in df.iterrows():
        raw_item = row[col_item]
        if pd.isna(raw_item):
            continue
        item_val = clean_zws(str(raw_item))

        # SECCION header?
        m_sec = RE_SECTION.match(item_val)
        if m_sec:
            current_section = int(m_sec.group(1))
            continue

        # Skip repeated header row that says "ITEM"
        if item_val.lower() in ('item', 'ítem', 'itém'):
            continue

        # Item N.M?
        m_item = RE_ITEM.match(item_val)
        if not m_item:
            continue

        # Skip duplicate items within the same sheet
        if item_val in seen_items:
            continue
        seen_items.add(item_val)

        seccion = int(m_item.group(1))
        if current_section is None:
            current_section = seccion

        crit_int = parse_criticidad(clean_zws(row[col_crit])) if col_crit else None

        rec = {
            'seccion': seccion,
            'item': item_val,
            'lado': clean_zws(str(row[col_lado])) if col_lado and pd.notna(row[col_lado]) else None,
            'observacion': clean_zws(str(row[col_obs])) if col_obs and pd.notna(row[col_obs]) else '',
            'aviso': clean_zws(str(row[col_aviso])) if col_aviso and pd.notna(row[col_aviso]) else None,
            'fecha': clean_zws(str(row[col_fecha])) if col_fecha and pd.notna(row[col_fecha]) else None,
            'estado': parse_estado(clean_zws(row[col_estado])) if col_estado else 'N/D',
            'criticidad_nivel': float(crit_int) if crit_int is not None else None,
            'fecha_reparacion': clean_zws(str(row[col_repar])) if col_repar and pd.notna(row[col_repar]) else None,
        }

        # Regex features
        obs = rec['observacion'].lower()

        rec['tipo_falla'] = 'Otro'
        for tipo, pat in FALLA_PATTERNS:
            if re.search(pat, obs):
                rec['tipo_falla'] = tipo
                break

        rec['componente'] = 'Otro'
        for comp, pat in COMP_PATTERNS:
            if re.search(pat, obs):
                rec['componente'] = comp
                break

        rec['zona'] = 'General'
        for z, pat in ZONA_PATTERNS:
            if re.search(pat, obs):
                rec['zona'] = z
                break

        rec['recurrencia'] = 1 if RE_RECURRENCIA.search(obs) else 0

        # Date parsing
        rec['fecha_dt'] = None
        rec['year_month'] = None
        rec['year'] = None
        if rec['fecha']:
            try:
                d = pd.to_datetime(rec['fecha'], errors='coerce', dayfirst=True)
                if pd.notna(d):
                    rec['fecha_dt']   = d.isoformat()
                    rec['year_month'] = d.strftime('%Y-%m')
                    rec['year']       = int(d.year)
            except Exception:
                pass

        # priority_score placeholder, computed later
        rec['priority_score'] = None
        records.append(rec)

    log(f"   ✓ {len(records)} ítems extraídos")
    # Quick stats
    ests = Counter(r['estado'] for r in records)
    log(f"   Estados: {dict(ests)}")
    return records


# ================================================================================
# PCA via SVD
# ================================================================================
def compute_pca(records, log=print):
    """One-hot + z-score standardize + SVD."""
    log("📊 PCA via SVD...")
    valid = [r for r in records if r['estado'] != 'N/D']
    if not valid:
        log("   ⚠ Sin registros válidos")
        return {}

    df = pd.DataFrame(valid)
    cats = [c for c in ('estado', 'lado', 'tipo_falla', 'componente', 'zona') if c in df.columns]
    df_cats = pd.get_dummies(df[cats].fillna('NA'), drop_first=False)

    # Numeric
    num_cols = []
    if 'criticidad_nivel' in df.columns:
        df['crit_num'] = df['criticidad_nivel'].fillna(df['criticidad_nivel'].median())
        num_cols.append('crit_num')
    if 'recurrencia' in df.columns: num_cols.append('recurrencia')
    if 'seccion' in df.columns:     num_cols.append('seccion')

    X = pd.concat([df_cats.astype(float), df[num_cols].astype(float)], axis=1).fillna(0)

    # Standardize (z-score)
    Xn = (X - X.mean()) / (X.std() + 1e-9)
    Xn = Xn.fillna(0).values

    # SVD
    try:
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    except Exception as e:
        log(f"   ⚠ SVD falló: {e}")
        return {}

    explained = (S**2) / (S**2).sum()
    scores = U * S
    feat_names = list(X.columns)
    n_pc = min(5, scores.shape[1])

    loadings_pc1 = sorted(
        [[feat_names[i], float(Vt[0, i])] for i in range(len(feat_names))],
        key=lambda x: -abs(x[1])
    )[:15]
    loadings_pc2 = sorted(
        [[feat_names[i], float(Vt[1, i])] for i in range(len(feat_names))],
        key=lambda x: -abs(x[1])
    )[:15]

    pca = {
        'pc_scores': scores[:, :n_pc].tolist(),
        'pc_seccion':    [r['seccion'] for r in valid],
        'pc_criticidad': [r.get('criticidad_nivel') for r in valid],
        'pc_componente': [r['componente'] for r in valid],
        'pc_tipo_falla': [r['tipo_falla'] for r in valid],
        'explained_var': explained.tolist(),
        'loadings_pc1':  loadings_pc1,
        'loadings_pc2':  loadings_pc2,
        'all_cols':      feat_names,
        'pc_labels':     [f'PC{i+1}' for i in range(n_pc)],
    }
    log(f"   PC1+PC2: {(explained[0]+explained[1])*100:.1f}% var · acum PC1-5: {sum(explained[:5])*100:.1f}%")
    return pca


# ================================================================================
# Bayesian NHPP — Gamma-Poisson conjugate
# ================================================================================
def compute_forecast(records, alpha=2, beta=2, log=print):
    """λ posterior = (α+n)/(β+T) per section."""
    log(f"🔮 NHPP Bayesiano · prior Gamma(α={alpha}, β={beta})")

    years = [r['year'] for r in records if r.get('year')]
    years_span = (max(years) - min(years) + 1) if years else 1.65

    forecast = {}
    for s in range(1, 11):
        n = sum(1 for r in records if r['seccion'] == s)
        lam = (alpha + n) / (beta + years_span)
        forecast[str(s)] = {
            'lambda_anual': round(float(lam), 2),
            'n_observed':   int(n),
            'years_span':   round(float(years_span), 2)
        }
    log(f"   λ máx: S{max(forecast, key=lambda k: forecast[k]['lambda_anual'])} = "
        f"{max(f['lambda_anual'] for f in forecast.values()):.2f}/año")
    return forecast


# ================================================================================
# Multicriteria Scoring + buckets
# ================================================================================
COMP_WEIGHTS_DEF = {
    'Tirante': 0.95, 'Cuerda superior': 0.95, 'Cuerda inferior': 0.95,
    'Viga long. inf.': 1.00, 'Viga long. sup.': 1.00,
    'Susp. estructural': 0.90,
    'Tubular diagonal': 0.80, 'Tubular transversal': 0.75, 'Tubular horizontal': 0.70,
    'Perfil vertical': 0.70, 'Corbata oruga': 0.70,
    'Arriostramiento': 0.60, 'Otro': 0.50
}
FALLA_WEIGHTS_DEF = {
    'Grieta pasante': 1.00, 'Rotura': 0.95, 'Falla soldadura': 0.85,
    'Grieta': 0.70, 'Fisura': 0.50, 'Otro': 0.40
}

# Expert judgment (GMT senior engineer calibration)
EXPERT_WEIGHTS = {
    'w_crit': 0.30, 'w_rec': 0.25, 'w_comp': 0.30, 'w_falla': 0.15,
    'comp_weights': {
        'Tirante': 1.00, 'Cuerda superior': 0.98, 'Cuerda inferior': 0.97,
        'Viga long. inf.': 0.97, 'Viga long. sup.': 0.92,
        'Susp. estructural': 0.95,
        'Tubular diagonal': 0.85, 'Tubular transversal': 0.78, 'Tubular horizontal': 0.75,
        'Perfil vertical': 0.72, 'Corbata oruga': 0.80,
        'Arriostramiento': 0.65, 'Otro': 0.55
    },
    'falla_weights': {
        'Grieta pasante': 1.00, 'Rotura': 0.98, 'Falla soldadura': 0.92,
        'Grieta': 0.75, 'Fisura': 0.55, 'Otro': 0.40
    },
    'bonus_zona': {
        'Cabezal cabeza': 8, 'Cabezal cola': 5, 'Sobre oruga': 4,
        'Semi-sección': 2, 'General': 0
    }
}


def compute_score(rec, w, comp_w, falla_w, bonus_zona=None):
    """Score = 100·Σ wᵢ·xᵢ + bonus(zona)."""
    crit = rec.get('criticidad_nivel') or 4.0
    w_crit  = (7 - crit) / 6
    w_rec   = 1.0 if rec.get('recurrencia') else 0.0
    w_comp  = comp_w.get(rec.get('componente', 'Otro'), 0.5)
    w_falla = falla_w.get(rec.get('tipo_falla', 'Otro'), 0.4)
    p = 100 * (w['wc']*w_crit + w['wr']*w_rec + w['wk']*w_comp + w['wf']*w_falla)
    if bonus_zona:
        p += bonus_zona.get(rec.get('zona', 'General'), 0)
    return round(float(p), 2)


def bucket_from_score(p):
    if p >= 80: return 'INMEDIATO (≤7d)'
    if p >= 65: return 'CORTO (≤30d)'
    if p >= 50: return 'MEDIO (≤90d)'
    return 'PROGRAMABLE (>90d)'


def compute_scoring(records, log=print):
    """Score all 'pendiente' records with default weights."""
    log("⚖️  Scoring multicriterio (default)...")
    weights = {'wc': 0.40, 'wr': 0.20, 'wk': 0.25, 'wf': 0.15}

    pendientes = [r for r in records if str(r.get('estado', '')).lower().startswith('pendient')]

    scoring = []
    for r in pendientes:
        s = compute_score(r, weights, COMP_WEIGHTS_DEF, FALLA_WEIGHTS_DEF)
        r['priority_score'] = s
        scoring.append({
            'item':        r['item'],
            'seccion':     r['seccion'],
            'lado':        r.get('lado'),
            'componente':  r.get('componente'),
            'tipo_falla':  r.get('tipo_falla'),
            'criticidad_nivel': r.get('criticidad_nivel'),
            'recurrencia': r.get('recurrencia', 0),
            'priority_score': s,
            'bucket':         bucket_from_score(s),
            'observacion':    r.get('observacion', '')
        })
    scoring.sort(key=lambda x: -x['priority_score'])
    log(f"   {len(scoring)} ítems pendientes puntuados")
    return scoring


# ================================================================================
# CPM scheduling — greedy assignment
# ================================================================================
def compute_cpm(scoring, n_crews=3, hours_per_repair=8, cost_per_repair=7000, log=print):
    log(f"⏱️  CPM · {n_crews} cuadrillas · {hours_per_repair}h/reparación")
    crews_finish = [0] * n_crews
    schedule = []
    for r in scoring:
        ci = min(range(n_crews), key=lambda i: crews_finish[i])
        start = crews_finish[ci]
        finish = start + hours_per_repair
        crews_finish[ci] = finish
        schedule.append({
            'item': r['item'], 'seccion': r['seccion'],
            'componente': r['componente'], 'criticidad': r['criticidad_nivel'],
            'tipo': r['tipo_falla'], 'lado': r['lado'],
            'priority': r['priority_score'],
            'start_hr': start, 'finish_hr': finish,
            'crew': ci + 1,
            'start_day': start / 8.0, 'finish_day': finish / 8.0
        })
    makespan_hr = max(crews_finish)
    cpm = {
        'schedule': schedule,
        'critical_path_items': [s['item'] for s in schedule[:5]],
        'critical_path_crew': 1,
        'makespan_hr': makespan_hr,
        'makespan_dias': makespan_hr / 8.0,
        'total_items': len(scoring),
        'horas_total': len(scoring) * hours_per_repair,
        'costo_total': len(scoring) * cost_per_repair,
        'cuadrillas': n_crews,
        'edt': []
    }
    log(f"   Makespan: {cpm['makespan_dias']:.1f} días · Costo directo: ${cpm['costo_total']:,}")
    return cpm


# ================================================================================
# Active defects S2-S9 + Geometry
# ================================================================================
def compute_active_defects(scoring, records, log=print):
    """Filter pending defects in S2-S9, compute expert score & delta."""
    log("🎯 Defectos activos S2-S9 + juicio experto...")
    by_item = {r['item']: r for r in records}
    expert_w = {'wc': EXPERT_WEIGHTS['w_crit'], 'wr': EXPERT_WEIGHTS['w_rec'],
                'wk': EXPERT_WEIGHTS['w_comp'], 'wf': EXPERT_WEIGHTS['w_falla']}
    active = []
    for s in scoring:
        if not (2 <= s['seccion'] <= 9):
            continue
        r = by_item.get(s['item'], {})
        zona = r.get('zona', 'General')
        es = compute_score(
            r, expert_w,
            EXPERT_WEIGHTS['comp_weights'], EXPERT_WEIGHTS['falla_weights'],
            EXPERT_WEIGHTS['bonus_zona']
        )
        active.append({
            **s,
            'zona': zona,
            'expert_score': es,
            'delta': round(es - s['priority_score'], 2)
        })
    active.sort(key=lambda x: -x['priority_score'])
    log(f"   {len(active)} defectos activos S2-S9")
    return active


def compute_geometry(scoring, n_sections=10):
    """3D geometry coordinates: each section's length proportional to defect count."""
    sec_count = Counter(s['seccion'] for s in scoring if 2 <= s['seccion'] <= 9)
    geometry = []
    x = 0.0
    for s in range(1, n_sections + 1):
        n = sec_count.get(s, 0)
        length = 6.0 + (n / 30.0) * 4.0
        geometry.append({
            'seccion':  s,
            'x_start':  round(x, 3),
            'x_end':    round(x + length, 3),
            'length':   round(length, 2),
            'n_defects': int(n)
        })
        x += length + 0.5
    return geometry, {str(k): int(v) for k, v in sec_count.items()}


# ================================================================================
# WEEKLY EVOLUTION ANALYSIS
# ================================================================================
def compute_weekly_evolution(weekly_records, log=print):
    """Compare consecutive weekly snapshots. Returns evolution dict with:
       - weeks: ordered list of week names
       - summaries: per-week aggregates (n_total, by_state, by_section, avg_priority, lambda_avg)
       - transitions: between-week deltas (new, removed, repaired, recurrent, crit_changes)
       - lambda_timeline: per-section lambda values per week
    """
    week_names = sorted(
        weekly_records.keys(),
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    if len(week_names) < 1 or week_names == ['__single__']:
        log("📊 Sin análisis evolutivo (una sola pestaña)")
        return None

    log("=" * 64)
    log(f"📊 ANÁLISIS EVOLUTIVO · {' → '.join(week_names)}")
    log("=" * 64)

    # Per-week summaries
    summaries = {}
    lambda_timeline = {}
    weights_def = {'wc': 0.40, 'wr': 0.20, 'wk': 0.25, 'wf': 0.15}
    for w in week_names:
        recs = weekly_records[w]
        by_state = Counter(r['estado'] for r in recs)
        by_sec   = Counter(r['seccion'] for r in recs)
        by_sec_pend = Counter(r['seccion'] for r in recs if r['estado'] == 'Pendiente')
        # Avg priority on pending
        priors = []
        for r in recs:
            if r['estado'] == 'Pendiente':
                p = compute_score(r, weights_def, COMP_WEIGHTS_DEF, FALLA_WEIGHTS_DEF)
                priors.append(p)
        # Per-section lambda for this week
        years_obs = [r['year'] for r in recs if r.get('year')]
        T = (max(years_obs) - min(years_obs) + 1) if years_obs else 1.65
        lam_by_sec = {}
        for s in range(1, 11):
            n_s = sum(1 for r in recs if r['seccion'] == s and r['estado'] == 'Pendiente')
            lam_by_sec[s] = round((2 + n_s) / (2 + T), 2)   # Gamma(α=2,β=2) prior
        lambda_timeline[w] = lam_by_sec
        # Recurrencia
        n_recur = sum(1 for r in recs if r.get('recurrencia'))
        summaries[w] = {
            'n_total':     len(recs),
            'n_pendiente': by_state.get('Pendiente', 0),
            'n_reparado':  by_state.get('Reparado', 0),
            'n_recurrent': n_recur,
            'by_state':    dict(by_state),
            'by_section':  {str(k): int(v) for k, v in by_sec.items()},
            'by_section_pendiente': {str(k): int(v) for k, v in by_sec_pend.items()},
            'avg_priority': round(sum(priors) / len(priors), 2) if priors else 0,
            'max_lambda':   max(lam_by_sec.values()),
            'sum_lambda':   round(sum(lam_by_sec.values()), 2)
        }
        log(f"   {w}: n_total={len(recs)} · pendiente={by_state.get('Pendiente',0)} · "
            f"reparado={by_state.get('Reparado',0)} · Σλ={summaries[w]['sum_lambda']}")

    # Transitions (consecutive weeks)
    transitions = []
    for i in range(len(week_names) - 1):
        w_prev, w_curr = week_names[i], week_names[i + 1]
        prev_map = {r['item']: r for r in weekly_records[w_prev]}
        curr_map = {r['item']: r for r in weekly_records[w_curr]}
        keys_prev, keys_curr = set(prev_map.keys()), set(curr_map.keys())

        new_items = sorted(keys_curr - keys_prev)
        removed   = sorted(keys_prev - keys_curr)
        shared    = keys_curr & keys_prev

        repaired = [k for k in shared
                    if prev_map[k]['estado'] == 'Pendiente' and curr_map[k]['estado'] == 'Reparado']
        recurrent = [k for k in shared
                     if prev_map[k]['estado'] == 'Reparado' and curr_map[k]['estado'] == 'Pendiente']

        # Criticidad changes (lower Nv = MORE critical)
        crit_chg = []
        for k in shared:
            cp, cc = prev_map[k].get('criticidad_nivel'), curr_map[k].get('criticidad_nivel')
            if cp is not None and cc is not None and cp != cc:
                crit_chg.append({
                    'item': k, 'seccion': curr_map[k]['seccion'],
                    'from': cp, 'to': cc, 'componente': curr_map[k].get('componente'),
                    'direction': 'AGRAVADO' if cc < cp else 'MITIGADO'
                })

        # Per-section delta of pending
        sec_prev = Counter(r['seccion'] for r in weekly_records[w_prev] if r['estado'] == 'Pendiente')
        sec_curr = Counter(r['seccion'] for r in weekly_records[w_curr] if r['estado'] == 'Pendiente')
        sec_delta = {str(s): int(sec_curr.get(s, 0) - sec_prev.get(s, 0)) for s in range(1, 11)}

        def _detail(rec, src_map=None):
            base = {
                'item': rec.get('item'),
                'seccion': rec.get('seccion'),
                'componente': rec.get('componente'),
                'tipo_falla': rec.get('tipo_falla'),
                'zona': rec.get('zona'),
                'lado': rec.get('lado'),
                'criticidad_nivel': rec.get('criticidad_nivel'),
                'aviso': rec.get('aviso'),
                'fecha': rec.get('fecha_dt') or rec.get('fecha'),
                'estado_prev': rec.get('estado'),
                'observacion': (rec.get('observacion') or '')[:160]
            }
            return base

        transitions.append({
            'from': w_prev, 'to': w_curr,
            'n_new': len(new_items),
            'n_removed': len(removed),
            'n_repaired': len(repaired),
            'n_recurrent': len(recurrent),
            'n_crit_changes': len(crit_chg),
            'new_items':       [_detail(curr_map[k]) for k in new_items],
            'removed_items':   [_detail(prev_map[k]) for k in removed],
            'repaired_items':  [_detail(curr_map[k]) for k in repaired],
            'recurrent_items': [_detail(curr_map[k]) for k in recurrent],
            'crit_changes':    crit_chg[:50],
            'sec_delta': sec_delta
        })
        log(f"   {w_prev}→{w_curr}: +{len(new_items)} nuevos · −{len(removed)} removidos · "
            f"✓{len(repaired)} reparados · ↻{len(recurrent)} recurrentes · "
            f"Δcrit={len(crit_chg)}")

    return {
        'weeks': week_names,
        'summaries': summaries,
        'transitions': transitions,
        'lambda_timeline': lambda_timeline,
        'current_week':  week_names[-1],
        'previous_week': week_names[-2] if len(week_names) > 1 else None
    }


# ================================================================================
# PIPELINE ASSEMBLY
# ================================================================================
def build_data(filepath, bridge_id=None, n_crews=3, hours_per_repair=8,
               cost_per_repair=7000, alpha=2, beta=2, log=print):
    """Run full pipeline. Multi-sheet aware (SEM<n>). Returns dict ready for HTML injection."""
    log("=" * 64)
    log(f"  PIPELINE · {filepath}")
    log("=" * 64)

    # --- Multi-sheet parsing ---
    weekly_records = parse_excel_multi(filepath, log)
    if not weekly_records:
        raise ValueError("No se extrajeron registros — verifique formato del Excel.")

    # Pick latest week as "current state" for the rest of the pipeline
    week_keys = list(weekly_records.keys())
    if week_keys == ['__single__']:
        current_week = None
        records = weekly_records['__single__']
        log(f"\n📌 Modo single-sheet · {len(records)} ítems")
    else:
        # Sort by week number ascending; the LAST is the most recent
        sorted_weeks = sorted(week_keys, key=lambda x: int(re.search(r'\d+', x).group()))
        current_week = sorted_weeks[-1]
        records = weekly_records[current_week]
        log(f"\n📌 Semana actual = {current_week} · {len(records)} ítems")

    # --- Weekly evolution analysis ---
    evolution = compute_weekly_evolution(weekly_records, log) if current_week else None

    # --- Standard pipeline on current state ---
    pca      = compute_pca(records, log)
    forecast = compute_forecast(records, alpha, beta, log)
    scoring  = compute_scoring(records, log)
    cpm      = compute_cpm(scoring, n_crews, hours_per_repair, cost_per_repair, log)
    active   = compute_active_defects(scoring, records, log)
    geometry, sec_count = compute_geometry(scoring)

    if not bridge_id:
        bridge_id = Path(filepath).stem.replace('RESUMEN_GRIETAS', 'CV').replace('_', '-')
        if not bridge_id or bridge_id == 'CV':
            bridge_id = 'CV-XXX'

    data = {
        'meta': {
            'titulo':          f'Plan de Intervención · {bridge_id}',
            'autor':           'Dr. Ing. L. Rojas Valdivia',
            'analista':        'GMT Group / Innovaminer · CODELCO',
            'fecha_analisis':  datetime.now().isoformat(),
            'bridge_id':       bridge_id,
            'current_week':    current_week,
            'parametros': {
                'n_crews':            n_crews,
                'hours_per_repair':   hours_per_repair,
                'cost_per_repair_USD': cost_per_repair,
                'alpha_gamma_prior':   alpha,
                'beta_gamma_prior':    beta,
                'alpha_FEA':           1.2
            }
        },
        'enriched':  records,
        'pca':       pca,
        'forecast':  forecast,
        'cpm':       cpm,
        'scoring':   scoring,
        'active':    active,
        'expert':    EXPERT_WEIGHTS,
        'geometry':  geometry,
        'sec_count': sec_count,
        'evolution': evolution,                 # weekly summary + transitions + lambda timeline
    }
    log("=" * 64)
    log(f"  ✅ Pipeline completo · {len(records)} ítems · {len(active)} activos S2-S9"
        + (f" · semana actual: {current_week}" if current_week else ""))
    if evolution:
        log(f"  📊 Evolución: {len(evolution['weeks'])} semanas · {len(evolution['transitions'])} transiciones")
    log("=" * 64)
    return data


def generate_html(data, output_path, log=print):
    """Inject DATA into the embedded HTML template; write the result."""
    log(f"📦 Generando HTML interactivo...")
    template = get_template()
    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=str)
    fea_json = get_fea_model_json()
    html = template.replace('__FEA_MODEL_JSON__', fea_json).replace('__DATA_JSON__', json_str)
    Path(output_path).write_text(html, encoding='utf-8')
    size_kb = len(html) / 1024
    log(f"   ✅ Escrito: {output_path}  ({size_kb:.0f} KB · {len(html):,} chars)")
    return output_path


# ================================================================================
# GUI (tkinter) — only defined if tkinter is importable
# ================================================================================
if _TK_OK:
  class App(tk.Tk):
      def __init__(self):
          super().__init__()
          self.title("CV Bridge Generator v2.0  ·  GMT Group / Innovaminer")
          self.geometry("820x720")
          self.minsize(720, 600)
          self._configure_style()

          self.filepath_var   = tk.StringVar()
          self.outpath_var    = tk.StringVar()
          self.bridge_var     = tk.StringVar(value="CV-202")
          self.crews_var      = tk.IntVar(value=3)
          self.hours_var      = tk.IntVar(value=8)
          self.cost_var       = tk.IntVar(value=7000)
          self.alpha_var      = tk.IntVar(value=2)
          self.beta_var       = tk.IntVar(value=2)
          self.last_output    = None
          self._build_ui()
          self._log_initial()

      # --------------- styling ---------------
      def _configure_style(self):
          self.configure(bg='#0f141d')
          s = ttk.Style(self)
          try:
              s.theme_use('clam')
          except tk.TclError:
              pass
          s.configure('.', background='#0f141d', foreground='#e6edf3', fieldbackground='#1d2533')
          s.configure('TFrame',     background='#0f141d')
          s.configure('TLabel',     background='#0f141d', foreground='#e6edf3', font=('Segoe UI', 10))
          s.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#00d4ff')
          s.configure('SubTitle.TLabel', font=('Segoe UI', 9), foreground='#a8b3c1')
          s.configure('Section.TLabel', font=('Segoe UI', 10, 'bold'), foreground='#00d4ff')
          s.configure('Status.TLabel', font=('Consolas', 9), foreground='#a8b3c1', background='#0a0e14')
          s.configure('TLabelframe', background='#0f141d', foreground='#00d4ff', borderwidth=1)
          s.configure('TLabelframe.Label', background='#0f141d', foreground='#00d4ff', font=('Segoe UI', 10, 'bold'))
          s.configure('TButton', padding=(14, 7), font=('Segoe UI', 10), background='#1d2533', foreground='#e6edf3', borderwidth=0)
          s.map('TButton', background=[('active', '#2a3344')])
          s.configure('Accent.TButton', padding=(18, 9), font=('Segoe UI', 11, 'bold'), background='#00d4ff', foreground='#000000')
          s.map('Accent.TButton', background=[('active', '#33ddff'), ('disabled', '#444')])
          s.configure('Violet.TButton', padding=(14, 7), font=('Segoe UI', 10, 'bold'), background='#a855f7', foreground='#ffffff')
          s.map('Violet.TButton', background=[('active', '#bb6efc')])
          s.configure('TEntry', fieldbackground='#1d2533', foreground='#e6edf3', insertcolor='#00d4ff', borderwidth=1)
          s.configure('TSpinbox', fieldbackground='#1d2533', foreground='#e6edf3', arrowcolor='#00d4ff')
          s.configure('Horizontal.TProgressbar', background='#00d4ff', troughcolor='#1d2533', borderwidth=0)

      # --------------- UI construction ---------------
      def _build_ui(self):
          # Header
          header = ttk.Frame(self, padding=(18, 14))
          header.pack(fill='x')
          ttk.Label(header, text="CV BRIDGE GENERATOR  ·  v2.0", style='Title.TLabel').pack(anchor='w')
          ttk.Label(header,
                    text="GMT Group / Innovaminer · CODELCO  ·  Pipeline EDA → PCA → NHPP → Scoring → CPM → 3D/FEA",
                    style='SubTitle.TLabel').pack(anchor='w')
          ttk.Separator(self, orient='horizontal').pack(fill='x', padx=18)

          body = ttk.Frame(self, padding=(18, 12))
          body.pack(fill='both', expand=True)

          # ---- 1) File selection ----
          fr1 = ttk.LabelFrame(body, text="  1 · Archivo Excel de origen  ", padding=(12, 10))
          fr1.pack(fill='x', pady=(0, 10))
          rowf = ttk.Frame(fr1); rowf.pack(fill='x')
          ttk.Entry(rowf, textvariable=self.filepath_var, font=('Consolas', 9)).pack(side='left', fill='x', expand=True, ipady=4)
          ttk.Button(rowf, text="📁 Examinar…", command=self.on_browse_input).pack(side='left', padx=(8, 0))
          ttk.Label(fr1, text="Formato esperado: hoja con cabecera SECCION X seguida de filas con ítems N.M (ej. 9.3) "
                              "y columnas Lado, Observación, Aviso, Fecha, Estado, Criticidad.",
                    style='SubTitle.TLabel', wraplength=720).pack(anchor='w', pady=(8, 0))

          # ---- 2) Parameters ----
          fr2 = ttk.LabelFrame(body, text="  2 · Parámetros del proyecto  ", padding=(12, 10))
          fr2.pack(fill='x', pady=(0, 10))
          gr = ttk.Frame(fr2); gr.pack(fill='x')
          # Row 0
          ttk.Label(gr, text="ID Puente:").grid(row=0, column=0, sticky='w', padx=(0, 8), pady=4)
          ttk.Entry(gr, textvariable=self.bridge_var, width=14).grid(row=0, column=1, sticky='w', pady=4)
          ttk.Label(gr, text="N° Cuadrillas:").grid(row=0, column=2, sticky='w', padx=(20, 8), pady=4)
          ttk.Spinbox(gr, from_=1, to=8, textvariable=self.crews_var, width=6).grid(row=0, column=3, sticky='w', pady=4)
          ttk.Label(gr, text="Horas/reparación:").grid(row=0, column=4, sticky='w', padx=(20, 8), pady=4)
          ttk.Spinbox(gr, from_=2, to=24, textvariable=self.hours_var, width=6).grid(row=0, column=5, sticky='w', pady=4)
          # Row 1
          ttk.Label(gr, text="USD / reparación:").grid(row=1, column=0, sticky='w', padx=(0, 8), pady=4)
          ttk.Spinbox(gr, from_=1000, to=50000, increment=500, textvariable=self.cost_var, width=10).grid(row=1, column=1, sticky='w', pady=4)
          ttk.Label(gr, text="Prior Gamma α:").grid(row=1, column=2, sticky='w', padx=(20, 8), pady=4)
          ttk.Spinbox(gr, from_=1, to=10, textvariable=self.alpha_var, width=6).grid(row=1, column=3, sticky='w', pady=4)
          ttk.Label(gr, text="Prior Gamma β:").grid(row=1, column=4, sticky='w', padx=(20, 8), pady=4)
          ttk.Spinbox(gr, from_=1, to=10, textvariable=self.beta_var, width=6).grid(row=1, column=5, sticky='w', pady=4)

          # ---- 3) Output ----
          fr3 = ttk.LabelFrame(body, text="  3 · Archivo de salida HTML  ", padding=(12, 10))
          fr3.pack(fill='x', pady=(0, 10))
          rowo = ttk.Frame(fr3); rowo.pack(fill='x')
          ttk.Entry(rowo, textvariable=self.outpath_var, font=('Consolas', 9)).pack(side='left', fill='x', expand=True, ipady=4)
          ttk.Button(rowo, text="📁 Guardar como…", command=self.on_browse_output).pack(side='left', padx=(8, 0))

          # ---- 4) Action ----
          ar = ttk.Frame(body); ar.pack(fill='x', pady=(8, 8))
          self.btn_generate = ttk.Button(ar, text="▶  GENERAR HTML INTERACTIVO", style='Accent.TButton', command=self.on_generate)
          self.btn_generate.pack(side='left')
          self.btn_open = ttk.Button(ar, text="🌐 Abrir HTML", command=self.on_open_output, state='disabled')
          self.btn_open.pack(side='left', padx=(8, 0))
          self.btn_folder = ttk.Button(ar, text="📂 Carpeta", command=self.on_open_folder, state='disabled')
          self.btn_folder.pack(side='left', padx=(8, 0))

          # Progress
          self.progress = ttk.Progressbar(body, mode='determinate', maximum=100)
          self.progress.pack(fill='x', pady=(4, 8))

          # Log
          fr4 = ttk.LabelFrame(body, text="  4 · Log de procesamiento  ", padding=(8, 6))
          fr4.pack(fill='both', expand=True)
          self.log_text = scrolledtext.ScrolledText(
              fr4, height=14, bg='#0a0e14', fg='#a8b3c1',
              insertbackground='#00d4ff', font=('Consolas', 9),
              relief='flat', borderwidth=0
          )
          self.log_text.pack(fill='both', expand=True)
          self.log_text.tag_configure('ok',    foreground='#00e676')
          self.log_text.tag_configure('warn',  foreground='#ffaa00')
          self.log_text.tag_configure('err',   foreground='#ff3355')
          self.log_text.tag_configure('cyan',  foreground='#00d4ff')
          self.log_text.tag_configure('hdr',   foreground='#a855f7', font=('Consolas', 9, 'bold'))

          # Status bar
          self.status = ttk.Label(self, text=" Listo. Seleccione un archivo Excel para comenzar.",
                                  style='Status.TLabel', anchor='w')
          self.status.pack(side='bottom', fill='x', ipady=4)

      def _log_initial(self):
          self.log("=" * 70, 'cyan')
          self.log("  CV BRIDGE GENERATOR  ·  v2.0", 'hdr')
          self.log("  Dr. Ing. L. Rojas · GMT Group / Innovaminer · CODELCO", 'hdr')
          self.log("=" * 70, 'cyan')
          self.log("\nPipeline: Excel → Regex → PCA → NHPP → Scoring → CPM → 3D/FEA → HTML")
          self.log("Salida: archivo HTML autocontenido con 11 tabs interactivas (Three.js + jsPDF + I_RE).\n")
          if not _DEPS_OK:
              self.log(_DEPS_MSG, 'err')
              self.btn_generate.configure(state='disabled')

      # --------------- UI handlers ---------------
      def log(self, msg, tag=None):
          ts = datetime.now().strftime('%H:%M:%S')
          prefix = f'[{ts}]  '
          self.log_text.insert('end', prefix, 'cyan')
          self.log_text.insert('end', str(msg) + '\n', tag if tag else None)
          self.log_text.see('end')
          self.log_text.update_idletasks()

      def set_status(self, msg, color=None):
          self.status.configure(text=' ' + msg)
          if color:
              self.status.configure(foreground=color)

      def set_progress(self, pct):
          self.progress['value'] = pct
          self.update_idletasks()

      def on_browse_input(self):
          fp = filedialog.askopenfilename(
              title="Seleccione el archivo Excel de inspección",
              filetypes=[("Excel", "*.xlsx *.xlsm *.xls"), ("Todos", "*.*")]
          )
          if fp:
              self.filepath_var.set(fp)
              # auto-suggest output path
              stem = Path(fp).stem
              bridge_id = self.bridge_var.get() or 'CV-XXX'
              out = str(Path(fp).parent / f"{bridge_id}_Plan_Intervencion.html")
              self.outpath_var.set(out)
              self.set_status(f"Archivo cargado: {Path(fp).name}", '#00e676')

      def on_browse_output(self):
          fp = filedialog.asksaveasfilename(
              title="Guardar HTML generado como…",
              defaultextension=".html",
              filetypes=[("HTML", "*.html"), ("Todos", "*.*")],
              initialfile=f"{self.bridge_var.get()}_Plan_Intervencion.html"
          )
          if fp:
              self.outpath_var.set(fp)

      def on_generate(self):
          fp = self.filepath_var.get().strip()
          out = self.outpath_var.get().strip()
          if not fp or not Path(fp).exists():
              messagebox.showerror("Error", "Seleccione un archivo Excel válido.")
              return
          if not out:
              messagebox.showerror("Error", "Especifique el archivo de salida HTML.")
              return

          self.btn_generate.configure(state='disabled')
          self.btn_open.configure(state='disabled')
          self.btn_folder.configure(state='disabled')
          self.set_progress(0)
          self.set_status("Procesando…", '#00d4ff')
          threading.Thread(target=self._run_pipeline, args=(fp, out), daemon=True).start()

      def _run_pipeline(self, fp, out):
          try:
              self.log("\n" + "▶ INICIANDO PIPELINE", 'hdr')
              steps = [
                  (15, "Parsing Excel"),
                  (35, "Feature engineering"),
                  (50, "PCA via SVD"),
                  (65, "NHPP Bayesiano"),
                  (80, "Scoring + CPM"),
                  (90, "Geometría 3D"),
                  (98, "Inyección HTML")
              ]
              # Wrap log to capture and display
              def _log(msg):
                  self.log(str(msg))
              self.set_progress(5)
              data = build_data(
                  fp,
                  bridge_id=self.bridge_var.get(),
                  n_crews=int(self.crews_var.get()),
                  hours_per_repair=int(self.hours_var.get()),
                  cost_per_repair=int(self.cost_var.get()),
                  alpha=int(self.alpha_var.get()),
                  beta=int(self.beta_var.get()),
                  log=_log
              )
              self.set_progress(95)
              generate_html(data, out, log=_log)
              self.set_progress(100)
              self.last_output = out
              self.btn_open.configure(state='normal')
              self.btn_folder.configure(state='normal')
              self.set_status(f"✅ HTML generado: {Path(out).name}  ·  {os.path.getsize(out)/1024:.0f} KB", '#00e676')
              self.log("\n✅ COMPLETADO  ·  Use 🌐 Abrir HTML para visualizar el reporte.", 'ok')
          except Exception as e:
              tb = traceback.format_exc()
              self.log("\n❌ ERROR EN PIPELINE:", 'err')
              self.log(str(e), 'err')
              self.log(tb, 'err')
              self.set_status(f"❌ Error: {e}", '#ff3355')
              messagebox.showerror("Pipeline fallido", f"{e}\n\nRevise el log para detalle.")
          finally:
              self.btn_generate.configure(state='normal')

      def on_open_output(self):
          if self.last_output and Path(self.last_output).exists():
              webbrowser.open(Path(self.last_output).absolute().as_uri())

      def on_open_folder(self):
          if self.last_output:
              folder = Path(self.last_output).parent
              try:
                  if sys.platform.startswith('win'):
                      os.startfile(folder)
                  elif sys.platform == 'darwin':
                      os.system(f'open "{folder}"')
                  else:
                      os.system(f'xdg-open "{folder}"')
              except Exception as e:
                  messagebox.showerror("Error", f"No se pudo abrir la carpeta: {e}")


# ================================================================================
# CLI / Entry point
# ================================================================================
def cli_main():
    """Run from command line without GUI."""
    parser = argparse.ArgumentParser(
        description="CV Bridge Generator — Excel → HTML interactivo"
    )
    parser.add_argument('input',  help='Archivo Excel de inspección')
    parser.add_argument('--out',  '-o', help='HTML de salida (auto si no se especifica)')
    parser.add_argument('--bridge-id', help='ID del puente (ej: CV-202)')
    parser.add_argument('--crews',     type=int, default=3, help='N° cuadrillas (default 3)')
    parser.add_argument('--hours',     type=int, default=8, help='Horas/reparación (default 8)')
    parser.add_argument('--cost',      type=int, default=7000, help='USD/reparación (default 7000)')
    parser.add_argument('--alpha',     type=int, default=2, help='Prior Gamma α (default 2)')
    parser.add_argument('--beta',      type=int, default=2, help='Prior Gamma β (default 2)')
    args = parser.parse_args()

    if not _DEPS_OK:
        print(_DEPS_MSG, file=sys.stderr); sys.exit(1)

    if not Path(args.input).exists():
        print(f"❌ No existe: {args.input}", file=sys.stderr); sys.exit(1)

    bridge_id = args.bridge_id or Path(args.input).stem
    out = args.out or str(Path(args.input).parent / f"{bridge_id}_Plan_Intervencion.html")

    data = build_data(args.input, bridge_id=bridge_id, n_crews=args.crews,
                      hours_per_repair=args.hours, cost_per_repair=args.cost,
                      alpha=args.alpha, beta=args.beta, log=print)
    generate_html(data, out, log=print)
    print(f"\n✅ HTML generado: {out}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main()
    else:
        if not _TK_OK:
            print("ERROR: tkinter no disponible. Use modo CLI: python cv_bridge_generator.py archivo.xlsx",
                  file=sys.stderr); sys.exit(1)
        if not _DEPS_OK:
            # Show GUI anyway with disabled button + error message
            pass
        app = App()
        app.mainloop()
