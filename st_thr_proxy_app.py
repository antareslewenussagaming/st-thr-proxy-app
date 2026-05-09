"""
ST THR Proxy Calculator for PLTGU CCR Excel Files

Run:
    pip install -r requirements.txt
    streamlit run st_thr_proxy_app.py

Main method:
    Reheat-aware proxy THR and proxy efficiency from CCR data.

Important:
    This is NOT an official performance-test THR calculation.
    It is a proxy based on CCR data and IAPWS-IF97 steam properties.
"""

from __future__ import annotations

import io
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from iapws import IAPWS97
except Exception:  # pragma: no cover
    IAPWS97 = None


# ================================================================
# Defaults from the chat calculation. Used only when CCR P/T data
# are missing or steam-property calculation fails.
# ================================================================
DEFAULT_H_MS = 3438.39      # kJ/kg
DEFAULT_H_HRH = 3337.14     # kJ/kg. Corrected from 337.14.
DEFAULT_H_LP = 2981.13      # kJ/kg
DEFAULT_H_CRH = 3135.05     # kJ/kg
DEFAULT_H_COND = 164.61     # kJ/kg
CP_WATER = 4.18855          # kJ/(kg degC), condensate fallback


# ================================================================
# Text and numeric helpers
# ================================================================

def clean_text(value: Any) -> str:
    """Normalize cell text for fuzzy row matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_number(value: Any) -> Optional[float]:
    """Convert Excel-like numeric values to float. Returns None if invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # Remove common non-numeric marks while keeping decimal sign.
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    if text in {"", ".", "-", "+"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_time_like(value: Any) -> bool:
    if isinstance(value, (datetime, time, pd.Timestamp)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        # Some CCR tables use 1, 5, 9, 13, 17, 21 as hour labels.
        return 0 <= float(value) <= 24 and float(value).is_integer()
    text = str(value).strip().lower()
    if not text:
        return False
    return bool(re.match(r"^\d{1,2}\s*[:.]\s*\d{2}$", text))


def format_time_label(value: Any, col_index: int) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%H.%M")
    if isinstance(value, datetime):
        return value.strftime("%H.%M")
    if isinstance(value, time):
        return f"{value.hour:02d}.{value.minute:02d}"
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        hour = int(float(value))
        return f"{hour:02d}.00"
    text = str(value).strip()
    match = re.match(r"^(\d{1,2})\s*[:.]\s*(\d{2})$", text)
    if match:
        return f"{int(match.group(1)):02d}.{match.group(2)}"
    return f"COL_{col_index}"


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return None
    return numerator / denominator


def weighted_average(values_and_weights: Iterable[Tuple[Optional[float], Optional[float]]]) -> Optional[float]:
    total_w = 0.0
    total_vw = 0.0
    for value, weight in values_and_weights:
        if value is None or weight is None or pd.isna(value) or pd.isna(weight):
            continue
        if weight <= 0:
            continue
        total_w += weight
        total_vw += value * weight
    if total_w <= 0:
        return None
    return total_vw / total_w


# ================================================================
# Pressure and enthalpy calculation
# ================================================================

def pressure_to_mpa(value: Optional[float], unit: str, is_gauge: bool) -> Optional[float]:
    """Convert pressure value to absolute MPa."""
    if value is None or pd.isna(value):
        return None

    p = float(value)
    unit = unit.lower().strip()

    if unit == "kg/cm2":
        if is_gauge:
            p += 1.03323  # kg/cm2 atmospheric pressure
        return p * 0.0980665

    if unit == "bar":
        if is_gauge:
            p += 1.01325
        return p * 0.1

    if unit == "mpa":
        if is_gauge:
            p += 0.101325
        return p

    if unit == "kpa":
        if is_gauge:
            p += 101.325
        return p / 1000.0

    raise ValueError(f"Unsupported pressure unit: {unit}")


def steam_enthalpy_kjkg(
    pressure_value: Optional[float],
    temperature_c: Optional[float],
    pressure_unit: str,
    pressure_is_gauge: bool,
    fallback_h: float,
    phase_hint: str,
) -> Tuple[float, str]:
    """
    Calculate steam/water enthalpy in kJ/kg using IAPWS97.

    phase_hint:
        - "steam": use P/T, fallback saturated vapor if needed
        - "liquid": use P/T, fallback cp*T if only T is available
    """
    if IAPWS97 is None:
        if temperature_c is not None and phase_hint == "liquid":
            return CP_WATER * float(temperature_c), "fallback_cp_water_iapws_not_installed"
        return fallback_h, "fallback_iapws_not_installed"

    pressure_mpa = pressure_to_mpa(pressure_value, pressure_unit, pressure_is_gauge)
    t_k = None if temperature_c is None or pd.isna(temperature_c) else float(temperature_c) + 273.15

    # Condensate with no pressure is usually good enough as cp*T proxy.
    if phase_hint == "liquid" and t_k is not None and pressure_mpa is None:
        return CP_WATER * float(temperature_c), "fallback_cp_water_no_pressure"

    if pressure_mpa is not None and t_k is not None:
        try:
            h = IAPWS97(P=pressure_mpa, T=t_k).h
            if h is not None and math.isfinite(h):
                return float(h), "iapws_PT"
        except Exception:
            pass

    # If only pressure is valid for steam and P/T failed, try saturated vapor.
    if phase_hint == "steam" and pressure_mpa is not None:
        try:
            h = IAPWS97(P=pressure_mpa, x=1).h
            if h is not None and math.isfinite(h):
                return float(h), "iapws_sat_vapor_fallback"
        except Exception:
            pass

    if phase_hint == "liquid" and temperature_c is not None:
        return CP_WATER * float(temperature_c), "fallback_cp_water"

    return fallback_h, "fallback_constant"


# ================================================================
# CCR row detection
# ================================================================

@dataclass
class MatchedRow:
    key: str
    row_index: Optional[int]
    label: str
    score: int
    status: str


def detect_time_row(raw: pd.DataFrame) -> Tuple[Optional[int], List[int], List[str]]:
    """Find row with most time-like cells."""
    best_row = None
    best_cols: List[int] = []

    for r_idx in range(raw.shape[0]):
        cols = [c_idx for c_idx in range(raw.shape[1]) if is_time_like(raw.iat[r_idx, c_idx])]
        if len(cols) > len(best_cols):
            best_row = r_idx
            best_cols = cols

    if best_row is not None and len(best_cols) >= 2:
        labels = [format_time_label(raw.iat[best_row, c], c) for c in best_cols]
        return best_row, best_cols, labels

    # Fallback: use columns that contain enough numeric data.
    numeric_counts = []
    for c_idx in range(raw.shape[1]):
        count = 0
        for r_idx in range(raw.shape[0]):
            if parse_number(raw.iat[r_idx, c_idx]) is not None:
                count += 1
        numeric_counts.append((c_idx, count))

    cols = [c for c, count in numeric_counts if count >= 5]
    labels = [f"COL_{c}" for c in cols]
    return None, cols, labels


def build_row_labels(raw: pd.DataFrame, time_cols: List[int]) -> List[str]:
    if time_cols:
        first_time_col = min(time_cols)
    else:
        first_time_col = min(6, raw.shape[1])

    labels = []
    for r_idx in range(raw.shape[0]):
        left_cells = []
        for c_idx in range(first_time_col):
            text = clean_text(raw.iat[r_idx, c_idx])
            if text:
                left_cells.append(text)
        labels.append(" | ".join(left_cells))
    return labels


def contains_any(label: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, label, flags=re.IGNORECASE) for pattern in patterns)


def stream_patterns(stream: str) -> List[str]:
    return {
        "ms": [r"\bmain\b", r"\bhp\b"],
        "ip": [r"\bip\b", r"intermediate"],
        "lp": [r"\blp\b", r"low\s*pressure"],
        "hrh": [r"\bhrh\b", r"hot\s*reheat", r"hot\s*rh", r"reheat\s*hot"],
        "crh": [r"\bcrh\b", r"cold\s*reheat", r"cold\s*rh", r"hp\s*exhaust"],
        "cond": [r"condensate", r"hotwell", r"hot\s*well"],
    }[stream]


def measure_patterns(measure: str) -> List[str]:
    return {
        "flow": [r"flow", r"\bt/h\b", r"ton", r"m3/h", r"m\^?3/h"],
        "press": [r"press", r"pressure", r"kg/cm", r"bar", r"mpa"],
        "temp": [r"temp", r"temperature", r"deg", r"\bc\b", r"℃"],
        "power": [r"mw", r"power", r"generator", r"load"],
    }[measure]


def unit_patterns(unit: Optional[str]) -> List[str]:
    if unit == "31":
        return [r"\b31\b", r"\b3\.1\b", r"gt\s*3?1\b", r"hrsg\s*3?1\b"]
    if unit == "32":
        return [r"\b32\b", r"\b3\.2\b", r"gt\s*3?2\b", r"hrsg\s*3?2\b"]
    return []


def score_generic(label: str, stream: str, measure: str, unit: Optional[str] = None) -> int:
    if not label:
        return -999

    score = 0
    if contains_any(label, stream_patterns(stream)):
        score += 35
    else:
        score -= 30

    if contains_any(label, measure_patterns(measure)):
        score += 30
    else:
        score -= 25

    if unit:
        if contains_any(label, unit_patterns(unit)):
            score += 25
        else:
            score -= 20

    # Penalize wrong measure words.
    wrong_measures = {"flow", "press", "temp", "power"} - {measure}
    for wm in wrong_measures:
        if contains_any(label, measure_patterns(wm)):
            score -= 8

    # Penalize wrong stream words when obvious.
    wrong_streams = {"ms", "ip", "lp", "hrh", "crh", "cond"} - {stream}
    for ws in wrong_streams:
        if contains_any(label, stream_patterns(ws)):
            score -= 4

    return score


def score_power_st(label: str) -> int:
    if not label:
        return -999
    score = 0
    if re.search(r"steam\s*turbine|\bst\b|\bstg\b", label):
        score += 40
    if re.search(r"generator|gen", label):
        score += 25
    if re.search(r"3\.3|\b33\b", label):
        score += 20
    if re.search(r"mw|power|load", label):
        score += 15
    if re.search(r"gas\s*turbine|\bgt\b|3\.1|3\.2|\b31\b|\b32\b", label):
        score -= 35
    return score


def best_match(labels: List[str], key: str, stream: Optional[str] = None, measure: Optional[str] = None, unit: Optional[str] = None) -> MatchedRow:
    best_idx = None
    best_score = -999
    for idx, label in enumerate(labels):
        if key == "p_st":
            score = score_power_st(label)
        else:
            assert stream is not None and measure is not None
            score = score_generic(label, stream, measure, unit)
        if score > best_score:
            best_score = score
            best_idx = idx

    threshold = 35 if key == "p_st" else 45
    if best_idx is None or best_score < threshold:
        return MatchedRow(key, None, "", best_score, "not_found")
    return MatchedRow(key, best_idx, labels[best_idx], best_score, "auto")


def get_series_for_match(raw: pd.DataFrame, match: MatchedRow, time_cols: List[int], time_labels: List[str]) -> Dict[str, Optional[float]]:
    if match.row_index is None:
        return {t: None for t in time_labels}
    result = {}
    for col, time_label in zip(time_cols, time_labels):
        result[time_label] = parse_number(raw.iat[match.row_index, col])
    return result


# ================================================================
# Workbook processing
# ================================================================

def detect_mapping(raw: pd.DataFrame, time_cols: List[int]) -> Tuple[Dict[str, MatchedRow], List[str]]:
    labels = build_row_labels(raw, time_cols)
    mapping: Dict[str, MatchedRow] = {}

    mapping["p_st"] = best_match(labels, "p_st")

    for unit in ["31", "32"]:
        for stream in ["ms", "ip", "lp", "hrh", "crh"]:
            for measure in ["flow", "press", "temp"]:
                # HRH and CRH flow are optional. The proxy defaults to MS+IP and MS.
                key = f"{stream}_{measure}_{unit}"
                mapping[key] = best_match(labels, key, stream, measure, unit)

    mapping["cond_flow"] = best_match(labels, "cond_flow", "cond", "flow")
    mapping["cond_temp"] = best_match(labels, "cond_temp", "cond", "temp")
    mapping["cond_press"] = best_match(labels, "cond_press", "cond", "press")

    return mapping, labels


def row_value(series_map: Dict[str, Dict[str, Optional[float]]], key: str, time_label: str) -> Optional[float]:
    return series_map.get(key, {}).get(time_label)


def choose_flow(actual: Optional[float], proxy: float, allow_actual: bool) -> float:
    if allow_actual and actual is not None and not pd.isna(actual) and actual > 0:
        return float(actual)
    return float(proxy)


def calculate_sheet(
    sheet_name: str,
    raw: pd.DataFrame,
    pressure_unit: str,
    pressure_is_gauge: bool,
    use_actual_hrh_crh_flow_if_found: bool,
    defaults: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    time_row, time_cols, time_labels = detect_time_row(raw)
    mapping, labels = detect_mapping(raw, time_cols)

    series_map = {
        key: get_series_for_match(raw, match, time_cols, time_labels)
        for key, match in mapping.items()
    }

    rows = []
    mapping_rows = []
    for key, match in mapping.items():
        mapping_rows.append({
            "sheet": sheet_name,
            "key": key,
            "row_index_excel_1based": None if match.row_index is None else match.row_index + 1,
            "detected_label": match.label,
            "score": match.score,
            "status": match.status,
        })

    for t_label in time_labels:
        notes = []

        p_st = row_value(series_map, "p_st", t_label)
        if p_st is None or p_st <= 0:
            rows.append({
                "sheet": sheet_name,
                "jam": t_label,
                "status": "skipped_no_ST_power",
                "notes": "P_ST missing or <= 0",
            })
            continue

        # Flows from HRSG 31 and 32.
        m_hp_31 = row_value(series_map, "ms_flow_31", t_label)
        m_hp_32 = row_value(series_map, "ms_flow_32", t_label)
        m_ip_31 = row_value(series_map, "ip_flow_31", t_label) or 0.0
        m_ip_32 = row_value(series_map, "ip_flow_32", t_label) or 0.0
        m_lp_31 = row_value(series_map, "lp_flow_31", t_label) or 0.0
        m_lp_32 = row_value(series_map, "lp_flow_32", t_label) or 0.0

        if m_hp_31 is None or m_hp_32 is None:
            rows.append({
                "sheet": sheet_name,
                "jam": t_label,
                "P_ST_MW": p_st,
                "status": "skipped_no_main_steam_flow",
                "notes": "HP/Main steam flow for HRSG 31 or 32 missing",
            })
            continue

        # Actual HRH/CRH flow is optional. Default follows the chat proxy.
        m_hrh_31_actual = row_value(series_map, "hrh_flow_31", t_label)
        m_hrh_32_actual = row_value(series_map, "hrh_flow_32", t_label)
        m_crh_31_actual = row_value(series_map, "crh_flow_31", t_label)
        m_crh_32_actual = row_value(series_map, "crh_flow_32", t_label)

        m_hrh_31 = choose_flow(m_hrh_31_actual, (m_hp_31 or 0.0) + m_ip_31, use_actual_hrh_crh_flow_if_found)
        m_hrh_32 = choose_flow(m_hrh_32_actual, (m_hp_32 or 0.0) + m_ip_32, use_actual_hrh_crh_flow_if_found)
        m_crh_31 = choose_flow(m_crh_31_actual, (m_hp_31 or 0.0), use_actual_hrh_crh_flow_if_found)
        m_crh_32 = choose_flow(m_crh_32_actual, (m_hp_32 or 0.0), use_actual_hrh_crh_flow_if_found)

        m_cond = row_value(series_map, "cond_flow", t_label)
        if m_cond is None or m_cond <= 0:
            m_cond = (m_hp_31 or 0.0) + (m_hp_32 or 0.0) + m_ip_31 + m_ip_32 + m_lp_31 + m_lp_32
            notes.append("condensate_flow_missing_used_total_steam_flow_proxy")

        t_cond = row_value(series_map, "cond_temp", t_label)
        p_cond = row_value(series_map, "cond_press", t_label)

        # Stream P/T values.
        h_ms_31, src_ms_31 = steam_enthalpy_kjkg(
            row_value(series_map, "ms_press_31", t_label),
            row_value(series_map, "ms_temp_31", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_ms"],
            "steam",
        )
        h_ms_32, src_ms_32 = steam_enthalpy_kjkg(
            row_value(series_map, "ms_press_32", t_label),
            row_value(series_map, "ms_temp_32", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_ms"],
            "steam",
        )
        h_hrh_31, src_hrh_31 = steam_enthalpy_kjkg(
            row_value(series_map, "hrh_press_31", t_label),
            row_value(series_map, "hrh_temp_31", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_hrh"],
            "steam",
        )
        h_hrh_32, src_hrh_32 = steam_enthalpy_kjkg(
            row_value(series_map, "hrh_press_32", t_label),
            row_value(series_map, "hrh_temp_32", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_hrh"],
            "steam",
        )
        h_lp_31, src_lp_31 = steam_enthalpy_kjkg(
            row_value(series_map, "lp_press_31", t_label),
            row_value(series_map, "lp_temp_31", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_lp"],
            "steam",
        )
        h_lp_32, src_lp_32 = steam_enthalpy_kjkg(
            row_value(series_map, "lp_press_32", t_label),
            row_value(series_map, "lp_temp_32", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_lp"],
            "steam",
        )
        h_crh_31, src_crh_31 = steam_enthalpy_kjkg(
            row_value(series_map, "crh_press_31", t_label),
            row_value(series_map, "crh_temp_31", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_crh"],
            "steam",
        )
        h_crh_32, src_crh_32 = steam_enthalpy_kjkg(
            row_value(series_map, "crh_press_32", t_label),
            row_value(series_map, "crh_temp_32", t_label),
            pressure_unit,
            pressure_is_gauge,
            defaults["h_crh"],
            "steam",
        )
        h_cond, src_cond = steam_enthalpy_kjkg(
            p_cond,
            t_cond,
            pressure_unit,
            pressure_is_gauge,
            defaults["h_cond"],
            "liquid",
        )

        enthalpy_sources = {
            "ms31": src_ms_31,
            "ms32": src_ms_32,
            "hrh31": src_hrh_31,
            "hrh32": src_hrh_32,
            "lp31": src_lp_31,
            "lp32": src_lp_32,
            "crh31": src_crh_31,
            "crh32": src_crh_32,
            "cond": src_cond,
        }
        fallback_sources = [k for k, v in enthalpy_sources.items() if not str(v).startswith("iapws")]
        if fallback_sources:
            notes.append("fallback_enthalpy:" + ",".join(fallback_sources))

        # Reheat-aware proxy. Flow unit t/h and power unit MW cancel their 1000 factors.
        q_ms = h_ms_31 * m_hp_31 + h_ms_32 * m_hp_32
        q_hrh = h_hrh_31 * m_hrh_31 + h_hrh_32 * m_hrh_32
        q_lp = h_lp_31 * m_lp_31 + h_lp_32 * m_lp_32
        q_crh = h_crh_31 * m_crh_31 + h_crh_32 * m_crh_32
        q_cond = h_cond * m_cond
        q_proxy = q_ms + q_hrh + q_lp - q_crh - q_cond
        thr_proxy = safe_div(q_proxy, p_st)
        eta_proxy = None if thr_proxy is None or thr_proxy <= 0 else (3600.0 / thr_proxy) * 100.0

        m_ms_total = (m_hp_31 or 0.0) + (m_hp_32 or 0.0)
        m_ip_total = m_ip_31 + m_ip_32
        m_hrh_total = m_hrh_31 + m_hrh_32
        m_lp_total = m_lp_31 + m_lp_32
        m_crh_total = m_crh_31 + m_crh_32

        h_ms_weighted = weighted_average([(h_ms_31, m_hp_31), (h_ms_32, m_hp_32)])
        h_hrh_weighted = weighted_average([(h_hrh_31, m_hrh_31), (h_hrh_32, m_hrh_32)])
        h_lp_weighted = weighted_average([(h_lp_31, m_lp_31), (h_lp_32, m_lp_32)])
        h_crh_weighted = weighted_average([(h_crh_31, m_crh_31), (h_crh_32, m_crh_32)])

        # Data-quality flags.
        if eta_proxy is not None and eta_proxy > 45:
            notes.append("efficiency_proxy_high_check_mapping_or_boundary")
        if eta_proxy is not None and eta_proxy < 25:
            notes.append("efficiency_proxy_low_check_outlier_or_low_load")
        if thr_proxy is not None and (thr_proxy < 7500 or thr_proxy > 13000):
            notes.append("THR_proxy_outside_typical_range")

        rows.append({
            "sheet": sheet_name,
            "jam": t_label,
            "P_ST_MW": p_st,
            "m_HP_31_tph": m_hp_31,
            "m_HP_32_tph": m_hp_32,
            "m_MS_total_tph": m_ms_total,
            "m_IP_31_tph": m_ip_31,
            "m_IP_32_tph": m_ip_32,
            "m_IP_total_tph": m_ip_total,
            "m_HRH_proxy_tph": m_hrh_total,
            "m_LP_31_tph": m_lp_31,
            "m_LP_32_tph": m_lp_32,
            "m_LP_total_tph": m_lp_total,
            "m_CRH_proxy_tph": m_crh_total,
            "m_cond_total_tph": m_cond,
            "T_cond_C": t_cond,
            "h_MS_31_kJkg": h_ms_31,
            "h_MS_32_kJkg": h_ms_32,
            "h_MS_weighted_kJkg": h_ms_weighted,
            "h_HRH_31_kJkg": h_hrh_31,
            "h_HRH_32_kJkg": h_hrh_32,
            "h_HRH_weighted_kJkg": h_hrh_weighted,
            "h_LP_31_kJkg": h_lp_31,
            "h_LP_32_kJkg": h_lp_32,
            "h_LP_weighted_kJkg": h_lp_weighted,
            "h_CRH_31_kJkg": h_crh_31,
            "h_CRH_32_kJkg": h_crh_32,
            "h_CRH_weighted_kJkg": h_crh_weighted,
            "h_cond_kJkg": h_cond,
            "Q_MS": q_ms,
            "Q_HRH": q_hrh,
            "Q_LP": q_lp,
            "Q_CRH": q_crh,
            "Q_cond": q_cond,
            "Q_proxy": q_proxy,
            "THR_proxy_kJkWh": thr_proxy,
            "eta_proxy_percent": eta_proxy,
            "status": "ok",
            "notes": "; ".join(notes),
            "enthalpy_sources": "; ".join(f"{k}={v}" for k, v in enthalpy_sources.items()),
        })

    return pd.DataFrame(rows), pd.DataFrame(mapping_rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    ok = result_df[result_df["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()

    summary_rows = []
    for sheet, g in ok.groupby("sheet"):
        g_valid = g.dropna(subset=["THR_proxy_kJkWh", "eta_proxy_percent"])
        if g_valid.empty:
            continue
        best_thr_idx = g_valid["THR_proxy_kJkWh"].idxmin()
        best_eta_idx = g_valid["eta_proxy_percent"].idxmax()
        summary_rows.append({
            "sheet": sheet,
            "count_rows": len(g_valid),
            "THR_avg_kJkWh": g_valid["THR_proxy_kJkWh"].mean(),
            "eta_avg_percent": g_valid["eta_proxy_percent"].mean(),
            "THR_min_kJkWh": g_valid["THR_proxy_kJkWh"].min(),
            "THR_max_kJkWh": g_valid["THR_proxy_kJkWh"].max(),
            "eta_min_percent": g_valid["eta_proxy_percent"].min(),
            "eta_max_percent": g_valid["eta_proxy_percent"].max(),
            "best_THR_jam": g_valid.loc[best_thr_idx, "jam"],
            "best_eta_jam": g_valid.loc[best_eta_idx, "jam"],
        })
    return pd.DataFrame(summary_rows)


def to_excel_bytes(results: pd.DataFrame, summary: pd.DataFrame, mapping: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="THR_Proxy_Result")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        mapping.to_excel(writer, index=False, sheet_name="Detected_Mapping")
    return output.getvalue()


# ================================================================
# Streamlit UI
# ================================================================

def main() -> None:
    st.set_page_config(page_title="ST THR Proxy Calculator", layout="wide")
    st.title("Steam Turbine THR Proxy Calculator - PLTGU CCR")

    st.markdown(
        """
Aplikasi ini menghitung **enthalpy**, **THR proxy**, dan **efisiensi proxy** dari file CCR Excel.
Metode default adalah **reheat-aware proxy** sesuai rumus di room chat.

Rumus utama:

```
THR_proxy = (
    h_MS  * m_MS_total
  + h_HRH * m_HRH_proxy
  + h_LP  * m_LP_total
  - h_CRH * m_CRH_proxy
  - h_cond * m_cond_total
) / P_ST

eta_proxy = 3600 / THR_proxy * 100%
```

Catatan: ini **proxy**, bukan THR resmi performance test.
"""
    )

    if IAPWS97 is None:
        st.error(
            "Package `iapws` belum terpasang. Jalankan: `pip install iapws`. "
            "Tanpa package ini, aplikasi tetap bisa memakai fallback enthalpy, tetapi tidak menghitung entalpi dari P/T."
        )

    with st.sidebar:
        st.header("Setting Perhitungan")
        pressure_unit = st.selectbox("Satuan pressure di CCR", ["kg/cm2", "bar", "MPa", "kPa"], index=0)
        pressure_is_gauge = st.checkbox("Pressure CCR adalah gauge", value=True)
        use_actual_hrh_crh_flow_if_found = st.checkbox(
            "Pakai actual HRH/CRH flow jika row ditemukan",
            value=False,
            help="Default OFF agar persis dengan metode chat: m_HRH = m_MS + m_IP, m_CRH = m_MS.",
        )

        st.subheader("Fallback enthalpy jika P/T tidak ditemukan")
        h_ms_default = st.number_input("h_MS fallback kJ/kg", value=DEFAULT_H_MS, step=0.01)
        h_hrh_default = st.number_input("h_HRH fallback kJ/kg", value=DEFAULT_H_HRH, step=0.01)
        h_lp_default = st.number_input("h_LP fallback kJ/kg", value=DEFAULT_H_LP, step=0.01)
        h_crh_default = st.number_input("h_CRH fallback kJ/kg", value=DEFAULT_H_CRH, step=0.01)
        h_cond_default = st.number_input("h_cond fallback kJ/kg", value=DEFAULT_H_COND, step=0.01)

    defaults = {
        "h_ms": h_ms_default,
        "h_hrh": h_hrh_default,
        "h_lp": h_lp_default,
        "h_crh": h_crh_default,
        "h_cond": h_cond_default,
    }

    uploaded = st.file_uploader("Drag and drop file Excel CCR di sini", type=["xlsx", "xls"])
    if uploaded is None:
        st.info("Upload file CCR Excel untuk mulai menghitung.")
        return

    try:
        excel_file = pd.ExcelFile(uploaded)
    except Exception as exc:
        st.error(f"File Excel gagal dibaca: {exc}")
        return

    sheet_names = excel_file.sheet_names
    selected_sheets = st.multiselect("Sheet/hari yang dihitung", sheet_names, default=sheet_names)
    if not selected_sheets:
        st.warning("Pilih minimal satu sheet.")
        return

    all_results = []
    all_mapping = []
    with st.spinner("Membaca CCR dan menghitung THR proxy..."):
        for sheet_name in selected_sheets:
            try:
                raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                result_df, mapping_df = calculate_sheet(
                    sheet_name=sheet_name,
                    raw=raw,
                    pressure_unit=pressure_unit,
                    pressure_is_gauge=pressure_is_gauge,
                    use_actual_hrh_crh_flow_if_found=use_actual_hrh_crh_flow_if_found,
                    defaults=defaults,
                )
                all_results.append(result_df)
                all_mapping.append(mapping_df)
            except Exception as exc:
                all_results.append(pd.DataFrame([{
                    "sheet": sheet_name,
                    "jam": None,
                    "status": "sheet_error",
                    "notes": str(exc),
                }]))

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    mapping = pd.concat(all_mapping, ignore_index=True) if all_mapping else pd.DataFrame()
    summary = summarize_results(results)

    tab_result, tab_summary, tab_mapping, tab_formula = st.tabs([
        "Hasil Per Jam", "Ringkasan", "Mapping Terdeteksi", "Rumus"
    ])

    with tab_result:
        st.subheader("Hasil Per Jam")
        display_cols = [
            "sheet", "jam", "P_ST_MW",
            "m_MS_total_tph", "m_IP_total_tph", "m_HRH_proxy_tph", "m_LP_total_tph",
            "m_cond_total_tph", "T_cond_C",
            "h_MS_weighted_kJkg", "h_HRH_weighted_kJkg", "h_LP_weighted_kJkg",
            "h_CRH_weighted_kJkg", "h_cond_kJkg",
            "Q_proxy", "THR_proxy_kJkWh", "eta_proxy_percent", "status", "notes",
        ]
        available_cols = [c for c in display_cols if c in results.columns]
        st.dataframe(results[available_cols], use_container_width=True)

        if not results.empty:
            ok = results[results.get("status", "").eq("ok")]
            if not ok.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("THR rata-rata", f"{ok['THR_proxy_kJkWh'].mean():,.2f} kJ/kWh")
                c2.metric("Efisiensi rata-rata", f"{ok['eta_proxy_percent'].mean():,.2f}%")
                c3.metric("Jumlah data valid", f"{len(ok)}")

    with tab_summary:
        st.subheader("Ringkasan Per Sheet/Hari")
        if summary.empty:
            st.warning("Belum ada data valid untuk diringkas.")
        else:
            st.dataframe(summary, use_container_width=True)

    with tab_mapping:
        st.subheader("Mapping Row yang Terdeteksi Otomatis")
        st.markdown(
            "Jika hasil tidak masuk akal, cek tab ini dulu. Biasanya masalah ada pada row yang salah terdeteksi atau row CCR tidak konsisten."
        )
        st.dataframe(mapping, use_container_width=True)

    with tab_formula:
        st.subheader("Metode Perhitungan")
        st.markdown(
            f"""
### Flow total

```
m_MS_total = m_HP_31 + m_HP_32
m_IP_total = m_IP_31 + m_IP_32
m_HRH_proxy = m_MS_total + m_IP_total
m_LP_total = m_LP_31 + m_LP_32
m_CRH_proxy = m_MS_total
```

### Enthalpy

Jika pressure dan temperature stream tersedia di CCR, enthalpy dihitung menggunakan IAPWS-IF97:

```
h = IAPWS97(P=P_abs_MPa, T=T_C+273.15).h
```

Jika tidak tersedia, fallback yang dipakai:

```
h_MS   = {h_ms_default:.2f} kJ/kg
h_HRH  = {h_hrh_default:.2f} kJ/kg
h_LP   = {h_lp_default:.2f} kJ/kg
h_CRH  = {h_crh_default:.2f} kJ/kg
h_cond = {h_cond_default:.2f} kJ/kg
```

Untuk condensate, jika hanya temperature tersedia:

```
h_cond = {CP_WATER} * T_cond_C
```

### THR proxy

```
Q_proxy = h_MS*m_MS_total + h_HRH*m_HRH_proxy + h_LP*m_LP_total - h_CRH*m_CRH_proxy - h_cond*m_cond_total
THR_proxy = Q_proxy / P_ST
eta_proxy = 3600 / THR_proxy * 100%
```

Satuan langsung menjadi `kJ/kWh` karena flow `t/h` dan power `MW` sama-sama membawa faktor 1000 yang saling hilang.
"""
        )

    if not results.empty:
        excel_bytes = to_excel_bytes(results, summary, mapping)
        st.download_button(
            label="Download hasil Excel",
            data=excel_bytes,
            file_name="thr_proxy_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            label="Download hasil CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="thr_proxy_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
