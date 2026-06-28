# ================================================================
#  PV PLANT FAULT MONITORING SYSTEM
#  Hybrid AI Architecture for Solar PV Fault Diagnosis
#  University of Moratuwa | Dept. of Electrical Engineering
# ================================================================

import os
import io
import json
import re
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PV Fault Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CONSTANTS ────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
MODELS   = os.path.join(BASE, "models")
SAMPLES  = os.path.join(BASE, "sample_data")
N_SEQ    = 50     # IV curve subsampling points (Zone 1)
WIN_Z2   = 32     # Sliding window length (Zone 2)

Z1_S1_LABELS = {
    0: "Healthy",
    1: "Degradation (DG)",
    2: "Short-Circuit (SC)",
    3: "Open-Circuit (OC)",
}
Z1_S2_LABELS = {
    0: "SC — Array 1  |  R_LL = 0.05 Ω (Severe)",
    1: "SC — Array 1  |  R_LL = 10 Ω",
    2: "SC — Array 1  |  R_LL = 20 Ω",
    3: "SC — Array 1  |  R_LL = 30 Ω (Mild)",
    4: "SC — Array 2  |  R_LL = 0.05 Ω (Severe)",
    5: "SC — Array 2  |  R_LL = 10 Ω",
    6: "SC — Array 2  |  R_LL = 20 Ω",
    7: "SC — Array 2  |  R_LL = 30 Ω (Mild)",
    8: "Open-Circuit — Array 1",
    9: "Open-Circuit — Array 2",
}
Z2_LABELS = {
    0: "Healthy",
    1: "T1 Open-Circuit Fault",
    2: "T2 Open-Circuit Fault",
    3: "T3 Open-Circuit Fault",
    4: "T4 Open-Circuit Fault",
    5: "T5 Open-Circuit Fault",
    6: "T6 Open-Circuit Fault",
}

# ── CUSTOM CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');

:root {
    --term-bg: #05080d;
    --term-panel: #0a1018;
    --term-panel-2: #0e1624;
    --term-line: #17314a;
    --term-green: #22c55e;
    --term-cyan: #22d3ee;
    --term-amber: #f59e0b;
    --term-red: #ef4444;
    --term-text: #dbeafe;
    --term-muted: #7dd3fc;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(34,211,238,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 24%),
        linear-gradient(180deg, #030712 0%, #05080d 100%);
}
[data-testid="stHeader"]  { background: transparent; }
[data-testid="stSidebar"] { background: #070b12; }
.block-container { padding-top: 1.1rem; }

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.main-title {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 29px;
    font-weight: 800;
    color: #d1fae5;
    letter-spacing: 0.6px;
    text-shadow: 0 0 16px rgba(34,197,94,0.36);
    margin-bottom: 4px;
}
.main-sub {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #67e8f9;
    margin-bottom: 16px;
}
.zone-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 800;
    color: #cffafe;
    background: linear-gradient(90deg, rgba(34,211,238,0.16), rgba(34,197,94,0.06));
    border: 1px solid #164e63;
    border-left: 4px solid #22d3ee;
    border-radius: 8px;
    padding: 9px 10px;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.zone-title::after {
    content: " ONLINE";
    float: right;
    color: #22c55e;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}
.demo-caption {
    color: #67e8f9;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    margin: 4px 0 8px 0;
}
.report-card {
    background: linear-gradient(135deg, rgba(14,116,144,0.14), rgba(6,78,59,0.10));
    border: 1px solid #164e63;
    border-radius: 12px;
    padding: 12px;
    margin-top: 8px;
    color: #d1fae5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

.badge {
    border-radius: 8px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 800;
    text-align: center;
    margin: 6px 0;
    display: block;
    box-shadow: inset 0 0 18px rgba(255,255,255,0.03), 0 0 18px rgba(34,211,238,0.05);
}
.metric-row {
    background: #08111d;
    border-radius: 7px;
    padding: 8px 11px;
    margin: 5px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-left: 3px solid #22d3ee;
    color: #cbd5e1;
}
.prob-wrap { margin: 5px 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #93c5fd;
    margin-bottom: 3px;
}
.prob-track {
    height: 5px;
    background: #07111d;
    border: 1px solid #12263d;
    border-radius: 3px;
    overflow: hidden;
}
.llm-box, .terminal-report {
    background: #030712;
    border: 1px solid #14532d;
    border-radius: 12px;
    padding: 16px;
    color: #d1fae5;
    line-height: 1.75;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    box-shadow: 0 0 24px rgba(34,197,94,0.08), inset 0 0 18px rgba(34,197,94,0.03);
}
.terminal-card {
    background: linear-gradient(180deg, rgba(8,17,29,0.98), rgba(3,7,18,0.98));
    border: 1px solid #164e63;
    border-radius: 12px;
    padding: 12px 13px;
    margin: 8px 0;
    box-shadow: 0 0 22px rgba(34,211,238,0.07), inset 0 0 22px rgba(15,23,42,0.35);
}
.terminal-card-title {
    color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.terminal-line {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(34,211,238,0.12);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}
.terminal-line:last-child { border-bottom: none; }
.terminal-key { color: #7dd3fc; }
.terminal-value { color: #d1fae5; font-weight: 800; text-align: right; }
.small-chip {
    display: inline-block;
    border: 1px solid #164e63;
    color: #67e8f9;
    background: rgba(8,47,73,0.30);
    padding: 2px 7px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    margin: 2px 3px 2px 0;
}
div[data-testid="stMetric"] {
    background: #07111d;
    border: 1px solid #12324a;
    border-radius: 10px;
    padding: 7px 9px;
}
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    border: 1px solid #164e63;
    background: linear-gradient(180deg, #0e7490, #064e3b);
    color: #ecfeff;
}
hr { border-color: #12324a !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    """Two-layer LSTM classifier — Zone 1 Stage 1 (4-class) and Stage 2 (10-class)."""
    def __init__(self, input_size: int, hidden: int, n_layers: int,
                 n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class Zone2CNN1D(nn.Module):
    """Three-block CNN-1D — Zone 2 IGBT fault diagnosis (7-class, 32,679 params)."""
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(4,  32, 3, padding=1), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 7)
    def forward(self, x):
        return self.head(self.conv_stack(x).squeeze(-1))


# ════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — runs once per session)
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_models() -> dict:
    m = {}
    try:
        # Zone 1 — Stage 1 (4-class LSTM)
        s1 = LSTMClassifier(5, 128, 2, 4)
        s1.load_state_dict(torch.load(os.path.join(MODELS,"zone1_stage1_lstm.pth"), map_location="cpu"))
        s1.eval(); m["z1s1"] = s1
        with open(os.path.join(MODELS,"zone1_stage1_scaler.pkl"),"rb") as f: m["z1s1_sc"] = pickle.load(f)

        # Zone 1 — Stage 2 (10-class LSTM)
        s2 = LSTMClassifier(8, 128, 2, 10)
        s2.load_state_dict(torch.load(os.path.join(MODELS,"zone1_stage2_lstm.pth"), map_location="cpu"))
        s2.eval(); m["z1s2"] = s2
        with open(os.path.join(MODELS,"zone1_stage2_scaler.pkl"),"rb") as f: m["z1s2_sc"] = pickle.load(f)

        # Zone 2 — CNN-1D (7-class)
        z2 = Zone2CNN1D()
        z2.load_state_dict(torch.load(os.path.join(MODELS,"zone2_model.pt"), map_location="cpu"))
        z2.eval(); m["z2"] = z2
        with open(os.path.join(MODELS,"zone2_scaler.pkl"),"rb") as f: m["z2_sc"] = pickle.load(f)

        # Zone 3 — XGBoost ESR regressor
        with open(os.path.join(MODELS,"zone3_model.pkl"),"rb") as f: m["z3"] = pickle.load(f)
        with open(os.path.join(MODELS,"zone3_scaler.pkl"),"rb") as f: m["z3_sc"] = pickle.load(f)

        m["ok"] = True
    except Exception as e:
        m["ok"] = False; m["err"] = str(e)
    return m


# ════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════

def prep_z1_sys(df: pd.DataFrame, scaler) -> np.ndarray:
    """Subsample system IV sweep → scaled [1, N_SEQ, 5] tensor input."""
    df = df.sort_values("V").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N_SEQ).astype(int)
    seq = df.iloc[idx][["V", "I", "P", "Irr", "T"]].values.astype(np.float32)
    return scaler.transform(seq.reshape(-1, 5)).reshape(1, N_SEQ, 5)


def prep_z1_arr(df: pd.DataFrame, scaler) -> np.ndarray:
    """Subsample per-array IV data → scaled [1, N_SEQ, 8] tensor input."""
    df = df.sort_values("V_1").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N_SEQ).astype(int)
    seq = df.iloc[idx][["V_1","I_1","P_1","V_2","I_2","P_2","Irr","T"]].values.astype(np.float32)
    return scaler.transform(seq.reshape(-1, 8)).reshape(1, N_SEQ, 8)


def prep_z2(df: pd.DataFrame, scaler) -> np.ndarray:
    """Take WIN_Z2 rows → scale → transpose to [1, 4, WIN_Z2] for CNN-1D."""
    cols = ["I_alpha", "I_beta", "Irradiance", "Temperature"]
    win  = df[cols].values[:WIN_Z2].astype(np.float32)
    return scaler.transform(win).T[np.newaxis]   # (1, 4, 32)


def prep_z2_best_window(df: pd.DataFrame, scaler, model) -> tuple:
    """Sliding-window inference: find the WIN_Z2-row window with the highest
    softmax confidence, then return (best_x_tensor, best_probs_array).
    Falls back to the first window when data is exactly WIN_Z2 rows."""
    cols = ["I_alpha", "I_beta", "Irradiance", "Temperature"]
    data = df[cols].values.astype(np.float32)
    n    = len(data)
    best_probs, best_x = None, None
    for start in range(max(1, n - WIN_Z2 + 1)):
        win    = data[start:start + WIN_Z2]
        x      = torch.tensor(scaler.transform(win).T[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(model(x), 1)[0].numpy()
        if best_probs is None or float(probs.max()) > float(best_probs.max()):
            best_probs = probs
            best_x     = x
    return best_x, best_probs


def prep_z3(df: pd.DataFrame, scaler) -> np.ndarray:
    """Scale single-row Zone 3 features."""
    cols = ["Irradiance", "T_ambient", "I_rms_high", "V_avg", "V_ripple"]
    return scaler.transform(df[cols].values[:1].astype(np.float32))


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_z1(df_sys: pd.DataFrame, df_arr, m: dict) -> dict:
    x    = torch.tensor(prep_z1_sys(df_sys, m["z1s1_sc"]))
    logit = m["z1s1"](x)
    cls  = int(logit.argmax(1))
    prob = torch.softmax(logit, 1)[0].numpy()

    s2_cls, s2_lbl, s2_info = None, None, {}
    if cls in [2, 3] and df_arr is not None:
        x2     = torch.tensor(prep_z1_arr(df_arr, m["z1s2_sc"]))
        logit2 = m["z1s2"](x2)
        s2_cls = int(logit2.argmax(1))
        s2_lbl = Z1_S2_LABELS[s2_cls]
        s2_info = parse_z1_stage2(s2_lbl, s2_cls)

    return {"cls": cls, "label": Z1_S1_LABELS[cls],
            "probs": prob, "s2_cls": s2_cls, "s2_label": s2_lbl, **s2_info}


def run_z2(df: pd.DataFrame, m: dict) -> dict:
    """Use sliding-window max-confidence inference for Zone 2."""
    _, prob = prep_z2_best_window(df, m["z2_sc"], m["z2"])
    cls     = int(prob.argmax())
    return {"cls": cls, "label": Z2_LABELS[cls], "probs": prob}


def run_z3(df: pd.DataFrame, m: dict) -> dict:
    X_s   = prep_z3(df, m["z3_sc"])
    esr   = float(np.clip(m["z3"].predict(X_s)[0], 0.15, 0.40))
    deg   = esr_to_degradation_percentage(esr)
    fault = esr >= 0.30
    row   = df.iloc[0]
    return {
        "esr": esr, "deg": deg, "fault": fault,
        "label": zone3_health_label(esr, deg),
        "irr":  float(row.get("Irradiance", 0)),
        "tamb": float(row.get("T_ambient", 0)),
        "irms": float(row.get("I_rms_high", 0)),
        "vavg": float(row.get("V_avg", 0)),
        "vrip": float(row.get("V_ripple", 0)),
    }


# ════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════

_PLOT_CFG  = {"displayModeBar": False}
_BG_DARK   = "#0d1117"
_BG_PANEL  = "#131720"
_AXIS_COL  = "#475569"
_GRID_COL  = "#1a2133"

def _base_layout(**kwargs):
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_BG_PANEL,
                height=195, margin=dict(l=38, r=38, t=10, b=36),
                font=dict(color=_AXIS_COL, size=10), **kwargs)


def plot_iv(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or not all(col in df.columns for col in ["V", "I", "P"]):
        fig.add_annotation(
            text="Invalid or missing data columns.<br>Expected: V, I, P",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#ef4444")
        )
        fig.update_layout(**_base_layout())
        return fig
    fig.add_trace(go.Scatter(x=df["V"], y=df["I"], mode="lines", name="I–V",
                             line=dict(color="#3b82f6", width=2.5)))
    fig.add_trace(go.Scatter(x=df["V"], y=df["P"], mode="lines", name="P–V",
                             line=dict(color="#f59e0b", width=2, dash="dash"), yaxis="y2"))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Voltage (V)", gridcolor=_GRID_COL, color=_AXIS_COL),
        yaxis=dict(title="Current (A)", gridcolor=_GRID_COL, color=_AXIS_COL),
        yaxis2=dict(title="Power (W)", overlaying="y", side="right", color="#f59e0b"),
        legend=dict(x=0.02, y=0.97, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    ))
    return fig


def plot_ab(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or not all(col in df.columns for col in ["I_alpha", "I_beta"]):
        fig.add_annotation(
            text="Invalid or missing data columns.<br>Expected: I_alpha, I_beta",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#ef4444")
        )
        fig.update_layout(**_base_layout())
        return fig
    fig.add_trace(go.Scatter(x=df["I_alpha"], y=df["I_beta"], mode="lines",
                             line=dict(color="#a855f7", width=2.5)))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Iα (A)", gridcolor=_GRID_COL, color=_AXIS_COL, scaleanchor="y"),
        yaxis=dict(title="Iβ (A)", gridcolor=_GRID_COL, color=_AXIS_COL),
    ))
    return fig


def plot_gauge(esr: float, deg: float) -> go.Figure:
    c = "#ef4444" if esr >= 0.30 else ("#f59e0b" if deg >= 70 else "#22c55e")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(deg, 1),
        title={"text": f"ESR = {esr:.4f} Ω", "font": {"size": 11, "color": _AXIS_COL}},
        number={"suffix": "%", "font": {"size": 30, "color": c}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _AXIS_COL},
            "bar":  {"color": c},
            "bgcolor": _BG_PANEL, "bordercolor": _GRID_COL,
            "steps": [
                {"range": [0,  50], "color": "rgba(34, 197, 94, 0.10)"},
                {"range": [50, 80], "color": "rgba(245, 158, 11, 0.10)"},
                {"range": [80,100], "color": "rgba(239, 68, 68, 0.14)"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3},
                          "thickness": 0.8, "value": 100},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      height=195, margin=dict(l=16, r=16, t=28, b=8),
                      font=dict(color="#e2e8f0"))
    return fig


# ════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════

def badge(text: str, color: str) -> str:
    return (f'<div class="badge" style="background:{color}22;border:1px solid {color};'
            f'color:{color}">{text}</div>')


def prob_bars(labels: dict, probs: np.ndarray, pred_cls: int) -> str:
    html = ""
    for i, name in labels.items():
        p   = float(probs[i])
        hi  = i == pred_cls
        col = "#3b82f6" if hi else "#334155"
        fw  = "700" if hi else "400"
        w   = int(p * 100)
        html += (
            f'<div class="prob-wrap">'
            f'<div class="prob-label"><span>{name}</span>'
            f'<span style="color:{col};font-weight:{fw}">{p*100:.1f}%</span></div>'
            f'<div class="prob-track"><div style="width:{w}%;height:100%;'
            f'background:{col};border-radius:2px"></div></div></div>'
        )
    return html




def parse_z1_stage2(label: str, cls: int) -> dict:
    """Convert Stage 2 class label into engineering fields for UI/reporting."""
    if label is None:
        return {}
    info = {
        "stage2_raw": label,
        "stage2_cls": cls,
        "stage2_fault_type": None,
        "stage2_array": None,
        "r_ll": None,
        "r_ll_float": None,
        "sc_severity": None,
    }
    if "Array 1" in label:
        info["stage2_array"] = "Array 1"
    elif "Array 2" in label:
        info["stage2_array"] = "Array 2"

    if label.startswith("SC"):
        info["stage2_fault_type"] = "Short-Circuit"
        # Label format: "SC — Array 1  |  R_LL = 10 Ω"
        try:
            raw = label.split("R_LL =", 1)[1].split("Ω", 1)[0].strip()
            info["r_ll"] = f"{raw} Ω"
            info["r_ll_float"] = float(raw)
        except Exception:
            info["r_ll"] = "Unavailable"
            info["r_ll_float"] = None

        r = info["r_ll_float"]
        if r is not None:
            if r <= 0.1:
                info["sc_severity"] = "Critical / near-solid short"
            elif r <= 10:
                info["sc_severity"] = "High severity short"
            elif r <= 20:
                info["sc_severity"] = "Moderate severity short"
            else:
                info["sc_severity"] = "Mild short-circuit path"

    elif label.startswith("Open-Circuit"):
        info["stage2_fault_type"] = "Open-Circuit"
        info["sc_severity"] = "Not applicable"

    return info


def terminal_panel(title: str, rows: dict, color: str = "#22d3ee") -> str:
    """Small terminal-style key/value card."""
    html = f'<div class="terminal-card" style="border-color:{color}">'
    html += f'<div class="terminal-card-title">▣ {title}</div>'
    for k, v in rows.items():
        html += (
            '<div class="terminal-line">'
            f'<span class="terminal-key">{k}</span>'
            f'<span class="terminal-value">{v}</span>'
            '</div>'
        )
    html += '</div>'
    return html


def esr_to_degradation_percentage(esr: float) -> float:
    """Map ESR 0.15–0.30 Ω to 0–100%; clamp above 0.30 Ω as fault-level 100%."""
    return float(np.clip((esr - 0.15) / (0.30 - 0.15) * 100, 0, 100))


def zone3_health_label(esr: float, deg: float) -> str:
    if esr >= 0.30:
        return "Fault — ESR threshold exceeded / 100% degraded"
    if esr > 0.15:
        return f"Ageing — {deg:.1f}% degraded"
    return "Healthy — 0.0% degraded"

def zone_status(r, zone: int):
    if r is None:
        return "gray", "⚪ Awaiting Data"
    if zone == 1:
        m = {0: ("#22c55e","✅ Healthy"), 1: ("#f59e0b","⚠️ Degradation"),
             2: ("#ef4444","🔴 Short-Circuit"), 3: ("#ef4444","🔴 Open-Circuit")}
        return m[r["cls"]]
    if zone == 2:
        return ("#22c55e","✅ Healthy") if r["cls"]==0 else ("#ef4444","🔴 IGBT Fault")
    # zone 3
    if r["fault"]:         return "#ef4444", "🔴 Capacitor Degraded"
    if r["deg"] >= 70:     return "#f59e0b", "⚠️ Capacitor Ageing"
    return "#22c55e", "✅ Capacitor Healthy"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    # Strip whitespace and BOM from column names
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    
    # Map common variations to the expected names
    mapping = {
        "v": "V", "i": "I", "p": "P", "irr": "Irr", "t": "T",
        "v_1": "V_1", "i_1": "I_1", "p_1": "P_1",
        "v_2": "V_2", "i_2": "I_2", "p_2": "P_2",
        "i_alpha": "I_alpha", "i_beta": "I_beta",
        "irradiance": "Irradiance", "temperature": "Temperature",
        "t_ambient": "T_ambient", "i_rms_high": "I_rms_high",
        "v_avg": "V_avg", "v_ripple": "V_ripple"
    }
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in mapping and col != mapping[col_lower]:
            rename_dict[col] = mapping[col_lower]
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def sample(fname: str):
    """Load a sample CSV from the sample_data/ folder."""
    p = os.path.join(SAMPLES, fname)
    return clean_df(pd.read_csv(p, encoding="utf-8-sig")) if os.path.exists(p) else None


# ════════════════════════════════════════════════════════════════
# LLM — GPT-4o-mini
# ════════════════════════════════════════════════════════════════

def _local_report(r1, r2, r3) -> str:
    """Generate a structured maintenance report locally (no API required)."""
    lines = []

    # ── System Health Summary ────────────────────────────────────
    faults = []
    if r1 and r1["cls"] != 0: faults.append(f"Zone 1 PV Array ({r1['label']})");
    if r2 and r2["cls"] != 0: faults.append(f"Zone 2 IGBT Inverter ({r2['label']})");
    if r3 and r3["fault"]:   faults.append(f"Zone 3 DC-Link Capacitor ({r3['label']})");

    lines.append("**1. System Health Summary**")
    if not faults:
        lines.append(
            "All three diagnostic zones report nominal operating conditions. "
            "No corrective maintenance action is required at this time."
        )
    else:
        lines.append(
            f"The PV plant is operating with **{len(faults)} active fault(s)** detected across "
            f"{', '.join(faults)}. Immediate review of the flagged components is recommended."
        )

    # ── Fault Analysis ───────────────────────────────────────────
    lines.append("\n**2. Fault Analysis**")
    if r1 and r1["cls"] != 0:
        fault_detail = {
            1: ("Partial shading or cell mismatch (degradation/soiling)",
                "Reduced Isc, asymmetric I–V curve, estimated 5–15% power loss."),
            2: ("Low-impedance line-to-line short circuit between PV modules",
                "Reduced Voc and Isc, localised heating risk, up to 30% power loss."),
            3: ("Open-circuit in one or more module strings",
                "Current path interrupted; significant power loss depending on string configuration."),
        }.get(r1["cls"], ("Unknown fault type", "Impact unknown."))
        lines.append(f"- **Zone 1 ({r1['label']})**: {fault_detail[0]}. {fault_detail[1]}")
        if r1.get("s2_label"):
            lines.append(f"  - Localisation output: {r1['s2_label']}")
            if r1.get("stage2_fault_type") == "Short-Circuit":
                lines.append(
                    f"  - SC interpretation: affected array = {r1.get('stage2_array', 'Unknown')}; "
                    f"estimated R_LL = {r1.get('r_ll', 'Unavailable')}; "
                    f"severity = {r1.get('sc_severity', 'Unavailable')}. "
                    "Lower R_LL means a lower-impedance fault path, higher circulating current risk, "
                    "stronger thermal stress, and greater urgency for isolation."
                )
            elif r1.get("stage2_fault_type") == "Open-Circuit":
                lines.append(
                    f"  - OC interpretation: affected array = {r1.get('stage2_array', 'Unknown')}. "
                    "R_LL is not applicable because the diagnosed fault is an interrupted current path, "
                    "not a line-to-line short."
                )
    if r2 and r2["cls"] != 0:
        igbt_num = r2['label'].split()[0] if r2['label'] else "?"
        lines.append(
            f"- **Zone 2 ({r2['label']})**: Open-circuit failure of IGBT switch {igbt_num}. "
            "Causes asymmetric current waveforms (αβ-plane distortion), increased harmonic "
            "distortion, and risk of cascading thermal damage to adjacent switches."
        )
    if r3:
        if r3["fault"]:
            lines.append(
                f"- **Zone 3 ({r3['label']})**: ESR = {r3['esr']:.4f} Ω exceeds the 0.30 Ω fault "
                f"threshold (degradation: {r3['deg']:.1f}%). High ESR increases voltage ripple, "
                "raises capacitor temperature, and risks catastrophic failure under load transients."
            )
        elif r3["esr"] > 0.15:
            lines.append(
                f"- **Zone 3 (Capacitor ageing)**: ESR = {r3['esr']:.4f} Ω, which lies inside "
                f"the 0.15–0.30 Ω ageing band. Estimated degradation = {r3['deg']:.1f}%. "
                "This is not yet a hard fault, but it gives a continuous health index for planning maintenance."
            )
        else:
            lines.append(
                f"- **Zone 3 (Healthy)**: ESR = {r3['esr']:.4f} Ω and estimated degradation = {r3['deg']:.1f}%."
            )
    if not faults:
        lines.append("No faults detected — all zones within normal operating limits.")

    # ── Recommended Actions ──────────────────────────────────────
    lines.append("\n**3. Recommended Actions**")
    if r1 and r1["cls"] == 2:
        sc_target = f"{r1.get('stage2_array', 'affected array')}"
        rll_text = f" with estimated R_LL={r1.get('r_ll')}" if r1.get('r_ll') else ""
        lines.append(f"- [IMMEDIATE] Isolate {sc_target}{rll_text}; inspect string wiring, connectors, bypass diode paths, insulation damage, and module hotspots.")
    elif r1 and r1["cls"] == 3:
        lines.append("- [IMMEDIATE] Inspect Zone 1 for open-circuit string; check fuses, connectors, and bypass diodes.")
    elif r1 and r1["cls"] == 1:
        lines.append("- [WITHIN 48H] Inspect Zone 1 modules for soiling, shading, or degraded cells; schedule cleaning or module replacement.")
    if r2 and r2["cls"] != 0:
        lines.append(f"- [IMMEDIATE] Replace or test IGBT {r2['label']} in the Zone 2 inverter; measure gate-drive signals and check thermal imagery.")
    if r3 and r3["fault"]:
        lines.append("- [IMMEDIATE] Replace Zone 3 DC-link capacitor; verify bus voltage ripple after replacement.")
    elif r3 and r3["deg"] >= 70:
        lines.append("- [WITHIN 48H] Schedule Zone 3 DC-link capacitor replacement before next high-load period.")
    if not faults:
        lines.append("- [SCHEDULED] Continue routine monitoring and preventive maintenance per standard intervals.")

    # ── Risk Assessment ──────────────────────────────────────────
    lines.append("\n**4. Risk Assessment**")
    if faults:
        lines.append(
            "If the identified faults are left unaddressed: sustained operation with a shorted or "
            "open PV string risks permanent module damage and DC arc faults; an open IGBT switch "
            "will propagate thermal stress to neighbouring devices and may cause inverter shutdown; "
            "a degraded capacitor under ripple stress is prone to electrolyte dry-out and "
            "catastrophic short-circuit failure, potentially damaging the entire DC bus."
        )
    else:
        lines.append(
            "No immediate risks identified. Continue monitoring to detect early-stage degradation "
            "before it impacts generation output or equipment longevity."
        )

    lines.append("\n*Report generated by local AI engine (offline mode).*")
    return "\n".join(lines)


def gpt_explain(r1, r2, r3) -> str:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return _local_report(r1, r2, r3)
        from openai import OpenAI
        # Short timeout so the UI doesn't hang — fall back to local report on failure
        client = OpenAI(api_key=api_key, timeout=8.0)

        findings = []
        if r1:
            line = f"Zone 1 (PV Array): **{r1['label']}**"
            if r1.get("s2_label"):
                line += f" → Localised as: {r1['s2_label']}"
            if r1.get("stage2_fault_type") == "Short-Circuit":
                line += (
                    f" | Parsed SC result: faulty_array={r1.get('stage2_array')}, "
                    f"R_LL={r1.get('r_ll')}, severity={r1.get('sc_severity')}"
                )
            elif r1.get("stage2_fault_type") == "Open-Circuit":
                line += f" | Parsed OC result: faulty_array={r1.get('stage2_array')}; R_LL=not applicable"
            findings.append(line)
        if r2:
            findings.append(f"Zone 2 (IGBT Inverter): **{r2['label']}**")
        if r3:
            findings.append(
                f"Zone 3 (DC-Link Capacitor): **{r3['label']}** | "
                f"ESR={r3['esr']:.4f}Ω | Degradation={r3['deg']:.1f}%"
            )

        findings_str = "\n".join(f"- {f}" for f in findings) if findings else "- No faults detected."

        prompt = f"""You are a senior solar PV plant maintenance engineer reviewing AI-generated fault diagnosis results.

FAULT DIAGNOSIS RESULTS:
{findings_str}

Write a polished engineering maintenance report suitable for a final-year project viva/demo. Use clear section headings and concise technical language. Include:

1. **Executive System Health Summary** — overall status and most urgent issue in 2-3 sentences.
2. **Evidence From AI Models** — mention the diagnostic zone, predicted class, confidence where available, and the sensor/trajectory evidence implied by the model.
3. **Fault Physics Explanation** — explain the likely electrical mechanism, including how the fault changes current/voltage/ripple behaviour.
4. **Short-Circuit Detail** — when Zone 1 is SC, explicitly explain the faulty array, predicted R_LL value, severity, and what low vs high R_LL means electrically. When Zone 1 is OC, state that R_LL is not applicable.
5. **IGBT Inverter Detail** — when Zone 2 is T1-T6, identify the failed open switch and describe the expected alpha-beta current trajectory distortion, harmonic effect, and inverter risk.
6. **Capacitor Health Detail** — when ESR is between 0.15 and 0.30 Ohm, report degradation percentage as an ageing/health-index result, not as a hard fault. If ESR >= 0.30 Ohm, classify it as fault-level degradation.
7. **Prioritised Maintenance Plan** — actions tagged [IMMEDIATE], [WITHIN 48H], or [SCHEDULED].
8. **Viva-Ready Note** — one sentence explaining why this hybrid architecture uses LSTM/CNN-1D/XGBoost instead of one model for all zones.

Keep the total response under 650 words. Avoid vague statements. Do not invent numerical values not supplied in the diagnosis results."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        return resp.choices[0].message.content

    except Exception:
        # API unreachable or timed out — return a rich local report instead
        return _local_report(r1, r2, r3)


# ════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════

def make_report(r1, r2, r3, llm_text: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 62
    lines = [
        sep,
        "  PV PLANT FAULT DIAGNOSIS REPORT",
        f"  Timestamp  : {ts}",
        "  System     : 5 kW Grid-Connected PV Installation",
        "  AI Engine  : Hybrid Architecture — Univ. of Moratuwa",
        sep, "",
        "ZONE 1 — PV ARRAY FAULT DIAGNOSIS",
        f"  Fault Type : {r1['label'] if r1 else 'No data provided'}",
    ]
    if r1 and r1.get("s2_label"):
        lines.append(f"  Stage 2    : {r1['s2_label']}")
        if r1.get("stage2_array"):
            lines.append(f"  Array      : {r1['stage2_array']}")
        if r1.get("stage2_fault_type") == "Short-Circuit":
            lines.append(f"  R_LL       : {r1.get('r_ll', 'Unavailable')}")
            lines.append(f"  SC Severity: {r1.get('sc_severity', 'Unavailable')}")
        elif r1.get("stage2_fault_type") == "Open-Circuit":
            lines.append("  R_LL       : Not applicable for open-circuit fault")

    lines += [
        "", "ZONE 2 — IGBT INVERTER FAULT DIAGNOSIS",
        f"  Status     : {r2['label'] if r2 else 'No data provided'}",
        "", "ZONE 3 — DC-LINK CAPACITOR HEALTH",
    ]
    if r3:
        lines += [
            f"  ESR        : {r3['esr']:.4f} Ω  (Fault threshold: 0.30 Ω)",
            f"  Degradation: {r3['deg']:.1f}%",
            f"  Status     : {r3['label']}",
            f"  Sensors    : Irr={r3['irr']:.0f} W/m² | T={r3['tamb']:.1f}°C | "
            f"V_avg={r3['vavg']:.1f} V | V_ripple={r3['vrip']:.4f} V",
        ]
    else:
        lines.append("  Status     : No data provided")

    if llm_text:
        lines += ["", sep, "  AI MAINTENANCE RECOMMENDATIONS", sep, "", llm_text]
    lines += ["", sep]
    return "\n".join(lines)


def _pdf_safe(text) -> str:
    """Make strings safe for ReportLab built-in fonts."""
    text = "" if text is None else str(text)
    replacements = {
        "Ω": " Ohm", "≥": ">=", "≤": "<=", "–": "-", "—": "-",
        "α": "alpha", "β": "beta", "°": " deg ", "•": "-",
        "→": "->", "×": "x", "²": "2",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def _confidence(result) -> str:
    if not result or "probs" not in result:
        return "N/A"
    try:
        return f"{float(np.max(result['probs'])) * 100:.1f}%"
    except Exception:
        return "N/A"


def make_pdf_report(r1, r2, r3, llm_text: str) -> bytes:
    """Create a presentation-quality PDF report for the dashboard download."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=colors.HexColor("#064e3b"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSub", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#475569"), spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.HexColor("#0e7490"), spaceBefore=10, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["Normal"], fontSize=8.4, leading=11,
        textColor=colors.HexColor("#0f172a"), spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Body", parent=styles["Normal"], fontSize=9.2, leading=12.5,
        textColor=colors.HexColor("#0f172a"), spaceAfter=5,
    ))

    story = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("PV Plant Fault Diagnosis Report", styles["ReportTitle"]))
    story.append(Paragraph(
        _pdf_safe(f"Hybrid AI Architecture for Solar PV Fault Diagnosis | 5 kW Grid-Connected PV System | Generated: {ts}"),
        styles["ReportSub"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0e7490")))
    story.append(Spacer(1, 6))

    z1_status = r1["label"] if r1 else "No data provided"
    z2_status = r2["label"] if r2 else "No data provided"
    z3_status = r3["label"] if r3 else "No data provided"
    overall = "FAULT DETECTED" if ((r1 and r1.get("cls") != 0) or (r2 and r2.get("cls") != 0) or (r3 and r3.get("fault"))) else "NORMAL / MONITORING"

    story.append(Paragraph("Executive Summary", styles["Section"]))
    summary_data = [
        ["System Status", _pdf_safe(overall)],
        ["Zone 1 - PV Array", _pdf_safe(z1_status), "Confidence", _confidence(r1)],
        ["Zone 2 - IGBT Inverter", _pdf_safe(z2_status), "Confidence", _confidence(r2)],
        ["Zone 3 - DC-Link Capacitor", _pdf_safe(z3_status), "Confidence", "Regression output"],
    ]
    table = Table(summary_data, colWidths=[42*mm, 76*mm, 28*mm, 30*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#ecfeff")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#0f172a")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8.2),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(table)

    story.append(Paragraph("Zone-Specific Diagnostic Outputs", styles["Section"]))
    detail_rows = [
        ["Zone", "Primary Output", "Engineering Detail"],
        ["Zone 1", _pdf_safe(z1_status), "No Stage 2 output"],
        ["Zone 2", _pdf_safe(z2_status), _pdf_safe("Healthy inverter" if not r2 or r2.get("cls") == 0 else "Open-circuit IGBT switch detected from alpha-beta current trajectory")],
        ["Zone 3", _pdf_safe(z3_status), "No capacitor sample"],
    ]
    if r1 and r1.get("s2_label"):
        if r1.get("stage2_fault_type") == "Short-Circuit":
            detail_rows[1][2] = _pdf_safe(f"Localised to {r1.get('stage2_array')}; estimated R_LL={r1.get('r_ll')}; severity={r1.get('sc_severity')}")
        else:
            detail_rows[1][2] = _pdf_safe(f"Localised to {r1.get('stage2_array')}; R_LL not applicable for open-circuit")
    if r3:
        detail_rows[3][2] = _pdf_safe(
            f"ESR={r3['esr']:.4f} Ohm; estimated degradation={r3['deg']:.1f}%; "
            f"Irr={r3['irr']:.0f} W/m2; T={r3['tamb']:.1f} deg C; V_ripple={r3['vrip']:.4f} V"
        )
    diag_table = Table(detail_rows, colWidths=[24*mm, 58*mm, 94*mm])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8.0),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(diag_table)

    story.append(Paragraph("Engineering Analysis", styles["Section"]))
    if not llm_text:
        llm_text = _local_report(r1, r2, r3)
    clean = _pdf_safe(llm_text)
    clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean)
    for block in clean.split("\n"):
        block = block.strip()
        if not block:
            story.append(Spacer(1, 4))
            continue
        if block.startswith("#"):
            story.append(Paragraph(block.replace("#", "").strip(), styles["Section"]))
        else:
            story.append(Paragraph(block, styles["Body"]))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1")))
    story.append(Paragraph(
        "Note: This report is an AI-assisted diagnostic aid. Final maintenance decisions should be verified with electrical measurements, insulation testing, thermal inspection, and site safety procedures.",
        styles["Small"],
    ))
    doc.build(story)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────
    st.markdown('<div class="main-title">▣ PV-FAULT//ENGINEERING_DIAGNOSTIC_TERMINAL</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-sub">HYBRID AI ARCHITECTURE · 5 kW GRID-CONNECTED PV SYSTEM · UOM ELECTRICAL ENGINEERING</div>',
        unsafe_allow_html=True
    )

    # ── Load models ──────────────────────────────────────────────
    M = load_models()
    if not M.get("ok"):
        st.error(f"Model loading failed: {M.get('err')}")
        st.info("Ensure all files are in the `models/` folder. See README.md.")
        return

    # ── Session state ────────────────────────────────────────────
    defaults = {"r1": None, "r2": None, "r3": None,
                "df1": None, "df1a": None, "df2": None, "df3": None,
                "llm": ""}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Overall status strip ─────────────────────────────────────
    oc1, oc2, oc3 = st.columns(3)
    for col, r, zone in [(oc1, st.session_state.r1, 1),
                         (oc2, st.session_state.r2, 2),
                         (oc3, st.session_state.r3, 3)]:
        c, lbl = zone_status(r, zone)
        zone_names = {1:"Zone 1 — PV Array", 2:"Zone 2 — IGBT Inverter", 3:"Zone 3 — Capacitor"}
        col.markdown(badge(f"{zone_names[zone]}: {lbl}", c), unsafe_allow_html=True)

    active_faults = sum([
        1 if st.session_state.r1 and st.session_state.r1["cls"] != 0 else 0,
        1 if st.session_state.r2 and st.session_state.r2["cls"] != 0 else 0,
        1 if st.session_state.r3 and st.session_state.r3["fault"] else 0,
    ])
    terminal_color = "#ef4444" if active_faults else "#22c55e"
    st.markdown(terminal_panel("system bus", {
        "MODE": "LIVE INFERENCE / DEMO",
        "MODEL STACK": "Zone1 LSTM · Zone2 CNN-1D · Zone3 XGBoost",
        "ACTIVE FAULTS": active_faults,
        "TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, terminal_color), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Three-zone layout ────────────────────────────────────────
    z1, z2, z3 = st.columns(3, gap="medium")

    # ══════════════════════════════════════════════════
    # ZONE 1 — PV Array
    # ══════════════════════════════════════════════════
    with z1:
        st.markdown('<div class="zone-title">🌞 Zone 1 — PV Array (LSTM)</div>', unsafe_allow_html=True)

        a, b, c, d = st.columns(4)
        if a.button("Healthy",  key="z1h"):  st.session_state.df1 = sample("z1_healthy.csv")
        if b.button("SC Fault", key="z1sc"): st.session_state.df1 = sample("z1_sc.csv")
        if c.button("OC Fault", key="z1oc"): st.session_state.df1 = sample("z1_oc.csv")
        if d.button("DG Fault", key="z1dg"): st.session_state.df1 = sample("z1_dg.csv")

        uf1 = st.file_uploader("Upload IV Sweep CSV", type="csv", key="uf1",
                                help="Required columns: V, I, P, Irr, T")
        if uf1:
            df_temp = clean_df(pd.read_csv(uf1, encoding="utf-8-sig"))
            required = ["V", "I", "P", "Irr", "T"]
            missing = [col for col in required if col not in df_temp.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
                if "V_1" in df_temp.columns or "I_1" in df_temp.columns:
                    st.info("💡 It looks like you uploaded a per-array CSV (Stage 2 data) here. Please upload it in the 'Stage 2 — Array Localisation' section below after running Stage 1.")
                st.session_state.df1 = None
            else:
                st.session_state.df1 = df_temp

        if st.session_state.df1 is not None:
            st.plotly_chart(plot_iv(st.session_state.df1),
                            use_container_width=True, config=_PLOT_CFG)
            if st.button("🔍 Run Zone 1 Diagnosis", key="r1btn",
                         type="primary", use_container_width=True):
                with st.spinner("Analysing IV curve…"):
                    try:
                        st.session_state.r1 = run_z1(
                            st.session_state.df1,
                            st.session_state.df1a,
                            M,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Zone 1 error: {e}")

        if st.session_state.r1:
            r = st.session_state.r1
            c_col, lbl = zone_status(r, 1)
            st.markdown(badge(r["label"], c_col), unsafe_allow_html=True)
            st.markdown(prob_bars(Z1_S1_LABELS, r["probs"], r["cls"]), unsafe_allow_html=True)

            # Stage 2 localisation
            if r["cls"] in [2, 3]:
                st.markdown("**Stage 2 — Array Localisation / R_LL Severity**")
                st.markdown('<div class="demo-caption">Select the matching per-array sample after Stage 1 detects SC or OC</div>', unsafe_allow_html=True)
                if r["cls"] == 2:
                    st.caption("SC localisation samples - output includes Array and R_LL")
                    sc_cols = st.columns(4)
                    sc_opts = [
                        ("A1 0.05Ω", "z1_sc_array1_rll_0p05.csv"),
                        ("A1 10Ω",   "z1_sc_array1_rll_10p0.csv"),
                        ("A1 20Ω",   "z1_sc_array1_rll_20p0.csv"),
                        ("A1 30Ω",   "z1_sc_array1_rll_30p0.csv"),
                        ("A2 0.05Ω", "z1_sc_array2_rll_0p05.csv"),
                        ("A2 10Ω",   "z1_sc_array2_rll_10p0.csv"),
                        ("A2 20Ω",   "z1_sc_array2_rll_20p0.csv"),
                        ("A2 30Ω",   "z1_sc_array2_rll_30p0.csv"),
                    ]
                    for i, (txt, fname) in enumerate(sc_opts):
                        if sc_cols[i % 4].button(txt, key=f"z1s2_{fname}", use_container_width=True):
                            st.session_state.df1a = sample(fname); st.session_state.llm = ""
                elif r["cls"] == 3:
                    st.caption("OC localisation samples - R_LL is not applicable")
                    oc_cols = st.columns(2)
                    if oc_cols[0].button("OC Array 1", key="z1oc_a1", use_container_width=True):
                        st.session_state.df1a = sample("z1_oc_array1.csv"); st.session_state.llm = ""
                    if oc_cols[1].button("OC Array 2", key="z1oc_a2", use_container_width=True):
                        st.session_state.df1a = sample("z1_oc_array2.csv"); st.session_state.llm = ""

                uf1a = st.file_uploader("Upload Per-Array CSV", type="csv", key="uf1a",
                                         help="Columns: V_1, I_1, P_1, V_2, I_2, P_2, Irr, T")
                if uf1a:
                    df_temp = clean_df(pd.read_csv(uf1a, encoding="utf-8-sig"))
                    required = ["V_1", "I_1", "P_1", "V_2", "I_2", "P_2", "Irr", "T"]
                    missing = [col for col in required if col not in df_temp.columns]
                    if missing:
                        st.error(f"Uploaded Stage 2 CSV is missing required columns: {', '.join(missing)}")
                        st.session_state.df1a = None
                    else:
                        st.session_state.df1a = df_temp
                if st.session_state.df1a is not None:
                    if st.button("🔍 Localise Fault", key="r1s2btn",
                                 type="secondary", use_container_width=True):
                        with st.spinner("Localising…"):
                            try:
                                st.session_state.r1 = run_z1(
                                    st.session_state.df1,
                                    st.session_state.df1a, M)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Stage 2 error: {e}")
            if r.get("s2_label"):
                st.markdown(badge(f"📍 {r['s2_label']}", "#8b5cf6"), unsafe_allow_html=True)
                detail_rows = {
                    "FAULT ROUTE": r.get("stage2_fault_type", "Unknown"),
                    "LOCALISATION": r.get("stage2_array", "Unknown"),
                }
                if r.get("stage2_fault_type") == "Short-Circuit":
                    detail_rows["R_LL OUTPUT"] = r.get("r_ll", "Unavailable")
                    detail_rows["SC SEVERITY"] = r.get("sc_severity", "Unavailable")
                else:
                    detail_rows["R_LL OUTPUT"] = "N/A for open-circuit"
                st.markdown(terminal_panel("zone 1 stage-2 decoded output", detail_rows, "#8b5cf6"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # ZONE 2 — IGBT Inverter
    # ══════════════════════════════════════════════════
    with z2:
        st.markdown('<div class="zone-title">⚡ Zone 2 — IGBT Inverter (CNN-1D)</div>', unsafe_allow_html=True)

        st.markdown('<div class="demo-caption">Demo samples - all 7 inverter classes</div>', unsafe_allow_html=True)
        z2r1 = st.columns(4)
        z2r2 = st.columns(3)
        if z2r1[0].button("Healthy", key="z2h", use_container_width=True):
            st.session_state.df2 = sample("z2_healthy.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r1[1].button("T1 Fault", key="z2t1", use_container_width=True):
            st.session_state.df2 = sample("z2_t1_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r1[2].button("T2 Fault", key="z2t2", use_container_width=True):
            st.session_state.df2 = sample("z2_t2_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r1[3].button("T3 Fault", key="z2t3", use_container_width=True):
            st.session_state.df2 = sample("z2_t3_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r2[0].button("T4 Fault", key="z2t4", use_container_width=True):
            st.session_state.df2 = sample("z2_t4_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r2[1].button("T5 Fault", key="z2t5", use_container_width=True):
            st.session_state.df2 = sample("z2_t5_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""
        if z2r2[2].button("T6 Fault", key="z2t6", use_container_width=True):
            st.session_state.df2 = sample("z2_t6_fault.csv"); st.session_state.r2 = None; st.session_state.llm = ""

        uf2 = st.file_uploader("Upload Current Trajectory CSV", type="csv", key="uf2",
                                help="Columns: I_alpha, I_beta, Irradiance, Temperature (≥32 rows)")
        if uf2:
            df_temp = clean_df(pd.read_csv(uf2, encoding="utf-8-sig"))
            required = ["I_alpha", "I_beta", "Irradiance", "Temperature"]
            missing = [col for col in required if col not in df_temp.columns]
            if missing:
                st.error(f"Uploaded Zone 2 CSV is missing required columns: {', '.join(missing)}")
                st.session_state.df2 = None
            else:
                st.session_state.df2 = df_temp

        if st.session_state.df2 is not None:
            if len(st.session_state.df2) < WIN_Z2:
                st.warning(f"CSV needs at least {WIN_Z2} rows. Found {len(st.session_state.df2)}.")
            else:
                st.plotly_chart(plot_ab(st.session_state.df2),
                                use_container_width=True, config=_PLOT_CFG)
                if st.button("🔍 Run Zone 2 Diagnosis", key="r2btn",
                             type="primary", use_container_width=True):
                    with st.spinner("Analysing αβ trajectory…"):
                        try:
                            st.session_state.r2 = run_z2(st.session_state.df2, M)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Zone 2 error: {e}")

        if st.session_state.r2:
            r = st.session_state.r2
            c_col, _ = zone_status(r, 2)
            st.markdown(badge(r["label"], c_col), unsafe_allow_html=True)
            st.markdown(prob_bars(Z2_LABELS, r["probs"], r["cls"]), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # ZONE 3 — DC-Link Capacitor
    # ══════════════════════════════════════════════════
    with z3:
        st.markdown('<div class="zone-title">🔋 Zone 3 — DC-Link Capacitor (XGBoost)</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        if a.button("Healthy",  key="z3h"): st.session_state.df3 = sample("z3_healthy.csv")
        if b.button("Degraded", key="z3d"): st.session_state.df3 = sample("z3_degraded.csv")

        uf3 = st.file_uploader("Upload Measurement CSV", type="csv", key="uf3",
                                help="Columns: Irradiance, T_ambient, I_rms_high, V_avg, V_ripple")
        if uf3:
            df_temp = clean_df(pd.read_csv(uf3, encoding="utf-8-sig"))
            required = ["Irradiance", "T_ambient", "I_rms_high", "V_avg", "V_ripple"]
            missing = [col for col in required if col not in df_temp.columns]
            if missing:
                st.error(f"Uploaded Zone 3 CSV is missing required columns: {', '.join(missing)}")
                st.session_state.df3 = None
            else:
                st.session_state.df3 = df_temp

        if st.session_state.df3 is not None:
            row = st.session_state.df3.iloc[0]
            ma, mb = st.columns(2)
            ma.metric("Irradiance",  f"{row.get('Irradiance',0):.0f} W/m²")
            ma.metric("V_avg",       f"{row.get('V_avg',0):.1f} V")
            mb.metric("T_ambient",   f"{row.get('T_ambient',0):.1f} °C")
            mb.metric("V_ripple",    f"{row.get('V_ripple',0):.4f} V")
            if st.button("🔍 Run Zone 3 Diagnosis", key="r3btn",
                         type="primary", use_container_width=True):
                with st.spinner("Estimating ESR…"):
                    try:
                        st.session_state.r3 = run_z3(st.session_state.df3, M)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Zone 3 error: {e}")

        if st.session_state.r3:
            r = st.session_state.r3
            st.plotly_chart(plot_gauge(r["esr"], r["deg"]),
                            use_container_width=True, config=_PLOT_CFG)
            c_col, _ = zone_status(r, 3)
            st.markdown(badge(r["label"], c_col), unsafe_allow_html=True)
            st.markdown(
                terminal_panel("zone 3 health-index output", {
                    "PREDICTED ESR": f'{r["esr"]:.4f} Ω',
                    "AGEING BAND": "0.15–0.30 Ω",
                    "FAULT THRESHOLD": "≥ 0.30 Ω",
                    "DEGRADATION %": f'{r["deg"]:.1f}%',
                    "INTERPRETATION": "Hard fault" if r["fault"] else ("Ageing / warning" if r["esr"] > 0.15 else "Healthy"),
                }, c_col),
                unsafe_allow_html=True,
            )

    # ── LLM + Report ─────────────────────────────────────────────
    any_result = any([st.session_state.r1, st.session_state.r2, st.session_state.r3])
    if any_result:
        st.markdown("---")
        llm_col, rep_col = st.columns([3, 1])

        with llm_col:
            st.markdown("### ▣ ENGINEERING ANALYSIS CONSOLE")
            if st.button("Generate Maintenance Analysis",
                         type="primary", use_container_width=True):
                with st.spinner("Generating analysis… (up to 8 s)"):
                    st.session_state.llm = gpt_explain(
                        st.session_state.r1,
                        st.session_state.r2,
                        st.session_state.r3,
                    )
            if st.session_state.llm:
                formatted = st.session_state.llm.replace("\n", "<br>").replace("**","<b>",1)
                # simple bold markdown conversion
                import re
                text = st.session_state.llm
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                text = text.replace("\n", "<br>")
                st.markdown(f'<div class="llm-box">{text}</div>', unsafe_allow_html=True)

        with rep_col:
            st.markdown("### 📄 Report")
            report = make_report(
                st.session_state.r1,
                st.session_state.r2,
                st.session_state.r3,
                st.session_state.llm,
            )
            ts_str = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report,
                file_name=f"pv_fault_report_{ts_str}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            try:
                pdf_bytes = make_pdf_report(
                    st.session_state.r1,
                    st.session_state.r2,
                    st.session_state.r3,
                    st.session_state.llm,
                )
                st.download_button(
                    label="⬇️ Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"pv_fault_report_{ts_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF report generation failed: {e}")
            st.markdown("&nbsp;")
            if st.button("🔄 Reset All Zones", use_container_width=True):
                for k in defaults:
                    st.session_state[k] = defaults[k]
                st.rerun()


if __name__ == "__main__":
    main()
