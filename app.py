# ================================================================
#  PV PLANT FAULT MONITORING SYSTEM
#  Hybrid AI Architecture for Solar PV Fault Diagnosis
#  University of Moratuwa | Dept. of Electrical Engineering
# ================================================================

import os
import json
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
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stSidebar"]          { background: #161b27; }

.main-title {
    text-align: center; font-size: 30px; font-weight: 900;
    color: #f1f5f9; letter-spacing: -0.5px; margin-bottom: 2px;
}
.main-sub {
    text-align: center; font-size: 12px; color: #475569; margin-bottom: 18px;
}
.zone-title {
    font-size: 15px; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid #1d4ed8; padding-bottom: 5px; margin-bottom: 10px;
}
.badge {
    border-radius: 7px; padding: 7px 12px; font-size: 13px;
    font-weight: 600; text-align: center; margin: 6px 0; display: block;
}
.metric-row {
    background: #1a2133; border-radius: 5px; padding: 6px 11px;
    margin: 3px 0; font-size: 12px; border-left: 3px solid #334155; color: #cbd5e1;
}
.prob-wrap { margin: 2px 0; }
.prob-label {
    display: flex; justify-content: space-between;
    font-size: 11px; color: #94a3b8; margin-bottom: 2px;
}
.prob-track {
    height: 3px; background: #1e2433; border-radius: 2px;
}
.llm-box {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
    padding: 16px; color: #e2e8f0; line-height: 1.75; font-size: 13px;
}
div[data-testid="stButton"] > button { border-radius: 7px; font-weight: 600; }
div[data-testid="stDownloadButton"] > button {
    border-radius: 7px; font-weight: 600; width: 100%;
}
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

    s2_cls, s2_lbl = None, None
    if cls in [2, 3] and df_arr is not None:
        x2    = torch.tensor(prep_z1_arr(df_arr, m["z1s2_sc"]))
        logit2 = m["z1s2"](x2)
        s2_cls = int(logit2.argmax(1))
        s2_lbl = Z1_S2_LABELS[s2_cls]

    return {"cls": cls, "label": Z1_S1_LABELS[cls],
            "probs": prob, "s2_cls": s2_cls, "s2_label": s2_lbl}


def run_z2(df: pd.DataFrame, m: dict) -> dict:
    """Use sliding-window max-confidence inference for Zone 2."""
    _, prob = prep_z2_best_window(df, m["z2_sc"], m["z2"])
    cls     = int(prob.argmax())
    return {"cls": cls, "label": Z2_LABELS[cls], "probs": prob}


def run_z3(df: pd.DataFrame, m: dict) -> dict:
    X_s   = prep_z3(df, m["z3_sc"])
    esr   = float(np.clip(m["z3"].predict(X_s)[0], 0.15, 0.40))
    deg   = float(np.clip((esr - 0.15) / (0.30 - 0.15) * 100, 0, 100))
    fault = esr >= 0.30
    row   = df.iloc[0]
    return {
        "esr": esr, "deg": deg, "fault": fault,
        "label": "Degraded — Replacement Recommended" if fault else f"Healthy ({deg:.1f}% degraded)",
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
                {"range": [0,  50], "color": "#14532d18"},
                {"range": [50, 80], "color": "#71400a18"},
                {"range": [80,100], "color": "#450a0a28"},
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
            lines.append(f"  - Localised to: {r1['s2_label']}")
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
        elif r3["deg"] >= 70:
            lines.append(
                f"- **Zone 3 (Ageing)**: ESR = {r3['esr']:.4f} Ω ({r3['deg']:.1f}% degraded). "
                "Approaching fault threshold — plan replacement within the next maintenance cycle."
            )
    if not faults:
        lines.append("No faults detected — all zones within normal operating limits.")

    # ── Recommended Actions ──────────────────────────────────────
    lines.append("\n**3. Recommended Actions**")
    if r1 and r1["cls"] == 2:
        lines.append("- [IMMEDIATE] Inspect Zone 1 PV array for short-circuit fault; isolate affected string and measure I–V curves at module level.")
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
            if r1.get("s2_label"): line += f" → Localised as: {r1['s2_label']}"
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

Write a concise maintenance report including:
1. **System Health Summary** — overall plant status in 2 sentences
2. **Fault Analysis** — for each detected fault: the physical cause, effect on system performance, and estimated power loss impact
3. **Recommended Actions** — prioritised list with urgency tag: [IMMEDIATE], [WITHIN 48H], or [SCHEDULED]
4. **Risk Assessment** — what happens if faults are left unaddressed

Be specific, use technical language appropriate for an electrical engineer, and keep the total response under 350 words."""

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
        lines.append(f"  Localised  : {r1['s2_label']}")

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


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────
    st.markdown('<div class="main-title">⚡ PV Plant Fault Monitoring System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-sub">Hybrid AI Architecture for Solar PV Fault Diagnosis &nbsp;|&nbsp; '
        'University of Moratuwa, Department of Electrical Engineering</div>',
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
                st.markdown("**Stage 2 — Array Localisation**")
                a2, b2 = st.columns(2)
                if a2.button("Sample Arrays",  key="z1arr_s"):
                    st.session_state.df1a = sample("z1_sc_arrays.csv")
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

    # ══════════════════════════════════════════════════
    # ZONE 2 — IGBT Inverter
    # ══════════════════════════════════════════════════
    with z2:
        st.markdown('<div class="zone-title">⚡ Zone 2 — IGBT Inverter (CNN-1D)</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        if a.button("Healthy",   key="z2h"):  st.session_state.df2 = sample("z2_healthy.csv")
        if b.button("T1 Fault",  key="z2t1"): st.session_state.df2 = sample("z2_t1_fault.csv")

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
                f'<div class="metric-row">'
                f'ESR: <b>{r["esr"]:.4f} Ω</b> &nbsp;|&nbsp; '
                f'Threshold: 0.30 Ω &nbsp;|&nbsp; '
                f'Degradation: <b>{r["deg"]:.1f}%</b></div>',
                unsafe_allow_html=True,
            )

    # ── LLM + Report ─────────────────────────────────────────────
    any_result = any([st.session_state.r1, st.session_state.r2, st.session_state.r3])
    if any_result:
        st.markdown("---")
        llm_col, rep_col = st.columns([3, 1])

        with llm_col:
            st.markdown("### 🤖 AI Maintenance Assistant")
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
            st.markdown("&nbsp;")
            if st.button("🔄 Reset All Zones", use_container_width=True):
                for k in defaults:
                    st.session_state[k] = defaults[k]
                st.rerun()


if __name__ == "__main__":
    main()
