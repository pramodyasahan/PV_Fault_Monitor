# PV Fault Monitoring System — Model Context

Hybrid AI Architecture for Solar PV Fault Diagnosis
University of Moratuwa | 5 kW Grid-Connected PV System

---

## Zone 1 — PV Array Fault Diagnosis (Two-Stage LSTM)

### Stage 1 — System-Level Fault Classifier

**Model:** Two-layer LSTM | 209,732 params  
**Input CSV columns:** `V, I, P, Irr, T`  
- Full IV sweep (~200–900 rows), sorted by voltage ascending
- Preprocessed: subsampled to 50 equally-spaced points, Z-score scaled
- Tensor shape fed to model: `[1, 50, 5]`

**Output:** One of 4 classes

| Class ID | Label | Meaning |
|---|---|---|
| 0 | Healthy | No fault detected |
| 1 | Degradation (DG) | Series resistance degradation in PV strings |
| 2 | Short-Circuit (SC) | Line-to-line fault — triggers Stage 2 |
| 3 | Open-Circuit (OC) | String disconnection — triggers Stage 2 |

**If output is SC or OC → pass to Stage 2**

---

### Stage 2 — Array Localisation + SC Severity Grader

**Model:** Two-layer LSTM | 211,658 params  
**Input CSV columns:** `V_1, I_1, P_1, V_2, I_2, P_2, Irr, T`  
- Dual-array IV sweep (both arrays measured simultaneously)
- Preprocessed: subsampled to 50 points, Z-score scaled
- Tensor shape: `[1, 50, 8]`

**Output:** One of 10 classes

| Class ID | Label |
|---|---|
| 0 | SC — Array 1, R_LL = 0.05 Ω (Severe) |
| 1 | SC — Array 1, R_LL = 10 Ω |
| 2 | SC — Array 1, R_LL = 20 Ω |
| 3 | SC — Array 1, R_LL = 30 Ω (Mild) |
| 4 | SC — Array 2, R_LL = 0.05 Ω (Severe) |
| 5 | SC — Array 2, R_LL = 10 Ω |
| 6 | SC — Array 2, R_LL = 20 Ω |
| 7 | SC — Array 2, R_LL = 30 Ω (Mild) |
| 8 | Open-Circuit — Array 1 |
| 9 | Open-Circuit — Array 2 |

---

## Zone 2 — IGBT Inverter Fault Diagnosis (CNN-1D)

**Model:** Three-block CNN-1D | 32,679 params  
**Input CSV columns:** `I_alpha, I_beta, Irradiance, Temperature`  
- Time-series of Clarke-transformed αβ currents (minimum 32 rows required)
- Preprocessed: take first 32 rows, Z-score scaled per feature, transposed
- Tensor shape: `[1, 4, 32]` (4 channels, window of 32)

**Output:** One of 7 classes

| Class ID | Label | Meaning |
|---|---|---|
| 0 | Healthy | All IGBTs operating normally |
| 1 | T1 Open-Circuit | IGBT T1 open-circuit fault |
| 2 | T2 Open-Circuit | IGBT T2 open-circuit fault |
| 3 | T3 Open-Circuit | IGBT T3 open-circuit fault |
| 4 | T4 Open-Circuit | IGBT T4 open-circuit fault |
| 5 | T5 Open-Circuit | IGBT T5 open-circuit fault |
| 6 | T6 Open-Circuit | IGBT T6 open-circuit fault |

---

## Zone 3 — DC-Link Capacitor Health (XGBoost Regressor)

**Model:** XGBoost Regressor | Optuna-tuned (904 trees)  
**Input CSV columns:** `Irradiance, T_ambient, I_rms_high, V_avg, V_ripple`  
- Single row (one operating point) or batch
- Preprocessed: Z-score scaled
- Array shape: `[1, 5]`

**Output:** Continuous ESR value in Ohms → derived outputs

| Output | Formula | Meaning |
|---|---|---|
| ESR (Ω) | Direct model output, clipped to [0.15, 0.40] | Equivalent Series Resistance of capacitor |
| Degradation % | `(ESR − 0.15) / (0.30 − 0.15) × 100`, clipped [0, 100] | 0% = new, 100% = at fault threshold |
| Fault flag | `ESR ≥ 0.30 Ω` | True = degraded, False = healthy |

**ESR threshold:** 0.30 Ω = 2× nominal (0.15 Ω)

---

## File Structure

```
pv_fault_monitor/
├── app.py                         # Streamlit dashboard
├── models/
│   ├── zone1_stage1_lstm.pth      # Zone 1 Stage 1 weights
│   ├── zone1_stage1_scaler.pkl    # Zone 1 Stage 1 StandardScaler (5 features)
│   ├── zone1_stage2_lstm.pth      # Zone 1 Stage 2 weights
│   ├── zone1_stage2_scaler.pkl    # Zone 1 Stage 2 StandardScaler (8 features)
│   ├── zone2_model.pt             # Zone 2 CNN-1D weights
│   ├── zone2_scaler.pkl           # Zone 2 StandardScaler (4 features)
│   ├── zone3_model.pkl            # Zone 3 XGBoost regressor
│   └── zone3_scaler.pkl           # Zone 3 StandardScaler (5 features)
└── sample_data/
    ├── z1_healthy.csv             # Zone 1: healthy IV sweep
    ├── z1_sc.csv                  # Zone 1: short-circuit IV sweep
    ├── z1_oc.csv                  # Zone 1: open-circuit IV sweep
    ├── z1_dg.csv                  # Zone 1: degradation IV sweep
    ├── z1_sc_arrays.csv           # Zone 1 Stage 2: per-array SC data
    ├── z2_healthy.csv             # Zone 2: healthy αβ trajectory
    ├── z2_t1_fault.csv            # Zone 2: T1 fault trajectory
    ├── z3_healthy.csv             # Zone 3: healthy capacitor reading
    ├── z3_degraded.csv            # Zone 3: degraded capacitor reading
    └── z3_warning.csv             # Zone 3: ageing capacitor (ESR=0.28Ω)
```

---

## Key Preprocessing Rules

- **Zone 1:** Always sort IV sweep by `V` ascending before subsampling
- **Zone 2:** Column names must be lowercase: `I_alpha`, `I_beta` (not `I_Alpha`)
- **Zone 2:** Scaler was fit on `[I_alpha, I_beta, Irradiance, Temperature]` in that column order
- **Zone 3:** ESR column is NOT an input — model predicts it from the 5 sensor features only
- **All zones:** Scalers were fit on training data only — always use the saved `.pkl` scaler, never refit
