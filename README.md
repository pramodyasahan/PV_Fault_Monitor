# PV Plant Fault Monitoring System

**Hybrid AI Architecture for Solar PV Fault Diagnosis**
University of Moratuwa | Department of Electrical Engineering | June 2026

A real-time fault monitoring dashboard combining three AI models across three
diagnostic zones of a 5 kW grid-connected PV installation.

---

## Quick Start (Local — Run Today)

### Step 1 — Clone / create the project folder

```
pv_fault_monitor/
├── app.py
├── requirements.txt
├── generate_samples.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml        ← fill in your API key
├── models/                 ← paste all 8 model files here
│   ├── zone1_stage1_lstm.pth
│   ├── zone1_stage1_scaler.pkl
│   ├── zone1_stage2_lstm.pth
│   ├── zone1_stage2_scaler.pkl
│   ├── zone2_model.pt
│   ├── zone2_scaler.pkl
│   ├── zone3_model.pkl
│   └── zone3_scaler.pkl
└── sample_data/            ← paste CSVs from generate_samples.py here
```

### Step 2 — Place model files

Copy all 8 model files you downloaded from Colab into the `models/` folder.

| File name | Source |
|---|---|
| `zone1_stage1_lstm.pth` | Zone 1 Colab (downloaded via `files.download`) |
| `zone1_stage1_scaler.pkl` | Zone 1 Colab |
| `zone1_stage2_lstm.pth` | Zone 1 Colab |
| `zone1_stage2_scaler.pkl` | Zone 1 Colab |
| `zone2_model.pt` | Teammate 210474R — rename from `best_model_checkpoint.pt` |
| `zone2_scaler.pkl` | Teammate 210474R — rename from `scaler.pkl` |
| `zone3_model.pkl` | Zone 3 Colab — rename from `xgboost_esr_zone3.pkl` |
| `zone3_scaler.pkl` | Zone 3 Colab — rename from `scaler_zone3.pkl` |

### Step 3 — Generate sample data

Run `generate_samples.py` in Google Colab (open it, paste into a cell, run).
It downloads `sample_data.zip` — extract and place all CSVs in `sample_data/`.

### Step 4 — Add your OpenAI API key

Edit `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-actual-key-here"
```

### Step 5 — Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Deploy to Streamlit Cloud (Public URL)

### Step 1 — Create a GitHub repository

```bash
git init
git add app.py requirements.txt .streamlit/config.toml .gitignore
git add models/ sample_data/
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pv-fault-monitor.git
git push -u origin main
```

> **Model file sizes**: If any model file exceeds 50 MB, use
> [Git LFS](https://git-lfs.com) or store models on Google Drive and
> download them at app startup. For files under 50 MB each, direct
> commit is fine.

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **Advanced settings** → **Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   ```
5. Click **Deploy** — you get a public URL in ~2 minutes.

---

## How to Use the App

### Running a diagnosis

1. **Click a sample button** (Healthy / SC Fault / OC Fault) to load example data, or
2. **Upload your own CSV** with the required columns.
3. **Click "Run Diagnosis"** — the model predicts and displays results.
4. For Zone 1 SC/OC: a Stage 2 localisation panel appears — upload per-array CSV.
5. **Click "Generate Maintenance Analysis"** — GPT-4o-mini generates a full report.
6. **Click "Download Report"** — saves a `.txt` report file.

### Required CSV formats

| Zone | Columns | Notes |
|---|---|---|
| Zone 1 (System) | V, I, P, Irr, T | Full IV sweep, 200+ rows |
| Zone 1 (Arrays) | V_1, I_1, P_1, V_2, I_2, P_2, Irr, T | Per-array IV sweep |
| Zone 2 | I_alpha, I_beta, Irradiance, Temperature | At least 32 rows |
| Zone 3 | Irradiance, T_ambient, I_rms_high, V_avg, V_ripple | Single row or batch |

---

## Architecture Summary

| Zone | Fault Types | Model | Accuracy |
|---|---|---|---|
| Zone 1 Stage 1 | Healthy / DG / SC / OC | 2-layer LSTM (209,732 params) | 97.22% |
| Zone 1 Stage 2 | Array localisation + SC severity (10 classes) | 2-layer LSTM (211,658 params) | 100.00% |
| Zone 2 | IGBT T1–T6 open-circuit faults (7 classes) | CNN-1D (32,679 params) | 99.88% |
| Zone 3 | ESR regression → degradation % + fault flag | XGBoost (Optuna-tuned) | 96.06% |

---

## Troubleshooting

**Model loading fails**
→ Check all 8 files are in `models/` with exact names as listed above.

**"Sample data not found"**
→ Run `generate_samples.py` in Colab and place CSVs in `sample_data/`.

**GPT explanation says "Add OPENAI_API_KEY"**
→ Edit `.streamlit/secrets.toml` and add your key, then restart the app.

**Zone 2 predictions seem wrong on synthetic data**
→ The Zone 2 model was trained on Simulink-generated αβ currents. For accurate
  predictions, use real measurement data from the IGBT simulation dataset.

**XGBoost version warning**
→ Safe to ignore. The model was saved with an older XGBoost version.
  Predictions remain correct.
