# ================================================================
#  generate_samples.py
#  Run this script in Google Colab BEFORE running the Streamlit app.
#  It extracts representative sample CSV files from your original
#  Simulink datasets and saves them to sample_data/ folder.
#
#  Usage:
#    1. Upload this file to Colab (or paste into a cell)
#    2. Mount Google Drive
#    3. Update DATA_DIR to your data folder
#    4. Run — it creates 7 CSV files
#    5. Download all CSVs to pv_fault_monitor/sample_data/
# ================================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np

# ── UPDATE THIS to your folder on Google Drive ──────────────────
DATA_DIR = '/content/drive/MyDrive/'   # e.g. '/content/drive/MyDrive/FYP_Data/'
OUT_DIR  = '/content/sample_data/'
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# ZONE 1 SAMPLES (IV sweep data)
# ────────────────────────────────────────────────────────────────

def extract_z1_sweep(df_raw, group_keys, first_n_points=None, label='', out_name=''):
    """Take the first complete IV sweep from a dataset and rename columns."""
    df = df_raw.dropna(subset=['V','I','P','Irr','T'])
    keys = df.groupby(group_keys).groups
    first_key = list(keys.keys())[0]
    sweep = df[df[group_keys[0]] == first_key[0]]
    if len(group_keys) > 1:
        for i, k in enumerate(group_keys[1:], 1):
            sweep = sweep[sweep[k] == first_key[i]]
    sweep = sweep[['V','I','P','Irr','T']].sort_values('V').reset_index(drop=True)
    if first_n_points:
        sweep = sweep.iloc[:first_n_points]
    sweep.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    print(f'  Saved {out_name}: {len(sweep)} rows  {label}')
    return sweep


print('\n── Zone 1: Healthy ──')
df = pd.read_csv(os.path.join(DATA_DIR, 'Combined_PV_Sweep_HEALTHY_SYSTEM.csv'))
df = df.dropna(subset=['Irr','T','V','I','P'])
# Take one (Irr, T) group
g = df.groupby(['Irr','T'])
key = list(g.groups.keys())[5]   # pick Irr=300, T=35 or similar
sw = df[(df['Irr']==key[0]) & (df['T']==key[1])][['V','I','P']].copy()
sw['Irr'] = key[0]; sw['T'] = key[1]
sw = sw.sort_values('V').reset_index(drop=True)
sw.to_csv(os.path.join(OUT_DIR,'z1_healthy.csv'), index=False)
print(f'  Saved z1_healthy.csv: {len(sw)} rows')


print('\n── Zone 1: Short-Circuit (system level) ──')
df = pd.read_csv(os.path.join(DATA_DIR, 'PV_Sweep_SYSTEM_R_LL.csv'))
g = df.groupby(['Irr','T','R_LL'])
# Pick R_LL=10 (moderate severity), first Irr/T combo
key = [k for k in g.groups.keys() if k[2] == 10.0][5]
sw = df[(df['Irr']==key[0]) & (df['T']==key[1]) & (df['R_LL']==key[2])][['V','I','P']].copy()
sw['Irr'] = key[0]; sw['T'] = key[1]
sw = sw.sort_values('V').reset_index(drop=True)
sw.to_csv(os.path.join(OUT_DIR,'z1_sc.csv'), index=False)
print(f'  Saved z1_sc.csv: {len(sw)} rows')


print('\n── Zone 1: Open-Circuit (system level) ──')
df = pd.read_csv(os.path.join(DATA_DIR, 'Combined_PV_Sweep_R_OC_SYSTEM.csv'))
g = df.groupby(['Irr','T'])
key = list(g.groups.keys())[5]
sw = df[(df['Irr']==key[0]) & (df['T']==key[1])][['V','I','P']].copy()
sw['Irr'] = key[0]; sw['T'] = key[1]
sw = sw.sort_values('V').reset_index(drop=True)
sw.to_csv(os.path.join(OUT_DIR,'z1_oc.csv'), index=False)
print(f'  Saved z1_oc.csv: {len(sw)} rows')


print('\n── Zone 1: DG (system level) ──')
df = pd.read_csv(os.path.join(DATA_DIR, 'Combined_PV_Sweep_R_DG_SYSTEM_ONLY.csv'))
df = df[df['R_DG'] != 0.05]   # exclude R_DG=0.05 (indistinct from healthy)
g = df.groupby(['Irr','T','R_DG'])
key = [k for k in g.groups.keys() if k[2] == 10.0][3]
sw = df[(df['Irr']==key[0]) & (df['T']==key[1]) & (df['R_DG']==key[2])][['V','I','P']].copy()
sw['Irr'] = key[0]; sw['T'] = key[1]
sw = sw.sort_values('V').reset_index(drop=True)
sw.to_csv(os.path.join(OUT_DIR,'z1_dg.csv'), index=False)
print(f'  Saved z1_dg.csv: {len(sw)} rows')


print('\n── Zone 1: SC per-array (Stage 2 input) ──')
try:
    df = pd.read_excel(os.path.join(DATA_DIR, 'Combined_PV_Sweep_R_LL_reduced.xlsx'))
    g = df.groupby(['Part','Irr','T','R_LL'])
    key = [k for k in g.groups.keys() if k[0]=='ARRAY1' and k[3]==10.0][3]
    sw = df[(df['Part']==key[0]) & (df['Irr']==key[1]) &
            (df['T']==key[2])   & (df['R_LL']==key[3])][['V_1','I_1','P_1','V_2','I_2','P_2']].copy()
    sw['Irr'] = key[1]; sw['T'] = key[2]
    sw = sw.sort_values('V_1').reset_index(drop=True)
    sw.to_csv(os.path.join(OUT_DIR,'z1_sc_arrays.csv'), index=False)
    print(f'  Saved z1_sc_arrays.csv: {len(sw)} rows')
except Exception as e:
    print(f'  z1_sc_arrays.csv skipped: {e}')


# ────────────────────────────────────────────────────────────────
# ZONE 2 SAMPLES (αβ current trajectory)
# ────────────────────────────────────────────────────────────────
print('\n── Zone 2: NOTE ──')
print('  Zone 2 sample data must come from the IGBT simulation dataset.')
print('  Ask teammate 210474R for the Zone 2 CSV files with columns:')
print('  I_alpha, I_beta, Irradiance, Temperature  (≥32 rows per sample)')
print()
print('  If you have them, save as:')
print('    sample_data/z2_healthy.csv')
print('    sample_data/z2_t1_fault.csv')
print()
print('  As a fallback, synthetic samples are generated below (may not predict correctly):')

def synthetic_z2_healthy(n=100):
    t = np.linspace(0, 4*np.pi, n)
    Ipk = 5.0
    return pd.DataFrame({
        'I_alpha':     Ipk * np.cos(t),
        'I_beta':      Ipk * np.sin(t),
        'Irradiance':  np.full(n, 800.0),
        'Temperature': np.full(n, 35.0),
    })

def synthetic_z2_t1(n=100):
    t = np.linspace(0, 4*np.pi, n)
    return pd.DataFrame({
        'I_alpha':     4.2 * np.cos(t) + 0.6,    # T1 shifts the αβ circle
        'I_beta':      5.0 * np.sin(t),
        'Irradiance':  np.full(n, 800.0),
        'Temperature': np.full(n, 35.0),
    })

z2_h_path = os.path.join(DATA_DIR, 'z2_healthy.csv')
z2_t_path = os.path.join(DATA_DIR, 'z2_t1_fault.csv')

if os.path.exists(z2_h_path):
    import shutil
    shutil.copy(z2_h_path, os.path.join(OUT_DIR, 'z2_healthy.csv'))
    print('  Copied z2_healthy.csv from Drive')
else:
    synthetic_z2_healthy().to_csv(os.path.join(OUT_DIR,'z2_healthy.csv'), index=False)
    print('  Generated synthetic z2_healthy.csv (replace with real data if available)')

if os.path.exists(z2_t_path):
    shutil.copy(z2_t_path, os.path.join(OUT_DIR, 'z2_t1_fault.csv'))
    print('  Copied z2_t1_fault.csv from Drive')
else:
    synthetic_z2_t1().to_csv(os.path.join(OUT_DIR,'z2_t1_fault.csv'), index=False)
    print('  Generated synthetic z2_t1_fault.csv (replace with real data if available)')


# ────────────────────────────────────────────────────────────────
# ZONE 3 SAMPLES (ESR sensor readings)
# ────────────────────────────────────────────────────────────────
print('\n── Zone 3: Capacitor samples ──')
df = pd.read_csv(os.path.join(DATA_DIR, 'Timeseries_Capacitor_degradation_data_Pramodya.csv'))

# Healthy sample (ESR = 0.15, low degradation)
healthy = df[df['ESR (Round)'] == 0.15].iloc[[10]]
healthy_out = healthy[['Irradiance','T_ambient','I_rms_high','V_avg','V_ripple']].copy()
healthy_out.to_csv(os.path.join(OUT_DIR,'z3_healthy.csv'), index=False)
print(f'  Saved z3_healthy.csv: ESR={healthy["ESR (Round)"].values[0]:.2f}Ω')

# Degraded sample (ESR >= 0.30)
degraded = df[df['ESR (Round)'] >= 0.35].iloc[[10]]
degraded_out = degraded[['Irradiance','T_ambient','I_rms_high','V_avg','V_ripple']].copy()
degraded_out.to_csv(os.path.join(OUT_DIR,'z3_degraded.csv'), index=False)
print(f'  Saved z3_degraded.csv: ESR={degraded["ESR (Round)"].values[0]:.2f}Ω')

# ────────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('  DONE — Download all files from /content/sample_data/')
print('  and place them in pv_fault_monitor/sample_data/')
print('='*55)

# Zip for easy download
import shutil
shutil.make_archive('/content/sample_data', 'zip', '/content/sample_data')
print('\n  Also saved as /content/sample_data.zip for bulk download.')

from google.colab import files
files.download('/content/sample_data.zip')
