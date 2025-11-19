import ssl
import pandas as pd
import numpy as np

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

print("Connecting to VitalDB API...")

# Get all cases and tracks
df_trks = pd.read_csv('https://api.vitaldb.net/trks')
df_cases = pd.read_csv('https://api.vitaldb.net/cases')

print(f"Total VitalDB cases: {len(df_cases):,}")
print(f"Total track records: {len(df_trks):,}")

# Filter cases: General Anesthesia
ga_cases = df_cases[df_cases['ane_type'] == 'General']['caseid'].unique()
print(f"\nGeneral Anesthesia cases: {len(ga_cases):,}")

# Filter tracks: Primus/AWP (Airway Pressure)
awp_cases = df_trks[df_trks['tname'] == 'Primus/AWP']['caseid'].unique()
print(f"Cases with Primus/AWP (Airway Pressure): {len(awp_cases):,}")

# Intersection - our target population
valid_case_ids = np.intersect1d(ga_cases, awp_cases)
print(f"\n✅ MAXIMUM AVAILABLE: {len(valid_case_ids):,} cases")
print(f"   (General Anesthesia + Airway Pressure waveforms)")

# Additional filtering for other signals (optional)
df_trks_valid = df_trks[df_trks['caseid'].isin(valid_case_ids)]

flow_cases = df_trks[df_trks['tname'] == 'Primus/FLOW_AIR']['caseid'].unique()
flow_intersection = np.intersect1d(valid_case_ids, flow_cases)
print(f"\nWith Flow data: {len(flow_intersection):,} cases")

co2_cases = df_trks[df_trks['tname'] == 'Primus/CO2']['caseid'].unique()
co2_intersection = np.intersect1d(valid_case_ids, co2_cases)
print(f"With CO2 data: {len(co2_intersection):,} cases")

# All three
all_three = np.intersect1d(flow_intersection, co2_intersection)
print(f"With all three (AWP + Flow + CO2): {len(all_three):,} cases")

print("\n" + "="*60)
print(f"RECOMMENDATION:")
print(f"  - Minimum requirement (AWP only): {len(valid_case_ids):,} cases")
print(f"  - Your current scan captured: 100 cases")
print(f"  - You validated on: 10 cases")
print("="*60)

