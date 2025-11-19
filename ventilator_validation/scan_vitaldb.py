import ssl
import vitaldb
import pandas as pd
import numpy as np

# CONFIG
OUTPUT_FILE = 'valid_vitaldb_cases.csv'

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

def scan_vitaldb():
    print("Connecting to VitalDB API...")
    
    # 1. Get Track List
    try:
        # Note: pandas read_csv with https url uses urllib which respects the global ssl context
        df_trks = pd.read_csv('https://api.vitaldb.net/trks')
        df_cases = pd.read_csv('https://api.vitaldb.net/cases')
    except Exception as e:
        print(f"Error connecting to VitalDB: {e}")
        return

    print("Filtering for General Anesthesia cases with Primus/AWP...")
    
    # Filter cases: General Anesthesia
    # Note: 'ane_type' column in cases table
    ga_cases = df_cases[df_cases['ane_type'] == 'General']['caseid'].unique()
    
    # Filter tracks: Primus/AWP (Airway Pressure)
    awp_cases = df_trks[df_trks['tname'] == 'Primus/AWP']['caseid'].unique()
    
    # Intersection
    valid_case_ids = np.intersect1d(ga_cases, awp_cases)
    
    print(f"Found {len(valid_case_ids)} valid General Anesthesia cases with Airway Pressure.")
    
    # Create manifesto
    # We also want to check for Flow and CO2 availability for these cases
    # This might be slow to check one by one if we iterate, but we can check the track list dataframe
    
    valid_records = []
    
    # Optimize: Filter df_trks for our valid cases first
    df_trks_valid = df_trks[df_trks['caseid'].isin(valid_case_ids)]
    
    for caseid in valid_case_ids[:100]: # Limit to 100 for now to be fast, or remove limit for full scan
        case_tracks = df_trks_valid[df_trks_valid['caseid'] == caseid]['tname'].values
        
        has_flow = 'Primus/FLOW_AIR' in case_tracks
        has_co2 = 'Primus/CO2' in case_tracks
        
        valid_records.append({
            'caseid': caseid,
            'has_flow': has_flow,
            'has_co2': has_co2
        })
        
    df_out = pd.DataFrame(valid_records)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_out)} cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    scan_vitaldb()
