import wfdb
import os
import pandas as pd

# CONFIG
# Adjust this path to where the data is actually downloaded relative to this script
# The setup script downloads to ./mimic_data/physionet.org/files/mimic4wdb/0.1.0/
# We need to verify the exact structure after download, but usually wget -r creates the full host path.
# Let's assume the user runs this from the ventilator_validation directory.
DATA_DIR = './mimic_data/physionet.org/files/mimic3wdb/1.0'

def scan_for_ventilator_data(root_dir):
    """
    Walks through the directory, reads .hea headers, and finds records
    with Airway Pressure (Paw) and Flow channels.
    """
    valid_records = []
    
    # Keyword match list
    target_signals = {
        'pressure': ['Paw', 'AWP', 'P_airway', 'Pressure', 'Vent'],
        'flow': ['Flow', 'V_dot', 'VentFlow']
    }

    print(f"Scanning {root_dir} for ventilator waveforms...")

    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.hea') and '_' not in file: # Root header files only
                record_name = file.split('.')[0]
                full_path = os.path.join(dirpath, record_name)
                
                try:
                    # Read header ONLY (fast)
                    header = wfdb.rdheader(full_path)
                    
                    signals = header.sig_name
                    if signals is None:
                        # Handle multi-segment records
                        if header.seg_name is not None:
                            # Try to read the first valid segment to get signal names
                            # We assume signals are consistent across segments (or at least the layout implies it)
                            # Actually, different segments might have different signals, but usually for MIMIC they are consistent in layout records
                            # or we just check the first one that has signals.
                            found_signals = None
                            for seg in header.seg_name:
                                if seg == '~': continue
                                try:
                                    seg_path = os.path.join(dirpath, seg)
                                    seg_header = wfdb.rdheader(seg_path)
                                    if seg_header.sig_name is not None:
                                        found_signals = seg_header.sig_name
                                        break
                                except:
                                    continue
                            
                            if found_signals:
                                signals = found_signals
                            else:
                                print(f"Skipping {record_name}: No signals found in segments.")
                                continue
                        else:
                            print(f"Skipping {record_name}: No signals found in header.")
                            continue
                        
                    found_pressure = any(sub in s for s in signals for sub in target_signals['pressure'])
                    found_flow = any(sub in s for s in signals for sub in target_signals['flow'])

                    if found_pressure: # Flow is optional but preferred
                        print(f"[MATCH] Record: {record_name} | Signals: {signals}")
                        valid_records.append({
                            'record_path': full_path,
                            'record_name': record_name,
                            'signals': signals,
                            'has_flow': found_flow
                        })
                except Exception as e:
                    print(f"Error reading {record_name}: {e}")
                    continue

    # Save manifesto
    if valid_records:
        df = pd.DataFrame(valid_records)
        df.to_csv('valid_ventilator_patients.csv', index=False)
        print(f"Scan complete. Found {len(valid_records)} valid patients.")
    else:
        print("Scan complete. No valid patients found.")

if __name__ == "__main__":
    scan_for_ventilator_data(DATA_DIR)
