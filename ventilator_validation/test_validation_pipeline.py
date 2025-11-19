"""
Comprehensive pre-flight test of the validation pipeline.
Tests all components before running the full validation.
"""
import pandas as pd
import numpy as np
import os
import joblib
import data_loader
import ssl
from datetime import datetime

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

print("="*80)
print("VALIDATION PIPELINE PRE-FLIGHT CHECK")
print("="*80)

# Test 1: Check required files
print("\n1️⃣ CHECKING REQUIRED FILES...")
required_files = {
    'valid_vitaldb_cases.csv': 'Case list',
    'models/trajectory_model.pkl': 'Trained model',
    'models/scaler_trajectory.pkl': 'Feature scaler'
}

all_files_ok = True
for file, desc in required_files.items():
    if os.path.exists(file):
        print(f"   ✅ {desc}: {file}")
    else:
        print(f"   ❌ MISSING {desc}: {file}")
        all_files_ok = False

if not all_files_ok:
    print("\n❌ Missing required files. Cannot proceed.")
    exit(1)

# Test 2: Load model and scaler
print("\n2️⃣ LOADING MODEL AND SCALER...")
try:
    model = joblib.load('models/trajectory_model.pkl')
    scaler = joblib.load('models/scaler_trajectory.pkl')
    print(f"   ✅ Model type: {type(model).__name__}")
    print(f"   ✅ Model expects: {model.n_features_in_} features")
    print(f"   ✅ Scaler expects: {scaler.n_features_in_} features")
    
    if model.n_features_in_ != scaler.n_features_in_:
        print(f"   ❌ Model/scaler feature mismatch!")
        exit(1)
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

# Test 3: Check case list
print("\n3️⃣ CHECKING CASE LIST...")
try:
    df_cases = pd.read_csv('valid_vitaldb_cases.csv')
    print(f"   ✅ Total cases available: {len(df_cases):,}")
    print(f"   ✅ Columns: {list(df_cases.columns)}")
    print(f"   ✅ First case ID: {df_cases['caseid'].iloc[0]}")
    print(f"   ✅ Last case ID: {df_cases['caseid'].iloc[-1]}")
except Exception as e:
    print(f"   ❌ Error loading case list: {e}")
    exit(1)

# Test 4: Test data loading for 3 sample cases
print("\n4️⃣ TESTING DATA LOADING ON SAMPLE CASES...")
test_case_ids = df_cases['caseid'].head(5).tolist()
successful_loads = 0
failed_loads = 0

for caseid in test_case_ids:
    try:
        print(f"\n   Testing case {caseid}...")
        breath_df = data_loader.load_and_process_case(caseid)
        
        if breath_df is None or len(breath_df) == 0:
            print(f"      ⚠️  No valid breaths extracted")
            failed_loads += 1
            continue
        
        print(f"      ✅ Loaded {len(breath_df)} breaths")
        print(f"      ✅ Features created: {len(breath_df.columns)}")
        
        # Check feature names
        required_features = [
            'R', 'C', 'PIP', 'mean_pressure', 'pressure_range', 'pressure_std',
            'max_flow_in', 'mean_flow_in', 'flow_variability',
            'current_PIP_high', 'pressure_flow_ratio',
            'PIP_lag1', 'PIP_lag2', 'PIP_lag3', 'PIP_lag4', 'PIP_lag5',
            'mean_pressure_lag1', 'mean_pressure_lag2', 'mean_pressure_lag3',
            'pressure_range_lag1', 'pressure_range_lag2', 'pressure_range_lag3',
            'PIP_slope_3', 'PIP_slope_5', 'PIP_acceleration',
            'PIP_volatility_3', 'PIP_volatility_5',
            'PIP_trend_3', 'PIP_trend_5',
            'consecutive_rises', 'range_volatility',
            'C_change', 'R_change', 'compliance_pressure_risk'
        ]
        
        missing = [f for f in required_features if f not in breath_df.columns]
        if missing:
            print(f"      ❌ Missing features: {missing}")
            failed_loads += 1
            continue
        
        print(f"      ✅ All {len(required_features)} required features present")
        
        # Test model prediction
        X = breath_df[required_features].values
        X_scaled = scaler.transform(X)
        y_pred = model.predict_proba(X_scaled)[:, 1]
        
        print(f"      ✅ Prediction successful: {len(y_pred)} predictions")
        print(f"      ✅ Mean predicted risk: {y_pred.mean():.3f}")
        print(f"      ✅ Escalation rate: {breath_df['high_escalation_risk'].mean():.1%}")
        
        successful_loads += 1
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        failed_loads += 1

print(f"\n   Summary: {successful_loads} successful, {failed_loads} failed")

if successful_loads == 0:
    print("\n   ❌ No cases loaded successfully. Cannot proceed.")
    exit(1)

# Test 5: Check disk space (rough estimate)
print("\n5️⃣ ESTIMATING DISK SPACE...")
try:
    import shutil
    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024**3)
    print(f"   ✅ Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 1:
        print(f"   ⚠️  WARNING: Low disk space. May need more for large runs.")
    else:
        print(f"   ✅ Sufficient disk space available")
except Exception as e:
    print(f"   ⚠️  Could not check disk space: {e}")

# Test 6: Test checkpoint directory creation
print("\n6️⃣ TESTING CHECKPOINT SYSTEM...")
try:
    checkpoint_dir = 'validation_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Test write permissions
    test_file = f'{checkpoint_dir}/test_write.txt'
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    
    print(f"   ✅ Checkpoint directory: {checkpoint_dir}")
    print(f"   ✅ Write permissions: OK")
except Exception as e:
    print(f"   ❌ Error with checkpoint system: {e}")
    exit(1)

# Test 7: Network connectivity to VitalDB
print("\n7️⃣ TESTING VITALDB CONNECTIVITY...")
try:
    import vitaldb
    # Try to load a small amount of data
    test_vals = vitaldb.load_case(1, ['Primus/AWP'], interval=1/62.5, start=0, end=10)
    if test_vals is not None:
        print(f"   ✅ VitalDB connection: OK")
        print(f"   ✅ Data download: OK")
    else:
        print(f"   ⚠️  VitalDB returned None (may be expected for this case)")
except Exception as e:
    print(f"   ❌ VitalDB connection error: {e}")
    print(f"   ⚠️  May have issues downloading data during validation")

# Final summary
print("\n" + "="*80)
print("PRE-FLIGHT CHECK SUMMARY")
print("="*80)

issues = []

if successful_loads < len(test_case_ids) * 0.6:
    issues.append("Low success rate loading test cases")

if model.n_features_in_ != len(required_features):
    issues.append(f"Feature count mismatch: model expects {model.n_features_in_}, we have {len(required_features)}")

if issues:
    print("\n⚠️  WARNINGS FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nYou may want to investigate these before running full validation.")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nReady to run validation with:")
    print("   python3 validate_with_checkpoints.py [max_cases]")
    print("\nRecommended options:")
    print("   python3 validate_with_checkpoints.py 50    # Quick (1 hour)")
    print("   python3 validate_with_checkpoints.py 100   # Recommended (2 hours)")
    print("   python3 validate_with_checkpoints.py 500   # Comprehensive (10 hours)")

print("\n" + "="*80)

