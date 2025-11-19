import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
import data_loader
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# CONFIG
VALID_CASES_FILE = 'valid_vitaldb_cases.csv'
MODEL_PATH = 'models/trajectory_model.pkl'
SCALER_PATH = 'models/scaler_trajectory.pkl'
RESULTS_FILE = 'validation_results.csv'

# Feature list (must match training exactly)
trajectory_features = [
    'R', 'C',
    'PIP', 'mean_pressure', 'pressure_range', 'pressure_std',
    'max_flow_in', 'mean_flow_in', 'flow_variability',
    'current_PIP_high', 'pressure_flow_ratio',
    'PIP_lag1', 'PIP_lag2', 'PIP_lag3', 'PIP_lag4', 'PIP_lag5',
    'mean_pressure_lag1', 'mean_pressure_lag2', 'mean_pressure_lag3',
    'pressure_range_lag1', 'pressure_range_lag2', 'pressure_range_lag3',
    'PIP_slope_3', 'PIP_slope_5', 'PIP_acceleration',
    'PIP_volatility_3', 'PIP_volatility_5',
    'PIP_trend_3', 'PIP_trend_5',
    'consecutive_rises', 'range_volatility',
    'C_change', 'R_change',
    'compliance_pressure_risk'
]

def run_validation():
    if not os.path.exists(VALID_CASES_FILE):
        print(f"Error: {VALID_CASES_FILE} not found. Run scan_vitaldb.py first.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    # Load model and scaler
    print(f"\n📂 Loading trained model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")

    df_cases = pd.read_csv(VALID_CASES_FILE)
    
    all_predictions = []
    all_labels = []
    all_escalations = []
    case_summary = []  # Track per-case results
    
    # Process first 50 cases (expanded from 10)
    test_cases = df_cases.head(50)
    
    print(f"\n🔄 Processing {len(test_cases)} VitalDB cases...")
    
    for index, row in test_cases.iterrows():
        caseid = row['caseid']
        print(f"\nCase {caseid}:")
        
        # Load and extract breath-level features
        breath_df = data_loader.load_and_process_case(caseid)
        
        if breath_df is None or len(breath_df) == 0:
            print(f"  ❌ No valid breaths extracted")
            continue
            
        print(f"  ✅ Extracted {len(breath_df)} breaths")
        
        # Prepare features
        X = breath_df[trajectory_features].values
        y_true = breath_df['high_escalation_risk'].values
        escalations = breath_df['PIP_escalation'].values
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        y_pred = model.predict_proba(X_scaled)[:, 1]
        
        # Store
        all_predictions.extend(y_pred)
        all_labels.extend(y_true)
        all_escalations.extend(escalations)
        
        # Track per-case summary
        case_summary.append({
            'caseid': caseid,
            'n_breaths': len(breath_df),
            'escalation_rate': y_true.mean(),
            'mean_pred_risk': y_pred.mean()
        })
        
        print(f"  📊 Escalation rate: {y_true.mean():.1%}")
        print(f"  📊 Mean predicted risk: {y_pred.mean():.3f}")
        
    if not all_labels:
        print("\n❌ No data processed.")
        return
        
    # Calculate metrics
    print("\n" + "="*80)
    print("VALIDATION RESULTS ON VITALDB")
    print("="*80)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_escalations = np.array(all_escalations)
    
    auroc = roc_auc_score(all_labels, all_predictions)
    auprc = average_precision_score(all_labels, all_predictions)
    
    print(f"\n📊 Model Performance:")
    print(f"   Total Breaths: {len(all_labels):,}")
    print(f"   Escalation Rate: {all_labels.mean():.1%}")
    print(f"   Mean Escalation: {all_escalations.mean():.2f} cmH2O")
    print(f"   AUROC: {auroc:.3f}")
    print(f"   AUPRC: {auprc:.3f}")
    
    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Clinical metrics
    y_pred_binary = (all_predictions >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    print(f"\n📊 Clinical Metrics (threshold={optimal_threshold:.3f}):")
    print(f"   Sensitivity: {sensitivity:.1%}")
    print(f"   Specificity: {specificity:.1%}")
    print(f"   PPV: {ppv:.1%}")
    print(f"   NPV: {npv:.1%}")
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Total_Breaths', 'Escalation_Rate', 'N_Cases'],
        'value': [auroc, auprc, sensitivity, specificity, ppv, npv, len(all_labels), all_labels.mean(), len(case_summary)]
    })
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n💾 Results saved to: {RESULTS_FILE}")
    
    # Save per-case summary
    case_df = pd.DataFrame(case_summary)
    case_df.to_csv('validation_results_per_case.csv', index=False)
    print(f"💾 Per-case results saved to: validation_results_per_case.csv")
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    
    if auroc > 0.70:
        print("\n🎉 EXCELLENT! Model generalizes well to real VitalDB data!")
    elif auroc > 0.60:
        print("\n✅ GOOD! Model shows reasonable performance on VitalDB.")
    else:
        print("\n⚠️  Model performance degraded on VitalDB. Consider domain adaptation.")

if __name__ == "__main__":
    run_validation()
