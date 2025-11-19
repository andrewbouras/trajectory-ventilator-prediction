"""
Checkpoint-based validation for VitalDB cases.
Saves progress after each case and generates reports at milestones.
Can resume from interruption.
"""
import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime, timedelta
import pytz
import time
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, brier_score_loss
import data_loader
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# CONFIG
VALID_CASES_FILE = 'valid_vitaldb_cases.csv'
MODEL_PATH = 'models/trajectory_model.pkl'
SCALER_PATH = 'models/scaler_trajectory.pkl'

# Output files
CHECKPOINT_DIR = 'validation_checkpoints'
PROGRESS_FILE = f'{CHECKPOINT_DIR}/progress.json'
PREDICTIONS_FILE = f'{CHECKPOINT_DIR}/predictions.csv'
PER_CASE_FILE = f'{CHECKPOINT_DIR}/per_case_results.csv'

# Checkpoint milestones
CHECKPOINT_MILESTONES = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000]

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

def setup_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_progress():
    """Load progress from previous run if exists."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        'processed_cases': [],
        'failed_cases': [],
        'last_milestone': 0,
        'start_time': datetime.now().isoformat()
    }

def save_progress(progress):
    """Save current progress."""
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def append_predictions(caseid, predictions, labels, escalations):
    """Append predictions for a case to CSV (incremental save)."""
    df = pd.DataFrame({
        'caseid': caseid,
        'breath_idx': range(len(predictions)),
        'y_pred': predictions,
        'y_true': labels,
        'escalation_magnitude': escalations
    })
    
    # Append to file
    if os.path.exists(PREDICTIONS_FILE):
        df.to_csv(PREDICTIONS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(PREDICTIONS_FILE, mode='w', header=True, index=False)

def append_case_summary(case_info):
    """Append case summary to CSV (incremental save)."""
    df = pd.DataFrame([case_info])
    
    if os.path.exists(PER_CASE_FILE):
        df.to_csv(PER_CASE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(PER_CASE_FILE, mode='w', header=True, index=False)

def calculate_metrics(predictions_file):
    """Calculate overall metrics from predictions file."""
    df = pd.read_csv(predictions_file)
    
    y_pred = df['y_pred'].values
    y_true = df['y_true'].values
    escalations = df['escalation_magnitude'].values
    
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Clinical metrics
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    
    return {
        'n_breaths': len(y_true),
        'n_cases': df['caseid'].nunique(),
        'escalation_rate': y_true.mean(),
        'mean_escalation_magnitude': escalations[y_true == 1].mean(),
        'auroc': auroc,
        'auprc': auprc,
        'brier_score': brier,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1
    }

def generate_checkpoint_report(n_cases_processed):
    """Generate and save checkpoint report."""
    print(f"\n{'='*80}")
    print(f"CHECKPOINT REPORT: {n_cases_processed} CASES PROCESSED")
    print(f"{'='*80}")
    
    if not os.path.exists(PREDICTIONS_FILE):
        print("No predictions file found yet.")
        return
    
    metrics = calculate_metrics(PREDICTIONS_FILE)
    
    print(f"\n📊 Overall Performance:")
    print(f"   Cases Processed: {metrics['n_cases']}")
    print(f"   Total Breaths: {metrics['n_breaths']:,}")
    print(f"   Escalation Rate: {metrics['escalation_rate']:.1%}")
    print(f"   Mean Escalation: {metrics['mean_escalation_magnitude']:.2f} cmH₂O")
    print(f"\n🎯 Discrimination:")
    print(f"   AUROC: {metrics['auroc']:.3f}")
    print(f"   AUPRC: {metrics['auprc']:.3f}")
    print(f"   Brier Score: {metrics['brier_score']:.3f}")
    print(f"\n🏥 Clinical Metrics (threshold={metrics['optimal_threshold']:.3f}):")
    print(f"   Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"   Specificity: {metrics['specificity']:.1%}")
    print(f"   PPV: {metrics['ppv']:.1%}")
    print(f"   NPV: {metrics['npv']:.1%}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    # Save checkpoint report
    report_file = f'{CHECKPOINT_DIR}/checkpoint_{n_cases_processed}_cases.json'
    with open(report_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Checkpoint saved: {report_file}")
    print(f"{'='*80}\n")

def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def get_est_time():
    """Get current time in EST."""
    est = pytz.timezone('US/Eastern')
    return datetime.now(est)

def run_validation(max_cases=None):
    """
    Run validation with checkpointing.
    
    Args:
        max_cases: Maximum number of cases to process (None = all available)
    """
    setup_checkpoint_dir()
    
    # Check required files
    if not os.path.exists(VALID_CASES_FILE):
        print(f"❌ Error: {VALID_CASES_FILE} not found. Run scan_vitaldb.py first.")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return
    
    # Get start time
    start_time = time.time()
    start_time_est = get_est_time()
    
    # Load model and scaler
    print(f"\n{'='*80}")
    print("CHECKPOINT-BASED VITALDB VALIDATION")
    print(f"{'='*80}")
    print(f"\n⏰ Start Time (EST): {start_time_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"\n📂 Loading trained model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")
    
    # Load progress
    progress = load_progress()
    processed_cases = set(progress['processed_cases'])
    failed_cases = set(progress['failed_cases'])
    
    if processed_cases:
        print(f"\n🔄 Resuming from previous run...")
        print(f"   Already processed: {len(processed_cases)} cases")
        print(f"   Failed: {len(failed_cases)} cases")
    
    # Load available cases
    df_cases = pd.read_csv(VALID_CASES_FILE)
    
    if max_cases:
        df_cases = df_cases.head(max_cases)
    
    total_to_process = len(df_cases)
    already_done = len(processed_cases)
    remaining = total_to_process - already_done
    
    print(f"\n📋 Validation Plan:")
    print(f"   Total cases in dataset: {len(df_cases):,}")
    print(f"   Already completed: {already_done:,}")
    print(f"   Remaining to process: {remaining:,}")
    print(f"   Next milestone: {[m for m in CHECKPOINT_MILESTONES if m > already_done][:1]}")
    
    # For tracking timing
    case_times = []
    total_breaths_processed = 0
    
    # Process cases
    print(f"\n{'='*80}")
    print("STARTING VALIDATION")
    print(f"{'='*80}\n")
    
    for idx, row in df_cases.iterrows():
        caseid = row['caseid']
        
        # Skip if already processed
        if caseid in processed_cases or caseid in failed_cases:
            continue
        
        n_processed = len(processed_cases)
        case_start_time = time.time()
        
        # Calculate progress and time estimates
        elapsed_time = time.time() - start_time
        progress_pct = (n_processed / total_to_process) * 100 if total_to_process > 0 else 0
        
        # Estimate remaining time based on average case processing time
        if case_times:
            avg_case_time = np.mean(case_times)
            remaining_cases = remaining - (n_processed - already_done)
            est_remaining_seconds = avg_case_time * remaining_cases
            est_completion = get_est_time() + timedelta(seconds=est_remaining_seconds)
        else:
            avg_case_time = None
            est_completion = None
        
        # Verbose progress header
        print(f"\n{'─'*80}")
        print(f"📊 PROGRESS: {n_processed}/{total_to_process} cases ({progress_pct:.1f}%)")
        print(f"🔄 Processing Case ID: {caseid}")
        print(f"⏱️  Elapsed: {format_time(elapsed_time)}")
        if avg_case_time:
            print(f"⚡ Avg time/case: {format_time(avg_case_time)}")
            print(f"⏳ Est. remaining: {format_time(est_remaining_seconds)}")
            print(f"🎯 Est. completion (EST): {est_completion.strftime('%I:%M:%S %p')}")
        if total_breaths_processed > 0:
            print(f"📈 Total breaths processed: {total_breaths_processed:,}")
        print(f"{'─'*80}")
        
        try:
            # Load and extract breath-level features
            print(f"   📥 Downloading waveform data from VitalDB...")
            breath_df = data_loader.load_and_process_case(caseid)
            
            if breath_df is None or len(breath_df) == 0:
                print(f"   ❌ No valid breaths extracted (empty or insufficient data)")
                failed_cases.add(caseid)
                progress['failed_cases'].append(caseid)
                save_progress(progress)
                case_times.append(time.time() - case_start_time)
                continue
            
            n_breaths = len(breath_df)
            print(f"   ✅ Extracted {n_breaths:,} breaths")
            
            # Prepare features
            print(f"   🔧 Computing features and predictions...")
            X = breath_df[trajectory_features].values
            y_true = breath_df['high_escalation_risk'].values
            escalations = breath_df['PIP_escalation'].values
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            y_pred = model.predict_proba(X_scaled)[:, 1]
            
            # Save predictions incrementally
            print(f"   💾 Saving predictions to checkpoint...")
            append_predictions(caseid, y_pred, y_true, escalations)
            
            # Save case summary
            case_info = {
                'caseid': caseid,
                'n_breaths': len(breath_df),
                'escalation_rate': y_true.mean(),
                'mean_pred_risk': y_pred.mean(),
                'mean_escalation_magnitude': escalations[y_true == 1].mean() if y_true.sum() > 0 else 0,
                'processing_time_seconds': time.time() - case_start_time,
                'processed_at': datetime.now().isoformat()
            }
            append_case_summary(case_info)
            
            # Update progress
            processed_cases.add(caseid)
            progress['processed_cases'].append(caseid)
            save_progress(progress)
            
            # Track timing
            case_processing_time = time.time() - case_start_time
            case_times.append(case_processing_time)
            total_breaths_processed += n_breaths
            
            # Show results
            print(f"   ✅ Case completed in {format_time(case_processing_time)}")
            print(f"   📊 Escalation rate: {y_true.mean():.1%}")
            print(f"   📊 Mean predicted risk: {y_pred.mean():.3f}")
            print(f"   📊 Mean escalation magnitude: {escalations[y_true == 1].mean():.2f} cmH₂O" if y_true.sum() > 0 else "   📊 No escalations in this case")
            
            # Check for milestone checkpoints
            n_processed = len(processed_cases)
            if n_processed in CHECKPOINT_MILESTONES and n_processed > progress['last_milestone']:
                generate_checkpoint_report(n_processed)
                progress['last_milestone'] = n_processed
                save_progress(progress)
        
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            print(f"   ⚠️  Case {caseid} failed after {format_time(time.time() - case_start_time)}")
            failed_cases.add(caseid)
            progress['failed_cases'].append(caseid)
            save_progress(progress)
            case_times.append(time.time() - case_start_time)
            continue
    
    # Final report
    end_time = time.time()
    end_time_est = get_est_time()
    total_elapsed = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n⏰ Start Time (EST): {start_time_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"⏰ End Time (EST):   {end_time_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"⏱️  Total Duration:   {format_time(total_elapsed)}")
    print(f"\n📊 Final Statistics:")
    print(f"   ✅ Successfully processed: {len(processed_cases):,} cases")
    print(f"   ❌ Failed: {len(failed_cases):,} cases")
    print(f"   📈 Total breaths: {total_breaths_processed:,}")
    if case_times:
        print(f"   ⚡ Average time per case: {format_time(np.mean(case_times))}")
        print(f"   🚀 Processing rate: {len(processed_cases) / (total_elapsed / 3600):.1f} cases/hour")
    
    if os.path.exists(PREDICTIONS_FILE):
        print(f"\n📊 Generating final report...")
        generate_checkpoint_report(len(processed_cases))
    
    print(f"\n💾 All results saved in: {CHECKPOINT_DIR}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import sys
    
    # Allow command-line argument for max cases
    max_cases = None
    if len(sys.argv) > 1:
        max_cases = int(sys.argv[1])
        print(f"Limiting to first {max_cases} cases")
    
    run_validation(max_cases=max_cases)

