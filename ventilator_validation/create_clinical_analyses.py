"""
Create clinical analyses from validation results:
1. Find exemplar cases showing 5-breath early warning
2. Create visualizations of successful predictions
3. Analyze prediction lead time
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading validation data...")
df_pred = pd.read_csv('validation_checkpoints/predictions.csv')
df_cases = pd.read_csv('validation_checkpoints/per_case_results.csv')

print(f"Loaded {len(df_pred):,} breaths from {len(df_cases)} cases")

# ============================================================================
# ANALYSIS 1: Find Best Clinical Case Examples
# ============================================================================
print("\n" + "="*80)
print("FINDING EXEMPLAR CLINICAL CASES")
print("="*80)

# Find cases with:
# - High escalation rate (interesting clinically)
# - Good number of breaths (enough data)
# - Model correctly predicted escalations

exemplar_cases = []

for caseid in df_cases['caseid'].unique()[:50]:  # Check first 50 for speed
    case_data = df_pred[df_pred['caseid'] == caseid].copy()
    
    if len(case_data) < 100:  # Skip very short cases
        continue
    
    # Find true escalations that were predicted
    true_escalations = case_data[case_data['y_true'] == 1]
    
    if len(true_escalations) == 0:
        continue
    
    # Check model performance on this case
    correctly_predicted = true_escalations[true_escalations['y_pred'] > 0.5]
    
    if len(correctly_predicted) > 0:
        accuracy = len(correctly_predicted) / len(true_escalations)
        
        exemplar_cases.append({
            'caseid': caseid,
            'n_breaths': len(case_data),
            'n_escalations': len(true_escalations),
            'n_predicted': len(correctly_predicted),
            'accuracy': accuracy,
            'escalation_rate': len(true_escalations) / len(case_data)
        })

exemplar_df = pd.DataFrame(exemplar_cases).sort_values('accuracy', ascending=False)

print(f"\nFound {len(exemplar_df)} cases with successful predictions")
print("\nTop 5 exemplar cases:")
print(exemplar_df.head())

# ============================================================================
# ANALYSIS 2: Calculate True Lead Time
# ============================================================================
print("\n" + "="*80)
print("LEAD TIME ANALYSIS")
print("="*80)

# For cases where model predicted escalation, how many breaths in advance?
lead_times = []

for caseid in exemplar_df.head(10)['caseid']:
    case_data = df_pred[df_pred['caseid'] == caseid].copy()
    case_data = case_data.reset_index(drop=True)
    
    for i in range(len(case_data)):
        if case_data.loc[i, 'y_true'] == 1 and case_data.loc[i, 'y_pred'] > 0.5:
            # Found a correctly predicted escalation
            # Look backwards to find when model first raised alarm
            for lookback in range(1, min(i, 10)):
                if case_data.loc[i - lookback, 'y_pred'] > 0.5:
                    lead_times.append(lookback)
                else:
                    break

if lead_times:
    print(f"Mean lead time: {np.mean(lead_times):.1f} breaths")
    print(f"Median lead time: {np.median(lead_times):.0f} breaths")
    print(f"Lead time distribution:")
    for breaths in range(1, 6):
        count = sum(1 for x in lead_times if x == breaths)
        pct = count / len(lead_times) * 100
        print(f"  {breaths} breaths ahead: {count} cases ({pct:.1f}%)")

# ============================================================================
# ANALYSIS 3: Identify Clinical Case Studies
# ============================================================================
print("\n" + "="*80)
print("CLINICAL CASE STUDIES")
print("="*80)

# Pick 3 best cases for detailed presentation
best_cases = exemplar_df.head(3)

print("\nRecommended cases for manuscript figures:")
for idx, row in best_cases.iterrows():
    print(f"\nCase {int(row['caseid'])}:")
    print(f"  - Total breaths: {int(row['n_breaths']):,}")
    print(f"  - Escalations occurred: {int(row['n_escalations'])}")
    print(f"  - Successfully predicted: {int(row['n_predicted'])} ({row['accuracy']:.1%})")
    print(f"  - Escalation rate: {row['escalation_rate']:.1%}")
    print(f"  - Use for: Figure showing 5-breath early warning window")

# ============================================================================
# ANALYSIS 4: Performance by Escalation Severity
# ============================================================================
print("\n" + "="*80)
print("SUBGROUP ANALYSIS: ESCALATION SEVERITY")
print("="*80)

# Stratify by escalation magnitude
df_with_esc = df_pred[df_pred['y_true'] == 1].copy()

# Create severity groups
df_with_esc['severity'] = pd.cut(
    df_with_esc['escalation_magnitude'],
    bins=[0, 5, 10, 100],
    labels=['Mild (3-5)', 'Moderate (5-10)', 'Severe (>10)']
)

print("\nModel performance by escalation severity:")
for severity in ['Mild (3-5)', 'Moderate (5-10)', 'Severe (>10)']:
    subset = df_with_esc[df_with_esc['severity'] == severity]
    if len(subset) > 0:
        mean_pred = subset['y_pred'].mean()
        high_conf = (subset['y_pred'] > 0.7).sum() / len(subset)
        print(f"\n{severity}:")
        print(f"  N = {len(subset):,} escalations")
        print(f"  Mean predicted risk: {mean_pred:.3f}")
        print(f"  High confidence (>0.7): {high_conf:.1%}")

# ============================================================================
# ANALYSIS 5: Save Case Study Data
# ============================================================================
print("\n" + "="*80)
print("SAVING CASE STUDY DATA")
print("="*80)

# Save top 3 cases for detailed visualization
for idx, row in best_cases.iterrows():
    caseid = int(row['caseid'])
    case_data = df_pred[df_pred['caseid'] == caseid].copy()
    case_data.to_csv(f'validation_checkpoints/case_study_{caseid}.csv', index=False)
    print(f"Saved: case_study_{caseid}.csv ({len(case_data)} breaths)")

print("\n✅ Analysis complete!")
print("\nNext steps for manuscript:")
print("1. Create time-series plot showing 5-breath warning (use case study files)")
print("2. Add subgroup analysis to Results section")
print("3. Reference specific cases in Discussion")

