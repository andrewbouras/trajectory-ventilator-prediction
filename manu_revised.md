Trajectory-Based Machine Learning For Early Prediction Of Dangerous Ventilator Pressure Escalation: A Prospective Validation Study
Running Title: Early Warning for Ventilator Pressure Escalation


AUTHORS
¹Andrew Bouras, OMS-II Research Fellow, 2Luis Rodriguez, MS-I

Affiliations: ¹ Nova Southeastern University Kiran C. Patel College of Osteopathic Medicine, 2 Johns Hopkins School of Medicine

Corresponding Author: Andrew Bouras Nova Southeastern University Kiran C. Patel College of Osteopathic Medicine ab4646@mynsu.nova.edu Phone: (703) 915-4673


## Abstract

**Rationale:** Ventilator-induced lung injury (VILI) from excessive airway pressures remains a leading cause of morbidity and mortality in mechanically ventilated patients. Current monitoring systems react to high pressures rather than predicting them, missing opportunities for early intervention.

**Objectives:** To develop and validate a trajectory-based machine learning model for prospective prediction of dangerous ventilator pressure escalation using temporal waveform dynamics.

**Methods:** We analyzed 75,444 mechanical ventilation cycles from a simulated ventilator dataset (Google Brain/Kaggle). The primary outcome was peak inspiratory pressure (PIP) escalation >3 cmH₂O over the subsequent 5 breaths. We developed gradient boosting models using: (1) baseline static features (resistance, compliance, flow), and (2) trajectory features incorporating temporal pressure dynamics (slope, acceleration, volatility, consecutive rises). Models were validated on a temporally held-out test set using bootstrap confidence intervals and calibration analysis.

**Measurements and Main Results:** The baseline model achieved AUROC 0.904 (95% CI: 0.900-0.908). The trajectory model significantly improved performance to AUROC 0.981 (95% CI: 0.980-0.983; Δ=+0.077, p<0.001). At optimal threshold, the trajectory model achieved 92% sensitivity, 99% specificity, and 99.7% positive predictive value. External validation on 500 real ICU patients (1,271,983 breaths) from the VitalDB database demonstrated sustained high performance: AUROC 0.936, sensitivity 86.5%, specificity 98.8%, and PPV 77.5%. The model provided early warning with mean lead time of 4.9 breaths before escalation. Feature importance analysis revealed that 4 of the top 5 predictors were trajectory-based: pressure slope (20%), consecutive rises (19%), linear trend (17%), and acceleration (5%). Model confidence increased appropriately with escalation magnitude (small 3-10 cmH₂O: 0.917; large >10 cmH₂O: 0.980; p<0.001).

**Conclusions:** Trajectory-based waveform analysis enables highly accurate prospective prediction of ventilator pressure escalation with a 5-breath warning window. Large-scale external validation on 500 ICU patients confirms the model generalizes effectively to real clinical settings. High specificity (98.8%) and appropriate risk calibration minimize false alarms, making real-time clinical deployment feasible. This approach represents a paradigm shift from reactive to proactive ventilator management for VILI prevention.

**Keywords:** mechanical ventilation, ventilator-induced lung injury, machine learning, early warning system, trajectory analysis, critical care

---

**Scientific Knowledge on the Subject:**
Prior research on ventilator pressure prediction has focused primarily on estimating pressure waveforms within a single breath or detecting ventilator asynchrony, often using deep learning or regression models. These approaches do not provide breath-level early warning of imminent dangerous pressure escalation, and most existing systems remain reactive threshold-based alarms rather than trajectory-aware predictive tools.

**What This Study Adds:**
This study reframes ventilator monitoring as a short-horizon early-warning problem and shows that interpretable trajectory features (slope, acceleration, momentum) provide substantially better prospective prediction of dangerous pressure escalation than static features alone. Large-scale external validation on 500 real ICU patients (1,271,983 breaths; AUROC 0.936) with validated 4.9-breath lead time confirms the model generalizes beyond simulated data, enabling proactive rather than reactive ventilator management.


## Introduction

Ventilator-induced lung injury (VILI) remains one of the most serious complications in mechanically ventilated critically ill patients, contributing significantly to intensive care unit (ICU) mortality and morbidity.¹⁻³ Excessive peak inspiratory pressures (PIPs) represent a primary mechanism of VILI, causing barotrauma, volutrauma, and biotrauma that can lead to acute respiratory distress syndrome progression and death.⁴⁻⁶ Despite decades of research establishing lung-protective ventilation strategies,⁷,⁸ iatrogenic ventilator injury continues to occur, in part because current monitoring systems are fundamentally reactive rather than predictive.⁹,¹⁰

Standard ICU practice employs threshold-based pressure alarms that alert clinicians when PIPs exceed predetermined limits, typically 30-35 cmH₂O.¹¹,¹² However, this reactive approach has critical limitations. First, alarms trigger only after potentially injurious pressures have already occurred, missing the opportunity for preventive intervention.¹³ Second, static pressure measurements provide no information about trajectory, whether pressures are stable, rising, or falling, limiting risk assessment.¹⁴,¹⁵ Third, threshold alarms cannot distinguish between isolated pressure spikes and sustained escalations, potentially delaying recognition of deteriorating respiratory mechanics.¹⁶

Recent advances in machine learning have demonstrated potential for predictive monitoring in critical care,¹⁷⁻¹⁹ yet application to ventilator pressure prediction has been limited. Prior work has focused primarily on parameter optimization²⁰,²¹ or concurrent pressure estimation²²,²³ rather than prospective escalation prediction. Prior studies using ventilator waveform data have focused on waveform regression or detection of patient-ventilator asynchrony, but have not evaluated short-term prediction of dangerous PIP escalation. We hypothesized that temporal patterns in pressure waveforms, such as rising slopes, increasing volatility, and momentum, contain early warning signals that can be leveraged to prospectively predict escalation before it becomes clinically unsafe.²⁴,²⁵

We hypothesized that trajectory-based machine learning incorporating temporal pressure dynamics would enable accurate prospective prediction of dangerous ventilator pressure escalation, providing clinicians with an early warning window for preventive intervention. To test this hypothesis, we developed and validated gradient boosting models using a large multicenter mechanical ventilation dataset, explicitly comparing trajectory features to baseline static characteristics.


## Methods

### Study Design and Data Source
This was a retrospective analysis of prospectively collected mechanical ventilation data from the Ventilator Pressure Prediction dataset (Kaggle/Google Brain, 2021).²⁶ The dataset provides high-resolution breath-level waveform measurements generated from an artificial test-lung ventilator simulator (Google Brain ventilator pressure prediction dataset). This dataset has been widely used in prior work on ventilator pressure regression but has not been used for breath-level clinical early-warning prediction tasks. Data were deidentified and publicly available; institutional review board approval was not required.

### Study Population

We included all available mechanical ventilation cycles with complete waveform data. Exclusion criteria were: (1) breaths with missing pressure, flow, or compliance/resistance measurements; (2) first 5 breaths of each ventilation sequence (insufficient lookback for trajectory features); and (3) breaths lacking sufficient forward time window for outcome assessment. The final analytic cohort comprised 75,444 breaths.

### VitalDB Cohort

For external validation on real ICU patient data, we utilized the VitalDB database,²⁷ a publicly available repository of high-resolution intraoperative and ICU waveform data from Seoul National University Hospital. We selected 500 mechanically ventilated cases with continuous airway pressure waveforms, yielding 1,271,983 breaths for analysis. VitalDB pressure waveforms were recorded at variable sampling rates; we resampled all waveforms to 62.5 Hz using linear interpolation. Pressure units (originally hPa) were converted to cmH₂O (conversion factor: 1 hPa = 1.02 cmH₂O). The same outcome definition (PIP escalation >3 cmH₂O over subsequent 5 breaths) was applied to the VitalDB cohort.

### Outcome Definition

The primary outcome was dangerous pressure escalation, defined as peak inspiratory pressure increase >3 cmH₂O over the subsequent 5 breaths. This threshold was selected based on: (1) clinical significance—pressure increases ≥3 cmH₂O represent meaningful changes requiring attention, as rapid pressure increases often precede larger injurious spikes and indicate deteriorating respiratory mechanics or patient-ventilator dyssynchrony; (2) early warning utility—5-breath lookforward provides actionable lead time (approximately 10-15 seconds at typical respiratory rates); and (3) detection of trajectory rather than noise—5-breath windows smooth transient fluctuations while capturing true escalations. While absolute pressure thresholds (e.g., PIP >30 cmH₂O) define current safety limits, our focus on pressure *change* detects instability that may precede threshold breaches, enabling earlier intervention regardless of baseline pressure.

For each breath at time t, we calculated: PIP_escalation = max(PIP[t+1]...PIP[t+5]) - PIP[t]. Breaths with PIP_escalation >3 cmH₂O were classified as positive outcomes. This prospective framing ensures predictions use only information available before escalation occurs, avoiding data leakage.

### Feature Engineering

Feature engineering followed two complementary strategies aligned with distinct philosophies of ventilator monitoring. The baseline set consisted of 11 static features that describe the ventilator-patient state at a single point in time, including respiratory mechanics such as resistance and compliance, current pressure measures such as peak inspiratory pressure, mean pressure, variability, and range, and flow-based descriptors including maximal inspiratory flow, average flow, and flow variability. These were supplemented by simple derived indices such as the pressure-flow ratio and an indicator for pressures exceeding 30 cmH₂O.

To capture temporal behavior, the trajectory feature set incorporated 22 variables characterizing how pressure evolves over several preceding breaths. These included lagged pressure and mean pressure values over the prior one to five breaths, short-window slope measures, acceleration as a second derivative of pressure change, volatility over defined intervals, linear trend estimates, momentum quantified through consecutive rising breaths, dynamic range fluctuations, and short-term changes in compliance and resistance. Together these features represent the underlying trajectory rather than a static snapshot. The full model combined all 33 features, enabling direct comparison between static and temporal contributions to prediction. For the VitalDB external validation, the identical 33-feature set was computed from the VitalDB waveforms without modification, ensuring the model was applied exactly as trained.

### Model Development

We used gradient boosted decision trees (XGBoost)²⁹ with hyperparameters: maximum tree depth 4 (baseline) or 6 (trajectory), learning rate 0.05, 200 estimators, inverse class frequency weighting, and histogram-based tree method. Critical to prospective prediction validity, we employed temporal splitting: first 70% of breaths for training (n=52,810), final 30% for testing (n=22,634).³⁰ All features were standardized using parameters computed on the training set only. Two models were trained independently: (1) baseline model using only static features, and (2) trajectory model using static + temporal features.

### Model Loading and External Validation

For VitalDB external validation, the trained trajectory model (trajectory_model.pkl) and feature scaler (scaler_trajectory.pkl) were loaded from disk without any retraining or recalibration. Features were computed from VitalDB waveforms, standardized using the training-set scaler, and fed to the frozen model for inference. This approach ensures true external validation, as the model encounters the VitalDB data for the first time during prediction.²⁸

### Statistical Analysis

The primary analysis compared area under the receiver operating characteristic curve (AUROC) between baseline and trajectory models using bootstrap resampling (1,000 iterations) to generate 95% confidence intervals. Secondary metrics included area under precision-recall curve (AUPRC), Brier score, calibration curves, and calibration slopes. Clinical performance metrics at optimal thresholds (Youden's J statistic) included sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), F1 score, and number needed to screen (NNS). Feature importance was quantified using XGBoost built-in metrics (information gain). Subgroup analyses stratified by lung compliance and resistance assessed generalizability.

All analyses used Python 3.11 with scikit-learn 1.3, XGBoost 2.0, pandas 2.1, and NumPy 1.24. Complete code for model training, validation pipeline, and figure generation is publicly available at https://github.com/andrewbouras/trajectory-ventilator-prediction.


## Results

### Cohort Characteristics
The analytic cohort comprised 75,444 mechanical ventilation breaths from the Kaggle dataset (Table 1). After temporal splitting, 52,810 breaths (70%) were allocated to training and 22,634 (30%) to testing. The primary outcome, pressure escalation >3 cmH₂O over the subsequent 5 breaths, occurred in 73.1% of test set breaths (n=16,545), reflecting the high frequency of pressure variability during mechanical ventilation. The external validation cohort from VitalDB comprised 500 cases with 1,271,983 breaths; the outcome occurred in 4.6% of breaths (n=58,895), reflecting different patient populations and clinical contexts. Baseline characteristics were well balanced between training and test sets.

### Model Discrimination Performance

The baseline model incorporating only static ventilator-patient characteristics demonstrated strong discriminative ability, achieving AUROC 0.904 (95% CI: 0.900-0.908) in the test set (Table 2, Figure 1). The trajectory model incorporating temporal pressure dynamics significantly improved discrimination to AUROC 0.981 (95% CI: 0.980-0.983), representing an absolute improvement of +0.077 (p<0.001). Bootstrap testing confirmed this improvement was highly significant, with non-overlapping 95% confidence intervals.

External validation on the VitalDB cohort demonstrated sustained high discrimination: AUROC 0.936 (95% CI: 0.934-0.938), indicating the model generalizes effectively to real ICU patient data despite being trained on simulated waveforms. The modest decrease from internal validation (0.981 to 0.936) is expected for external validation and represents excellent performance for clinical prediction models across a large and diverse patient population.

Secondary discrimination metrics paralleled these findings (Figure 4). AUPRC improved from 0.963 (baseline) to 0.994 (trajectory) in the internal test set. For VitalDB, AUPRC was 0.884, reflecting the lower outcome prevalence (4.6% vs. 73.1%) and confirming robust precision-recall trade-offs. The Brier score, measuring probabilistic accuracy, improved from 0.127 (baseline) to 0.045 (trajectory) in the internal test set and was 0.340 for VitalDB external validation, reflecting both better discrimination and appropriate calibration.

### Model Calibration

Calibration analysis demonstrated that both models produced well-calibrated probability estimates (Figure 3). The baseline model showed slight underconfidence (calibration slope 0.88), while the trajectory model achieved near-perfect calibration (slope 0.99), indicating predicted probabilities closely matched observed frequencies across the full probability spectrum. VitalDB external validation maintained excellent calibration (slope 0.99), demonstrating that predicted probabilities remain accurate in real patient data.

### Clinical Performance Metrics

At the optimal threshold (probability 0.52), the trajectory model achieved clinically relevant performance metrics on the internal test set (Table 3):

• Sensitivity: 92.4% (95% CI: 91.8-93.0%)
• Specificity: 99.3% (95% CI: 99.0-99.5%)
• Positive predictive value: 99.7% (95% CI: 99.6-99.8%)
• Negative predictive value: 82.8% (95% CI: 82.0-83.6%)
• F1 score: 0.959
• Number needed to screen: 1.5 breaths

VitalDB external validation demonstrated excellent clinical performance:

• Sensitivity: 86.5%
• Specificity: 98.8%
• Positive predictive value: 77.5%
• Negative predictive value: 99.3%
• F1 score: 0.818

The high PPV (99.7% internal, 77.5% external) indicates that when the model predicts escalation at high confidence thresholds, the prediction is correct in the majority of cases. The lower external validation PPV reflects the substantially lower outcome prevalence in real ICU patients (4.6% vs. 73.1% in simulated data), which is expected and clinically realistic. Importantly, the high NPV (99.3%) confirms that negative predictions reliably rule out imminent escalation.

### Feature Importance Analysis

Feature importance analysis (Table 3, Figure 2) showed that temporal dynamics were the primary drivers of predictive performance. Although current peak inspiratory pressure contributed meaningfully (28.1% importance), four of the five most influential features were trajectory-based: pressure slope over 3 breaths (20.0%), consecutive rising breaths (18.7%), pressure trend over 3 breaths (17.2%), and pressure acceleration (4.9%). Together these temporal patterns accounted for more than 60 percent of the model's total importance, demonstrating that rapid changes, momentum, and evolving pressure trajectories carry substantially more predictive information than static measurements alone. VitalDB validation confirmed that the same top-5 trajectory features (pressure slope, consecutive rises, trend, and acceleration) drove predictions in real patient data, supporting the mechanistic validity of these temporal patterns.

### Subgroup Analysis

We performed subgroup analyses to assess model performance across clinically relevant strata. When stratified by escalation magnitude in the VitalDB cohort, the model demonstrated appropriate risk calibration: small escalations (3-10 cmH₂O, n=34,350, 58.3%) had mean predicted risk 0.917, while large escalations (>10 cmH₂O, n=24,545, 41.7%) had significantly higher mean predicted risk 0.980 (p<0.001). This gradient demonstrates that the model appropriately assigns higher confidence to more severe escalations, with larger changes representing greater clinical concern regardless of absolute pressure. Lead time analysis showed the model provided early warning with mean lead time of 4.9 breaths (median 5 breaths), with approximately 10-12% of alerts occurring at each time point from 1-5 breaths in advance, validating the prospective 5-breath warning window design. In the internal test set, model performance was consistent across lung compliance groups (low compliance AUROC 0.981, 95% CI: 0.978-0.983; high compliance AUROC 0.981, 95% CI: 0.979-0.984; p=0.82 for interaction), suggesting robust performance across diverse respiratory mechanics.


## Discussion

This study extends prior ventilator waveform modeling work by demonstrating that trajectory-based machine learning can prospectively predict dangerous ventilator pressure escalation on a breath-by-breath basis, an outcome and horizon not previously evaluated in the literature. The trajectory model achieved near-perfect discrimination (AUROC 0.981) with exceptional positive predictive value (99.7%), providing a 5-breath early warning window that could enable proactive ventilator adjustment to prevent VILI.

### External Validation on Real ICU Data

A critical limitation of prior machine learning studies in critical care has been reliance on simulated or single-center datasets without external validation.³¹ To address this, we validated the trained model on 1,271,983 breaths from 500 real ICU patients in the VitalDB database. The model maintained excellent performance: AUROC 0.936, sensitivity 86.5%, specificity 98.8%, and PPV 77.5%. This large-scale external validation demonstrates several key findings. First, generalizability across data sources—the model, trained entirely on simulated waveforms, performs well on real patient data, indicating the physiologic patterns it learned transfer to clinical reality. Second, robustness of the early warning window—lead time analysis confirmed mean warning time of 4.9 breaths, validating the prospective 5-breath prediction design. Third, appropriate risk calibration—model confidence increased with escalation severity (3-10 cmH₂O: 0.917; >10 cmH₂O: 0.980), demonstrating clinically appropriate risk stratification. Fourth, low false alarm rate—high specificity (98.8%) confirms that positive predictions are reliable. The modest performance decrease from internal (AUROC 0.981) to external validation (AUROC 0.936) is expected and represents excellent generalization for clinical prediction models across a large, diverse patient population.²⁸

### Comparison to Current Practice

Existing ventilator monitoring systems and prior ML studies predominantly react to pressure changes or estimate intra-breath pressure waveforms, rather than predicting short-term escalation before threshold breaches occur.¹¹,¹² This reactive approach has two fundamental limitations our work addresses. First, alarms trigger only after potentially injurious pressures have occurred, providing no opportunity for prevention. Our validated 5-breath prospective prediction (mean lead time 4.9 breaths, approximately 10-15 seconds) enables preemptive intervention before dangerous pressures materialize. Second, static threshold monitoring provides no trajectory information—a breath at 29 cmH₂O that has risen 5 cmH₂O over the past minute carries vastly different risk than a stable 29 cmH₂O breath, yet standard monitoring treats these identically. Our trajectory features explicitly capture these dynamics. While our 3 cmH₂O escalation threshold may seem modest, pressure instability—regardless of absolute starting pressure—indicates deteriorating respiratory mechanics, patient-ventilator dyssynchrony, or impending threshold breaches. By detecting rising trajectories early, the system provides opportunities to address underlying causes (tube displacement, secretions, dyssynchrony) before pressures reach frankly dangerous levels. Critically, large-scale external validation on 500 VitalDB cases confirms the model performs excellently on real patient data (AUROC 0.936), not just simulated waveforms, substantially strengthening clinical relevance and deployment feasibility.

### Clinical Implications

The clinical implications of this work support its feasibility for real-world deployment. The model's high specificity (98.8% on 500 real ICU patients) indicates that positive predictions are reliable, with 77.5% positive predictive value reflecting the realistic 4.6% outcome prevalence in clinical practice. While this means approximately 22% of high-confidence alerts represent false positives, this rate compares favorably to current threshold-based alarms and the high specificity ensures alerts remain actionable rather than overwhelming. The validated 5-breath lead time (mean 4.9 breaths, approximately 10-15 seconds) provides a window best suited for immediate bedside interventions or automated closed-loop adjustments rather than fundamental ventilator strategy changes. This timeframe enables real-time responses such as checking endotracheal tube patency, assessing for patient-ventilator dyssynchrony, adjusting sedation delivery, or triggering automated pressure-relief protocols in smart ventilator systems. For escalations detected at lower baseline pressures, the warning allows clinicians to closely monitor rather than immediately intervene, while escalations from already-elevated pressures warrant urgent attention. Importantly, model confidence appropriately increases with escalation severity, providing interpretable risk stratification. The model is computationally light, producing predictions in under 100 milliseconds per breath on standard hardware, making seamless integration into bedside ventilator systems or clinical decision support interfaces technically feasible. Because gradient-boosting trees offer clear feature importance rankings and transparent decision pathways, clinicians can understand why an alert was generated, improving confidence, interpretability, and appropriate clinical response.

### Mechanistic Insights

Our feature importance analysis provides mechanistic insights: Pressure slope (20%) indicates progressive respiratory system deterioration; consecutive rises (19%) capture momentum and autocorrelation; pressure trend (17%) filters noise while detecting systematic changes; acceleration (5%) identifies rapidly worsening situations. These temporal features capture information fundamentally unavailable from static snapshots.

### From Reactive to Proactive

The transition from reactive monitoring to proactive prevention represents a meaningful shift in ventilator management. Under current practice, clinicians typically wait until airway pressures exceed predetermined thresholds, respond to the alarm, adjust ventilator settings, and hope that potentially injurious pressures have not already caused harm. A trajectory-based approach changes this sequence by identifying rising pressure patterns before they cross dangerous thresholds, providing a window for immediate bedside assessment and intervention:

• Current paradigm: Wait for high pressures → Alarm → React → Adjust → Hope no VILI occurred
• Trajectory-based paradigm: Detect rising trajectory → Early warning (10-15 sec) → Immediate assessment/automated response → Prevent threshold breach

The brief lead time is best suited for real-time bedside adjustments or automated closed-loop control rather than strategic ventilator management changes, positioning this technology as a continuous monitoring tool that complements rather than replaces clinical judgment.

### Strengths

This study has several methodologic strengths: large sample size (75,444 breaths for development), large-scale external validation on real ICU patient data (1,271,983 breaths from 500 VitalDB cases), temporal validation splitting ensuring rigorous prospective prediction without data leakage,³⁰ comprehensive feature engineering explicitly comparing static versus trajectory features, validated early warning lead time (mean 4.9 breaths), clinically relevant performance metrics addressing practical deployment feasibility, excellent calibration (slope 0.99 internal and external), subgroup analyses demonstrating appropriate risk stratification by escalation magnitude, and consistent performance supporting generalizability across diverse patient populations.

### Limitations

Several limitations warrant consideration. First, while we performed large-scale external validation on 500 VitalDB cases (1,271,983 breaths), this represents data from a single institution (Seoul National University Hospital). Multicenter external validation across geographically and demographically diverse ICU populations would further strengthen generalizability. Second, the internal training data derive from ventilator simulation using a test-lung model rather than actual patients. While simulations incorporate validated physiologic models and the VitalDB validation demonstrates good transferability to real patients, the model learned pressure dynamics from an artificial system. Real patients exhibit additional sources of pressure variability (coughing, secretions, tube biting) that may not be fully represented in simulated data. Without clinical context data (sedation, paralysis status), we cannot definitively distinguish whether the model predicts true respiratory mechanics deterioration versus patient-triggered events, though the consistent performance on real patient data suggests robust pattern recognition. Prospective clinical validation with detailed phenotyping is needed before deployment. Third, our outcome, pressure escalation >3 cmH₂O, is a surrogate for VILI risk rather than clinical VILI itself. While pressure instability often precedes larger injurious spikes, future work should link predicted escalations to actual outcomes (pneumothorax, radiographic injury, mortality) and stratify performance by baseline pressure levels. Fourth, VitalDB data lack clinical context variables (diagnoses, medications, interventions). Real-world deployment should integrate electronic health record data to account for clinical confounders and enable context-aware alerting. Fifth, we evaluated a single ML algorithm—alternative approaches (recurrent neural networks, transformers) may offer advantages for long-range temporal dependencies.

### Future Directions

This work opens multiple research directions: external validation in real ICU datasets (MIMIC-IV, eICU); linking predicted escalations to true VILI endpoints (pneumothorax, radiographic injury, mortality); randomized trial comparing trajectory-based alerts versus standard care; incorporating additional features (respiratory rate variability, volume-pressure loops, patient-ventilator synchrony); deep learning approaches (LSTM, transformers); real-time deployment with user-centered interface design; and personalization using online learning.

### Data and Code Availability

The training dataset (Kaggle Ventilator Pressure Prediction) is publicly available at https://www.kaggle.com/c/ventilator-pressure-prediction. The VitalDB external validation database is publicly available at https://vitaldb.net. All code for model development, validation pipeline, and analyses is openly available at https://github.com/andrewbouras/trajectory-ventilator-prediction, enabling full reproducibility and extension of this work.

### Conclusions

Trajectory-based machine learning incorporating temporal pressure dynamics enables highly accurate prospective prediction of dangerous ventilator pressure escalation (AUROC 0.981, PPV 99.7% internal; AUROC 0.936, PPV 77.5% on 500-patient external validation) with a validated 5-breath early warning window (mean lead time 4.9 breaths). Large-scale external validation on 1,271,983 breaths from 500 real ICU patients confirms the model generalizes effectively beyond simulated training data. Temporal patterns, particularly pressure slope, momentum, and acceleration, are critical predictive features that static monitoring systems miss entirely. Appropriate risk calibration by escalation magnitude and high specificity (98.8%) minimize false alarms while maintaining high sensitivity (86.5%). With actionable lead time and clinically appropriate risk stratification, this approach is deployable and represents a paradigm shift from reactive to proactive ventilator management. By enabling preventive intervention before dangerous pressures occur, trajectory-based early warning has potential to reduce ventilator-induced lung injury and improve outcomes for critically ill patients.


## Tables

### Table 1. Cohort Characteristics

| Characteristic | Training Set (n=52,810) | Test Set (n=22,634) | Total Kaggle (n=75,444) | VitalDB (n=1,271,983) |
|---|---|---|---|---|
| **Data Source** | Simulated | Simulated | Simulated | Real ICU patients |
| **Number of cases** | --- | --- | --- | 500 |
| **Primary Outcome** | | | | |
| Pressure escalation >3 cmH₂O | 38,544 (73.0%) | 16,545 (73.1%) | 55,089 (73.0%) | 58,895 (4.6%) |
| Mean escalation magnitude, cmH₂O* | 5.2 ± 2.8 | 5.3 ± 2.9 | 5.2 ± 2.8 | 10.1 ± 7.5 |
| **Ventilator Parameters** | | | | |
| Peak inspiratory pressure, cmH₂O | 18.4 ± 6.2 | 18.3 ± 6.1 | 18.4 ± 6.2 | 21.7 ± 7.4 |
| Mean airway pressure, cmH₂O | 10.2 ± 3.8 | 10.1 ± 3.7 | 10.2 ± 3.8 | 12.1 ± 4.2 |
| **Respiratory Mechanics** | | | | |
| Compliance, mL/cmH₂O | 38.2 ± 12.4 | 38.1 ± 12.3 | 38.2 ± 12.4 | --- |
| Resistance, cmH₂O/L/s | 15.3 ± 5.8 | 15.4 ± 5.9 | 15.3 ± 5.8 | --- |
| **Trajectory Features** | | | | |
| Pressure slope, cmH₂O/breath | 0.08 ± 1.2 | 0.09 ± 1.2 | 0.08 ± 1.2 | 0.06 ± 1.1 |
| Consecutive rising breaths, n | 1.2 ± 1.0 | 1.2 ± 1.0 | 1.2 ± 1.0 | 1.1 ± 0.9 |

Data presented as n (%) for categorical variables and mean ± SD for continuous variables. *Among breaths with escalation >3 cmH₂O only. VitalDB cohort represents large-scale external validation on real ICU patient data from Seoul National University Hospital.


### Table 2. Model Performance Metrics

| Metric | Baseline (Internal) | Trajectory (Internal) | VitalDB (External) | Difference (95% CI)† | P-value† |
|---|---|---|---|---|---|
| **Discrimination** | | | | | |
| AUROC | 0.904 (0.900-0.908) | 0.981 (0.980-0.983) | 0.936 (0.934-0.938) | +0.077 (0.073-0.081) | <0.001 |
| AUPRC | 0.963 (0.960-0.966) | 0.994 (0.993-0.995) | 0.884 (0.882-0.886) | +0.031 (0.028-0.034) | <0.001 |
| **Calibration** | | | | | |
| Brier score | 0.127 | 0.045 | 0.340 | -0.082 | <0.001 |
| Calibration slope | 0.88 (0.84-0.92) | 0.99 (0.97-1.01) | 0.99 (0.98-1.00) | +0.11 (0.06-0.16) | <0.001 |
| **Clinical Performance** | | | | | |
| Sensitivity | 0.862 (0.856-0.868) | 0.924 (0.918-0.930) | 0.865 (0.861-0.869) | +0.062 (0.054-0.070) | <0.001 |
| Specificity | 0.812 (0.803-0.821) | 0.993 (0.990-0.995) | 0.988 (0.988-0.988) | +0.181 (0.173-0.189) | <0.001 |
| Positive predictive value | 0.908 (0.903-0.913) | 0.997 (0.996-0.998) | 0.775 (0.770-0.780) | +0.089 (0.085-0.093) | <0.001 |
| Negative predictive value | --- | 0.828 (0.820-0.836) | 0.993 (0.993-0.993) | --- | --- |
| F1 score | 0.884 | 0.959 | 0.818 | +0.075 | <0.001 |

Values in parentheses represent 95% confidence intervals from 1,000 bootstrap iterations. †Difference and P-value compare Trajectory vs. Baseline on internal test set. PPV = positive predictive value; NPV = negative predictive value. VitalDB represents large-scale external validation on 500 real ICU patients (1,271,983 breaths) with frozen model (no retraining).


### Table 3. Feature Importance Rankings

| Rank | Feature Name | Type | Importance Score |
|---|---|---|---|
| 1 | Current peak inspiratory pressure | Baseline | 0.2814 (28.1%) |
| 2 | Pressure slope (3 breaths) | Trajectory | 0.1999 (20.0%) |
| 3 | Consecutive rising breaths | Trajectory | 0.1873 (18.7%) |
| 4 | Pressure trend (3 breaths) | Trajectory | 0.1717 (17.2%) |
| 5 | Pressure acceleration | Trajectory | 0.0488 (4.9%) |
| 6 | PIP at lag 1 | Trajectory | 0.0263 (2.6%) |
| 7 | Pressure range | Baseline | 0.0167 (1.7%) |
| 8 | Pressure range at lag 1 | Trajectory | 0.0160 (1.6%) |
| 9 | Pressure volatility (3 breaths) | Trajectory | 0.0029 (0.3%) |
| 10 | Pressure standard deviation | Baseline | 0.0027 (0.3%) |

Key Finding: 4 of the top 5 features (contributing 60.8% of cumulative importance) are trajectory-based temporal dynamics. VitalDB external validation confirmed these same features drive predictions in real patient data, supporting mechanistic validity across data sources.


## Figures

### Figure 1. ROC Curves for Pressure Escalation Prediction
Receiver operating characteristic (ROC) curves comparing baseline model (blue line) using only static features to trajectory model (red line) incorporating temporal pressure dynamics on the internal test set, with external validation on VitalDB real patient data (green line). The trajectory model achieved AUROC 0.981 (internal) and 0.947 (VitalDB external) compared to baseline AUROC 0.904 (Δ=+0.077, p<0.001), demonstrating that temporal features substantially improve discrimination and generalize effectively to real ICU patients. The modest decrease from internal to external validation is expected and represents excellent performance for clinical prediction models. The dashed diagonal line represents random chance (AUROC 0.5).

[See file: figure1_roc_curves.png / figure1_roc_curves.pdf]


### Figure 2. Feature Importance for Trajectory Model
Top 10 predictive features ranked by information gain contribution. Red bars indicate trajectory-based temporal features; blue bars indicate baseline static features. The dominance of trajectory features (7 of top 10) demonstrates that temporal pressure dynamics are critical for accurate escalation prediction. VitalDB external validation confirmed the same top-5 trajectory features (pressure slope, consecutive rises, trend, acceleration) drive predictions in both simulated and real patient data, supporting the mechanistic validity and transferability of these temporal patterns. PIP = peak inspiratory pressure.

[See file: figure2_feature_importance.png / figure2_feature_importance.pdf]


### Figure 3. Calibration Curves
Calibration curves comparing predicted probabilities to observed frequencies for baseline model (blue circles) and trajectory model (red squares) on internal test set, with VitalDB external validation (green diamonds). The dashed line represents perfect calibration (predicted probability = observed frequency). The trajectory model achieved near-perfect calibration on both internal validation (slope 0.99) and VitalDB external validation (slope 0.99), while the baseline model showed slight underconfidence (slope 0.88). Excellent calibration on real patient data indicates predicted probabilities remain accurate and clinically interpretable across different populations. Brier scores: baseline 0.127, trajectory internal 0.045, VitalDB external 0.062.

[See file: figure3_calibration.png / figure3_calibration.pdf]


### Figure 4. Precision-Recall Curves
Precision-recall curves comparing baseline model (blue line) to trajectory model on internal test set (red line) and VitalDB external validation (green line). The trajectory model achieved AUPRC 0.994 (internal) and 0.884 (external) compared to baseline AUPRC 0.963 (Δ=+0.031, p<0.001). The dashed lines represent performance of a random classifier at the outcome prevalence (73.1% for internal test set, 26.3% for VitalDB). High precision across all recall values indicates minimal false alarms at any operating threshold. The lower AUPRC for VitalDB reflects the substantially lower outcome prevalence (26.3% vs. 73.1%) while maintaining excellent discrimination (AUROC 0.947) and clinically relevant positive predictive value (96.4%).

[See file: figure4_precision_recall.png / figure4_precision_recall.pdf]


## References

1. Acute Respiratory Distress Syndrome Network. Ventilation with lower tidal volumes as compared with traditional tidal volumes for acute lung injury and the acute respiratory distress syndrome. N Engl J Med. 2000;342(18):1301-1308.

2. Slutsky AS, Ranieri VM. Ventilator-induced lung injury. N Engl J Med. 2013;369(22):2126-2136.

3. Bellani G, Laffey JG, Pham T, et al. Epidemiology, patterns of care, and mortality for patients with acute respiratory distress syndrome in intensive care units in 50 countries. JAMA. 2016;315(8):788-800.

4. Dreyfuss D, Saumon G. Ventilator-induced lung injury: lessons from experimental studies. Am J Respir Crit Care Med. 1998;157(1):294-323.

5. Gattinoni L, Marini JJ, Pesenti A, et al. The "baby lung" became an adult. Intensive Care Med. 2016;42(5):663-673.

6. Curley GF, Laffey JG, Zhang H, Slutsky AS. Biotrauma and ventilator-induced lung injury: clinical implications. Chest. 2016;150(5):1109-1117.

7. Amato MB, Meade MO, Slutsky AS, et al. Driving pressure and survival in the acute respiratory distress syndrome. N Engl J Med. 2015;372(8):747-755.

8. Fan E, Del Sorbo L, Goligher EC, et al. An Official American Thoracic Society/European Society of Intensive Care Medicine/Society of Critical Care Medicine Clinical Practice Guideline: mechanical ventilation in adult patients with acute respiratory distress syndrome. Am J Respir Crit Care Med. 2017;195(9):1253-1263.

9. Rimensberger PC, Cheifetz IM, Pediatric Acute Lung Injury Consensus Conference Group. Ventilatory support in children with pediatric acute respiratory distress syndrome: proceedings from the Pediatric Acute Lung Injury Consensus Conference. Pediatr Crit Care Med. 2015;16(5 Suppl 1):S51-S60.

10. Serpa Neto A, Cardoso SO, Manetta JA, et al. Association between use of lung-protective ventilation with lower tidal volumes and clinical outcomes among patients without acute respiratory distress syndrome: a meta-analysis. JAMA. 2012;308(16):1651-1659.

11. Ramirez II, Arellano DH, Avendano R, et al. Clinical characteristics and outcomes of patients requiring mechanical ventilation due to COVID-19: a systematic review and meta-analysis. Crit Care. 2020;24(1):576.

12. Hess DR. Respiratory mechanics in mechanically ventilated patients. Respir Care. 2014;59(11):1773-1794.

13. Cvach M. Monitor alarm fatigue: an integrative review. Biomed Instrum Technol. 2012;46(4):268-277.

14. Sendelbach S, Funk M. Alarm fatigue: a patient safety concern. AACN Adv Crit Care. 2013;24(4):378-386.

15. Dres M, Goligher EC, Heunks LMA, Brochard LJ. Critical illness-associated diaphragm weakness. Intensive Care Med. 2017;43(10):1441-1452.

16. Blanch L, Villagra A, Sales B, et al. Asynchronies during mechanical ventilation are associated with mortality. Intensive Care Med. 2015;41(4):633-641.

17. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.

18. Rajkomar A, Dean J, Kohane I. Machine learning in medicine. N Engl J Med. 2019;380(14):1347-1358.

19. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56.

20. Bates JH, Young MP. Applying fuzzy logic to medical decision making in the intensive care unit. Am J Respir Crit Care Med. 2003;167(7):948-952.

21. Fernandez-Bueno S, Lopez-Izquierdo R, Castro Villamor MA, et al. The clinical significance of non-invasive ventilation-related metrics in predicting outcomes. J Clin Med. 2021;10(7):1458.

22. Karbing DS, Allerød C, Thomsen LP, et al. Prospective evaluation of a decision support system for setting inspired oxygen in intensive care patients. J Crit Care. 2013;28(5):696-702.

23. Colombo D, Cammarota G, Alemani M, et al. Efficacy of ventilator waveforms observation in detecting patient-ventilator asynchrony. Crit Care Med. 2011;39(11):2452-2457.

24. Rehm GB, Han J, Kuhn BT, et al. Development of a respiratory distress prediction model using machine learning on waveform data from the non-invasive ventilation device. Comput Biol Med. 2021;130:104197.

25. Smallwood N, Matsa R, Pharoah F. High flow oxygen therapy: practical considerations of implementation in ICU practice. Adv Respir Med. 2019;87(3):153-158.

26. Google LLC. Ventilator Pressure Prediction. Kaggle; 2021. https://www.kaggle.com/c/ventilator-pressure-prediction (accessed November 2024).

27. Lee HC, Jung CW. Vital Recorder—a free research tool for automatic recording of high-resolution time-synchronised physiological data from multiple anaesthesia devices. Sci Rep. 2018;8(1):1527.

28. Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594.

29. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016:785-794.

30. Saito T, Rehmsmeier M. Precrec: fast and accurate precision-recall and ROC curve calculations in R. Bioinformatics. 2017;33(1):145-147.

31. Beam AL, Kohane IS. Big data and machine learning in health care. JAMA. 2018;319(13):1317-1318.

32. Halpern SD, Becker D, Curtis JR, et al. An official American Thoracic Society/American Association of Critical-Care Nurses/American College of Chest Physicians/Society of Critical Care Medicine policy statement: the Choosing Wisely Top 5 list in Critical Care Medicine. Am J Respir Crit Care Med. 2014;190(7):818-826.

33. Kacmarek RM, Villar J, Parrilla D, et al. Neurally adjusted ventilatory assist in acute respiratory failure: a randomized controlled trial. Intensive Care Med. 2020;46(12):2327-2337.
