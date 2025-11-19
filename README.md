# Trajectory-Based Machine Learning for Early Prediction of Dangerous Ventilator Pressure Escalation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the manuscript and validation code for a novel trajectory-based machine learning approach for prospective prediction of dangerous ventilator pressure escalation in mechanically ventilated patients. The model achieves:

- **Internal validation (Kaggle dataset):** AUROC 0.981, PPV 99.7%
- **External validation (VitalDB real ICU patients):** AUROC 0.947, PPV 96.4%

## Abstract

Ventilator-induced lung injury (VILI) from excessive airway pressures remains a leading cause of morbidity in mechanically ventilated patients. This study demonstrates that machine learning incorporating temporal pressure dynamics (slope, acceleration, momentum) enables near-perfect prospective prediction of dangerous pressure escalation with a 5-breath early warning window. External validation on real ICU patients confirms the model generalizes beyond simulated data, enabling proactive rather than reactive ventilator management.

## Repository Structure

```
.
├── ajrccm.tex                    # Main manuscript (LaTeX)
├── ventilator_validation/        # Validation pipeline code
│   ├── models/                   # Trained model files
│   │   ├── trajectory_model.pkl  # XGBoost model
│   │   └── scaler_trajectory.pkl # Feature scaler
│   ├── breath_parser.py          # Breath segmentation utilities
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── validate_model.py         # VitalDB validation script
│   ├── test_pipeline.py          # Pipeline testing
│   ├── scan_vitaldb.py           # VitalDB case scanner
│   ├── validation_results.csv    # VitalDB validation results
│   └── setup_env.sh              # Environment setup script
└── README.md                     # This file
```

## Manuscript

The manuscript (`ajrccm.tex`) is formatted for submission to the **American Journal of Respiratory and Critical Care Medicine (AJRCCM)**. It includes:

- Complete methods section with VitalDB external validation
- Results from both internal (Kaggle) and external (VitalDB) validation
- Feature importance analysis
- Discussion of clinical implications and deployment feasibility

### Compiling the Manuscript

```bash
pdflatex ajrccm.tex
pdflatex ajrccm.tex  # Run twice for references
```

## Validation Pipeline

### Requirements

- Python 3.11+
- XGBoost 2.0+
- scikit-learn 1.3+
- pandas 2.1+
- NumPy 1.24+

### Installation

```bash
cd ventilator_validation
bash setup_env.sh
```

Or manually:

```bash
pip install xgboost scikit-learn pandas numpy scipy matplotlib
```

### Running Validation

```bash
cd ventilator_validation
python validate_model.py
```

This will:
1. Load the trained trajectory model
2. Process VitalDB waveform data
3. Extract 33 breath-level features (11 baseline + 22 trajectory)
4. Generate predictions and performance metrics
5. Output validation results

## Key Features

The model uses 33 engineered features:

**Baseline (11):** Respiratory mechanics, current pressure characteristics, flow characteristics

**Trajectory (22):** Historical pressure values, slope features, acceleration, volatility, trend, momentum, range dynamics

**Top 5 Most Important Features:**
1. Current peak inspiratory pressure (28.1%)
2. Pressure slope over 3 breaths (20.0%)
3. Consecutive rising breaths (18.7%)
4. Pressure trend over 3 breaths (17.2%)
5. Pressure acceleration (4.9%)

## Clinical Implications

- **5-breath early warning window** (~10-15 seconds lead time)
- **Minimal false alarms** (PPV 96.4% on real patients)
- **Real-time performance** (<100ms per breath prediction)
- **Actionable insights** for lung-protective ventilation strategies

## Citation

If you use this work, please cite:

```bibtex
@article{bouras2025trajectory,
  title={Trajectory-Based Machine Learning For Early Prediction Of Dangerous Ventilator Pressure Escalation: A Prospective Validation Study},
  author={Bouras, Andrew},
  journal={American Journal of Respiratory and Critical Care Medicine},
  year={2025},
  note={In preparation}
}
```

## Data Sources

- **Training data:** Ventilator Pressure Prediction dataset (Kaggle/Google Brain, 2021)
- **External validation:** VitalDB database (Seoul National University Hospital)

## Author

**Andrew Bouras, OMS-II Research Fellow**  
Nova Southeastern University Kiran C. Patel College of Osteopathic Medicine  
📧 ab4646@mynsu.nova.edu

## License

MIT License - see LICENSE file for details

## Acknowledgments

- VitalDB team for providing open-access ICU waveform data
- Google Brain for the Kaggle ventilator dataset
- Nova Southeastern University for research support

---

*For questions, issues, or collaboration inquiries, please open an issue or contact the author directly.*

