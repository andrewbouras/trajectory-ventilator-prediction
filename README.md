# Trajectory-Based Machine Learning for Early Prediction of Dangerous Ventilator Pressure Escalation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**A prospective validation study demonstrating that trajectory-based waveform analysis enables highly accurate prediction of ventilator pressure escalation with a 5-breath early warning window.**

📄 **Manuscript:** *Trajectory-Based Machine Learning For Early Prediction Of Dangerous Ventilator Pressure Escalation: A Prospective Validation Study*  
👤 **Authors:** Andrew Bouras (Nova Southeastern University), Luis Rodriguez (Johns Hopkins School of Medicine)

---

## 🎯 Key Findings

- **AUROC 0.981** on internal test set (75,444 breaths)
- **AUROC 0.936** on external validation (500 ICU patients, 1,271,983 breaths)
- **4.9 breath mean lead time** - validated prospective early warning
- **98.8% specificity** - minimal false alarms
- **Trajectory features** (slope, acceleration, momentum) provide 60% of predictive power

---

## 📊 What This Repository Contains

### Core Code
- **`ventilator_validation/`** - Complete validation pipeline
  - `validate_with_checkpoints.py` - Robust validation with automatic resume
  - `scan_vitaldb.py` - VitalDB case scanning
  - `data_loader.py` - Breath-level feature extraction
  - `models/` - Trained XGBoost model and scaler

### Manuscript
- **`manu_revised.md`** - Full manuscript (Markdown format)
- **`figure*_updated.*`** - All figures (PNG & PDF)
- **`TABLES_AND_FIGURES_README.md`** - Figure descriptions

### Analysis Scripts
- `regenerate_figures.py` - Reproduce all figures
- `create_clinical_analyses.py` - Subgroup analyses
- `verify_authenticity.py` - Data integrity verification

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run 100-Case Validation
```bash
cd ventilator_validation
python3 validate_with_checkpoints.py 100
```

This will:
- Download waveforms from VitalDB API
- Extract breath-level features
- Run predictions with the frozen model
- Save checkpoints at 10, 25, 50, 75, 100 cases
- Generate performance reports

**Output:** Results saved in `validation_checkpoints/`

---

## 📖 Methodology

### Training Data
- **Source:** Kaggle/Google Brain Ventilator Pressure Prediction Dataset
- **Size:** 75,444 breaths (simulated test-lung data)
- **Split:** 70% training, 30% temporal hold-out test

### External Validation
- **Source:** VitalDB Database (Seoul National University Hospital)
- **Size:** 500 cases, 1,271,983 breaths
- **Type:** Real ICU patient data

### Model
- **Algorithm:** XGBoost gradient boosting
- **Features:** 34 total (11 static + 23 trajectory)
- **Outcome:** PIP escalation >3 cmH₂O over next 5 breaths

### Top Trajectory Features
1. Pressure slope (3 breaths) - 20.0%
2. Consecutive rising breaths - 18.7%
3. Pressure trend (3 breaths) - 17.2%
4. Current PIP - 28.1%
5. Pressure acceleration - 4.9%

---

## 📈 Results Summary

### Internal Test Set
| Metric | Baseline | Trajectory |
|--------|----------|------------|
| AUROC | 0.904 | **0.981** |
| Sensitivity | 86.2% | 92.4% |
| Specificity | 81.2% | 99.3% |
| PPV | 90.8% | 99.7% |

### External Validation (500 VitalDB Cases)
| Metric | Value |
|--------|-------|
| AUROC | 0.936 |
| Sensitivity | 86.5% |
| Specificity | 98.8% |
| PPV | 77.5% |
| NPV | 99.3% |

### Subgroup Analysis
- **Small escalations (3-10 cmH₂O):** Mean confidence 0.917
- **Large escalations (>10 cmH₂O):** Mean confidence 0.980

---

## 🔬 Reproducibility

### Regenerate All Figures
```bash
python3 ventilator_validation/regenerate_figures.py
```

### Run Clinical Analyses
```bash
python3 ventilator_validation/create_clinical_analyses.py
```

### Verify Data Authenticity
```bash
python3 ventilator_validation/verify_authenticity.py
```

---

## 📁 Repository Structure

```
.
├── manu_revised.md                      # Main manuscript
├── figure*_updated.{png,pdf}            # Updated figures (500 cases)
├── TABLES_AND_FIGURES_README.md         # Figure descriptions
├── requirements.txt                     # Python dependencies
├── ventilator_validation/
│   ├── validate_with_checkpoints.py     # Main validation script
│   ├── scan_vitaldb.py                  # Scan VitalDB cases
│   ├── data_loader.py                   # Feature extraction
│   ├── models/
│   │   ├── trajectory_model.pkl         # Trained XGBoost model
│   │   └── scaler_trajectory.pkl        # Feature scaler
│   ├── validation_checkpoints/          # Results & checkpoints
│   │   ├── checkpoint_*.json            # Performance at milestones
│   │   ├── per_case_results.csv         # Per-case summary
│   │   └── progress.json                # Resume state
│   └── README_VALIDATION.md             # Validation guide
└── README.md                            # This file
```

---

## 💡 Clinical Implications

### What This Enables
- **Proactive vs. Reactive:** Shift from threshold alarms to trajectory-based early warning
- **Actionable Lead Time:** 10-15 seconds for immediate bedside interventions
- **Low False Alarms:** 98.8% specificity reduces alarm fatigue
- **Automated Integration:** Compatible with closed-loop ventilator systems

### Use Cases
- Real-time bedside monitoring
- Smart ventilator pressure-relief protocols
- Early detection of patient-ventilator dyssynchrony
- Closed-loop automated adjustments

---

## 📚 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{bouras2025trajectory,
  title={Trajectory-Based Machine Learning For Early Prediction Of Dangerous Ventilator Pressure Escalation: A Prospective Validation Study},
  author={Bouras, Andrew and Rodriguez, Luis},
  journal={[Under Review]},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Andrew Bouras**  
Nova Southeastern University Kiran C. Patel College of Osteopathic Medicine  
Email: ab4646@mynsu.nova.edu

---

## 🙏 Acknowledgments

- **VitalDB** for providing open access to ICU waveform data
- **Google Brain/Kaggle** for the ventilator pressure prediction dataset
- **Open-source community** for XGBoost, scikit-learn, and related tools

---

## ⚠️ Disclaimer

This is a research tool and has not been validated for clinical use. It is provided for academic and research purposes only.
