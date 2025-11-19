# VitalDB Validation with Checkpoints

## Overview
Robust validation system with automatic checkpointing and resume capability.
**Never lose progress** - saves after every case processed.

## Available: 6,020 VitalDB cases with airway pressure data

---

## Quick Start Commands

### Option 1: 50 Cases (Conservative - ~1 hour)
```bash
python3 validate_with_checkpoints.py 50
```
**Checkpoints at:** 10, 25, 50 cases

### Option 2: 100 Cases (Recommended - ~2 hours)
```bash
python3 validate_with_checkpoints.py 100
```
**Checkpoints at:** 10, 25, 50, 75, 100 cases

### Option 3: 200 Cases (Strong - ~4 hours)
```bash
python3 validate_with_checkpoints.py 200
```
**Checkpoints at:** 10, 25, 50, 75, 100, 150, 200 cases

### Option 4: 500 Cases (Comprehensive - ~10 hours)
```bash
python3 validate_with_checkpoints.py 500
```
**Checkpoints at:** 10, 25, 50, 75, 100, 150, 200, 300, 500 cases

### Option 5: ALL Cases (Maximum - days, run overnight/weekend)
```bash
python3 validate_with_checkpoints.py
```
**Checkpoints at:** 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000 cases

---

## Safety Features

### 1. **Automatic Resume**
If interrupted (crash, ctrl-C, power loss):
```bash
# Just run the same command again - it will resume automatically!
python3 validate_with_checkpoints.py 500
```

### 2. **Incremental Saving**
- Progress saved after EVERY case
- Predictions saved to CSV continuously
- No data loss even if stopped

### 3. **Checkpoint Reports**
Generated automatically at milestones in `validation_checkpoints/`:
- `checkpoint_50_cases.json`
- `checkpoint_100_cases.json`
- `checkpoint_200_cases.json`
- etc.

### 4. **Real-time Monitoring**
Watch progress in another terminal:
```bash
# See how many cases processed
wc -l validation_checkpoints/per_case_results.csv

# See latest checkpoint
tail validation_checkpoints/progress.json

# Watch real-time
tail -f validation_checkpoints/per_case_results.csv
```

---

## Output Files

All saved in `validation_checkpoints/`:

| File | Description |
|------|-------------|
| `progress.json` | Current state (resume point) |
| `predictions.csv` | All breath-level predictions |
| `per_case_results.csv` | Summary per case |
| `checkpoint_N_cases.json` | Performance report at N cases |

---

## Before Starting

1. **First time? Scan available cases:**
```bash
python3 scan_vitaldb.py
```
This will take ~10 minutes and scan all 6,020 cases.

2. **Check you have the model:**
```bash
ls models/trajectory_model.pkl
ls models/scaler_trajectory.pkl
```

---

## Estimated Times & Breaths

| Cases | Est. Breaths | Est. Time | Checkpoints |
|-------|-------------|-----------|-------------|
| 10 | ~27K | 15 min | 10 |
| 50 | ~135K | 1 hour | 10, 25, 50 |
| 100 | ~270K | 2 hours | 10, 25, 50, 75, 100 |
| 200 | ~540K | 4 hours | 10, 25, 50, 75, 100, 150, 200 |
| 500 | ~1.3M | 10 hours | 10, 25, 50, 75, 100, 150, 200, 300, 500 |
| 1000 | ~2.7M | 20 hours | Every 50-100 up to 1000 |
| 6020 | ~16M | ~5 days | Major milestones |

---

## Recommended Strategy

### For Paper Revision (Quick Turn-around)
1. Start with **100 cases** today
2. Update manuscript with results
3. If reviewers want more, add 200-500 in revision

### For Maximum Impact (If you have time)
1. Start **500 cases** overnight
2. Will give you landmark external validation
3. Checkpoints let you stop at 200 if needed

### For Definitive Study
Run all 6,020 cases over a weekend

