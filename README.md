# synthetic-vibration-spectral-analysis

End-to-end vibration spectral analysis pipeline using **synthetic data**:
**data generation + FFT/ASD (Welch) + peak detection + PCA + ML**.

This repository is NDA-safe and fully reproducible locally.

---

## Project overview

This pipeline simulates multi-axis acceleration signals and extracts spectral features to recover a hidden
**condition level** (0-100) using signal processing + machine learning.

Main steps:
1. Generate synthetic vibration data (time-domain)
2. Compute spectra (FFT and ASD/Welch)
3. Detect spectral peaks
4. Build a tidy dataset (long format)
5. Run PCA + regression/classification models

---

## Repository structure

- `01_generate_synthetic_data.py`  
  Generates synthetic acceleration signals and saves CSV files + metadata.

- `02_extract_spectra_and_peaks.py`  
  Computes FFT + ASD, detects peaks, exports tidy tables and figures.

- `04_pca_and_ml_analysis.py`  
  PCA + RandomForest regression/classification to predict the condition level.

- `run_all.py`  
  Runs the full pipeline end-to-end.

- `src/utils_signal.py`  
  Signal processing utilities (FFT, ASD/Welch, peak detection).

---

## Requirements

Python 3.10+ recommended.

Install dependencies:
```bash
pip install -r requirements.txt
