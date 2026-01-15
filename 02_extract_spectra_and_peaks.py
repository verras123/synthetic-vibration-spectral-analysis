# 02_extract_spectra_and_peaks.py
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils_signal import unilateral_fft, amplitude_spectral_density, detect_peaks

AXES = ["x", "y", "z", "total"]

def make_dirs(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_signal_csv(path):
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError(f"Missing time_s in {path}")
    for col in ["ax", "ay", "az"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {path}")

    t = pd.to_numeric(df["time_s"], errors="coerce").dropna().values
    if t.size < 2:
        raise ValueError(f"Not enough time samples in {path}")

    dt = float(t[1] - t[0])
    if dt <= 0:
        raise ValueError(f"Non-positive dt in {path}")
    fs = 1.0 / dt

    ax = pd.to_numeric(df["ax"], errors="coerce").dropna().values
    ay = pd.to_numeric(df["ay"], errors="coerce").dropna().values
    az = pd.to_numeric(df["az"], errors="coerce").dropna().values

    n = min(t.size, ax.size, ay.size, az.size)
    t = t[:n]
    ax = ax[:n]
    ay = ay[:n]
    az = az[:n]
    atot = np.sqrt(ax*ax + ay*ay + az*az)

    return fs, t, {"x": ax, "y": ay, "z": az, "total": atot}

def plot_spectrum(freqs, vals, peak_idx, title, out_png):
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, vals, linewidth=1.0)
    if peak_idx.size > 0:
        plt.scatter(freqs[peak_idx], vals[peak_idx], s=10)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_path", default="data/metadata.csv", help="Metadata CSV from generator")
    ap.add_argument("--out_dir", default="outputs", help="Outputs folder")
    ap.add_argument("--max_freq_hz", type=float, default=250.0, help="Max frequency to keep in outputs")
    ap.add_argument("--prominence_ratio", type=float, default=0.05, help="Peak prominence ratio")
    ap.add_argument("--nperseg", type=int, default=256, help="Welch nperseg for ASD")
    ap.add_argument("--max_plots", type=int, default=40, help="Limit number of saved plots")
    args = ap.parse_args()

    meta_path = Path(args.meta_path)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    make_dirs(out_dir)
    make_dirs(fig_dir)

    meta = pd.read_csv(meta_path)
    if meta.empty:
        raise ValueError("Metadata is empty. Run 01_generate_synthetic_data.py first.")

    tidy_rows = []
    peak_rows = []

    plot_count = 0

    for _, row in meta.iterrows():
        sample_id = int(row["sample_id"])
        fpath = row["file"]
        condition_level = float(row["condition_level"])
        position_m = float(row["position_m"])

        fs, t, sigs = load_signal_csv(fpath)

        for transform in ["FFT", "ASD"]:
            for axis in AXES:
                sig = sigs[axis]

                if transform == "FFT":
                    freqs, vals = unilateral_fft(sig, fs_hz=fs)
                    ylab = "Amplitude (m/s^2)"
                else:
                    freqs, vals = amplitude_spectral_density(sig, fs_hz=fs, nperseg=args.nperseg)
                    ylab = "ASD ((m/s^2)/sqrt(Hz))"

                # Cut max freq
                m = freqs <= float(args.max_freq_hz)
                freqs = freqs[m]
                vals = vals[m]

                # Store tidy
                for f, v in zip(freqs, vals):
                    tidy_rows.append({
                        "sample_id": sample_id,
                        "condition_level": condition_level,
                        "position_m": position_m,
                        "axis": axis,
                        "transform": transform,
                        "frequency_hz": float(f),
                        "value": float(v)
                    })

                # Peaks
                pidx = detect_peaks(vals, prominence_ratio=float(args.prominence_ratio))
                for i in pidx:
                    peak_rows.append({
                        "sample_id": sample_id,
                        "condition_level": condition_level,
                        "position_m": position_m,
                        "axis": axis,
                        "transform": transform,
                        "peak_frequency_hz": float(freqs[i]),
                        "peak_value": float(vals[i])
                    })

                # Save limited plots
                if plot_count < int(args.max_plots):
                    title = f"{transform} - sample {sample_id} - axis {axis} - cond {int(condition_level)} - pos {position_m:.1f}m"
                    out_png = fig_dir / f"spectrum_{transform.lower()}_s{sample_id:04d}_{axis}.png"
                    plot_spectrum(freqs, vals, pidx, title, out_png)
                    plot_count += 1

    tidy = pd.DataFrame(tidy_rows)
    peaks = pd.DataFrame(peak_rows)

    tidy_path = out_dir / "tidy_spectra.csv"
    peaks_path = out_dir / "peaks_table.csv"
    tidy.to_csv(tidy_path, index=False)
    peaks.to_csv(peaks_path, index=False)

    info = {
        "tidy_path": str(tidy_path.as_posix()),
        "peaks_path": str(peaks_path.as_posix()),
        "n_tidy_rows": int(tidy.shape[0]),
        "n_peaks_rows": int(peaks.shape[0]),
        "plots_saved": int(plot_count)
    }
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
