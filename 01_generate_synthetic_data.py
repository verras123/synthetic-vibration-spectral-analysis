# 01_generate_synthetic_data.py
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def make_dirs(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def synth_signal(t, base_freq_hz, condition_level, position_m, seed):
    """
    Generates a synthetic acceleration signal:
    - baseline harmonics + resonance peak + colored-ish noise + mild nonstationarity
    condition_level: 0..100
    """
    rng = np.random.default_rng(int(seed))

    # Map condition_level into continuous factor
    c = float(condition_level) / 100.0
    pos = float(position_m)

    # Frequency shifts with condition and position (small, realistic)
    f0 = base_freq_hz * (1.0 + 0.15 * c) * (1.0 + 0.02 * np.sin(2.0 * np.pi * pos / 10.0))

    # Components
    a1 = 0.8 + 1.5 * c
    a2 = 0.3 + 0.6 * c
    a3 = 0.15 + 0.4 * c

    # Mild amplitude modulation (nonstationary)
    mod = 1.0 + 0.15 * np.sin(2.0 * np.pi * (0.7 + 0.3 * c) * t)

    sig = (
        a1 * np.sin(2.0 * np.pi * f0 * t) +
        a2 * np.sin(2.0 * np.pi * (2.0 * f0) * t + 0.3) +
        a3 * np.sin(2.0 * np.pi * (3.0 * f0) * t + 1.1)
    ) * mod

    # Add a "resonance-like" narrowband component
    f_res = 90.0 + 40.0 * c + 3.0 * pos
    sig += (0.4 + 1.2 * c) * np.sin(2.0 * np.pi * f_res * t + rng.uniform(0, 2*np.pi))

    # Noise (white + a crude low-frequency drift)
    white = rng.normal(0.0, 0.35 + 0.25 * (1.0 - c), size=t.size)
    drift = 0.10 * np.sin(2.0 * np.pi * (0.2 + 0.1 * rng.random()) * t + rng.uniform(0, 2*np.pi))
    sig = sig + white + drift

    return sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw", help="Output directory for CSV signals")
    ap.add_argument("--meta_path", default="data/metadata.csv", help="Metadata CSV output")
    ap.add_argument("--fs_hz", type=float, default=1000.0, help="Sampling frequency (Hz)")
    ap.add_argument("--duration_s", type=float, default=10.0, help="Signal duration (s)")
    ap.add_argument("--n_samples_per_combo", type=int, default=3, help="Replicates per condition/position")
    ap.add_argument("--seed", type=int, default=42, help="Global seed")

    # Generic, NDA-safe "condition" and "position"
    ap.add_argument("--condition_levels", default="0,25,50,75,100", help="Comma-separated condition levels (0..100)")
    ap.add_argument("--positions_m", default="2,4,6,8", help="Comma-separated positions (meters)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_path)
    make_dirs(out_dir)
    make_dirs(meta_path.parent)

    fs = float(args.fs_hz)
    dur = float(args.duration_s)
    n = int(round(fs * dur))
    if n < 10:
        raise ValueError("duration_s too small for given fs_hz")

    t = np.arange(n, dtype=float) / fs

    cond_levels = [int(x.strip()) for x in args.condition_levels.split(",") if x.strip() != ""]
    positions = [float(x.strip()) for x in args.positions_m.split(",") if x.strip() != ""]

    rng = np.random.default_rng(int(args.seed))

    rows = []
    sample_id = 0

    for cond in cond_levels:
        for pos in positions:
            for k in range(int(args.n_samples_per_combo)):
                sample_id += 1

                # Per-sample seed
                sseed = int(rng.integers(0, 2**31 - 1))

                # Three axes have slightly different base frequencies
                ax = synth_signal(t, base_freq_hz=30.0, condition_level=cond, position_m=pos, seed=sseed + 11)
                ay = synth_signal(t, base_freq_hz=33.0, condition_level=cond, position_m=pos, seed=sseed + 22)
                az = synth_signal(t, base_freq_hz=27.0, condition_level=cond, position_m=pos, seed=sseed + 33)

                df = pd.DataFrame({
                    "time_s": t,
                    "ax": ax,
                    "ay": ay,
                    "az": az
                })

                fname = f"accel_condition{cond:03d}_pos{int(round(pos)):02d}_sample{sample_id:04d}.csv"
                fpath = out_dir / fname
                df.to_csv(fpath, index=False)

                rows.append({
                    "sample_id": sample_id,
                    "file": str(fpath.as_posix()),
                    "condition_level": cond,
                    "position_m": pos,
                    "fs_hz": fs,
                    "duration_s": dur,
                    "seed": sseed
                })

    meta = pd.DataFrame(rows)
    meta.to_csv(meta_path, index=False)

    info = {
        "out_dir": str(out_dir.as_posix()),
        "meta_path": str(meta_path.as_posix()),
        "n_rows_metadata": int(meta.shape[0])
    }
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
