# 04_pca_and_ml_analysis.py
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

def make_dirs(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def pivot_wide(df_sub):
    """
    df_sub columns: sample_id, condition_level, position_m, frequency_hz, value
    Wide: index=sample_id, columns=frequency_hz, values=value
    """
    wide = df_sub.pivot_table(index="sample_id", columns="frequency_hz", values="value", aggfunc="mean")
    # Fill NaNs by column mean, then ffill (safe)
    wide = wide.apply(lambda col: col.fillna(col.mean()), axis=0)
    wide = wide.ffill(axis=1).bfill(axis=1)
    return wide

def save_pca_plot(pcs, y, title, out_png):
    plt.figure(figsize=(7, 5))
    # y can be continuous; color by bins for readability
    y = np.asarray(y, dtype=float)
    bins = np.unique(np.percentile(y, [0, 25, 50, 75, 100]))
    if bins.size < 3:
        bins = np.array([y.min(), y.max() + 1e-9])
    yb = np.digitize(y, bins[1:-1], right=True)

    sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=yb, s=18)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy_path", default="outputs/tidy_spectra.csv", help="Tidy spectra CSV")
    ap.add_argument("--out_dir", default="outputs", help="Outputs folder")
    ap.add_argument("--n_components", type=int, default=5, help="PCA components")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--target", default="condition_level", choices=["condition_level", "position_m"])
    args = ap.parse_args()

    tidy_path = Path(args.tidy_path)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    make_dirs(out_dir)
    make_dirs(fig_dir)

    df = pd.read_csv(tidy_path)
    if df.empty:
        raise ValueError("Tidy spectra is empty. Run 02_extract_spectra_and_peaks.py first.")

    metrics = {
        "target": args.target,
        "by_group": []
    }

    # Analyze by (transform, axis)
    for transform in sorted(df["transform"].unique()):
        for axis in sorted(df["axis"].unique()):
            sub = df[(df["transform"] == transform) & (df["axis"] == axis)].copy()
            if sub.empty:
                continue

            # Targets per sample_id
            y_tbl = sub.groupby("sample_id")[["condition_level", "position_m"]].first().reset_index()
            y = y_tbl[args.target].values

            wide = pivot_wide(sub[["sample_id", "frequency_hz", "value"]])
            # Align
            wide = wide.loc[y_tbl["sample_id"].values]

            X = wide.values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            ncomp = int(min(args.n_components, Xs.shape[1], Xs.shape[0]))
            if ncomp < 2:
                continue

            pca = PCA(n_components=ncomp, random_state=int(args.random_state))
            pcs = pca.fit_transform(Xs)

            # PCA plot
            pca_png = fig_dir / f"pca_{transform.lower()}_{axis}_{args.target}.png"
            save_pca_plot(pcs, y, f"PCA - {transform} - {axis} - target={args.target}", pca_png)

            # Regression (continuous)
            Xtr, Xte, ytr, yte = train_test_split(
                Xs, y, test_size=float(args.test_size), random_state=int(args.random_state)
            )
            reg = RandomForestRegressor(n_estimators=200, random_state=int(args.random_state))
            reg.fit(Xtr, ytr)
            yp = reg.predict(Xte)
            r2 = float(r2_score(yte, yp))
            rmse = float(np.sqrt(mean_squared_error(yte, yp)))

            # Classification (bin the target)
            # For condition_level it will be exact classes; for position_m it will be exact too in our synthetic setup.
            y_classes = pd.Series(y).astype(str).values
            if len(np.unique(y_classes)) >= 2:
                Xtrc, Xtec, ytrc, ytec = train_test_split(
                    Xs, y_classes, test_size=float(args.test_size), random_state=int(args.random_state)
                )
                clf = RandomForestClassifier(n_estimators=300, random_state=int(args.random_state))
                clf.fit(Xtrc, ytrc)
                ypc = clf.predict(Xtec)
                acc = float(accuracy_score(ytec, ypc))
                report = classification_report(ytec, ypc, zero_division=0)
            else:
                acc = None
                report = "Not enough classes for classification."

            metrics["by_group"].append({
                "transform": transform,
                "axis": axis,
                "n_samples": int(wide.shape[0]),
                "n_features": int(wide.shape[1]),
                "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
                "regression_r2": r2,
                "regression_rmse": rmse,
                "classification_accuracy": acc,
                "classification_report": report
            })

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({
        "metrics_path": str(metrics_path.as_posix()),
        "n_groups": int(len(metrics["by_group"]))
    }, indent=2))

if __name__ == "__main__":
    main()
