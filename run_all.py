# run_all.py
import argparse
import subprocess
import sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run generate + extract + analyze")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    do_all = args.all or (not args.generate and not args.extract and not args.analyze)

    py = sys.executable

    if do_all or args.generate:
        run([py, "01_generate_synthetic_data.py"])

    if do_all or args.extract:
        run([py, "02_extract_spectra_and_peaks.py"])

    if do_all or args.analyze:
        run([py, "04_pca_and_ml_analysis.py", "--target", "condition_level"])

if __name__ == "__main__":
    main()
