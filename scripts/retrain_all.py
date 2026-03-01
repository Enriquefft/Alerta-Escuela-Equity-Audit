"""Orchestration script to retrain LR, RF, MLP with 31-feature matrix.

Handles the critical ordering:
1. Save existing model_results.json entries (lightgbm, xgboost, test_2023_calibrated)
2. Run baseline.py (overwrites model_results.json with only LR)
3. Merge saved entries back
4. Run random_forest.py (adds RF entry)
5. Run mlp.py (adds MLP entry)
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "exports" / "model_results.json"


def main():
    # Step 1: Save existing entries
    print("=" * 60)
    print("STEP 1: Saving existing model_results.json entries")
    print("=" * 60)
    with open(RESULTS_PATH) as f:
        saved = json.load(f)
    saved_keys = {}
    for key in ["lightgbm", "xgboost", "test_2023_calibrated"]:
        if key in saved:
            saved_keys[key] = saved[key]
            print(f"  Saved: {key}")
    print()

    # Step 2: Run baseline.py (LR)
    print("=" * 60)
    print("STEP 2: Retraining Logistic Regression")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "models" / "baseline.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("ERROR: baseline.py failed!")
        sys.exit(1)

    # Step 3: Merge saved entries back
    print("\n" + "=" * 60)
    print("STEP 3: Merging saved entries back into model_results.json")
    print("=" * 60)
    with open(RESULTS_PATH) as f:
        lr_results = json.load(f)
    for key, value in saved_keys.items():
        lr_results[key] = value
        print(f"  Restored: {key}")
    with open(RESULTS_PATH, "w") as f:
        json.dump(lr_results, f, indent=2)
    print(f"  model_results.json keys: {sorted(lr_results.keys())}")
    print()

    # Step 4: Run random_forest.py
    print("=" * 60)
    print("STEP 4: Retraining Random Forest")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "models" / "random_forest.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("ERROR: random_forest.py failed!")
        sys.exit(1)
    print()

    # Step 5: Run mlp.py
    print("=" * 60)
    print("STEP 5: Retraining MLP")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "models" / "mlp.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("ERROR: mlp.py failed!")
        sys.exit(1)

    # Final verification
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    with open(RESULTS_PATH) as f:
        final = json.load(f)
    print(f"  Keys: {sorted(final.keys())}")
    for model in ["logistic_regression", "lightgbm", "xgboost", "random_forest", "mlp"]:
        nf = final[model]["metadata"]["n_features"]
        prauc = final[model]["metrics"]["validate_2022"]["weighted"]["pr_auc"]
        print(f"  {model}: n_features={nf}, val PR-AUC={prauc:.4f}")
    print("\nALL MODELS RETRAINED SUCCESSFULLY")


if __name__ == "__main__":
    main()
