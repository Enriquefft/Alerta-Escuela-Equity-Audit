#!/usr/bin/env python3
"""
Download ENAHO microdata and supplementary datasets for the Alerta Escuela audit.

Uses the `enahodata` library (https://github.com/MaykolMedrano/enahodata_py)
to download from INEI's microdatos portal, then reorganizes files into the
expected project directory structure.

Usage:
    uv run python src/data/download.py

Output structure:
    data/raw/
    ├── enaho/
    │   ├── 2018/    # Contains Enaho01a-2018-300.csv, Enaho01-2018-200.csv, etc.
    │   ├── 2019/
    │   ├── ...
    │   └── 2024/
    └── admin/       # District-level dropout rates from datosabiertos
"""

import os
import sys
import shutil
import glob
import requests
from pathlib import Path

from utils import find_project_root

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]

# ENAHO module codes (cross-sectional):
# 02 = Caracteristicas de los Miembros del Hogar (Module 200 -- demographics, age, sex, UBIGEO)
# 03 = Educacion (Module 300 -- P300A mother tongue, P303/P306 enrollment, FACTOR07)
# 05 = Empleo e Ingresos (Module 500 -- employment status, income)
# 34 = Sumarias (calculated variables -- household income/expenditure, poverty)
# 37 = Programas Sociales (JUNTOS participation)
MODULES = ["02", "03", "05", "34", "37"]


PROJECT_ROOT = find_project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
ENAHO_DIR = DATA_RAW / "enaho"
ADMIN_DIR = DATA_RAW / "admin"
TEMP_DIR = PROJECT_ROOT / ".tmp_enaho_download"

# datosabiertos.gob.pe URLs for district dropout rates
# These may change -- verify at https://www.datosabiertos.gob.pe if 404
ADMIN_URLS = {
    "primaria_2023": "https://www.datosabiertos.gob.pe/sites/default/files/Tasa%20y%20N%C3%BAmero%20de%20desertores%20de%20EBR%20primaria%202023.csv",
    "secundaria_2023": "https://www.datosabiertos.gob.pe/sites/default/files/Tasa%20y%20N%C3%BAmero%20de%20desertores%20de%20EBR%20secundaria%202023.csv",
}


# ---------------------------------------------------------------------------
# Step 1: Download ENAHO modules via enahodata
# ---------------------------------------------------------------------------

def download_enaho():
    """Download ENAHO modules using the enahodata library."""
    try:
        from enahodata import enahodata
    except ImportError:
        print("ERROR: enahodata not installed. Run: pip install enahodata")
        sys.exit(1)

    print("=" * 60)
    print("STEP 1: Downloading ENAHO modules from INEI")
    print(f"  Modules: {MODULES}")
    print(f"  Years: {YEARS}")
    print("=" * 60)

    # enahodata downloads into the current working directory,
    # so we temporarily chdir to a temp location
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(TEMP_DIR)

    try:
        enahodata(
            modulos=MODULES,
            anios=YEARS,
            descomprimir=True,   # Extract ZIPs
            only_dta=False,      # We want ALL files (CSVs + DTAs + docs)
            overwrite=False,     # Don't re-download existing
            output_dir="enaho_raw",
            panel=False,         # Cross-sectional, not panel
        )
    except Exception as e:
        print(f"ERROR during enahodata download: {e}")
        print("\nIf this fails, download manually from https://proyectos.inei.gob.pe/microdatos/")
        print("Place CSV files in data/raw/enaho/{YEAR}/ directories.")
        os.chdir(original_dir)
        return False
    finally:
        os.chdir(original_dir)

    return True


# ---------------------------------------------------------------------------
# Step 2: Reorganize into project structure
# ---------------------------------------------------------------------------

def reorganize_enaho():
    """
    enahodata creates: .tmp_enaho_download/enaho_raw/modulo_XX_YYYY_extract/...
    We need: data/raw/enaho/{YEAR}/Enaho01a-{YEAR}-300.csv, etc.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Reorganizing files into project structure")
    print("=" * 60)

    source_base = TEMP_DIR / "enaho_raw"
    if not source_base.exists():
        print(f"ERROR: Download directory not found at {source_base}")
        return False

    for year in YEARS:
        year_dir = ENAHO_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)

        # Find all extracted directories for this year
        year_patterns = list(source_base.glob(f"*_{year}_extract"))
        year_patterns += list(source_base.glob(f"*_{year}_dta_only"))

        for extract_dir in year_patterns:
            if not extract_dir.is_dir():
                continue

            # Recursively find all CSV and DTA files
            for ext in ("*.csv", "*.CSV", "*.dta", "*.DTA", "*.sav", "*.SAV"):
                for filepath in extract_dir.rglob(ext):
                    dest = year_dir / filepath.name
                    if not dest.exists():
                        print(f"  {filepath.name} -> {year}/")
                        shutil.copy2(filepath, dest)

        # Verify key files exist
        csv_files = list(year_dir.glob("*.csv")) + list(year_dir.glob("*.CSV"))
        dta_files = list(year_dir.glob("*.dta")) + list(year_dir.glob("*.DTA"))

        if csv_files or dta_files:
            print(f"  OK {year}: {len(csv_files)} CSVs, {len(dta_files)} DTAs")
        else:
            print(f"  FAIL {year}: NO DATA FILES FOUND -- check manually")

    return True


# ---------------------------------------------------------------------------
# Step 3: Verify expected files
# ---------------------------------------------------------------------------

def verify_enaho():
    """Check that the critical files exist for each year."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying expected files")
    print("=" * 60)

    all_ok = True
    for year in YEARS:
        year_dir = ENAHO_DIR / year
        if not year_dir.exists():
            print(f"  FAIL {year}: directory missing")
            all_ok = False
            continue

        # Look for education module (Module 300) -- could be various name patterns
        mod300_patterns = [
            f"Enaho01a-{year}-300*",
            f"enaho01a-{year}-300*",
            f"ENAHO01A-{year}-300*",
            f"enaho01a_{year}_300*",  # Some years use underscores
        ]

        mod200_patterns = [
            f"Enaho01-{year}-200*",   # Note: Module 200 is Enaho01, not Enaho01a
            f"enaho01-{year}-200*",
            f"ENAHO01-{year}-200*",
            f"Enaho01a-{year}-200*",  # Some years inconsistent
            f"enaho01a-{year}-200*",
        ]

        found_300 = False
        found_200 = False

        for pattern in mod300_patterns:
            matches = list(year_dir.glob(pattern))
            if matches:
                found_300 = True
                print(f"  OK {year} Module 300 (Education): {matches[0].name}")
                break

        for pattern in mod200_patterns:
            matches = list(year_dir.glob(pattern))
            if matches:
                found_200 = True
                print(f"  OK {year} Module 200 (Members):   {matches[0].name}")
                break

        if not found_300:
            print(f"  FAIL {year} Module 300 (Education): NOT FOUND")
            all_ok = False
        if not found_200:
            print(f"  FAIL {year} Module 200 (Members):   NOT FOUND")
            all_ok = False

    # Also list all files per year for debugging
    print("\n--- Full file listing ---")
    for year in YEARS:
        year_dir = ENAHO_DIR / year
        if year_dir.exists():
            files = sorted([f.name for f in year_dir.iterdir() if f.is_file()])
            print(f"\n  {year}/ ({len(files)} files):")
            for f in files[:15]:  # Show first 15
                print(f"    {f}")
            if len(files) > 15:
                print(f"    ... and {len(files) - 15} more")

    return all_ok


# ---------------------------------------------------------------------------
# Step 4: Download admin dropout rates
# ---------------------------------------------------------------------------

def download_admin_data():
    """Download district-level dropout rates from datosabiertos.gob.pe."""
    print("\n" + "=" * 60)
    print("STEP 4: Downloading admin dropout rates from datosabiertos")
    print("=" * 60)

    ADMIN_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in ADMIN_URLS.items():
        dest = ADMIN_DIR / f"{name}.csv"
        if dest.exists():
            print(f"  OK {name}.csv already exists, skipping")
            continue

        print(f"  Downloading {name}...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  OK {name}.csv ({len(resp.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"  FAIL {name}: {e}")
            print(f"    Download manually from: {url}")
            print(f"    Place in: {dest}")

    return True


# ---------------------------------------------------------------------------
# Step 5: Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    """Remove temporary download directory."""
    if TEMP_DIR.exists():
        response = input(f"\nRemove temp directory {TEMP_DIR}? (y/N): ").strip().lower()
        if response == "y":
            shutil.rmtree(TEMP_DIR)
            print("  Temp directory removed")
        else:
            print(f"  Temp files preserved at {TEMP_DIR}")
            print("  (Contains original ZIPs -- useful if you need to re-extract)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("""
==========================================================
  Alerta Escuela Audit -- Data Download Script

  Downloads ENAHO 2018-2024 (Modules 02, 03, 05, 34, 37)
  + district dropout rates from datosabiertos
==========================================================
    """)

    # Create directory structure
    for year in YEARS:
        (ENAHO_DIR / year).mkdir(parents=True, exist_ok=True)
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)

    # Check what already exists
    existing = sum(1 for y in YEARS if any((ENAHO_DIR / y).glob("*300*")))
    if existing == len(YEARS):
        print(f"All {len(YEARS)} years already have Module 300 files.")
        response = input("Re-download anyway? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping ENAHO download. Running verification only.\n")
            verify_enaho()
            download_admin_data()
            return

    # Download
    if download_enaho():
        reorganize_enaho()

    # Verify
    all_ok = verify_enaho()

    # Admin data
    download_admin_data()

    # Cleanup
    cleanup()

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL DATA READY. You can proceed with GSD Phase 1.")
    else:
        print("SOME FILES MISSING. Check the output above.")
        print("  You may need to download missing years manually from:")
        print("  https://proyectos.inei.gob.pe/microdatos/")
        print(f"  Place files in: {ENAHO_DIR}/{{YEAR}}/")
    print("=" * 60)

    # Print note about additional data sources
    print("""
NOTE: This script downloads ENAHO + admin dropout rates only.
You still need to manually obtain:

  1. Census 2017 district-level data -> data/raw/census/
     Source: INEI (https://censos2017.inei.gob.pe/redatam/)

  2. VIIRS Nighttime Lights (pre-aggregated) -> data/raw/nightlights/
     Source: Jiaxiong Yao's research site or Google Earth Engine

  3. Censo Escolar aggregates -> data/raw/escolar/
     Source: https://www.datosabiertos.gob.pe/dataset/censo-escolar

These are supplementary enrichment data (M1.4). You can start
M1.1 and M1.2 with just ENAHO data.
    """)


if __name__ == "__main__":
    main()
