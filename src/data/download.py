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
    ├── admin/       # District-level dropout rates from datosabiertos
    ├── census/      # Census 2017 district-level indicators
    └── nightlights/ # VIIRS district-level nighttime radiance
"""

import os
import subprocess
import sys
import shutil
import glob
import requests
import zipfile
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
CENSUS_DIR = DATA_RAW / "census"
NIGHTLIGHTS_DIR = DATA_RAW / "nightlights"
TEMP_DIR = PROJECT_ROOT / ".tmp_enaho_download"

# Census 2017 REDATAM database (647 MB ZIP from CELADE/ECLAC)
CENSUS_REDATAM_URL = "https://redatam.org/cdr/descargas/censos/poblacion/CP2017PER.zip"

# GADM Peru Level 3 shapefile (needed for GEE nightlights extraction)
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_PER_shp.zip"

# UBIGEO reference table (for GADM→UBIGEO name matching and poverty data)
UBIGEO_REF_URL = "https://raw.githubusercontent.com/jmcastagnetto/ubigeo-peru-aumentado/main/ubigeo_distrito.csv"

# datosabiertos.gob.pe URLs for district dropout rates
# Real data from www.datosabiertos.gob.pe (note: www. prefix required)
ADMIN_URLS = {
    "admin_dropout_primaria": "https://www.datosabiertos.gob.pe/sites/default/files/Educacion_Primaria.csv",
    "admin_dropout_secundaria": "https://www.datosabiertos.gob.pe/sites/default/files/Educacion_Secundaria.csv",
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
        # enahodata creates: modulo_XX_YYYY/ (e.g., modulo_02_2023/)
        year_patterns = list(source_base.glob(f"*_{year}"))
        year_patterns += list(source_base.glob(f"*_{year}_extract"))
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
# Step 5: Download Census 2017 REDATAM data
# ---------------------------------------------------------------------------

def download_census_data():
    """Download Census 2017 REDATAM ZIP and process to district-level CSV.

    Downloads the Census 2017 microdata from redatam.org, extracts to CSV
    using open-redatam, then aggregates to district-level indicators using
    scripts/process_census_redatam.py.

    If census CSV already exists, skips download.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Census 2017 district-level data")
    print("=" * 60)

    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    census_csv = CENSUS_DIR / "census_2017_districts.csv"

    if census_csv.exists():
        print(f"  OK census_2017_districts.csv already exists, skipping")
        return True

    # Download Census REDATAM ZIP
    census_zip = Path("/tmp/CP2017PER.zip")
    if not census_zip.exists():
        print(f"  Downloading Census 2017 REDATAM ({CENSUS_REDATAM_URL})...")
        print("  (647 MB -- this will take a few minutes)")
        try:
            resp = requests.get(CENSUS_REDATAM_URL, timeout=600, stream=True)
            resp.raise_for_status()
            with open(census_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  OK Downloaded ({census_zip.stat().st_size // (1024*1024)} MB)")
        except requests.RequestException as e:
            print(f"  FAIL Census download: {e}")
            print(f"    Download manually from: {CENSUS_REDATAM_URL}")
            print(f"    Save to: {census_zip}")
            return False
    else:
        print(f"  Census ZIP already at {census_zip}")

    # Process using the REDATAM script
    script = PROJECT_ROOT / "scripts" / "process_census_redatam.py"
    if script.exists():
        print("  Processing Census REDATAM data...")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )
        if result.returncode != 0:
            print("  FAIL: Census processing script failed")
            return False
    else:
        print(f"  WARNING: {script} not found")
        print("  Run: uv run python scripts/process_census_redatam.py")
        return False

    return census_csv.exists()


# ---------------------------------------------------------------------------
# Step 6: Download GADM shapefile + UBIGEO reference for nightlights
# ---------------------------------------------------------------------------

def download_nightlights_prereqs():
    """Download GADM shapefile and UBIGEO reference needed for nightlights.

    Nightlights extraction requires:
    1. GADM Peru Level 3 shapefile (uploaded to Google Earth Engine)
    2. UBIGEO reference table (for GADM name → UBIGEO matching)
    3. User runs GEE script manually (scripts/gee_viirs_peru.js)
    4. User places GEE CSV export as data/raw/nightlights/viirs_districts_gee.csv
    5. scripts/match_gadm_ubigeo.py converts to viirs_districts.csv
    """
    print("\n" + "=" * 60)
    print("STEP 6: Nightlights data prerequisites")
    print("=" * 60)

    NIGHTLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    nightlights_csv = NIGHTLIGHTS_DIR / "viirs_districts.csv"

    if nightlights_csv.exists():
        print(f"  OK viirs_districts.csv already exists, skipping")
        return True

    # Download GADM shapefile
    gadm_zip = Path("/tmp/gadm41_PER_shp.zip")
    if not gadm_zip.exists():
        print(f"  Downloading GADM Peru shapefile...")
        try:
            resp = requests.get(GADM_URL, timeout=120)
            resp.raise_for_status()
            gadm_zip.write_bytes(resp.content)
            print(f"  OK GADM shapefile ({len(resp.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"  FAIL GADM download: {e}")
    else:
        print(f"  GADM shapefile already at {gadm_zip}")

    # Extract GADM
    gadm_dir = Path("/tmp/gadm_peru")
    if not gadm_dir.exists() and gadm_zip.exists():
        gadm_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(gadm_zip, "r") as zf:
            zf.extractall(gadm_dir)
        print(f"  Extracted to {gadm_dir}")

    # Download UBIGEO reference
    ubigeo_csv = Path("/tmp/ubigeo_distrito.csv")
    if not ubigeo_csv.exists():
        print(f"  Downloading UBIGEO reference table...")
        try:
            resp = requests.get(UBIGEO_REF_URL, timeout=30)
            resp.raise_for_status()
            ubigeo_csv.write_bytes(resp.content)
            print(f"  OK UBIGEO reference ({len(resp.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"  FAIL UBIGEO download: {e}")
    else:
        print(f"  UBIGEO reference already at {ubigeo_csv}")

    # Check for GEE export
    gee_csv = NIGHTLIGHTS_DIR / "viirs_districts_gee.csv"
    if gee_csv.exists():
        # Run matching script
        script = PROJECT_ROOT / "scripts" / "match_gadm_ubigeo.py"
        if script.exists():
            print("  Running GADM→UBIGEO matching...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(PROJECT_ROOT),
                timeout=60,
            )
            if result.returncode == 0:
                return nightlights_csv.exists()
        return False

    print("""
  MANUAL STEPS REQUIRED for nightlights data:

  1. Upload /tmp/gadm_peru/gadm41_PER_3.* to Google Earth Engine
     (Assets > New > Shape files)

  2. Paste scripts/gee_viirs_peru.js into GEE Code Editor
     Update GADM_ASSET path, then click Run

  3. In Tasks tab, click Run next to 'viirs_peru_districts'

  4. Download CSV from Google Drive and place at:
     data/raw/nightlights/viirs_districts_gee.csv

  5. Run: uv run python scripts/match_gadm_ubigeo.py
    """)

    return False


# ---------------------------------------------------------------------------
# Step 7: Cleanup
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
  + Census 2017 district-level indicators
  + VIIRS nightlights prerequisites
==========================================================
    """)

    # Create directory structure
    for year in YEARS:
        (ENAHO_DIR / year).mkdir(parents=True, exist_ok=True)
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    NIGHTLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Census data
    download_census_data()

    # Nightlights prerequisites
    download_nightlights_prereqs()

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

    # Print status of supplementary data
    census_ok = (CENSUS_DIR / "census_2017_districts.csv").exists()
    nightlights_ok = (NIGHTLIGHTS_DIR / "viirs_districts.csv").exists()
    print(f"\n  Census 2017:  {'OK' if census_ok else 'MISSING'}")
    print(f"  Nightlights:  {'OK' if nightlights_ok else 'MISSING (manual GEE step needed)'}")
    if not nightlights_ok:
        print("  See instructions above for Google Earth Engine workflow.")


if __name__ == "__main__":
    main()
