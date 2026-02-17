#!/usr/bin/env python3
"""Process Peru Census 2017 REDATAM CSVs into district-level indicators.

Reads the CSVs produced by open-redatam from the Census 2017 REDATAM
database, joins hierarchical tables to construct UBIGEO codes, then
aggregates person- and household-level microdata to district-level
indicators needed for the equity audit model.

Usage:
    uv run python scripts/process_census_redatam.py

Prerequisites:
    1. Census ZIP extracted and converted to CSV by open-redatam at
       /tmp/census_2017_csv/
    2. UBIGEO reference at /tmp/ubigeo_distrito.csv

Output:
    data/raw/census/census_2017_districts.csv

Census variables used:
    PERSONA.C5P11 - Mother tongue (1-9,15-45 = indigenous)
    PERSONA.C5P12 - Literacy (1 = can read/write)
    VIVIENDA.C2P11 - Electric lighting (1 = has electricity)
    VIVIENDA.C2P7 - Water service daily (1 = yes all days)

District indicators produced:
    poverty_rate - from UBIGEO reference (INEI 2018 poverty map)
    indigenous_lang_pct - % with indigenous mother tongue
    water_access_pct - % dwellings with daily water service
    electricity_pct - % dwellings with electric lighting
    literacy_rate - % population 3+ who can read/write
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "census" / "census_2017_districts.csv"
CSV_DIR = Path("/tmp/census_2017_csv")
UBIGEO_REF = Path("/tmp/ubigeo_distrito.csv")

# Indigenous mother tongue codes in C5P11
# 1-9 and 15-45 are indigenous languages
# 10 = Castellano, 11 = Portuguese, 12 = Other foreign
# 13 = Sign language, 14 = deaf/mute, 46 = don't know
INDIGENOUS_CODES = set(range(1, 10)) | set(range(15, 46))
VALID_LANG_CODES = INDIGENOUS_CODES | {10, 11, 12}  # Include non-indigenous for denominator


def build_ubigeo_lookup() -> dict[int, str]:
    """Build mapping from DISTRITO_REF_ID to 6-digit UBIGEO.

    Joins DEPARTAM (CCDD) → PROVINCI (CCPP) → DISTRITO (CCDI)
    to construct the full UBIGEO code for each district.
    """
    print("Building UBIGEO lookup from DEPARTAM→PROVINCI→DISTRITO...")

    # DEPARTAM: REF_ID → CCDD
    dept_ccdd: dict[int, str] = {}
    with open(CSV_DIR / "DEPARTAM.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            dept_ccdd[int(row["DEPARTAM_REF_ID"])] = row["CCDD"].strip().zfill(2)

    # PROVINCI: REF_ID → (DEPARTAM_REF_ID, CCPP)
    prov_data: dict[int, tuple[int, str]] = {}
    with open(CSV_DIR / "PROVINCI.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            prov_data[int(row["PROVINCI_REF_ID"])] = (
                int(row["DEPARTAM_REF_ID"]),
                row["CCPP"].strip().zfill(2),
            )

    # DISTRITO: REF_ID → (PROVINCI_REF_ID, CCDI)
    dist_data: dict[int, tuple[int, str]] = {}
    with open(CSV_DIR / "DISTRITO.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            dist_data[int(row["DISTRITO_REF_ID"])] = (
                int(row["PROVINCI_REF_ID"]),
                row["CCDI"].strip().zfill(2),
            )

    # Build full UBIGEO: CCDD + CCPP + CCDI
    lookup: dict[int, str] = {}
    for dist_id, (prov_id, ccdi) in dist_data.items():
        dept_id, ccpp = prov_data[prov_id]
        ccdd = dept_ccdd[dept_id]
        ubigeo = f"{ccdd}{ccpp}{ccdi}"
        lookup[dist_id] = ubigeo

    print(f"  {len(lookup)} districts mapped")
    return lookup


def build_vivienda_to_distrito() -> dict[int, int]:
    """Build mapping from VIVIENDA_REF_ID to DISTRITO_REF_ID."""
    print("Building VIVIENDA→DISTRITO lookup...")
    lookup: dict[int, int] = {}

    with open(CSV_DIR / "VIVIENDA.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            lookup[int(row["VIVIENDA_REF_ID"])] = int(row["DISTRITO_REF_ID"])

    print(f"  {len(lookup)} viviendas mapped")
    return lookup


def build_hogar_to_vivienda() -> dict[int, int]:
    """Build mapping from HOGAR_REF_ID to VIVIENDA_REF_ID."""
    print("Building HOGAR→VIVIENDA lookup...")
    lookup: dict[int, int] = {}

    with open(CSV_DIR / "HOGAR.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            lookup[int(row["HOGAR_REF_ID"])] = int(row["VIVIENDA_REF_ID"])

    print(f"  {len(lookup)} hogares mapped")
    return lookup


def process_vivienda(
    dist_lookup: dict[int, str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Aggregate VIVIENDA-level indicators to district level.

    Returns (water_data, electricity_data) dicts keyed by UBIGEO.
    """
    print("\nProcessing VIVIENDA data (water + electricity)...")
    water: dict[str, dict] = defaultdict(lambda: {"total": 0, "has": 0})
    elec: dict[str, dict] = defaultdict(lambda: {"total": 0, "has": 0})

    count = 0
    with open(CSV_DIR / "VIVIENDA.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            count += 1
            dist_id = int(row["DISTRITO_REF_ID"])
            ubigeo = dist_lookup.get(dist_id)
            if not ubigeo:
                continue

            # Only count occupied dwellings (C2P2 == 1 = occupied with people present)
            c2p2 = row.get("C2P2", "0").strip()
            if c2p2 != "1":
                continue

            # Water: C2P7 == 1 means daily water service
            c2p7 = row.get("C2P7", "0").strip()
            if c2p7 in ("1", "2"):  # 1=daily, 2=not daily but has service
                water[ubigeo]["total"] += 1
                if c2p7 == "1":
                    water[ubigeo]["has"] += 1
            elif c2p7 != "0":  # 0 = not applicable
                water[ubigeo]["total"] += 1

            # Electricity: C2P11 == 1 means has electric lighting
            c2p11 = row.get("C2P11", "0").strip()
            if c2p11 in ("1", "2"):
                elec[ubigeo]["total"] += 1
                if c2p11 == "1":
                    elec[ubigeo]["has"] += 1
            elif c2p11 != "0":
                elec[ubigeo]["total"] += 1

    print(f"  Processed {count:,} viviendas")
    print(f"  Water data: {len(water)} districts")
    print(f"  Electricity data: {len(elec)} districts")

    return dict(water), dict(elec)


def process_persona(
    dist_lookup: dict[int, str],
    viv_lookup: dict[int, int],
    hog_lookup: dict[int, int],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Aggregate PERSONA-level indicators to district level.

    Returns (language_data, literacy_data) dicts keyed by UBIGEO.

    Because PERSONA.csv is ~5GB, we process it in streaming fashion.
    """
    print("\nProcessing PERSONA data (language + literacy)...")
    print("  (This may take a few minutes for ~30M records)")

    lang: dict[str, dict] = defaultdict(lambda: {"total": 0, "indigenous": 0})
    lit: dict[str, dict] = defaultdict(lambda: {"total": 0, "literate": 0})

    count = 0
    skipped = 0

    with open(CSV_DIR / "PERSONA.csv", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            count += 1
            if count % 5_000_000 == 0:
                print(f"    ... {count:,} records processed")

            # Navigate: PERSONA → HOGAR → VIVIENDA → DISTRITO → UBIGEO
            hogar_id = int(row["HOGAR_REF_ID"])
            viv_id = hog_lookup.get(hogar_id)
            if viv_id is None:
                skipped += 1
                continue
            dist_id = viv_lookup.get(viv_id)
            if dist_id is None:
                skipped += 1
                continue
            ubigeo = dist_lookup.get(dist_id)
            if not ubigeo:
                skipped += 1
                continue

            # Language: C5P11 (mother tongue)
            c5p11 = row.get("C5P11", "0").strip()
            try:
                lang_code = int(c5p11)
                if lang_code in VALID_LANG_CODES:
                    lang[ubigeo]["total"] += 1
                    if lang_code in INDIGENOUS_CODES:
                        lang[ubigeo]["indigenous"] += 1
            except (ValueError, TypeError):
                pass

            # Literacy: C5P12 (can read/write, for population 3+)
            c5p12 = row.get("C5P12", "0").strip()
            try:
                lit_code = int(c5p12)
                if lit_code in (1, 2):  # 1=literate, 2=illiterate
                    lit[ubigeo]["total"] += 1
                    if lit_code == 1:
                        lit[ubigeo]["literate"] += 1
            except (ValueError, TypeError):
                pass

    print(f"  Processed {count:,} personas ({skipped:,} skipped)")
    print(f"  Language data: {len(lang)} districts")
    print(f"  Literacy data: {len(lit)} districts")

    return dict(lang), dict(lit)


def load_poverty_from_ubigeo_ref() -> dict[str, float]:
    """Load poverty rates from UBIGEO reference table (real INEI data)."""
    if not UBIGEO_REF.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/jmcastagnetto/ubigeo-peru-aumentado/main/ubigeo_distrito.csv"
        print(f"Downloading UBIGEO reference...")
        urllib.request.urlretrieve(url, UBIGEO_REF)

    poverty = {}
    with open(UBIGEO_REF, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ubigeo = row["inei"]
            pct = row.get("pct_pobreza_total", "")
            if pct and pct != "NA":
                try:
                    poverty[ubigeo] = round(float(pct), 2)
                except ValueError:
                    pass
    print(f"Loaded poverty rates for {len(poverty)} districts from UBIGEO reference")
    return poverty


def merge_and_write(
    lang_data: dict[str, dict],
    lit_data: dict[str, dict],
    water_data: dict[str, dict],
    elec_data: dict[str, dict],
    poverty_data: dict[str, float],
    dist_lookup: dict[int, str],
):
    """Merge all indicators and write output CSV."""
    # Use all UBIGEOs from district lookup as canonical set
    all_ubigeos = set(dist_lookup.values())

    print(f"\nMerging {len(all_ubigeos)} districts...")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ubigeo in sorted(all_ubigeos):
        if len(ubigeo) != 6:
            continue

        # Indigenous language percentage
        lang = lang_data.get(ubigeo, {})
        indigenous_pct = ""
        if lang.get("total", 0) > 0:
            indigenous_pct = round(lang["indigenous"] / lang["total"] * 100, 2)

        # Literacy rate
        lit = lit_data.get(ubigeo, {})
        literacy = ""
        if lit.get("total", 0) > 0:
            literacy = round(lit["literate"] / lit["total"] * 100, 2)

        # Water access
        water = water_data.get(ubigeo, {})
        water_pct = ""
        if water.get("total", 0) > 0:
            water_pct = round(water["has"] / water["total"] * 100, 2)

        # Electricity
        elec = elec_data.get(ubigeo, {})
        elec_pct = ""
        if elec.get("total", 0) > 0:
            elec_pct = round(elec["has"] / elec["total"] * 100, 2)

        # Poverty
        poverty = poverty_data.get(ubigeo, "")

        rows.append({
            "UBIGEO": ubigeo,
            "poverty_rate": poverty,
            "indigenous_lang_pct": indigenous_pct,
            "water_access_pct": water_pct,
            "electricity_pct": elec_pct,
            "literacy_rate": literacy,
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "UBIGEO", "poverty_rate", "indigenous_lang_pct",
            "water_access_pct", "electricity_pct", "literacy_rate",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Report coverage
    n = len(rows)
    for col in ["poverty_rate", "indigenous_lang_pct", "water_access_pct", "electricity_pct", "literacy_rate"]:
        filled = sum(1 for r in rows if r[col] != "")
        print(f"  {col}: {filled}/{n} ({filled/n*100:.1f}%)")

    print(f"\nWritten {n} districts to {OUTPUT_CSV}")

    # Sanity checks
    if lang_data:
        # Total indigenous percentage nationally
        total_indig = sum(d["indigenous"] for d in lang_data.values())
        total_pop = sum(d["total"] for d in lang_data.values())
        print(f"\n  National indigenous language: {total_indig/total_pop*100:.1f}%")
    if lit_data:
        total_lit = sum(d["literate"] for d in lit_data.values())
        total_pop = sum(d["total"] for d in lit_data.values())
        print(f"  National literacy rate: {total_lit/total_pop*100:.1f}%")
    if elec_data:
        total_elec = sum(d["has"] for d in elec_data.values())
        total_viv = sum(d["total"] for d in elec_data.values())
        print(f"  National electricity: {total_elec/total_viv*100:.1f}%")


def main():
    print("=" * 60)
    print("  Census 2017 REDATAM → District-Level Indicators")
    print("=" * 60)

    # Check prerequisites
    if not CSV_DIR.exists():
        print(f"ERROR: Census CSV directory not found at {CSV_DIR}")
        print("Run open-redatam first:")
        print(f"  /tmp/open-redatam-extract/usr/local/bin/redatam "
              "/tmp/census_2017_extract/BaseD/BaseR/CPVPER2017D.dic "
              f"{CSV_DIR}/")
        sys.exit(1)

    # Step 1: Build hierarchical lookups
    dist_lookup = build_ubigeo_lookup()
    viv_lookup = build_vivienda_to_distrito()
    hog_lookup = build_hogar_to_vivienda()

    # Step 2: Process VIVIENDA (water + electricity)
    water_data, elec_data = process_vivienda(dist_lookup)

    # Step 3: Process PERSONA (language + literacy)
    lang_data, lit_data = process_persona(dist_lookup, viv_lookup, hog_lookup)

    # Step 4: Load poverty from UBIGEO reference
    poverty_data = load_poverty_from_ubigeo_ref()

    # Step 5: Merge and write
    merge_and_write(lang_data, lit_data, water_data, elec_data, poverty_data, dist_lookup)

    print("\nDone!")


if __name__ == "__main__":
    main()
