#!/usr/bin/env python3
"""Match GEE VIIRS export (GADM names) to INEI UBIGEO codes.

Reads the GEE-exported CSV with GADM district names (NAME_1/NAME_2/NAME_3)
and the UBIGEO reference table, then performs name matching to produce
data/raw/nightlights/viirs_districts.csv with UBIGEO + mean_radiance.

Usage:
    uv run python scripts/match_gadm_ubigeo.py

Prerequisites:
    1. GEE export CSV at data/raw/nightlights/viirs_districts_gee.csv
    2. UBIGEO reference at /tmp/ubigeo_distrito.csv (or auto-downloaded)

Output:
    data/raw/nightlights/viirs_districts.csv
"""

import csv
import re
import sys
import unicodedata
from pathlib import Path

# GEE exports include .geo column with huge polygon WKT strings
csv.field_size_limit(sys.maxsize)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEE_CSV = PROJECT_ROOT / "data" / "raw" / "nightlights" / "viirs_districts_gee.csv"
UBIGEO_REF = Path("/tmp/ubigeo_distrito.csv")
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "nightlights" / "viirs_districts.csv"


def normalize(name: str) -> str:
    """Normalize a name for comparison: uppercase, strip accents, collapse whitespace."""
    # Uppercase
    name = name.upper().strip()
    # Remove accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def load_ubigeo_reference() -> dict[str, str]:
    """Load UBIGEO reference table. Returns {(dept, prov, dist): ubigeo}."""
    if not UBIGEO_REF.exists():
        # Try to download it
        import urllib.request

        url = "https://raw.githubusercontent.com/jmcastagnetto/ubigeo-peru-aumentado/main/ubigeo_distrito.csv"
        print(f"Downloading UBIGEO reference from {url}...")
        urllib.request.urlretrieve(url, UBIGEO_REF)

    lookup: dict[tuple[str, str, str], str] = {}
    with open(UBIGEO_REF, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                normalize(row["departamento"]),
                normalize(row["provincia"]),
                normalize(row["distrito"]),
            )
            lookup[key] = row["inei"]
    return lookup


def load_gee_export() -> list[dict]:
    """Load GEE VIIRS export CSV."""
    if not GEE_CSV.exists():
        print(f"ERROR: GEE export not found at {GEE_CSV}")
        print("  1. Run the GEE script (scripts/gee_viirs_peru.js)")
        print("  2. Download CSV from Google Drive")
        print(f"  3. Place at {GEE_CSV}")
        sys.exit(1)

    rows = []
    with open(GEE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} districts from GEE export")
    return rows


def match_districts(
    gee_rows: list[dict],
    ubigeo_lookup: dict[tuple[str, str, str], str],
) -> tuple[list[dict], list[dict]]:
    """Match GEE district names to UBIGEO codes.

    Returns (matched, unmatched) lists.
    """
    matched = []
    unmatched = []

    # Build reverse lookup for fuzzy matching
    ubigeo_by_dept_prov = {}
    for (dept, prov, dist), ubigeo in ubigeo_lookup.items():
        key = (dept, prov)
        if key not in ubigeo_by_dept_prov:
            ubigeo_by_dept_prov[key] = {}
        ubigeo_by_dept_prov[key][dist] = ubigeo

    for row in gee_rows:
        dept = normalize(row["NAME_1"])
        prov = normalize(row["NAME_2"])
        dist = normalize(row["NAME_3"])
        radiance = row.get("mean_radiance", row.get("mean", ""))

        # Try exact match
        key = (dept, prov, dist)
        if key in ubigeo_lookup:
            matched.append({
                "UBIGEO": ubigeo_lookup[key],
                "mean_radiance": radiance,
                "gadm_name": f"{row['NAME_1']}/{row['NAME_2']}/{row['NAME_3']}",
            })
            continue

        # Department-level name variations (GADM vs INEI naming)
        dept_aliases = {
            "CALLAO": "CALLAO",
            "PROV. CONST. DEL CALLAO": "CALLAO",
            "PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO",
            "LIMA METROPOLITANA": "LIMA",
            "LIMA PROVINCIAS": "LIMA",
            "LIMA REGION": "LIMA",
            "LIMA PROVINCE": "LIMA",
            "HUENUCO": "HUANUCO",
            "JUNIN": "JUNIN",
        }
        # Province-level aliases
        prov_aliases = {
            ("HUANUCO", "HUENUCO"): "HUANUCO",
            ("JUNIN", "CHUPACA"): "CHUPACA",
        }
        alt_dept = dept_aliases.get(dept, dept)
        alt_prov = prov_aliases.get((alt_dept, prov), prov)

        # Province-level remap (GADM uses old province names)
        prov_remap = {
            ("LORETO", "ALTO AMAZONAS", "BARRANCA"): ("LORETO", "DATEM DEL MARANON", "BARRANCA"),
            ("LORETO", "ALTO AMAZONAS", "CAHUAPANAS"): ("LORETO", "DATEM DEL MARANON", "CAHUAPANAS"),
            ("LORETO", "ALTO AMAZONAS", "MANSERICHE"): ("LORETO", "DATEM DEL MARANON", "MANSERICHE"),
            ("LORETO", "ALTO AMAZONAS", "MORONA"): ("LORETO", "DATEM DEL MARANON", "MORONA"),
            ("LORETO", "ALTO AMAZONAS", "PASTAZA"): ("LORETO", "DATEM DEL MARANON", "PASTAZA"),
        }
        remap_key = (alt_dept, alt_prov, dist)
        if remap_key in prov_remap:
            alt_dept, alt_prov, dist = prov_remap[remap_key]

        # District-level overrides for known mismatches
        dist_aliases = {
            "3 DE DICIEMBRE": "TRES DE DICIEMBRE",
            "MAGDALENA VIEJA": "MAGDALENA DEL MAR",
            "SAN FCO.DE ASIS DE YARUSYACAN": "SAN FRANCISCO DE ASIS DE YARUSYACAN",
            "SAN JUAN DE ISCOS": "SAN JUAN DE YSCOS",
            "CAPASO": "CAPAZO",
            "CALLARIA": "CALLERIA",
        }
        alt_dist = dist_aliases.get(dist, dist)

        # Try all alias combinations
        found = False
        for d in {dept, alt_dept}:
            for p in {prov, alt_prov}:
                for di in {dist, alt_dist}:
                    key = (d, p, di)
                    if key in ubigeo_lookup:
                        matched.append({
                            "UBIGEO": ubigeo_lookup[key],
                            "mean_radiance": radiance,
                            "gadm_name": f"{row['NAME_1']}/{row['NAME_2']}/{row['NAME_3']}",
                        })
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue

        # Try substring match within same dept+prov
        for d in {dept, alt_dept}:
            for p in {prov, alt_prov}:
                dp_key = (d, p)
                candidates = ubigeo_by_dept_prov.get(dp_key, {})
                if candidates:
                    for cand_dist, cand_ubigeo in candidates.items():
                        if dist in cand_dist or cand_dist in dist:
                            matched.append({
                                "UBIGEO": cand_ubigeo,
                                "mean_radiance": radiance,
                                "gadm_name": f"{row['NAME_1']}/{row['NAME_2']}/{row['NAME_3']}",
                            })
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if not found:
            unmatched.append({
                "NAME_1": row["NAME_1"],
                "NAME_2": row["NAME_2"],
                "NAME_3": row["NAME_3"],
                "mean_radiance": radiance,
            })

    return matched, unmatched


def write_output(matched: list[dict]) -> None:
    """Write matched results as UBIGEO,mean_radiance CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Check for duplicate UBIGEOs
    seen = {}
    deduped = []
    for row in matched:
        ubigeo = row["UBIGEO"]
        if ubigeo in seen:
            print(f"  WARNING: duplicate UBIGEO {ubigeo}, keeping first")
            continue
        seen[ubigeo] = True
        deduped.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["UBIGEO", "mean_radiance"])
        for row in sorted(deduped, key=lambda r: r["UBIGEO"]):
            writer.writerow([row["UBIGEO"], row["mean_radiance"]])

    print(f"\nWritten {len(deduped)} districts to {OUTPUT_CSV}")


def main():
    print("=" * 60)
    print("  GADM → UBIGEO Nightlights Matching")
    print("=" * 60)

    # Load reference data
    ubigeo_lookup = load_ubigeo_reference()
    print(f"UBIGEO reference: {len(ubigeo_lookup)} districts")

    # Load GEE export
    gee_rows = load_gee_export()

    # Match
    matched, unmatched = match_districts(gee_rows, ubigeo_lookup)

    print(f"\nResults:")
    print(f"  Matched:   {len(matched)} ({len(matched)/len(gee_rows)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched)} ({len(unmatched)/len(gee_rows)*100:.1f}%)")

    if unmatched:
        print(f"\n  Unmatched districts:")
        for row in unmatched[:20]:
            print(f"    {row['NAME_1']} / {row['NAME_2']} / {row['NAME_3']}")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")

    # Write output
    write_output(matched)

    # Coverage check
    target = 1839  # Expected districts
    coverage = len(matched) / target
    if coverage < 0.90:
        print(f"\n  WARNING: Coverage {coverage:.1%} below 90% target")
        print("  Review unmatched districts and fix name mappings")
    else:
        print(f"\n  Coverage: {coverage:.1%} of {target} expected districts")

    print("\nDone! Next step: re-run pipeline with real nightlights data")


if __name__ == "__main__":
    main()
