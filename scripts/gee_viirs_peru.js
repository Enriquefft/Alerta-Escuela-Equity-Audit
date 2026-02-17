/**
 * Google Earth Engine script: VIIRS Nighttime Lights for Peru districts
 *
 * Computes mean annual nighttime radiance per district (GADM Level 3)
 * using VIIRS DNB monthly composites.
 *
 * Instructions:
 *   1. Upload gadm41_PER_3.shp (+ .shx, .dbf, .prj) to GEE as an asset
 *      - Go to https://code.earthengine.google.com/
 *      - Assets tab > New > Shape files
 *      - Upload all 4 files from /tmp/gadm_peru/gadm41_PER_3.*
 *      - Wait for ingestion to complete (few minutes)
 *   2. Update GADM_ASSET below with your asset path
 *   3. Paste this entire script into the Code Editor
 *   4. Click "Run"
 *   5. In the Tasks tab, click "Run" next to "viirs_peru_districts"
 *   6. Export will go to your Google Drive
 *   7. Download the CSV from Drive and place at:
 *      data/raw/nightlights/viirs_districts_gee.csv
 *   8. Then run: uv run python scripts/match_gadm_ubigeo.py
 */

// ──────────────────────────────────────────────────────────────
// CONFIGURATION — UPDATE THIS PATH
// ──────────────────────────────────────────────────────────────
var GADM_ASSET = "projects/alerta-escuela/assets/gadm41_PER_3";

// Year to compute (matches ENAHO survey years 2018-2023)
var YEAR = 2022; // Representative year for model features

// ──────────────────────────────────────────────────────────────
// Load data
// ──────────────────────────────────────────────────────────────
var districts = ee.FeatureCollection(GADM_ASSET);

// VIIRS DNB Monthly Composites (stray-light corrected)
var viirs = ee
    .ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
    .select("avg_rad");

// Annual mean composite
var annualMean = viirs.mean().rename("mean_radiance");

// Clip negative values to zero (background noise)
annualMean = annualMean.max(0);

// ──────────────────────────────────────────────────────────────
// Reduce regions: compute mean radiance per district
// ──────────────────────────────────────────────────────────────
var districtStats = annualMean.reduceRegions({
    collection: districts,
    reducer: ee.Reducer.mean(),
    scale: 500, // VIIRS native ~500m resolution
    crs: "EPSG:4326",
});

// Select only the columns we need
var output = districtStats.select(
    ["GID_3", "NAME_1", "NAME_2", "NAME_3", "mean"],
    ["GID_3", "NAME_1", "NAME_2", "NAME_3", "mean_radiance"],
);

// ──────────────────────────────────────────────────────────────
// Export to Google Drive
// ──────────────────────────────────────────────────────────────
Export.table.toDrive({
    collection: output,
    description: "viirs_peru_districts",
    fileNamePrefix: "viirs_peru_districts",
    fileFormat: "CSV",
});

// Quick preview
print("Districts:", districts.size());
print("VIIRS months:", viirs.size());
print("Sample output (first 5):", output.limit(5));

// Map visualization
Map.centerObject(districts, 6);
Map.addLayer(
    annualMean,
    { min: 0, max: 50, palette: ["black", "blue", "yellow", "white"] },
    "VIIRS Mean Radiance " + YEAR,
);
