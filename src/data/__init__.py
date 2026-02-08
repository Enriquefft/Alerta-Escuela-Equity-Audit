"""Data loading and merging modules for the Alerta Escuela Equity Audit.

Provides loaders for ENAHO microdata, administrative dropout rates,
Census 2017 district indicators, VIIRS nightlights, and a merge
pipeline that combines all sources.

Usage::

    from data.enaho import load_enaho_year, load_all_years
    from data.admin import load_admin_dropout_rates
    from data.census import load_census_2017
    from data.nightlights import load_viirs_nightlights
    from data.merge import merge_spatial_data
"""

from data.enaho import (
    ENAHOResult,
    PooledENAHOResult,
    load_enaho_year,
    load_all_years,
)
from data.admin import AdminResult, load_admin_dropout_rates
from data.census import CensusResult, load_census_2017
from data.nightlights import NightlightsResult, load_viirs_nightlights
from data.merge import MergeResult, merge_spatial_data, validate_merge_pipeline

__all__ = [
    # ENAHO
    "ENAHOResult",
    "PooledENAHOResult",
    "load_enaho_year",
    "load_all_years",
    # Admin
    "AdminResult",
    "load_admin_dropout_rates",
    # Census
    "CensusResult",
    "load_census_2017",
    # Nightlights
    "NightlightsResult",
    "load_viirs_nightlights",
    # Merge
    "MergeResult",
    "merge_spatial_data",
    "validate_merge_pipeline",
]
