"""Constants for Streamflow Pipeline."""

PIPELINE_STEPS = ["download", "validate", "cama", "hii", "merge"]

TIME_RESOLUTIONS = ["daily", "monthly", "hourly"]

CAMA_RESOLUTIONS = ["glb_15min", "glb_06min", "glb_05min", "glb_03min"]

# All registered dataset source names
VALID_SOURCES = [
    "caravan_core", "grdc", "grdc_caravan", "camels_br", "camels_gb_v2",
    "camels_fr", "camels_ch", "camels_dk", "camels_nz", "camels_es",
    "camels_se", "camels_fi", "camels_ind", "camels_col", "camels_lux",
    "caravan_de", "caravan_il", "gsha", "gsim", "camelsh",
    "robin", "adhi", "cabra", "mlit", "r_arcticnet",
    "grdd", "grdc_rseg", "dai_trenberth", "china_river",
    "estreams", "ca_discharge", "bull", "hydroch",
    "lamah_ice", "lamah_ce", "thousand_mile_eye", "yellow_river_commission",
]

# Unit conversion factors to standard units
DISCHARGE_CONVERSIONS = {
    "m3/s": 1.0,
    "ft3/s": 0.0283168,
    "L/s": 0.001,
    # mm/d requires area -- handled in unit_converter.py
}

AREA_CONVERSIONS = {
    "km2": 1.0,
    "m2": 1e-6,
    "ha": 0.01,
}
