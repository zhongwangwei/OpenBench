import logging
import threading


# Module-level cache for case-insensitive unit lookup
# Format: normalized_unit_lowercase -> (base_unit, conversion_func or None)
_UNIT_LOOKUP_CACHE = None
_UNIT_CACHE_LOCK = threading.Lock()


SECONDS_PER_DAY = 86400.0
LATENT_HEAT_VAPORIZATION_J_KG = 2.5e6


def _per_day_to_per_year(x):
    """Convert mm/day → mm/year using calendar-aware factor when possible.

    If `x` is an xarray DataArray with a time coordinate, multiply by
    the year length implied by each timestamp (365 or 366 for leap).
    Otherwise fall back to the Julian year (365.25), which averages
    leap-year drift out.
    """
    try:
        if hasattr(x, "time") and "time" in getattr(x, "coords", {}):
            years = x.time.dt.year
            # Days in year vector aligned with time axis
            is_leap = ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0)
            days = is_leap.astype("float64") + 365.0
            return x * days
    except Exception:
        pass
    return x * 365.25


def _per_month_to_per_day(x):
    """Convert mm/month → mm/day using calendar-aware days-in-month.

    Mirror of `_per_day_to_per_year`. The fixed 30.44 (mean Gregorian
    month length) used previously gave up to ±10% per-step error in
    months with 28/31 days. When `x` is an xarray DataArray with a
    time coordinate we use `x.time.dt.days_in_month`; otherwise fall
    back to 30.4375 (= 365.25/12) which averages leap drift.
    """
    try:
        if hasattr(x, "time") and "time" in getattr(x, "coords", {}):
            return x / x.time.dt.days_in_month
    except Exception:
        pass
    return x / 30.4375


class UnitProcessing:
    def __init__(self, info):
        self.name = "plotting"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Mar 2023"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)

    @staticmethod
    def convert_unit(data, input_unit):
        """
        Generic unit conversion method.
        Converts input unit to the base unit if possible.
        If data is None, only returns the converted unit without data conversion.

        Fully case-insensitive: All units are stored and compared in lowercase.
        Uses O(1) dictionary lookup for high performance.
        """
        # Define conversion factors in lowercase for case-insensitive matching
        # Format: base_unit_lowercase -> {input_unit_lowercase: conversion_func}
        conversion_factors = {
            "gc m-2 day-1": {  # Carbon flux unit (for GPP, NPP, NEE, etc.)
                "gc m-2 s-1": lambda x: x * 86400,  # Carbon-specific (explicit)
                "g c m-2 s-1": lambda x: x * 86400,  # Carbon-specific with space
                "g c m-2 day-1": lambda x: x,  # Carbon-specific with day
                "kg c m-2 s-1": lambda x: x * 1000 * 86400,
                "kgc m-2 s-1": lambda x: x * 1000 * 86400,
                "g m-2 s-1": lambda x: x * 86400,  # Carbon-implicit (common in models)
                "mol m-2 s-1": lambda x: x * (86400 * 12.01),  # Molar carbon
                "mumolco2 m-2 s-1": lambda x: x * (12e-6 * 86400),  # CO2 flux
                "umol/m2/s": lambda x: x * (12e-6 * 86400),  # micromol CO2/C flux
                "umol m-2 s-1": lambda x: x * (12e-6 * 86400),
                "umol m^-2 s^-1": lambda x: x * (12e-6 * 86400),
                "umolco2 m-2 s-1": lambda x: x * (12e-6 * 86400),
                "umol co2 m-2 s-1": lambda x: x * (12e-6 * 86400),
                "µmol/m2/s": lambda x: x * (12e-6 * 86400),
                "μmol/m2/s": lambda x: x * (12e-6 * 86400),
            },
            "mm": {
                # Water-equivalent depth/stock. Keep this separate from
                # rate units (mm day-1 / mm year-1) so SWE, soil water, and
                # other kg m-2 water-column state variables are not mislabeled
                # as annual fluxes.
                "kg m-2": lambda x: x,
                "kg/m2": lambda x: x,
                "kg m**-2": lambda x: x,
                # Equivalent-water-thickness depth (e.g. GRAiCE/GRACE TWSC in
                # centimetres). 1 cm = 10 mm. Matched on the full descriptive
                # string the catalogue uses so a bare "cm" stays a length
                # (mapped to metres below), not a water depth.
                "cm of equivalent water thickness": lambda x: x * 10,
                "cm of water": lambda x: x * 10,
                "cm water equivalent": lambda x: x * 10,
            },
            "mm day-1": {
                "kg m-2 s-1": lambda x: x * 86400,
                "kg/m2/s": lambda x: x * 86400,
                "mm s-1": lambda x: x * 86400,
                "mm h2o/s": lambda x: x * 86400,
                "mm h2o s-1": lambda x: x * 86400,
                "mm hr-1": lambda x: x * 24,
                "mm h-1": lambda x: x * 24,
                "mm hour-1": lambda x: x * 24,
                "mm mon-1": lambda x: _per_month_to_per_day(x),
                "mm month-1": lambda x: _per_month_to_per_day(x),
                "w m-2 heat": lambda x: x * SECONDS_PER_DAY / LATENT_HEAT_VAPORIZATION_J_KG,
                "mm 3hour-1": lambda x: x * 8,
                "mm 3h-1": lambda x: x * 8,
                "m hr-1": lambda x: x * 1000 * 24,
                # Daily runoff/flux depth expressed in metres of water
                # (e.g. ERA5-Land "ro"). 1 m = 1000 mm.
                "m day-1": lambda x: x * 1000,
                "m d-1": lambda x: x * 1000,
            },
            "w m-2": {
                "w/m2": lambda x: x,
                "watt/m2": lambda x: x,
                "watt m-2": lambda x: x,
                "w m**-2": lambda x: x,
                "mj m-2 day-1": lambda x: x * 11.574074074074074,  # 1 / 0.0864
                "mj m-2 d-1": lambda x: x * 11.574074074074074,  # 1 / 0.0864
            },
            "unitless": {
                "percent": lambda x: x / 100,
                "percentage": lambda x: x / 100,
                "%": lambda x: x / 100,
                "g kg-1": lambda x: x / 1000,
                "kg kg-1": lambda x: x,
                "kg/kg": lambda x: x,
                "fraction": lambda x: x,
                "m3 m-3": lambda x: x,
                "m2 m-2": lambda x: x,
                "g g-1": lambda x: x,
                "1": lambda x: x,  # Dimensionless (numeric representation)
                "0": lambda x: x,  # Sometimes used as placeholder for unitless
                "-": lambda x: x,  # Dimensionless ratio (e.g. albedo "f_sr/f_solarin")
                "none": lambda x: x,  # Sometimes written for unitless fields
            },
            "k": {
                "c": lambda x: x + 273.15,
                "degc": lambda x: x + 273.15,
                "degreec": lambda x: x + 273.15,
                "degree c": lambda x: x + 273.15,
                "degree_celsius": lambda x: x + 273.15,
                "celsius": lambda x: x + 273.15,
                "f": lambda x: (x - 32) * 5 / 9 + 273.15,
                "degf": lambda x: (x - 32) * 5 / 9 + 273.15,
                "degreef": lambda x: (x - 32) * 5 / 9 + 273.15,
                "degree f": lambda x: (x - 32) * 5 / 9 + 273.15,
                "fahrenheit": lambda x: (x - 32) * 5 / 9 + 273.15,
            },
            "m3 s-1": {
                "m3 day-1": lambda x: x / 86400,
                "m3 d-1": lambda x: x / 86400,
                "l s-1": lambda x: x / 1000,
            },
            "mcm": {
                "m3": lambda x: x / 1e6,
                "km3": lambda x: x * 1000,
                "million cubic meters": lambda x: x,
            },
            "mm year-1": {
                "m year-1": lambda x: x * 1000,
                "cm year-1": lambda x: x * 10,
                # Calendar-aware factor: use the actual day count of each
                # timestamp's year (365 or 366) when xarray supplies a
                # time coordinate; otherwise fall back to 365.25 (Julian
                # year) which averages out the leap-year drift instead of
                # the 0.27% systematic bias produced by a flat * 365.
                "mm day-1": lambda x: _per_day_to_per_year(x),
            },
            "m": {
                "cm": lambda x: x / 100,
                "mm": lambda x: x / 1000,
            },
            "pa": {
                "hpa": lambda x: x * 100,
                "mbar": lambda x: x * 100,
                "mb": lambda x: x * 100,
            },
            "km2": {
                "m2": lambda x: x / 1.0e6,
            },
            "m s-1": {
                "km h-1": lambda x: x / 3.6,
            },
            "t ha-1": {
                "kg ha-1": lambda x: x / 1000,
            },
            "kg c m-2": {
                "g c m-2": lambda x: x / 1000,
            },
            "kgc m-2": {
                "gc m-2": lambda x: x / 1000,
                "g c m-2": lambda x: x / 1000,
            },
        }

        # Build case-insensitive lookup dictionary (cached at module level)
        # This converts O(n) iteration to O(1) dictionary lookup
        # Use thread-safe initialization with double-checked locking
        global _UNIT_LOOKUP_CACHE
        if _UNIT_LOOKUP_CACHE is None:
            with _UNIT_CACHE_LOCK:
                # Double-check after acquiring lock
                if _UNIT_LOOKUP_CACHE is None:
                    temp_cache = {}

                    # All keys are already lowercase, just build the lookup
                    for base_unit, conversions in conversion_factors.items():
                        # Add base unit itself (None means no conversion needed)
                        if base_unit not in temp_cache:
                            temp_cache[base_unit] = (base_unit, None)

                        # Add all conversion units
                        for conv_unit, conv_func in conversions.items():
                            # Only add if not already present (prefer first match)
                            if conv_unit not in temp_cache:
                                temp_cache[conv_unit] = (base_unit, conv_func)

                    # Atomic assignment after cache is fully built
                    _UNIT_LOOKUP_CACHE = temp_cache
                    logging.info(f"Unit lookup cache initialized with {len(_UNIT_LOOKUP_CACHE)} entries")

        logging.info(f"Converting {input_unit} to base unit...")

        # Normalize input unit to lowercase for comparison
        input_key = input_unit.lower().strip()

        # O(1) dictionary lookup - much faster than O(n) iteration
        if input_key in _UNIT_LOOKUP_CACHE:
            base_unit, conv_func = _UNIT_LOOKUP_CACHE[input_key]

            # If no conversion needed (input is already a base unit)
            if conv_func is None:
                logging.info(f"No conversion needed for {input_unit} -> {base_unit}")
                return data, base_unit

            # Apply conversion
            if data is None:
                logging.info(f"Unit mapping found (case-insensitive): {input_unit} -> {base_unit}")
                return None, base_unit
            else:
                converted_data = conv_func(data)
                logging.info(f"Successfully converted (case-insensitive): {input_unit} -> {base_unit}")
                return converted_data, base_unit

        # If no conversion is found
        logging.warning(
            f"No conversion found for {input_unit} (case-insensitive search). Using original data and unit."
        )
        return data, input_unit

    def process_unit(self, data, unit):
        """
        Process unit conversion for a specific item.

        Note: this used to be decorated `@staticmethod` but kept `self` in
        its signature, which made it un-callable both as a method
        (TypeError on extra arg) and as a function (no `item` available).
        Removed the decorator so it works as a normal instance method.
        """
        if hasattr(UnitProcessing, f"Unit_{self.item}"):
            return getattr(UnitProcessing, f"Unit_{self.item}")(self, data, unit)
        else:
            logging.error(f"Unit conversion for {self.item} is not supported!")
            raise ValueError(f"Unit conversion for {self.item} is not supported!")

    @staticmethod
    def check_units(input_units, target_units):
        """
        Check if input units match target units.
        """
        return sorted(input_units.lower().split()) == sorted(target_units.lower().split())
