import logging
import threading

# Module-level cache for case-insensitive unit lookup
# Format: normalized_unit_lowercase -> (base_unit, conversion_func or None)
_UNIT_LOOKUP_CACHE = None
_UNIT_CACHE_LOCK = threading.Lock()

class UnitProcessing:
	def __init__(self, info):
		self.name = 'plotting'
		self.version = '0.1'
		self.release = '0.1'
		self.date = 'Mar 2023'
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
			'gc m-2 day-1': {  # Carbon flux unit (for GPP, NPP, NEE, etc.)
				'gc m-2 s-1': lambda x: x * 86400,  # Carbon-specific (explicit)
				'g c m-2 s-1': lambda x: x * 86400,  # Carbon-specific with space
				'g c m-2 day-1': lambda x: x ,  # Carbon-specific with day
				'g m-2 s-1': lambda x: x * 86400,   # Carbon-implicit (common in models)
				'mol m-2 s-1': lambda x: x * (86400 * 12.01),  # Molar carbon
				'mumolco2 m-2 s-1': lambda x: x * (12e-6 * 86400),  # CO2 flux
			},
			'mm day-1': {
				'kg m-2 s-1': lambda x: x * 86400,
				'mm s-1': lambda x: x * 86400,
				'mm hr-1': lambda x: x * 24,
				'mm h-1': lambda x: x * 24,
				'mm hour-1': lambda x: x * 24,
				'mm mon-1': lambda x: x / 30.44,
				'mm m-1': lambda x: x / 30.44,
				'mm month-1': lambda x: x / 30.44,
				'w m-2 heat': lambda x: x /28.4,
				'mm 3hour-1': lambda x: x * 8,
			},
			'w m-2': {
				'mj m-2 day-1': lambda x: x * 11.574074074074074,  # 1 / 0.0864
				'mj m-2 d-1': lambda x: x * 11.574074074074074,  # 1 / 0.0864
			},
			'unitless': {
				'percent': lambda x: x / 100,
				'percentage': lambda x: x / 100,
				'%': lambda x: x / 100,
				'g kg-1': lambda x: x / 1000,
				'fraction': lambda x: x,
				'm3 m-3': lambda x: x,
				'm2 m-2': lambda x: x,
				'g g-1': lambda x: x,
				'1': lambda x: x,  # Dimensionless (numeric representation)
				'0': lambda x: x,  # Sometimes used as placeholder for unitless
			},
			'k': {
				'c': lambda x: x + 273.15,
				'degc': lambda x: x + 273.15,
				'degreec': lambda x: x + 273.15,
				'degree c': lambda x: x + 273.15,
				'celsius': lambda x: x + 273.15,
				'f': lambda x: (x - 32) * 5 / 9 + 273.15,
				'degf': lambda x: (x - 32) * 5 / 9 + 273.15,
				'degreef': lambda x: (x - 32) * 5 / 9 + 273.15,
				'degree f': lambda x: (x - 32) * 5 / 9 + 273.15,
				'fahrenheit': lambda x: (x - 32) * 5 / 9 + 273.15,
			},
			'm3 s-1': {
				'm3 day-1': lambda x: x / 86400,
				'm3 d-1': lambda x: x / 86400,
				'l s-1': lambda x: x * 1000,
			},
			'mcm': {
				'm3': lambda x: x * 1e6,
				'km3': lambda x: x / 1000,
				'million cubic meters': lambda x: x,
			},
			'mm year-1': {
				'm year-1': lambda x: x / 1000,
				'cm year-1': lambda x: x / 10,
				'kg m-2': lambda x: x,
				'mm month-1': lambda x: x * 12,
				'mm mon-1': lambda x: x * 12,
				'mm day-1': lambda x: x * 365,
			},
			'm': {
				'cm': lambda x: x / 100,
				'mm': lambda x: x / 1000,
			},
			'km2': {
				'm2': lambda x: x / 1.e6,
			},
			'm s-1': {
				'km h-1': lambda x: x * 3.6,
			},
			't ha-1': {
				'kg ha-1': lambda x: x / 1000,
			},
			'kg c m-2': {
				'g c m-2': lambda x: x / 1000,
			},
			'kgc m-2': {
				'gc m-2': lambda x: x / 1000,
				'g c m-2': lambda x: x / 1000,
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
					logging.info(f'Unit lookup cache initialized with {len(_UNIT_LOOKUP_CACHE)} entries')
		
		logging.info(f'Converting {input_unit} to base unit...')
		
		# Normalize input unit to lowercase for comparison
		input_key = input_unit.lower().strip()
		
		# O(1) dictionary lookup - much faster than O(n) iteration
		if input_key in _UNIT_LOOKUP_CACHE:
			base_unit, conv_func = _UNIT_LOOKUP_CACHE[input_key]
			
			# If no conversion needed (input is already a base unit)
			if conv_func is None:
				logging.info(f'No conversion needed for {input_unit} -> {base_unit}')
				return data, base_unit
			
			# Apply conversion
			if data is None:
				logging.info(f'Unit mapping found (case-insensitive): {input_unit} -> {base_unit}')
				return None, base_unit
			else:
				converted_data = conv_func(data)
				logging.info(f'Successfully converted (case-insensitive): {input_unit} -> {base_unit}')
				return converted_data, base_unit
		
		# If no conversion is found
		logging.warning(f'No conversion found for {input_unit} (case-insensitive search). Using original data and unit.')
		return data, input_unit

	@staticmethod
	def process_unit(self, data, unit):
		"""
		Process unit conversion for a specific item.
		"""
		if hasattr(UnitProcessing, f'Unit_{self.item}'):
			return getattr(UnitProcessing, f'Unit_{self.item}')(self, data, unit)
		else:
			logging.error(f"Unit conversion for {self.item} is not supported!")
			raise ValueError(f"Unit conversion for {self.item} is not supported!")

	@staticmethod
	def check_units(input_units, target_units):
		"""
		Check if input units match target units.
		"""
		return sorted(input_units.lower().split()) == sorted(target_units.lower().split())
