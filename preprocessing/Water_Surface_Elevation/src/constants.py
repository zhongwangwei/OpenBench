"""WSE Pipeline constants"""

# CaMa resolutions
RESOLUTIONS = ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min']

# Data sources
VALID_SOURCES = ['hydrosat', 'hydroweb', 'cgls', 'icesat']

# Pipeline steps
PIPELINE_STEPS = ['download', 'validate', 'cama', 'reserved', 'merge']

# Data completeness thresholds
COMPLETENESS_THRESHOLDS = {
    'hydrosat': 2000,
    'hydroweb': 30000,
    'cgls': 10000,
    'icesat': 15000,
}
