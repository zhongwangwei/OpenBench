# Water Surface Elevation Pipeline Design

**Date**: 2026-01-27
**Status**: Approved
**Author**: Claude + User

## Overview

A 4-stage pipeline for processing satellite altimetry water surface elevation (WSE) data from multiple sources (HydroWeb, CGLS, ICESat, HydroSat) and allocating virtual stations to CaMa-Flood grid cells.

## Requirements

1. **4-stage pipeline** similar to Streamflow preprocessing
2. **Self-calculate EGM08/EGM96** - do not use metadata values from data sources
3. **Multi-source support** - HydroWeb, CGLS, ICESat, HydroSat (TODO)
4. **YAML configuration system** - global paths + dataset-specific configs
5. **Multi-resolution CaMa allocation** - glb_01min, glb_03min, glb_05min, glb_06min, glb_15min
6. **NC merge interface** - placeholder for future implementation

## Data Paths

| Data | Path |
|------|------|
| EGM Geoid | `/Volumes/Data01/AltiMaPpy-data/egm-geoids` |
| CaMa Maps | `/Volumes/Data01/2025` |
| HydroWeb | `/Volumes/Data01/Hydroweb` |
| CGLS | `/Volumes/Data01/Altimetry/CGLS/river` |
| ICESat | `/Volumes/Data01/Altimetry/ICESat_GLA14/txt_water` |
| HydroSat | TODO - needs web scraping |

## Project Structure

```
preprocessing/Water_Surface_Elevation/
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── pipeline.py                # Pipeline controller
│   │
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── step1_validate.py      # Validation + EGM calculation
│   │   ├── step2_cama.py          # CaMa allocation (5 resolutions)
│   │   ├── step3_reserved.py      # Reserved for future extensions
│   │   └── step4_merge.py         # NC merge interface (not implemented)
│   │
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── base_reader.py         # Abstract base class
│   │   ├── hydroweb_reader.py     # HydroWeb text format
│   │   ├── cgls_reader.py         # CGLS JSON format
│   │   ├── icesat_reader.py       # ICESat text format
│   │   └── hydrosat_reader.py     # HydroSat format (TODO)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── geoid_calculator.py    # EGM08/EGM96 calculation (existing)
│   │   ├── cama_allocator.py      # CaMa allocator (based on AllocateVS)
│   │   └── station_list.py        # Station list management
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py       # YAML config loader
│       ├── checkpoint.py          # Resume support
│       └── logger.py              # Logging system
│
├── config/
│   ├── global_paths.yaml          # Global path configuration
│   └── validation_rules.yaml      # Validation rules
│
├── templates/
│   └── dataset_config.yaml        # Dataset config template
│
└── geoid_data/                    # Symlink to EGM data
    └── -> /Volumes/Data01/AltiMaPpy-data/egm-geoids
```

## Configuration

### global_paths.yaml

```yaml
geoid_data:
  root: /Volumes/Data01/AltiMaPpy-data/egm-geoids
  egm96_model: egm96-5
  egm2008_model: egm2008-1

cama_data:
  root: /Volumes/Data01/2025
  resolutions:
    - glb_01min
    - glb_03min
    - glb_05min
    - glb_06min
    - glb_15min
  highres_tag: 1min

data_sources:
  hydroweb: /Volumes/Data01/Hydroweb
  cgls: /Volumes/Data01/Altimetry/CGLS/river
  icesat: /Volumes/Data01/Altimetry/ICESat_GLA14/txt_water
  hydrosat: null  # TODO

output:
  root: ./output
  station_list: altimetry_{source}_{date}.txt
  netcdf: OpenBench_WSE_{source}.nc
```

### dataset_config.yaml (template)

```yaml
dataset:
  name: "HydroWeb_2024"
  source: "hydroweb"
  version: "1.0"
  description: "HydroWeb river altimetry data"

processing:
  calculate_egm: true
  egm96_model: egm96-5
  egm2008_model: egm2008-1
  cama_resolutions:
    - glb_01min
    - glb_03min
    - glb_05min
    - glb_06min
    - glb_15min

filters:
  min_observations: 10
  start_date: null
  end_date: null
  bbox: null

output:
  format: txt
  include_timeseries: false
```

## Pipeline Stages

### Step 1: Validation + EGM Calculation

1. Scan data source directory using appropriate Reader
2. Parse station metadata (lon, lat, elevation, etc.)
3. Validate data completeness:
   - Coordinate range check (-180~180, -90~90)
   - Minimum observation count check
   - Duplicate station detection
4. Calculate EGM08/EGM96 using GeoidCalculator
   - Ignore metadata EGM values
   - Query geoid model by lon/lat

**Output**: `List[StationMetadata]` with calculated EGM values

### Step 2: CaMa Station Allocation

For each resolution (01min, 03min, 05min, 06min, 15min):
  For each 10°x10° tile:
    1. Load CaMa map data (uparea, nextxy, elevtn, etc.)
    2. Load high-resolution data (1min subdirectory)
    3. Execute AllocateVS algorithm for stations in tile:
       - Find nearest river centerline
       - Calculate kx1, ky1, kx2, ky2
       - Calculate dist1, dist2, rivwth
       - Allocate to CaMa grid (iXX, iYY)

**Output**: Each station with allocation results for all 5 resolutions

### Step 3: Reserved

Placeholder for future extensions:
- Human Impact Index (HII) calculation
- Upstream dam impact analysis
- Data quality scoring
- Cross-validation with in-situ data

### Step 4: NC Merge (Interface Only)

Placeholder interface for future implementation:
- Load existing NC file
- Detect duplicate stations
- Merge time series data
- Save updated NC

## Output Format

Single file with resolution-suffixed column names:

```
ID station dataname lon lat satellite flag elevation dist_to_mouth \
kx1_01min ky1_01min kx2_01min ky2_01min dist1_01min dist2_01min rivwth_01min ix_01min iy_01min lon_cama_01min lat_cama_01min \
kx1_03min ky1_03min kx2_03min ky2_03min dist1_03min dist2_03min rivwth_03min ix_03min iy_03min lon_cama_03min lat_cama_03min \
kx1_05min ky1_05min kx2_05min ky2_05min dist1_05min dist2_05min rivwth_05min ix_05min iy_05min lon_cama_05min lat_cama_05min \
kx1_06min ky1_06min kx2_06min ky2_06min dist1_06min dist2_06min rivwth_06min ix_06min iy_06min lon_cama_06min lat_cama_06min \
kx1_15min ky1_15min kx2_15min ky2_15min dist1_15min dist2_15min rivwth_15min ix_15min iy_15min lon_cama_15min lat_cama_15min \
EGM08 EGM96
```

## CLI Interface

```bash
# Full pipeline
python -m wse_pipeline.src.main --config my_dataset.yaml

# Single step
python -m wse_pipeline.src.main --config my_dataset.yaml --step validate
python -m wse_pipeline.src.main --config my_dataset.yaml --step cama

# Resume from checkpoint
python -m wse_pipeline.src.main --config my_dataset.yaml --resume

# Quick source selection
python -m wse_pipeline.src.main --source hydroweb
python -m wse_pipeline.src.main --source cgls
python -m wse_pipeline.src.main --source icesat
```

**Arguments**:

| Argument | Description |
|----------|-------------|
| `--config` | YAML config file path |
| `--source` | Quick source selection (uses default config) |
| `--step` | Run specific step only (validate/cama/reserved/merge) |
| `--resume` | Resume from last checkpoint |
| `--output-dir` | Override output directory |
| `--log-level` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `--dry-run` | Simulation mode, no file writes |

## Data Source Formats

| Source | Format | Key Parsing Logic |
|--------|--------|-------------------|
| HydroWeb | Text, `#KEY:: value` header | Parse 33-line header metadata |
| CGLS | JSON (GeoJSON) | Parse `properties` and `data` array |
| ICESat | Space-separated text, no header | Aggregate by lat/lon filename blocks |
| HydroSat | TODO | Pending web scraping |

## TODO Items

1. **HydroSat scraper** - Analyze web structure, write batch download script
2. **NC merge implementation** - Step 4 interface ready for future work

## References

- AltiMaP original Fortran code: `/Users/zhongwangwei/Desktop/Github/AltiMaP`
- Streamflow pipeline: `/Users/zhongwangwei/Desktop/Github/OpenBench-wei/preprocessing/Streamflow`
- Existing AllocateVS.py: `./AllocateVS.py`
- Existing geoid_calculator.py: `./geoid_calculator.py`
