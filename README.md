# OpenBench
The Open Land Surface Model Benchmark Evaluation System
## Overview

This system is designed for the evaluation of land surface model outputs against reference data. It handles configuration, data processing, validation, comparison, and statistical analysis.

## Key Features

- Supports multiple land surface models (e.g., CLM5, CoLM)
- Handles various data types (grid and station-based)
- Performs comprehensive evaluations across multiple variables
- Generates detailed metrics, scores, comparisons, and statistical analysis

## Main Components

1. 'run.py': Main entry point for the evaluation system
2. 'Mod_Validation.py': Handles validation of model outputs
3. 'Mod_DatasetProcessing.py': Processes datasets for evaluation
4. 'Mod_Comparison.py': Performs comparisons between models and reference data
5. 'Mod_Namelist.py': Manages configuration and namelist reading
6. 'Mod_Metrics.py': Calculates various statistical metrics
7. 'Mod_Scores.py': Computes performance scores
8. 'Mod_Statistics': Conducts statistical analysis
9. 'Lib_Unit': convert the unit 
10. Folder 'figlib': lib for vitalization
11. Folder 'regrid': lib for remap--modified from xarray-regrid
12. Floder 'custom': lib for dataset filter
 
## Setup and Configuration

1. Ensure all dependencies are installed (xarray, pandas, numpy, etc.)
2. Configure the system using the provided namelist files:
   - 'main.nml': configuration
   - 'ref.nml': Reference data configuration
   - 'sim.nml': Simulation data configuration
   - Model-specific namelists (e.g., 'CLM5.nml', 'CoLM.nml')
   - Reference-specific namelists (e.g., 'GRDC.nml', 'ILAMB.nml')

## Usage

Run the evaluation system using:

'''
python script/openbench.py nml/main.nml [or path_to_main_namelist]
'''

## Supported Evaluations

- Radiation and Energy Cycle
- Ecosystem and Carbon Cycle
- Hydrology Cycle
- Human Activity (Urban, Crop, Dam)
- Forcing Variables

## Output

The system generates various outputs including:
- Metric and score calculations
- Comparison plots and statistics
- Detailed logs and summaries

## Customization

- Custom filters can be added for handling specific variables, models or datasets
- Evaluation items, metrics, and scores can be configured in the namelists

## Contributors

- Zhongwang Wei (zhongwang007@gmail.com)
- Qingchen Xu (******)

## Version

Current Version: 0.1
Release Date: Sep 2024


