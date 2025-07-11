# OpenBench: The Open Source Land Surface Model Benchmarking System

## Overview

OpenBench is a comprehensive, open-source system designed to rigorously evaluate and compare the performance of land surface models (LSMs). It provides a standardized framework for benchmarking LSM outputs against a wide array of reference datasets, encompassing various physical processes and variables. This system automates critical aspects of model evaluation, including configuration management, data processing, validation, inter-comparison, and in-depth statistical analysis. By streamlining these complex tasks, OpenBench empowers researchers and model developers to efficiently assess model capabilities, identify areas for improvement, and advance the science of land surface modeling.

## Key Features

*   **Multi-Model Support:** Seamlessly evaluate and compare outputs from various land surface models, such as the Community Land Model (CLM5) and the Common Land Model (CoLM).
*   **Diverse Data Handling:** Accommodates a wide range of data types, including gridded datasets and station-based observations, ensuring comprehensive model assessment.
*   **Comprehensive Evaluation:** Performs thorough evaluations across multiple biophysical variables, covering crucial aspects of the energy cycle, carbon cycle, hydrological cycle, and human activity impacts.
*   **Flexible Configuration:** Utilizes a user-friendly namelist system for easy configuration of evaluation parameters, model settings, and dataset specifications.
*   **Advanced Metrics and Scoring:** Generates a suite of detailed metrics and performance scores, providing quantitative insights into model accuracy and behavior.
*   **In-depth Statistical Analysis:** Conducts robust statistical analyses to assess model performance, identify biases, and quantify uncertainties.
*   **Customizable Framework:** Allows users to define custom filters for specific variables, models, or datasets, and to tailor evaluation items, metrics, and scoring methods to their needs.
*   **Modular Design:** Built with a modular architecture, making it extensible and adaptable for future enhancements and community contributions.
*   **Visualization Support:** Includes libraries for generating visualizations to aid in the interpretation of evaluation results.
*   **Regridding Capabilities:** Provides tools for regridding data to common spatial resolutions, facilitating consistent comparisons.

## Main Components

OpenBench is comprised of several key modules that work together to deliver its powerful benchmarking capabilities:

1.  **`openbench.py`**: The main entry point and orchestrator of the evaluation system. It parses configurations and manages the overall workflow.
2.  **`Mod_Evaluation.py`**: Handles the core logic for evaluating model outputs against reference data based on the configured metrics and criteria.
3.  **`Mod_DatasetProcessing.py`**: Responsible for ingesting, processing, and preparing both model simulation outputs and reference datasets for evaluation. This includes tasks like unit conversion and temporal/spatial alignment.
4.  **`Mod_Comparison.py`**: Performs detailed comparisons between different model outputs and between models and reference datasets.
5.  **`Mod_Namelist.py`**: Manages the reading and interpretation of configuration files (namelists) that define the evaluation setup.
6.  **`Mod_Metrics.py`**: Calculates a variety of statistical metrics to quantify different aspects of model performance (e.g., bias, RMSE, correlation).
7.  **`Mod_Scores.py`**: Computes aggregated performance scores based on the calculated metrics, providing a summarized view of model skill.
8.  **`Mod_Statistics.py`**: Conducts further statistical analyses on the evaluation results, such as significance testing or trend analysis.
9.  **`Lib_Unit/`**: A library for handling unit conversions, ensuring consistency between different datasets.
10. **`figlib/`**: A collection of libraries and scripts dedicated to generating plots and visualizations of the evaluation results.
11. **`regrid/`**: A library for performing spatial regridding operations, modified from `xarray-regrid`, to align datasets to a common grid.
12. **`custom/`**: A directory for user-defined scripts or libraries, such as custom filters for specific dataset characteristics or variable manipulations.
13. **`statistic/`**: A library containing statistical functions and base classes used by `Mod_Statistics.py` and other analysis components.

## Benefits of using OpenBench

*   **Standardized Evaluation:** Provides a consistent and reproducible framework for benchmarking LSMs, reducing subjectivity and improving comparability of results across studies.
*   **Efficiency and Automation:** Automates many time-consuming tasks involved in model evaluation, freeing up researchers to focus on analysis and interpretation.
*   **Comprehensive Insights:** Delivers a holistic view of model performance through a wide range of metrics, scores, and statistical analyses.
*   **Facilitates Model Improvement:** Helps identify model strengths and weaknesses, guiding efforts for model development and refinement.
*   **Community Collaboration:** As an open-source project, it encourages collaboration, knowledge sharing, and the development of best practices in land surface model evaluation.
*   **Flexibility and Customization:** Adapts to diverse research needs through its configurable and extensible design.

## Setup and Configuration

1.  **Prerequisites:** Ensure all necessary Python dependencies are installed. Key dependencies include `xarray`, `pandas`, `numpy`, and others as required by specific modules. (A `requirements.txt` file will be provided in future versions for easier setup).
2.  **Configuration Files:** Configure the system using the provided configuration files located in the `nml/` directory. JSON format is shown primarily, but Fortran Namelist and YAML are also supported:
    *   `main.json`: Main configuration file defining the overall evaluation run, including paths to model and reference data, desired evaluation periods, and output settings.
    *   `ref.json`: Configures the reference datasets to be used in the evaluation.
    *   `sim.json`: Configures the simulation (model output) datasets to be evaluated.
    *   Model-specific definition files (e.g., `CLM5.json`, `CoLM.json` in `nml/Mod_variables_defination/`): Define parameters and variable mappings for each land surface model.
    *   Reference-specific definition files (e.g., `GRDC.json`, `ILAMB.json` in `nml/Ref_variables_defination/`): Define parameters and variable mappings for each reference dataset.
    Note: While JSON is shown, configuration files in traditional Fortran Namelist (.nml) and YAML (.yaml) formats are also supported by the system.

## Usage

To run the OpenBench evaluation system, execute the main script from your terminal:

```bash
python openbench/openbench.py nml/main-Debug.yaml
```
You can also use a main configuration file in .nml or .json format.

Alternatively, you can provide the full path to your main configuration file (e.g., `/path/to/your/main.json` or `/path/to/your/main.nml`):
```bash
python openbench/openbench.py /path/to/your/main.yaml
```

## Supported Evaluations

OpenBench supports the evaluation of a wide range of variables and processes, typically categorized into:

*   **Radiation and Energy Cycle:** (e.g., net radiation, sensible heat flux, latent heat flux)
*   **Ecosystem and Carbon Cycle:** (e.g., gross primary productivity, net ecosystem exchange, soil carbon)
*   **Hydrology Cycle:** (e.g., evapotranspiration, runoff, soil moisture, snow water equivalent)
*   **Human Activity:** (e.g., impacts of urban areas, crop management, dam operations on land surface processes)
*   **Forcing Variables:** (e.g., precipitation, air temperature, humidity - often used for input validation or understanding driver impacts)

The specific variables and evaluation items can be configured in the namelists.

## Output

The system generates a comprehensive set of outputs, typically organized into a user-defined output directory. These include:

*   **Metrics and Scores:** Text files or spreadsheets containing detailed metric calculations (e.g., bias, RMSE, correlation coefficients) and aggregated performance scores for each evaluated variable and model.
*   **Comparison Plots and Statistics:** Visualizations (e.g., time series plots, scatter plots, spatial maps) and statistical summaries comparing model outputs against reference data and against other models.
*   **Detailed Logs and Summaries:** Log files documenting the evaluation process, including any warnings or errors, and summary reports of the evaluation findings.

## Troubleshooting
### Catopy coastline data download
An internet connection is required for Cartopy coastline, while some HPC environments may not have internet connectivity. For offline HPC use, manually download Natural Earth datasets and save them to Cartopy's data directory.
```bash
import cartopy
print(cartopy.config['data_dir'])
```
Access Natural Earth official website download data (Cultural, Physical) on an Internet-enabled machine: https://www.naturalearthdata.com/downloads/
We recommend to download 110m resulation files and replace them on 'data dir'.
```bash
└── cartopy_data_dir/
    ├── shapefiles/
    │   ├── natural_earth/
    │   │   ├── cultural/
    │   │   └── physical/
    └── raster/
        └── natural_earth/
```
For example, the Cultural shapefile should be exist at /Users/zhongwangwei/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_110m_admin_0_boundary_lines_land.shp

## Customization

OpenBench is designed to be flexible:

*   **Custom Filters:** Users can develop and integrate custom Python scripts into the `custom/` directory to handle specific data processing needs, such as filtering data based on quality flags, selecting specific regions or time periods, or applying complex transformations.
*   **Configurable Evaluations:** Evaluation items (variables to be analyzed), metrics to be calculated, and scoring methodologies can be precisely defined and modified within the namelist files, allowing users to tailor the benchmarking process to their specific research questions.

## How to Contribute

We welcome contributions from the community to enhance and expand OpenBench! Whether you're interested in adding support for new models, incorporating additional reference datasets, developing new analysis metrics, or improving the software's usability, your input is valuable.

Here's how you can contribute:

1.  **Fork the Repository:** Start by forking the official OpenBench repository on GitHub.
2.  **Create a New Branch:** Create a dedicated branch for your changes (e.g., `feature/new-metric` or `bugfix/issue-123`).
3.  **Make Your Changes:** Implement your improvements, ensuring your code adheres to the project's coding style and includes clear comments.
4.  **Test Your Changes:** If applicable, add unit tests for any new functionality and ensure all existing tests pass.
5.  **Write Clear Commit Messages:** Follow good practices for writing informative commit messages.
6.  **Submit a Pull Request:** Once your changes are ready, submit a pull request to the main OpenBench repository. Provide a clear description of the changes you've made and why they are beneficial.

Please also consider opening an issue on GitHub to discuss significant changes or new features before investing a lot of time in development.

## License

OpenBench is released under the **MIT License**. See the `LICENSE` file in the repository for full details. This permissive license allows for broad use and modification of the software, encouraging open collaboration and dissemination.

## Contributors

*   Zhongwang Wei (zhongwang007@gmail.com)
*   Qingchen Xu (PhD student)
*   Fan Bai (MS student)
*   Zixin Wei (PhD student)
*   Xionghui Xu (PhD student)
*   Dr. Wenzong Dong

We are grateful for all contributions to the OpenBench project.

## Version

Current Version: 1.0
Release Date: June 2025
