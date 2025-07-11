# OpenBench: The Open Source Land Surface Model Benchmarking System

## Overview

OpenBench is a comprehensive, open-source system designed to rigorously evaluate and compare the performance of land surface models (LSMs). It provides a standardized framework for benchmarking LSM outputs against a wide array of reference datasets, encompassing various physical processes and variables. This system automates critical aspects of model evaluation, including configuration management, data processing, validation, inter-comparison, and in-depth statistical analysis. By streamlining these complex tasks, OpenBench empowers researchers and model developers to efficiently assess model capabilities, identify areas for improvement, and advance the science of land surface modeling.

**Latest Updates (v2.0):**
- ✅ **Multi-Format Configuration Support**: JSON, YAML, and Fortran Namelist formats
- ✅ **Enhanced Modular Architecture**: 9 core modules with standardized interfaces  
- ✅ **Cross-Platform Compatibility**: Windows, Linux, and macOS support
- ✅ **Intelligent Parallel Processing**: Automatic multi-core utilization with progress tracking
- ✅ **Advanced Caching System**: Memory and disk caching for improved performance
- ✅ **Unified Error Handling**: Consistent error reporting and recovery mechanisms
- ✅ **Enhanced Logging**: Structured JSON logs with performance metrics

## Key Features

*   **Multi-Model Support:** Seamlessly evaluate and compare outputs from various land surface models, such as the Community Land Model (CLM5) and the Common Land Model (CoLM).
*   **Multi-Format Configuration:** Full support for JSON, YAML, and Fortran Namelist configuration formats with automatic format detection and fallback mechanisms.
*   **Cross-Platform Compatibility:** Runs on Windows, Linux, and macOS with intelligent handling of platform-specific dependencies.
*   **Diverse Data Handling:** Accommodates a wide range of data types, including gridded datasets and station-based observations, ensuring comprehensive model assessment.
*   **Comprehensive Evaluation:** Performs thorough evaluations across multiple biophysical variables, covering crucial aspects of the energy cycle, carbon cycle, hydrological cycle, and human activity impacts.
*   **Intelligent Parallel Processing:** Automatic multi-core utilization with smart worker allocation, progress tracking, and resource monitoring.
*   **Advanced Caching System:** Multi-level caching (memory + disk) with LRU eviction and automatic invalidation for improved performance.
*   **Advanced Metrics and Scoring:** Generates a suite of detailed metrics and performance scores, providing quantitative insights into model accuracy and behavior.
*   **In-depth Statistical Analysis:** Conducts robust statistical analyses to assess model performance, identify biases, and quantify uncertainties.
*   **Enhanced Error Handling:** Unified error handling system with structured logging and graceful degradation for missing dependencies.
*   **Modular Architecture:** Built with 9 core modules featuring standardized interfaces, dependency injection, and plugin system for extensibility.
*   **Visualization Support:** Includes libraries for generating visualizations to aid in the interpretation of evaluation results.
*   **Regridding Capabilities:** Provides tools for regridding data to common spatial resolutions, facilitating consistent comparisons.

## Main Components

OpenBench features a modern modular architecture with 9 core modules and legacy integration:

### **Core Modules (New Architecture)**
1.  **`Mod_ConfigManager.py`**: Unified configuration management supporting JSON, YAML, and Fortran Namelist formats with schema validation
2.  **`Mod_Exceptions.py`**: Standardized error handling system with structured logging and graceful degradation
3.  **`Mod_Interfaces.py`**: Abstract base classes and interfaces ensuring component compatibility
4.  **`Mod_DataPipeline.py`**: Streamlined data processing pipeline with validation and caching
5.  **`Mod_EvaluationEngine.py`**: Modular evaluation engine with pluggable metrics and parallel processing
6.  **`Mod_OutputManager.py`**: Multi-format output management with metadata preservation
7.  **`Mod_LoggingSystem.py`**: Enhanced logging with JSON formatting and performance metrics
8.  **`Mod_ParallelEngine.py`**: Intelligent parallel processing with automatic resource management
9.  **`Mod_CacheSystem.py`**: Multi-level caching system with memory and disk storage

### **Enhanced Legacy Components**
- **`openbench.py`**: Main entry point with enhanced multi-format configuration support
- **`Mod_Evaluation.py`**: Core evaluation logic integrated with new modules while maintaining compatibility
- **`Mod_DatasetProcessing.py`**: Data processing enhanced with pipeline integration and caching
- **`Mod_Comparison.py`**: Cross-model comparison with improved parallel processing
- **`Mod_Namelist.py`**: Legacy namelist support with new ConfigManager integration
- **`Mod_Metrics.py`**: Statistical metrics calculation with enhanced parallel execution
- **`Mod_Scores.py`**: Performance scoring with new evaluation engine integration
- **`Mod_Statistics.py`**: Statistical analysis with enhanced logging and caching

### **Supporting Libraries**
- **`Lib_Unit/`**: Unit conversion library ensuring dataset consistency
- **`figlib/`**: Visualization libraries for evaluation result plots
- **`regrid/`**: Spatial regridding operations (modified from `xarray-regrid`)
- **`custom/`**: User-defined scripts and filters
- **`statistic/`**: Statistical functions and base classes
- **`config/`**: Configuration readers and validators for multiple formats

## Architecture Highlights

### **Enhanced Performance**
- **Parallel Processing**: Automatic multi-core utilization with smart worker allocation based on system resources
- **Intelligent Caching**: Memory and disk caching with LRU eviction policies for faster subsequent runs
- **Data Pipelines**: Streamlined processing with validation and error recovery
- **Resource Monitoring**: Real-time CPU and memory usage tracking with automatic optimization

### **Improved Reliability**
- **Unified Error Handling**: Consistent error reporting with structured logging and graceful degradation
- **Configuration Validation**: Schema-based validation for all configuration formats with helpful error messages
- **Format Flexibility**: Automatic detection and parsing of JSON, YAML, and Fortran Namelist formats
- **Interface Contracts**: Abstract base classes ensure component compatibility and extensibility

### **Better Maintainability**
- **Modular Design**: 9 independent, testable, and reusable core modules
- **Dependency Injection**: Configurable backends and implementations for different platforms
- **Plugin System**: Extensible metrics, formatters, and data processors
- **Legacy Integration**: 100% backward compatibility with existing workflows and configurations

## Benefits of using OpenBench

*   **Standardized Evaluation:** Provides a consistent and reproducible framework for benchmarking LSMs, reducing subjectivity and improving comparability of results across studies.
*   **Efficiency and Automation:** Automates many time-consuming tasks involved in model evaluation, freeing up researchers to focus on analysis and interpretation.
*   **Comprehensive Insights:** Delivers a holistic view of model performance through a wide range of metrics, scores, and statistical analyses.
*   **Facilitates Model Improvement:** Helps identify model strengths and weaknesses, guiding efforts for model development and refinement.
*   **Community Collaboration:** As an open-source project, it encourages collaboration, knowledge sharing, and the development of best practices in land surface model evaluation.
*   **Flexibility and Customization:** Adapts to diverse research needs through its configurable and extensible design.

## Setup and Configuration

### **Prerequisites**
1. **Python Environment**: Python 3.8+ recommended
2. **Core Dependencies**: `xarray`, `pandas`, `numpy`, `netCDF4`, `matplotlib`, `cartopy`
3. **Optional Dependencies**: 
   - `yaml` for YAML configuration support
   - `f90nml` for Fortran Namelist support  
   - `CDO` for advanced data operations (Linux/macOS)

### **Multi-Format Configuration Support**
OpenBench supports three configuration formats with automatic detection:

#### **Directory Structure**
```
nml/
├── nml-json/          # JSON configuration files
│   ├── main-Debug.json
│   ├── ref-Debug.json
│   └── sim-Debug.json
├── nml-yaml/          # YAML configuration files  
│   ├── main-Debug.yaml
│   ├── ref-Debug.yaml
│   └── sim-Debug.yaml
└── nml-Fortran/       # Fortran Namelist files
    ├── main-Debug.nml
    ├── ref-Debug.nml
    └── sim-Debug.nml
```

#### **Configuration Files**
- **Main Configuration** (`main.*`): Overall evaluation settings, paths, and processing options
- **Reference Configuration** (`ref.*`): Reference dataset definitions and sources
- **Simulation Configuration** (`sim.*`): Model output dataset configurations
- **Variable Definitions**: Model and reference variable mappings in respective subdirectories

### **Cross-Platform Compatibility**
- **Windows**: Automatic detection and graceful handling of missing CDO dependency
- **Linux/macOS**: Full functionality including CDO operations
- **All Platforms**: Core evaluation features work universally

## Usage

### **Basic Usage**
OpenBench automatically detects configuration format and runs the evaluation:

```bash

# JSON format
python openbench/openbench.py nml/nml-json/main-Debug.json

# YAML format  
python openbench/openbench.py nml/nml-yaml/main-Debug.yaml

# Fortran Namelist format
python openbench/openbench.py nml/nml-Fortran/main-Debug.nml
```

### **Advanced Usage**
```bash
# Using full paths
python openbench/openbench.py /path/to/your/config.json

# API Usage (Python script)
from openbench.openbench_api import OpenBenchAPI
api = OpenBenchAPI()
results = api.run_evaluation("nml/nml-json/main-Debug.json")
```

### **Configuration Format Examples**

#### **JSON Configuration**
```json
{
  "general": {
    "basename": "debug",
    "basedir": "./output",
    "reference_nml": "./nml/nml-json/ref-Debug.json",
    "simulation_nml": "./nml/nml-json/sim-Debug.json"
  },
  "evaluation_items": {
    "Evapotranspiration": true,
    "Latent_Heat": true
  }
}
```

#### **YAML Configuration**
```yaml
general:
  basename: debug
  basedir: ./output
  reference_nml: ./nml/nml-yaml/ref-Debug.yaml
  simulation_nml: ./nml/nml-yaml/sim-Debug.yaml
evaluation_items:
  Evapotranspiration: true
  Latent_Heat: true
```

#### **Fortran Namelist Configuration**
```fortran
&general
  basename = debug
  basedir = ./output
  reference_nml = ./nml/nml-Fortran/ref-Debug.nml
  simulation_nml = ./nml/nml-Fortran/sim-Debug.nml
/
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

### **Platform-Specific Issues**

#### **Windows Users**
- **CDO Dependency**: CDO is not available on Windows. OpenBench automatically detects this and gracefully skips CDO-dependent operations
- **Path Handling**: Use forward slashes (/) in configuration files for cross-platform compatibility
- **Memory Management**: Windows may require adjusting parallel workers for large datasets

#### **Linux/macOS Users**
- **CDO Installation**: Install CDO via package manager (`apt install cdo`, `brew install cdo`)
- **File Permissions**: Ensure write permissions for output directories

### **Configuration Issues**
- **Format Detection**: OpenBench automatically detects JSON/YAML/NML formats
- **Path Errors**: Use relative paths from the working directory or absolute paths
- **Comment Parsing**: In Fortran NML files, comments after values are automatically stripped

### **Performance Optimization**
- **Parallel Processing**: Adjust `num_cores` in main configuration based on system resources
- **Caching**: First runs may be slower due to cache building; subsequent runs will be faster
- **Memory Usage**: Monitor memory usage with large datasets; reduce parallel workers if needed

### **Cartopy Coastline Data Download**
An internet connection is required for Cartopy coastline data. For offline HPC environments:

```python
import cartopy
print(cartopy.config['data_dir'])
```

Download Natural Earth datasets from https://www.naturalearthdata.com/downloads/ and place them in:
```
└── cartopy_data_dir/
    ├── shapefiles/
    │   ├── natural_earth/
    │   │   ├── cultural/
    │   │   └── physical/
    └── raster/
        └── natural_earth/
```

### **Common Error Solutions**
- **Configuration File Not Found**: Check file paths and ensure files exist
- **Module Import Errors**: Install missing dependencies with `pip install package_name`
- **Memory Errors**: Reduce dataset size or adjust parallel processing settings
- **Plotting Errors**: Ensure matplotlib backend is properly configured

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

## Version History

**Current Version: 2.0**
- Release Date: July 2025
- Major refactoring with enhanced modular architecture
- Multi-format configuration support (JSON, YAML, Fortran NML)
- Cross-platform compatibility (Windows, Linux, macOS)
- Intelligent parallel processing and caching systems
- Unified error handling and enhanced logging

**Previous Version: 1.0**
- Release Date: June 2025
- Initial open-source release
- Basic evaluation framework
- JSON configuration support
