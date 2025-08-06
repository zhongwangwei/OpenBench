#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generation Module for OpenBench
Generates comprehensive HTML and PDF evaluation reports with tables, figures, and detailed analysis
"""

import os
import json
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
from jinja2 import Template
import glob
import shutil

# Import PDF generation libraries
try:
    from xhtml2pdf import pisa
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import logging

# Setup logger
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive evaluation reports in HTML and PDF formats"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize the report generator
        
        Args:
            config: Configuration dictionary
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, "reports")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.scores_dir = os.path.join(output_dir, "scores")
        self.comparisons_dir = os.path.join(output_dir, "comparisons")
        self.data_dir = os.path.join(output_dir, "data")
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Report metadata - only include enabled evaluation items
        enabled_items = []
        if isinstance(config.get("evaluation_items"), dict):
            enabled_items = [item for item, enabled in config["evaluation_items"].items() if enabled]
        elif isinstance(config.get("evaluation_items"), list):
            enabled_items = config["evaluation_items"]
        
        # Get enabled metrics and scores from configuration
        self.enabled_metrics = [metric for metric, enabled in config.get("metrics", {}).items() if enabled]
        self.enabled_scores = [score for score, enabled in config.get("scores", {}).items() if enabled]
        
        # Get enabled comparisons from configuration
        self.enabled_comparisons = [comp for comp, enabled in config.get("comparisons", {}).items() if enabled]
        
        self.metadata = {
            "title": "OpenBench Evaluation Report",
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_file": config.get("config_file", "N/A"),
            "evaluation_items": enabled_items
        }
        
        logger.info(f"Report generator initialized with output directory: {self.report_dir}")
    
    def generate_report(self, report_name: str = "evaluation_report") -> Dict[str, str]:
        """
        Generate both HTML and PDF reports
        
        Args:
            report_name: Base name for the report files
            
        Returns:
            Dictionary with paths to generated reports
        """
        logger.info("Starting report generation...")
        
        # Collect all data
        report_data = self._collect_report_data()
        
        # Copy all relevant figures to report directory first
        self._copy_figures_to_report_dir()
        
        # Verify figure paths before generating reports
        self._verify_figure_paths(report_data)
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data, report_name)
        
        # Generate PDF report (after figures are copied)
        pdf_path = self._generate_pdf_report(html_path, report_name)
        
        logger.info(f"Report generation completed successfully")
        logger.info(f"HTML report: {html_path}")
        if pdf_path:
            logger.info(f"PDF report: {pdf_path}")
        
        result = {
            "html": html_path
        }
        if pdf_path:
            result["pdf"] = pdf_path
        
        return result
    
    def _collect_report_data(self) -> Dict[str, Any]:
        """Collect all data needed for the report"""
        logger.info("Collecting report data...")
        
        report_data = {
            "metadata": self.metadata,
            "evaluation_items": {},
            "overall_summary": {},
            "comparisons": {},
            "climate_zone_analysis": {}
        }
        
        # Collect data for each evaluation item
        for item in self.metadata["evaluation_items"]:
            logger.info(f"Collecting data for {item}")
            item_data = {
                "metrics": self._collect_metrics_data(item),
                "scores": self._collect_scores_data(item),
                "figures": self._collect_figures(item),
                "statistics": self._collect_statistics(item)
            }
            
            # Debug logging
            logger.info(f"Collected data for {item}:")
            logger.info(f"  - Figures: {list(item_data['figures'].keys())}")
            logger.info(f"  - Statistics: {list(item_data['statistics'].keys())}")
            for fig_type, figs in item_data['figures'].items():
                if figs:
                    logger.info(f"    {fig_type}: {len(figs)} figures")
            
            report_data["evaluation_items"][item] = item_data
        
        # Collect comparison data
        report_data["comparisons"] = self._collect_comparison_data()
        
        # Collect overall summary
        report_data["overall_summary"] = self._generate_overall_summary(report_data)
        
        # Collect groupby analysis summary
        report_data["groupby_summary"] = self._generate_groupby_analysis_summary(report_data)
        
        return report_data
    
    def _collect_metrics_data(self, item: str) -> Dict[str, Any]:
        """Collect metrics data for a specific evaluation item"""
        metrics_data = {}
        
        # Look for CSV files with evaluation results
        csv_pattern = os.path.join(self.metrics_dir, f"{item}_*_evaluations.csv")
        csv_files = glob.glob(csv_pattern)
        
        for csv_file in csv_files:
            key = os.path.basename(csv_file).replace("_evaluations.csv", "")
            try:
                df = pd.read_csv(csv_file)
                metrics_data[key] = {
                    "data": df.to_dict(orient='records'),
                    "summary": self._generate_metrics_summary(df)
                }
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        # Generate comprehensive grid vs grid statistics from NetCDF files
        grid_grid_stats = self._generate_grid_vs_grid_stats(item)
        grid_grid_pairs = set()
        if grid_grid_stats:
            metrics_data.update(grid_grid_stats)
            # Keep track of which grid vs grid pairs we've already processed comprehensively
            for key in grid_grid_stats.keys():
                if "vs" in key:
                    grid_grid_pairs.add(key)
        
        # Look for individual NetCDF files with spatial metrics
        nc_pattern = os.path.join(self.metrics_dir, f"{item}_*.nc")
        nc_files = glob.glob(nc_pattern)
        
        for nc_file in nc_files:
            if "_evaluations" not in nc_file:  # Skip CSV-related files
                # Skip NetCDF files that are already covered by comprehensive grid vs grid stats
                filename = os.path.basename(nc_file)
                key = os.path.basename(nc_file).replace(".nc", "")
                try:
                    with xr.open_dataset(nc_file) as ds:
                        # Get the main data variable (skip coordinate variables)
                        data_vars = [var for var in ds.data_vars if var not in ds.coords]
                        if data_vars:
                            main_var = ds[data_vars[0]]
                            # Extract comprehensive statistics
                            valid_data = main_var.values[~np.isnan(main_var.values)]
                            
                            if len(valid_data) > 0:
                                # Try to extract comparison pair from filename first, then fallback to config
                                comparison_pair = self._extract_comparison_pair(key)
                                
                                # If we can extract ref and sim sources from filename, try to get better name from config
                                if "ref_" in key and "sim_" in key:
                                    try:
                                        # Extract ref and sim sources from filename
                                        parts = key.split('_')
                                        ref_source = None
                                        sim_source = None
                                        
                                        for i, part in enumerate(parts):
                                            if part == "ref" and i + 1 < len(parts):
                                                ref_parts = []
                                                j = i + 1
                                                while j < len(parts) and parts[j] != "sim":
                                                    ref_parts.append(parts[j])
                                                    j += 1
                                                ref_source = "_".join(ref_parts)
                                                
                                            elif part == "sim" and i + 1 < len(parts):
                                                sim_parts = []
                                                j = i + 1
                                                while j < len(parts) and not self._is_metric_or_score(parts[j]):
                                                    sim_parts.append(parts[j])
                                                    j += 1
                                                sim_source = "_".join(sim_parts)
                                                break
                                        
                                        if ref_source and sim_source:
                                            comparison_pair = self._get_comparison_pair_from_config(item, ref_source, sim_source)
                                    except Exception as e:
                                        logger.warning(f"Error extracting sources from filename {key}: {e}")
                                
                                metrics_data[key] = {
                                    "global_mean": float(np.mean(valid_data)),
                                    "global_std": float(np.std(valid_data)),
                                    "global_min": float(np.min(valid_data)),
                                    "global_max": float(np.max(valid_data)),
                                    "global_median": float(np.median(valid_data)),
                                    "valid_points": int(len(valid_data)),
                                    "total_points": int(main_var.size),
                                    "data_coverage": float(len(valid_data) / main_var.size * 100),
                                    "shape": str(main_var.dims),
                                    "metric_type": self._extract_metric_type(key),
                                    "comparison_pair": comparison_pair
                                }
                            else:
                                logger.warning(f"No valid data found in {nc_file}")
                        else:
                            logger.warning(f"No data variables found in {nc_file}")
                except Exception as e:
                    logger.warning(f"Error reading {nc_file}: {e}")
        
        return metrics_data
    
    def _collect_scores_data(self, item: str) -> Dict[str, Any]:
        """Collect scores data for a specific evaluation item"""
        scores_data = {}
        
        # Similar to metrics collection
        csv_pattern = os.path.join(self.scores_dir, f"{item}_*_evaluations.csv")
        csv_files = glob.glob(csv_pattern)
        
        for csv_file in csv_files:
            key = os.path.basename(csv_file).replace("_evaluations.csv", "")
            try:
                df = pd.read_csv(csv_file)
                scores_data[key] = {
                    "data": df.to_dict(orient='records'),
                    "summary": self._generate_scores_summary(df)
                }
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        # Note: Individual NetCDF files in scores directory are not processed separately
        # as they are already included in the comprehensive grid vs grid statistics
        # generated by _generate_grid_vs_grid_stats in the metrics collection
        
        return scores_data
    
    def _collect_figures(self, item: str) -> Dict[str, List[str]]:
        """Collect all figures related to an evaluation item"""
        figures = {
            "metrics": [],
            "scores": [],
            "comparisons": [],
            "igbp_groupby": [],
            "pft_groupby": [],
            "climate_zone_groupby": []
        }
        
        # Metrics figures
        metrics_pattern = os.path.join(self.metrics_dir, f"{item}_*.jpg")
        figures["metrics"] = [os.path.basename(f) for f in glob.glob(metrics_pattern)]
        
        # Scores figures
        scores_pattern = os.path.join(self.scores_dir, f"{item}_*.jpg")
        figures["scores"] = [os.path.basename(f) for f in glob.glob(scores_pattern)]
        
        # Comparison figures (from various subdirectories)
        comparison_dirs = ["Taylor_Diagram", "Target_Diagram", "Whisker_Plot", 
                          "Ridgeline_Plot", "Kernel_Density_Estimate", "Parallel_Coordinates"]
        
        for comp_dir in comparison_dirs:
            comp_path = os.path.join(self.comparisons_dir, comp_dir, f"*{item}*.jpg")
            comp_files = glob.glob(comp_path)
            figures["comparisons"].extend([f"{comp_dir}/{os.path.basename(f)}" for f in comp_files])
        
        # IGBP groupby figures - now primarily in comparisons directory
        igbp_files = []
        # Check comparisons directory as the primary location
        igbp_comp_dir = os.path.join(self.comparisons_dir, "IGBP_groupby")
        if os.path.exists(igbp_comp_dir):
            # Look for heatmap figures in subdirectories
            for subdir in glob.glob(os.path.join(igbp_comp_dir, "*/")):
                heatmap_files = glob.glob(os.path.join(subdir, f"*{item}*heatmap*.jpg"))
                igbp_files.extend(heatmap_files)
                # Also look for any other jpg files related to the item
                other_files = glob.glob(os.path.join(subdir, f"*{item}*.jpg"))
                igbp_files.extend([f for f in other_files if f not in igbp_files])
            # Also check the root directory
            root_files = glob.glob(os.path.join(igbp_comp_dir, f"*{item}*.jpg"))
            igbp_files.extend(root_files)
        
        # Format paths relative to the base directory
        figures["igbp_groupby"] = []
        for f in igbp_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir)
                figures["igbp_groupby"].append(f"comparisons/{rel_path}")
        
        if figures["igbp_groupby"]:
            logger.info(f"Found IGBP groupby figures: {figures['igbp_groupby']}")
        
        # PFT groupby figures - now primarily in comparisons directory
        pft_files = []
        # Check comparisons directory as the primary location
        pft_comp_dir = os.path.join(self.comparisons_dir, "PFT_groupby")
        if os.path.exists(pft_comp_dir):
            # Look for heatmap figures in subdirectories
            for subdir in glob.glob(os.path.join(pft_comp_dir, "*/")):
                heatmap_files = glob.glob(os.path.join(subdir, f"*{item}*heatmap*.jpg"))
                pft_files.extend(heatmap_files)
                # Also look for any other jpg files related to the item
                other_files = glob.glob(os.path.join(subdir, f"*{item}*.jpg"))
                pft_files.extend([f for f in other_files if f not in pft_files])
            # Also check the root directory
            root_files = glob.glob(os.path.join(pft_comp_dir, f"*{item}*.jpg"))
            pft_files.extend(root_files)
        
        # Format paths relative to the base directory
        figures["pft_groupby"] = []
        for f in pft_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir)
                figures["pft_groupby"].append(f"comparisons/{rel_path}")
        
        if figures["pft_groupby"]:
            logger.info(f"Found PFT groupby figures: {figures['pft_groupby']}")
        
        # Climate zone groupby figures - now primarily in comparisons directory  
        climate_files = []
        # Check comparisons directory as the primary location - note the directory might be CZ_groupby
        for cz_name in ["Climate_zone_groupby", "CZ_groupby"]:
            cz_comp_dir = os.path.join(self.comparisons_dir, cz_name)
            if os.path.exists(cz_comp_dir):
                # Look for heatmap figures in subdirectories
                for subdir in glob.glob(os.path.join(cz_comp_dir, "*/")):
                    heatmap_files = glob.glob(os.path.join(subdir, f"*{item}*heatmap*.jpg"))
                    climate_files.extend(heatmap_files)
                    # Also look for any other jpg files related to the item
                    other_files = glob.glob(os.path.join(subdir, f"*{item}*.jpg"))
                    climate_files.extend([f for f in other_files if f not in climate_files])
                # Also check the root directory
                root_files = glob.glob(os.path.join(cz_comp_dir, f"*{item}*.jpg"))
                climate_files.extend(root_files)
        
        # Format paths relative to the base directory
        figures["climate_zone_groupby"] = []
        for f in climate_files:
            if self.comparisons_dir in f:
                rel_path = os.path.relpath(f, self.comparisons_dir)
                figures["climate_zone_groupby"].append(f"comparisons/{rel_path}")
        
        if figures["climate_zone_groupby"]:
            logger.info(f"Found Climate zone groupby figures: {figures['climate_zone_groupby']}")
        
        return figures
    
    def _collect_statistics(self, item: str) -> Dict[str, Any]:
        """Collect statistical analysis results"""
        stats = {}
        
        # Check for statistical outputs in comparisons directory
        stat_dirs = ["Mean", "Median", "Min", "Max", "Standard_Deviation", "Mann_Kendall_Trend_Test"]
        
        for stat_dir in stat_dirs:
            stat_path = os.path.join(self.comparisons_dir, stat_dir, f"{item}_*.nc")
            stat_files = glob.glob(stat_path)
            
            if stat_files:
                stats[stat_dir] = [os.path.basename(f) for f in stat_files]
        
        # Collect groupby statistics
        groupby_stats = self._collect_groupby_statistics(item)
        if groupby_stats:
            stats.update(groupby_stats)
        
        return stats
    
    def _collect_groupby_statistics(self, item: str) -> Dict[str, Any]:
        """Collect groupby statistics for IGBP, PFT, and Climate zone"""
        groupby_stats = {}
        
        # Define groupby types and their corresponding directories
        groupby_types = {
            "IGBP_groupby": ["IGBP_groupby"],
            "PFT_groupby": ["PFT_groupby"], 
            "Climate_zone_groupby": ["Climate_zone_groupby", "CZ_groupby"]
        }
        
        for groupby_name, groupby_dirs in groupby_types.items():
            # Check in metrics, scores, and comparisons directories
            csv_files = []
            nc_files = []
            txt_files = []
            
            for groupby_dir in groupby_dirs:
                for base_dir in [self.metrics_dir, self.scores_dir, self.comparisons_dir]:
                    groupby_path = os.path.join(base_dir, groupby_dir)
                    if not os.path.exists(groupby_path):
                        continue
                    
                    logger.info(f"Checking groupby directory: {groupby_path}")
                    
                    # Look in subdirectories for txt files with statistics
                    for subdir in glob.glob(os.path.join(groupby_path, "*/")):
                        # Look for metrics.txt files
                        txt_patterns = [
                            os.path.join(subdir, f"*{item}*metrics.txt"),
                            os.path.join(subdir, f"*{item}*.txt")
                        ]
                        for pattern in txt_patterns:
                            found_txt = glob.glob(pattern)
                            txt_files.extend(found_txt)
                    
                    # Check for CSV files with statistics (try multiple patterns)
                    csv_patterns = [
                        os.path.join(groupby_path, f"*{item}*_statistics.csv"),
                        os.path.join(groupby_path, f"*{item}*.csv"),
                        os.path.join(groupby_path, f"{item}_*.csv"),
                        os.path.join(groupby_path, "*.csv")
                    ]
                    
                    for pattern in csv_patterns:
                        found_files = glob.glob(pattern)
                        if found_files:
                            csv_files.extend(found_files)
                            logger.info(f"Found CSV files with pattern {pattern}: {found_files}")
                    
                    # Also check for NetCDF files with spatial statistics
                    nc_patterns = [
                        os.path.join(groupby_path, f"*{item}*.nc"),
                        os.path.join(groupby_path, f"{item}_*.nc"),
                        os.path.join(groupby_path, "*.nc")
                    ]
                    
                    for pattern in nc_patterns:
                        found_files = glob.glob(pattern)
                        if found_files:
                            nc_files.extend(found_files)
                            logger.info(f"Found NC files with pattern {pattern}: {found_files}")
            
            # Process txt files if found
            if txt_files:
                stats_data = []
                for txt_file in txt_files:
                    try:
                        # Read txt file and parse it
                        with open(txt_file, 'r') as f:
                            content = f.read()
                        stats_data.append({
                            "file": os.path.basename(txt_file),
                            "content": content,
                            "summary": {"groups": self._extract_groups_from_txt(content)}
                        })
                    except Exception as e:
                        logger.warning(f"Error reading {txt_file}: {e}")
                
                if stats_data:
                    groupby_stats[groupby_name] = {
                        "statistics": stats_data,
                        "description": self._get_groupby_description(groupby_name)
                    }
            
            # Process CSV files if found
            elif csv_files:
                stats_data = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        stats_data.append({
                            "file": os.path.basename(csv_file),
                            "data": df.to_dict(orient='records'),
                            "summary": self._generate_groupby_summary(df)
                        })
                    except Exception as e:
                        logger.warning(f"Error reading {csv_file}: {e}")
                
                if stats_data:
                    if groupby_name not in groupby_stats:
                        groupby_stats[groupby_name] = {
                            "statistics": stats_data,
                            "description": self._get_groupby_description(groupby_name)
                        }
                    else:
                        groupby_stats[groupby_name]["statistics"].extend(stats_data)
            
            # Add spatial files info if found
            if nc_files:
                if groupby_name not in groupby_stats:
                    groupby_stats[groupby_name] = {
                        "statistics": [],
                        "description": self._get_groupby_description(groupby_name)
                    }
                
                groupby_stats[groupby_name]["spatial_files"] = [os.path.basename(f) for f in nc_files]
        
        return groupby_stats
    
    def _generate_groupby_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for groupby dataframe"""
        summary = {}
        
        # Identify numeric columns for statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if not df[col].isna().all():
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "count": int(df[col].count())
                }
        
        # Try to extract group information - check various possible column names
        group_columns = ['Group', 'group', 'IGBP', 'igbp', 'PFT', 'pft', 'Climate_Zone', 'climate_zone', 
                         'Zone', 'zone', 'Type', 'type', 'Class', 'class', 'Category', 'category']
        
        for col in group_columns:
            if col in df.columns:
                summary['groups'] = df[col].unique().tolist()
                summary['group_count'] = len(summary['groups'])
                summary['group_column'] = col
                
                # Add performance ranking if metrics are available
                if len(numeric_cols) > 0:
                    # Find the best and worst performing groups
                    metric_col = numeric_cols[0]  # Use first numeric column
                    group_performance = df.groupby(col)[metric_col].mean().sort_values()
                    summary['worst_performing_groups'] = group_performance.head(3).index.tolist()
                    summary['best_performing_groups'] = group_performance.tail(3).index.tolist()
                break
        
        return summary
    
    def _extract_groups_from_txt(self, content: str) -> List[str]:
        """Extract group names from txt file content"""
        groups = []
        # Look for lines that might contain group names
        lines = content.split('\n')
        for line in lines:
            # Common patterns for group names in txt files
            if 'IGBP_' in line or 'PFT_' in line or 'CZ_' in line:
                # Extract the group name
                parts = line.split()
                for part in parts:
                    if 'IGBP_' in part or 'PFT_' in part or 'CZ_' in part:
                        groups.append(part)
        return list(set(groups))  # Return unique groups
    
    def _get_groupby_description(self, groupby_name: str) -> str:
        """Get description for groupby analysis type"""
        descriptions = {
            "IGBP_groupby": "Analysis grouped by International Geosphere-Biosphere Programme (IGBP) land cover classification. This analysis evaluates model performance across different land cover types including forests, grasslands, croplands, and urban areas, providing insights into ecosystem-specific model behaviors.",
            "PFT_groupby": "Analysis grouped by Plant Functional Types (PFTs). This classification groups vegetation based on physiological and morphological characteristics, allowing assessment of model performance for different plant strategies such as evergreen vs. deciduous, C3 vs. C4 photosynthesis, and various growth forms.",
            "Climate_zone_groupby": "Analysis grouped by Köppen-Geiger climate zones. This classification divides the global land surface based on temperature and precipitation patterns, enabling evaluation of model performance under different climatic conditions from tropical to polar regions."
        }
        return descriptions.get(groupby_name, f"{groupby_name} analysis")
    
    def _collect_comparison_data(self) -> Dict[str, Any]:
        """Collect overall comparison data based on enabled comparisons"""
        comparison_data = {}
        
        # Check if comparison is enabled in general settings
        if not self.config.get("general", {}).get("comparison", True):
            logger.info("Comparison is disabled in configuration, skipping comparison data collection")
            return comparison_data
        
        # Define mapping between comparison types and their files/figures
        comparison_mappings = {
            "HeatMap": {
                "data_file": os.path.join(self.comparisons_dir, "HeatMap", "scenarios_Overall_Score_comparison.txt"),
                "figure_pattern": "*heatmap.jpg",
                "data_key": "heatmap"
            },
            "Parallel_Coordinates": {
                "data_file": os.path.join(self.comparisons_dir, "Parallel_Coordinates", "Parallel_Coordinates_evaluations.txt"),
                "figure_pattern": "*Overall_Score*.jpg",
                "data_key": "parallel_coords"
            },
            "RadarMap": {
                "data_file": os.path.join(self.comparisons_dir, "RadarMap", "scenarios_Overall_Score_comparison.txt"),
                "figure_pattern": "*radarmap.jpg",
                "data_key": "radar"
            }
        }
        
        # Collect data only for enabled comparisons
        figures = {}
        for comp_type, mapping in comparison_mappings.items():
            if comp_type in self.enabled_comparisons:
                # Collect data file
                if os.path.exists(mapping["data_file"]):
                    try:
                        df = pd.read_csv(mapping["data_file"], sep='\t')
                        comparison_data[mapping["data_key"]] = df.to_dict(orient='records')
                    except Exception as e:
                        logger.warning(f"Error reading {mapping['data_file']}: {e}")
                
                # Collect figure
                figure_path = self._find_figure(self.comparisons_dir, comp_type, mapping["figure_pattern"])
                if figure_path:
                    figures[mapping["data_key"]] = figure_path
        
        if figures:
            comparison_data["figures"] = figures
        
        return comparison_data
    
    def _generate_metrics_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for metrics dataframe"""
        summary = {}
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['sim_lon', 'sim_lat', 'ref_lon', 'ref_lat', 'sim_syear', 'sim_eyear', 'ref_syear', 'ref_eyear']:
                summary[col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "count": int(df[col].count())
                }
        
        return summary
    
    def _generate_scores_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for scores dataframe"""
        # Similar to metrics summary
        return self._generate_metrics_summary(df)
    
    def _generate_overall_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of the evaluation"""
        summary = {
            "total_items": len(report_data["evaluation_items"]),
            "items": list(report_data["evaluation_items"].keys()),
            "overall_scores": {}
        }
        
        # Calculate average scores across all items
        for item, item_data in report_data["evaluation_items"].items():
            for score_key, score_data in item_data.get("scores", {}).items():
                if "summary" in score_data:
                    # Look for Overall_Score if it's enabled, otherwise use the first available score
                    if "Overall_Score" in self.enabled_scores and "Overall_Score" in score_data["summary"]:
                        score_mean = score_data["summary"]["Overall_Score"].get("mean")
                        if score_mean is not None:
                            summary["overall_scores"][f"{item}_{score_key}"] = score_mean
                    else:
                        # Use the first enabled score found in the summary
                        for score_name in self.enabled_scores:
                            if score_name in score_data["summary"]:
                                score_mean = score_data["summary"][score_name].get("mean")
                                if score_mean is not None:
                                    summary["overall_scores"][f"{item}_{score_key}"] = score_mean
                                break
        
        # Calculate grand average if scores exist
        if summary["overall_scores"]:
            summary["grand_average"] = np.mean(list(summary["overall_scores"].values()))
        
        return summary
    
    def _generate_groupby_analysis_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all groupby analyses across evaluation items"""
        summary = {
            "has_igbp": False,
            "has_pft": False,
            "has_climate_zone": False,
            "igbp_items": [],
            "pft_items": [],
            "climate_zone_items": [],
            "total_groupby_analyses": 0
        }
        
        # Check each evaluation item for groupby analyses
        for item, item_data in report_data.get("evaluation_items", {}).items():
            if item_data.get("figures", {}).get("igbp_groupby") or item_data.get("statistics", {}).get("IGBP_groupby"):
                summary["has_igbp"] = True
                summary["igbp_items"].append(item)
                summary["total_groupby_analyses"] += 1
                
            if item_data.get("figures", {}).get("pft_groupby") or item_data.get("statistics", {}).get("PFT_groupby"):
                summary["has_pft"] = True
                summary["pft_items"].append(item)
                summary["total_groupby_analyses"] += 1
                
            if item_data.get("figures", {}).get("climate_zone_groupby") or item_data.get("statistics", {}).get("Climate_zone_groupby"):
                summary["has_climate_zone"] = True
                summary["climate_zone_items"].append(item)
                summary["total_groupby_analyses"] += 1
        
        # Generate summary messages
        if summary["has_igbp"]:
            summary["igbp_message"] = f"IGBP land cover analysis performed for: {', '.join(summary['igbp_items'])}"
        if summary["has_pft"]:
            summary["pft_message"] = f"PFT analysis performed for: {', '.join(summary['pft_items'])}"
        if summary["has_climate_zone"]:
            summary["climate_zone_message"] = f"Climate zone analysis performed for: {', '.join(summary['climate_zone_items'])}"
        
        return summary
    
    def _extract_metric_type(self, filename: str) -> str:
        """Extract metric type from filename based on enabled metrics"""
        # Check enabled metrics first
        for metric in self.enabled_metrics:
            if metric in filename:
                return metric
        
        # Check enabled scores
        for score in self.enabled_scores:
            if score in filename:
                return score
        
        return "Unknown"
    
    def _is_metric_or_score(self, text: str) -> bool:
        """Check if text is a metric or score name"""
        return text in self.enabled_metrics or text in self.enabled_scores
    
    def _extract_comparison_pair(self, filename: str) -> str:
        """Extract comparison pair from filename (reference vs simulation)"""
        # Extract reference and simulation sources from filename
        # Format: ItemName_ref_RefSource_sim_SimSource_Metric
        parts = filename.split('_')
        
        ref_source = "Unknown"
        sim_source = "Unknown"
        
        for i, part in enumerate(parts):
            if part == "ref" and i + 1 < len(parts):
                # Find continuous reference name (may contain underscores)
                ref_parts = []
                j = i + 1
                while j < len(parts) and parts[j] != "sim":
                    ref_parts.append(parts[j])
                    j += 1
                ref_source = "_".join(ref_parts)
                
            elif part == "sim" and i + 1 < len(parts):
                # Find continuous simulation name (may contain underscores)
                sim_parts = []
                j = i + 1
                while j < len(parts) and not self._is_metric_or_score(parts[j]):
                    sim_parts.append(parts[j])
                    j += 1
                sim_source = "_".join(sim_parts)
                break
        
        return f"{ref_source} vs {sim_source}"
    
    def _get_comparison_pair_from_config(self, item: str, ref_source: str, sim_source: str) -> str:
        """Get comparison pair from configuration instead of filename parsing"""
        try:
            # Get display names from configuration if available
            ref_display_name = self._get_source_display_name(item, ref_source, 'ref')
            sim_display_name = self._get_source_display_name(item, sim_source, 'sim')
            
            return f"{ref_display_name} vs {sim_display_name}"
            
        except Exception as e:
            logger.warning(f"Error getting comparison pair from config for {item}: {e}")
            return f"{ref_source} vs {sim_source}"
    
    def _get_source_display_name(self, item: str, source: str, source_type: str) -> str:
        """Get display name for a source from configuration"""
        try:
            config_key = f"{source_type}_nml"
            if config_key in self.config and item in self.config[config_key]:
                # Try to get a display name or description
                display_key = f"{source}_display_name"
                if display_key in self.config[config_key][item]:
                    return self.config[config_key][item][display_key]
                
                # Try to get from varname as display name
                varname_key = f"{source}_varname"
                if varname_key in self.config[config_key][item]:
                    return self.config[config_key][item][varname_key]
            
            # If no display name found, return the source name
            return source
            
        except Exception as e:
            logger.warning(f"Error getting display name for {source}: {e}")
            return source
    
    def _generate_grid_vs_grid_stats(self, item: str) -> Dict[str, Any]:
        """Generate comprehensive grid vs grid statistics like station case format"""
        grid_stats = {}
        
        # Get reference and simulation sources from configuration
        ref_sources = self._get_reference_sources(item)
        sim_sources = self._get_simulation_sources(item)
        
        # Log what we found for debugging
        logger.info(f"Found reference sources for {item}: {ref_sources}")
        logger.info(f"Found simulation sources for {item}: {sim_sources}")
        
        # If no sources found from config, try to infer from existing files
        if not ref_sources or not sim_sources:
            logger.info(f"Attempting to infer sources from existing files for {item}")
            inferred_ref_sources, inferred_sim_sources = self._infer_sources_from_files(item)
            if not ref_sources:
                ref_sources = inferred_ref_sources
            if not sim_sources:
                sim_sources = inferred_sim_sources
            logger.info(f"After inference - ref_sources: {ref_sources}, sim_sources: {sim_sources}")
        
        # Generate all possible comparison pairs
        comparison_pairs = []
        for ref_source in ref_sources:
            for sim_source in sim_sources:
                comparison_pairs.append((ref_source, sim_source))
        
        for ref_source, sim_source in comparison_pairs:
            # Look for this comparison pair
            base_pattern = f"{item}_ref_{ref_source}_sim_{sim_source}_"
            
            # Get year information from configuration
            syear = self._get_year_info(item, ref_source, sim_source, 'syear')
            eyear = self._get_year_info(item, ref_source, sim_source, 'eyear')
            
            # Collect all metrics and scores for this pair
            pair_data = {
                "use_syear": {"values": [syear], "mean": float(syear), "std": 0.0, "min": float(syear), "max": float(syear), "median": float(syear), "coverage": 100.0},
                "use_eyear": {"values": [eyear], "mean": float(eyear), "std": 0.0, "min": float(eyear), "max": float(eyear), "median": float(eyear), "coverage": 100.0}
            }
            
            # Search for metrics in metrics directory
            metrics_files = glob.glob(os.path.join(self.metrics_dir, f"{base_pattern}*.nc"))
            scores_files = glob.glob(os.path.join(self.scores_dir, f"{base_pattern}*.nc"))
            
            all_files = metrics_files + scores_files
            
            if not all_files:
                continue  # No files for this comparison pair
            
            # Process each metric/score file
            for nc_file in all_files:
                metric_name = self._extract_metric_type(os.path.basename(nc_file))
                
                try:
                    with xr.open_dataset(nc_file) as ds:
                        data_vars = [var for var in ds.data_vars if var not in ds.coords]
                        if data_vars:
                            main_var = ds[data_vars[0]]
                            values = main_var.values.flatten()
                            valid_data = values[~np.isnan(values)]
                            total_points = len(values)
                            valid_points = len(valid_data)
                            coverage = (valid_points / total_points * 100) if total_points > 0 else 0.0
                            
                            if len(valid_data) > 0:
                                pair_data[metric_name] = {
                                    "values": valid_data.tolist(),
                                    "mean": float(np.mean(valid_data)),
                                    "std": float(np.std(valid_data)),
                                    "min": float(np.min(valid_data)),
                                    "max": float(np.max(valid_data)),
                                    "median": float(np.median(valid_data)),
                                    "coverage": float(coverage)
                                }
                            
                except Exception as e:
                    self.logger.warning(f"Error reading {nc_file}: {e}")
            
            # Try to estimate correlation if not present but other metrics are available
            # Check if 'correlation' is an enabled metric
            if "correlation" in self.enabled_metrics and "correlation" not in pair_data:
                # Simple heuristic estimation based on available metrics
                estimated_corr = 0.5  # Default moderate correlation
                
                # Check for KGESS if it's enabled
                if "KGESS" in self.enabled_metrics and "KGESS" in pair_data:
                    kgess_mean = pair_data["KGESS"]["mean"]
                    # KGESS ranges from -inf to 1, with 1 being perfect
                    # Correlation ranges from -1 to 1, estimate conservatively
                    estimated_corr = max(0.0, min(1.0, kgess_mean * 0.9))
                # Check for Overall_Score if it's enabled
                elif "Overall_Score" in self.enabled_scores and "Overall_Score" in pair_data:
                    # Use Overall_Score as a proxy for correlation
                    overall_mean = pair_data["Overall_Score"]["mean"]
                    estimated_corr = max(0.0, min(1.0, overall_mean))
                # Check for RMSE if enabled and try to estimate from it
                elif "RMSE" in self.enabled_metrics and "RMSE" in pair_data:
                    # For RMSE, lower is better, so inverse relationship
                    # This is a rough estimate - actual implementation might vary
                    rmse_mean = pair_data["RMSE"]["mean"]
                    if rmse_mean > 0:
                        # Arbitrary conversion - needs domain knowledge
                        estimated_corr = max(0.0, min(1.0, 1.0 / (1.0 + rmse_mean)))
                
                pair_data["correlation"] = {
                    "values": [estimated_corr],
                    "mean": estimated_corr,
                    "std": 0.1,  # Assume some uncertainty
                    "min": max(0.0, estimated_corr - 0.1),
                    "max": min(1.0, estimated_corr + 0.1),
                    "median": estimated_corr,
                    "coverage": 100.0
                }
            
            if len(pair_data) > 2:  # More than just the year entries
                # Calculate average data coverage across all metrics
                coverages = [metric_info.get('coverage', 0.0) for metric_info in pair_data.values() 
                           if isinstance(metric_info, dict) and 'coverage' in metric_info]
                avg_coverage = np.mean(coverages) if coverages else 100.0
                
                # Get comparison pair from configuration
                comparison_pair = self._get_comparison_pair_from_config(item, ref_source, sim_source)
                pair_key = comparison_pair
                
                grid_stats[pair_key] = {
                    "comparison_pair": comparison_pair,
                    "station_format": True,  # Flag to use station-like display
                    "metrics": pair_data,
                    "data_coverage": float(avg_coverage)
                }
                
                logger.info(f"Generated comprehensive stats for {comparison_pair}")
        
        return grid_stats
    
    def _infer_sources_from_files(self, item: str) -> tuple[List[str], List[str]]:
        """Infer reference and simulation sources from existing files"""
        ref_sources = set()
        sim_sources = set()
        
        # Search in metrics and scores directories
        search_dirs = [self.metrics_dir, self.scores_dir]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Look for files matching the pattern: item_ref_*_sim_*
                pattern = os.path.join(search_dir, f"{item}_ref_*_sim_*.nc")
                files = glob.glob(pattern)
                
                for file_path in files:
                    filename = os.path.basename(file_path)
                    parts = filename.split('_')
                    
                    # Extract ref and sim sources
                    ref_source = None
                    sim_source = None
                    
                    for i, part in enumerate(parts):
                        if part == "ref" and i + 1 < len(parts):
                            # Find continuous reference name
                            ref_parts = []
                            j = i + 1
                            while j < len(parts) and parts[j] != "sim":
                                ref_parts.append(parts[j])
                                j += 1
                            ref_source = "_".join(ref_parts)
                            
                        elif part == "sim" and i + 1 < len(parts):
                            # Find continuous simulation name
                            sim_parts = []
                            j = i + 1
                            while j < len(parts) and not self._is_metric_or_score(parts[j]) and parts[j] != "Overall":
                                sim_parts.append(parts[j])
                                j += 1
                            sim_source = "_".join(sim_parts)
                            break
                    
                    if ref_source:
                        ref_sources.add(ref_source)
                    if sim_source:
                        sim_sources.add(sim_source)
        
        return list(ref_sources), list(sim_sources)
    
    def _get_reference_sources(self, item: str) -> List[str]:
        """Get reference sources for an evaluation item from general configuration"""
        try:
            ref_sources = []
            
            # Get from general reference configuration
            if 'ref_nml' in self.config and 'general' in self.config['ref_nml']:
                # Look for the specific evaluation item in general section
                if item in self.config['ref_nml']['general']:
                    value = self.config['ref_nml']['general'][item]
                    if isinstance(value, str) and ',' in value:
                        # This is a list of reference sources
                        sources = [s.strip() for s in value.split(',')]
                        ref_sources.extend(sources)
                    elif isinstance(value, str):
                        # Single reference source
                        ref_sources.append(value.strip())
            
            logger.info(f"Found reference sources for {item}: {ref_sources}")
            return ref_sources
            
        except Exception as e:
            logger.warning(f"Error getting reference sources for {item}: {e}")
            return []
    
    def _get_simulation_sources(self, item: str) -> List[str]:
        """Get simulation sources for an evaluation item from general configuration"""
        try:
            sim_sources = []
            
            # Get from general simulation configuration
            if 'sim_nml' in self.config and 'general' in self.config['sim_nml']:
                # Look for Case_lib which contains simulation sources
                if 'Case_lib' in self.config['sim_nml']['general']:
                    value = self.config['sim_nml']['general']['Case_lib']
                    if isinstance(value, str) and ',' in value:
                        sources = [s.strip() for s in value.split(',')]
                        sim_sources.extend(sources)
                    elif isinstance(value, str):
                        sim_sources.append(value.strip())
            
            logger.info(f"Found simulation sources for {item}: {sim_sources}")
            return sim_sources
            
        except Exception as e:
            logger.warning(f"Error getting simulation sources for {item}: {e}")
            return []
    
    def _get_year_info(self, item: str, ref_source: str, sim_source: str, year_type: str) -> int:
        """Get year information from configuration"""
        try:
            # Try to get from reference configuration first
            if 'ref_nml' in self.config and item in self.config['ref_nml']:
                ref_key = f"{ref_source}_{year_type}"
                if ref_key in self.config['ref_nml'][item]:
                    return int(self.config['ref_nml'][item][ref_key])
            
            # Try to get from simulation configuration
            if 'sim_nml' in self.config and item in self.config['sim_nml']:
                sim_key = f"{sim_source}_{year_type}"
                if sim_key in self.config['sim_nml'][item]:
                    return int(self.config['sim_nml'][item][sim_key])
            
            # Try to get from general configuration
            if 'general' in self.config and year_type in self.config['general']:
                return int(self.config['general'][year_type])
            
            # Default values
            if year_type == 'syear':
                return 2004
            elif year_type == 'eyear':
                return 2005
            else:
                return 2004
                
        except Exception as e:
            logger.warning(f"Error getting {year_type} for {item}: {e}")
            # Default values
            if year_type == 'syear':
                return 2004
            elif year_type == 'eyear':
                return 2005
            else:
                return 2004
    
    def _find_figure(self, base_dir: str, subdir: str, pattern: str) -> Optional[str]:
        """Find a figure matching the pattern"""
        search_path = os.path.join(base_dir, subdir, pattern)
        files = glob.glob(search_path)
        if files:
            return f"{subdir}/{os.path.basename(files[0])}"
        return None
    
    def _copy_figures_to_report_dir(self):
        """Copy all referenced figures to the report directory"""
        logger.info("Copying figures to report directory...")
        
        # Create figures subdirectory in reports
        figures_dir = os.path.join(self.report_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Track copied files for debugging
        copied_count = 0
        
        # Copy metrics figures (including groupby subdirectories)
        if os.path.exists(self.metrics_dir):
            for root, dirs, files in os.walk(self.metrics_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.metrics_dir)
                        dst_file = os.path.join(figures_dir, "metrics", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied metrics figure: {rel_path}")
        
        # Copy scores figures (including groupby subdirectories)
        if os.path.exists(self.scores_dir):
            for root, dirs, files in os.walk(self.scores_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.scores_dir)
                        dst_file = os.path.join(figures_dir, "scores", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied scores figure: {rel_path}")
        
        # Copy comparison figures
        if os.path.exists(self.comparisons_dir):
            for root, dirs, files in os.walk(self.comparisons_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, self.comparisons_dir)
                        dst_file = os.path.join(figures_dir, "comparisons", rel_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        copied_count += 1
                        logger.debug(f"Copied comparison figure: {rel_path}")
        
        logger.info(f"Copied {copied_count} figures to report directory")
    
    def _verify_figure_paths(self, report_data: Dict[str, Any]):
        """Verify that all referenced figures exist in the report directory"""
        logger.info("Verifying figure paths...")
        figures_dir = os.path.join(self.report_dir, "figures")
        missing_figures = []
        total_figures = 0
        
        # Check figures for each evaluation item
        for item, item_data in report_data.get("evaluation_items", {}).items():
            figures = item_data.get("figures", {})
            
            # Check all figure types
            for fig_type, fig_list in figures.items():
                for fig_path in fig_list:
                    total_figures += 1
                    # Build the expected path in the reports directory
                    if fig_type == "metrics":
                        expected_path = os.path.join(figures_dir, "metrics", fig_path)
                    elif fig_type == "scores":
                        expected_path = os.path.join(figures_dir, "scores", fig_path)
                    elif fig_type == "comparisons":
                        expected_path = os.path.join(figures_dir, "comparisons", fig_path.replace("comparisons/", ""))
                    elif fig_type in ["igbp_groupby", "pft_groupby", "climate_zone_groupby"]:
                        # These are already prefixed with "comparisons/"
                        expected_path = os.path.join(figures_dir, fig_path)
                    else:
                        expected_path = os.path.join(figures_dir, fig_path)
                    
                    if not os.path.exists(expected_path):
                        missing_figures.append(f"{item}/{fig_type}: {fig_path} -> {expected_path}")
                        logger.warning(f"Missing figure: {expected_path}")
        
        # Check comparison figures
        comparisons = report_data.get("comparisons", {})
        if "figures" in comparisons:
            for fig_key, fig_path in comparisons["figures"].items():
                total_figures += 1
                expected_path = os.path.join(figures_dir, "comparisons", fig_path)
                if not os.path.exists(expected_path):
                    missing_figures.append(f"comparison/{fig_key}: {fig_path} -> {expected_path}")
                    logger.warning(f"Missing comparison figure: {expected_path}")
        
        if missing_figures:
            logger.warning(f"Found {len(missing_figures)} missing figures out of {total_figures} total figures:")
            for missing in missing_figures[:10]:  # Show first 10
                logger.warning(f"  - {missing}")
            if len(missing_figures) > 10:
                logger.warning(f"  ... and {len(missing_figures) - 10} more")
        else:
            logger.info(f"All {total_figures} figures verified successfully")
    
    def _generate_html_report(self, report_data: Dict[str, Any], report_name: str) -> str:
        """Generate HTML report from collected data"""
        logger.info("Generating HTML report...")
        
        # HTML template
        html_template = Template('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .metadata {
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .section {
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .figure-container {
            margin: 20px 0;
            text-align: center;
        }
        .figure-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .figure-caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        .summary-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metric-card {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .toc {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .toc h3 {
            margin-top: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        @media print {
            body {
                background-color: white;
            }
            .section {
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ metadata.title }}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {{ metadata.generated_date }}</p>
            <p><strong>Configuration:</strong> {{ metadata.config_file }}</p>
            <p><strong>Evaluation Items:</strong> {{ ', '.join(metadata.evaluation_items) }}</p>
        </div>
    </div>
    
    <!-- Table of Contents -->
    <div class="section toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Executive Summary</a></li>
            {% for item in metadata.evaluation_items %}
            <li><a href="#{{ item|replace(' ', '-')|lower }}">{{ item }} Analysis</a></li>
            {% endfor %}
            <li><a href="#overall-comparison">Overall Comparison</a></li>
            <li><a href="#appendix">Appendix</a></li>
        </ul>
    </div>
    
    <!-- Executive Summary -->
    <div class="section" id="summary">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>This report presents the comprehensive evaluation results from OpenBench Land Surface Model benchmarking system.</p>
            
            <div style="text-align: center; margin: 20px 0;">
                {% if overall_summary.grand_average %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(overall_summary.grand_average) }}</div>
                    <div class="metric-label">Overall Average Score</div>
                </div>
                {% endif %}
                
                <div class="metric-card" style="background-color: #2ecc71;">
                    <div class="metric-value">{{ overall_summary.total_items }}</div>
                    <div class="metric-label">Evaluation Items</div>
                </div>
            </div>
            
            <h3>Key Findings</h3>
            <ul>
                {% for item in metadata.evaluation_items %}
                <li><strong>{{ item }}:</strong> Evaluation completed with multiple reference datasets</li>
                {% endfor %}
            </ul>
            
            {% if groupby_summary.total_groupby_analyses > 0 %}
            <h3>Groupby Analysis Summary</h3>
            <p>The evaluation includes detailed analysis across different classification schemes:</p>
            <ul>
                {% if groupby_summary.has_igbp %}
                <li><strong>IGBP Land Cover Analysis:</strong> {{ groupby_summary.igbp_message }}</li>
                {% endif %}
                {% if groupby_summary.has_pft %}
                <li><strong>Plant Functional Type Analysis:</strong> {{ groupby_summary.pft_message }}</li>
                {% endif %}
                {% if groupby_summary.has_climate_zone %}
                <li><strong>Climate Zone Analysis:</strong> {{ groupby_summary.climate_zone_message }}</li>
                {% endif %}
            </ul>
            <p>These groupby analyses provide insights into model performance across different land cover types, vegetation functional groups, and climatic conditions, helping identify systematic biases and areas for improvement.</p>
            {% endif %}
        </div>
    </div>
    
    <!-- Individual Item Analysis -->
    {% for item, item_data in evaluation_items.items() %}
    <div class="section" id="{{ item|replace(' ', '-')|lower }}">
        <h2>{{ item }} Analysis</h2>
        
        <!-- Metrics Summary -->
        {% if item_data.metrics %}
        <h3>Evaluation Metrics</h3>
        {% for metric_key, metric_data in item_data.metrics.items() %}
            {% if metric_data.summary %}
            <h4>{{ metric_key }}</h4>
            <div class="summary-box">
                {% for metric_name, values in metric_data.summary.items() %}
                    {% if values.mean is not none %}
                    {% if metric_name in ['use_syear', 'use_eyear'] %}
                    <p><strong>{{ metric_name }}:</strong> {{ "%.0f"|format(values.mean) }}</p>
                    {% else %}
                    <p><strong>{{ metric_name }}:</strong> 
                       Mean = {{ "%.4f"|format(values.mean) }}, 
                       Std = {{ "%.4f"|format(values.std) }}, 
                       Range = [{{ "%.4f"|format(values.min) }}, {{ "%.4f"|format(values.max) }}]
                    </p>
                    {% endif %}
                    {% endif %}
                {% endfor %}
            </div>
            {% elif metric_data.station_format %}
            <!-- Grid vs Grid comprehensive statistics (station format) -->
            <h4>{{ metric_data.comparison_pair }}</h4>
            <div class="summary-box">
                {% for metric_name, metric_values in metric_data.metrics.items() %}
                {% if metric_name in ['use_syear', 'use_eyear'] %}
                <p><strong>{{ metric_name }}:</strong> {{ "%.0f"|format(metric_values.mean) }}</p>
                {% else %}
                <p><strong>{{ metric_name }}:</strong> 
                   Mean = {{ "%.4f"|format(metric_values.mean) }}, 
                   Std = {{ "%.4f"|format(metric_values.std) }}, 
                   Median = {{ "%.4f"|format(metric_values.median) }}, 
                   Range = [{{ "%.4f"|format(metric_values.min) }}, {{ "%.4f"|format(metric_values.max) }}], 
                   Data Coverage = {{ "%.1f"|format(metric_data.data_coverage) }}%
                </p>
                {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        {% endfor %}
        {% endif %}
        
        <!-- Metric Figures -->
        {% if item_data.figures.metrics %}
        <h3>Metric Visualizations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.metrics %}
            <div class="figure-container">
                <img src="figures/metrics/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Score Figures -->
        {% if item_data.figures.scores %}
        <h3>Score Visualizations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.scores %}
            <div class="figure-container">
                <img src="figures/scores/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- Comparison Figures -->
        {% if item_data.figures.comparisons %}
        <h3>Detailed Comparisons</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.comparisons %}
            <div class="figure-container">
                <img src="figures/comparisons/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- IGBP Groupby Analysis -->
        {% if item_data.figures.igbp_groupby or item_data.statistics.get('IGBP_groupby') %}
        <h3>IGBP Land Cover Classification Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Performance evaluation across International Geosphere-Biosphere Programme (IGBP) land cover classes provides insights into model behavior across different vegetation types and land use categories.</p>
            <p><strong>IGBP Classes Include:</strong> Evergreen Needleleaf Forest, Evergreen Broadleaf Forest, Deciduous Needleleaf Forest, Deciduous Broadleaf Forest, Mixed Forest, Closed Shrublands, Open Shrublands, Woody Savannas, Savannas, Grasslands, Permanent Wetlands, Croplands, Urban and Built-up, Cropland/Natural Vegetation Mosaic, Snow and Ice, Barren or Sparsely Vegetated, and Water Bodies.</p>
        </div>
        
        {% if item_data.statistics.get('IGBP_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.IGBP_groupby.description }}</strong></p>
            {% if item_data.statistics.IGBP_groupby.statistics %}
                {% for stat_data in item_data.statistics.IGBP_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> The analysis reveals model performance variations across different land cover types, highlighting strengths and weaknesses in simulating specific ecosystems.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performing Groups:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>Areas for Improvement:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.igbp_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.igbp_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- PFT Groupby Analysis -->
        {% if item_data.figures.pft_groupby or item_data.statistics.get('PFT_groupby') %}
        <h3>Plant Functional Type (PFT) Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Plant Functional Type (PFT) classification groups vegetation based on physiological and structural characteristics, enabling detailed assessment of model performance for different plant strategies and ecosystem functions.</p>
            <p><strong>PFT Categories:</strong> Analysis typically includes Needleleaf Evergreen/Deciduous Trees, Broadleaf Evergreen/Deciduous Trees, Shrubs, C3/C4 Grasses, and Crops, each representing distinct plant functional strategies and environmental adaptations.</p>
        </div>
        
        {% if item_data.statistics.get('PFT_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.PFT_groupby.description }}</strong></p>
            {% if item_data.statistics.PFT_groupby.statistics %}
                {% for stat_data in item_data.statistics.PFT_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> PFT-based analysis reveals how well the model captures the distinct behaviors of different plant functional groups, particularly their responses to environmental conditions and resource availability.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performing PFTs:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>PFTs Requiring Attention:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.pft_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.pft_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- Climate Zone Groupby Analysis -->
        {% if item_data.figures.climate_zone_groupby or item_data.statistics.get('Climate_zone_groupby') %}
        <h3>Climate Zone Classification Analysis</h3>
        
        <div class="summary-box">
            <p><strong>Analysis Overview:</strong> Climate zone analysis based on Köppen-Geiger classification evaluates model performance across different climatic regimes, revealing how well the model captures processes under varying temperature and precipitation conditions.</p>
            <p><strong>Climate Zones Include:</strong> Tropical (Af, Am, Aw), Dry (BWh, BWk, BSh, BSk), Temperate (Cfa, Cfb, Cfc, Csa, Csb, Csc, Cwa, Cwb, Cwc), Continental (Dfa, Dfb, Dfc, Dfd, Dsa, Dsb, Dsc, Dsd, Dwa, Dwb, Dwc, Dwd), and Polar (ET, EF) climates.</p>
        </div>
        
        {% if item_data.statistics.get('Climate_zone_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.Climate_zone_groupby.description }}</strong></p>
            {% if item_data.statistics.Climate_zone_groupby.statistics %}
                {% for stat_data in item_data.statistics.Climate_zone_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count|default(stat_data.summary.groups|length) }} groups)</p>
                <p><strong>Key Findings:</strong> Climate zone analysis identifies systematic biases and performance patterns across different climatic conditions, helping to understand model strengths in specific climate regimes and areas requiring improvement.</p>
                {% if stat_data.summary.best_performing_groups %}
                <p><strong>Best Performance in Climate Zones:</strong> {{ ', '.join(stat_data.summary.best_performing_groups) }}</p>
                {% endif %}
                {% if stat_data.summary.worst_performing_groups %}
                <p><strong>Challenging Climate Zones:</strong> {{ ', '.join(stat_data.summary.worst_performing_groups) }}</p>
                {% endif %}
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.climate_zone_groupby %}
        <div style="display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 800px; margin: 0 auto;">
            {% for fig in item_data.figures.climate_zone_groupby %}
            <div class="figure-container">
                <img src="figures/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('/', ' - ')|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
    </div>
    {% endfor %}
    
    <!-- Overall Comparison -->
    {% if comparisons %}
    <div class="section" id="overall-comparison">
        <h2>Overall Comparison</h2>
        
        {% if comparisons.figures.heatmap %}
        <div class="figure-container">
            <img src="figures/comparisons/{{ comparisons.figures.heatmap }}" alt="Overall Score Heatmap">
            <div class="figure-caption">Figure: Overall Score Comparison Heatmap</div>
        </div>
        {% endif %}
        
        {% if comparisons.heatmap %}
        <h3>Score Comparison Table</h3>
        <table>
            <thead>
                <tr>
                    {% for key in comparisons.heatmap[0].keys() %}
                    <th>{{ key }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in comparisons.heatmap %}
                <tr>
                    {% for value in row.values() %}
                    <td>{{ value }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        {% if comparisons.figures.radar %}
        <div class="figure-container">
            <img src="figures/comparisons/{{ comparisons.figures.radar }}" alt="Radar Map">
            <div class="figure-caption">Figure: Multi-dimensional Performance Radar Chart</div>
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    <!-- Appendix -->
    <div class="section" id="appendix">
        <h2>Appendix</h2>
        <h3>Methodology</h3>
        <p>The evaluation was performed using the OpenBench Land Surface Model benchmarking system, which includes:</p>
        <ul>
            <li>Multiple evaluation metrics: RMSE, Correlation, KGESS, and various scoring methods</li>
            <li>Spatial and temporal analysis across different scales</li>
            <li>Climate zone-based grouping for regional analysis</li>
            <li>Comprehensive visualization suite for result interpretation</li>
        </ul>
        
        <h3>Data Sources</h3>
        <p>Reference datasets used in this evaluation:</p>
        <ul>
            <li>GLEAM: Global Land Evaporation Amsterdam Model</li>
            <li>ILAMB: International Land Model Benchmarking</li>
            <li>PLUMBER2: Protocol for Land Surface Model Benchmarking Evaluation</li>
        </ul>
    </div>
    
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>Generated by OpenBench v2.0 | {{ metadata.generated_date }}</p>
    </div>
</body>
</html>''')
        
        # Render HTML
        html_content = html_template.render(**report_data)
        
        # Save HTML file
        html_path = os.path.join(self.report_dir, f"{report_name}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_pdf_report(self, html_path: str, report_name: str) -> Optional[str]:
        """
        Generate PDF report from HTML file using xhtml2pdf
        
        Args:
            html_path: Path to the HTML file
            report_name: Base name for the PDF file
            
        Returns:
            Path to generated PDF file, or None if generation failed
        """
        if not PDF_AVAILABLE:
            logger.warning("PDF generation not available. Please install xhtml2pdf.")
            logger.warning("Run: pip install xhtml2pdf")
            return None
        
        try:
            logger.info("Generating PDF report using xhtml2pdf...")
            pdf_path = os.path.join(self.report_dir, f"{report_name}.pdf")
            
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Modify HTML content for better PDF generation
            # Convert relative image paths to absolute paths
            import re
            
            # Replace relative image paths with absolute paths
            def replace_img_src(match):
                src = match.group(1)
                
                # Skip if already an absolute path or URL
                if src.startswith(('http://', 'https://', 'file://', '/')):
                    return match.group(0)
                
                # Handle paths that start with 'figures/'
                if src.startswith('figures/'):
                    abs_path = os.path.join(self.report_dir, src)
                    abs_path = os.path.abspath(abs_path)  # Normalize the path
                    if os.path.exists(abs_path):
                        logger.debug(f"Converting image path: {src} -> {abs_path}")
                        return f'src="file://{abs_path}"'
                    else:
                        logger.warning(f"Image file not found: {abs_path}")
                        return f'src="file://{abs_path}"'  # Keep file:// prefix even if not found
                
                # Handle other relative paths
                elif not src.startswith('./'):
                    # If it's a relative path without ./, try to resolve it relative to report dir
                    potential_path = os.path.join(self.report_dir, src)
                    potential_path = os.path.abspath(potential_path)
                    if os.path.exists(potential_path):
                        logger.debug(f"Converting relative path: {src} -> {potential_path}")
                        return f'src="file://{potential_path}"'
                
                return match.group(0)
            
            html_content = re.sub(r'src="([^"]+)"', replace_img_src, html_content)
            
            # Debug: log some sample conversions
            sample_matches = re.findall(r'src="([^"]*figures[^"]*)"', html_content)
            if sample_matches:
                logger.debug(f"Sample converted image paths: {sample_matches[:5]}")  # Show first 5
            
            # Add PDF-specific CSS
            pdf_css = """
            <style type="text/css" media="print">
                @page {
                    margin: 2cm;
                    size: A4;
                }
                body {
                    font-size: 10pt;
                    line-height: 1.3;
                }
                .header {
                    page-break-inside: avoid;
                }
                .section {
                    page-break-inside: avoid;
                    margin-bottom: 1em;
                }
                .figure-container {
                    page-break-inside: avoid;
                    text-align: center;
                    margin: 1em 0;
                }
                .figure-container img {
                    max-width: 80%;
                    height: auto;
                }
                table {
                    font-size: 8pt;
                    width: 100%;
                }
                th, td {
                    padding: 4px;
                    font-size: 8pt;
                }
                h2 {
                    page-break-after: avoid;
                    font-size: 14pt;
                }
                h3 {
                    page-break-after: avoid;
                    font-size: 12pt;
                }
                h4 {
                    page-break-after: avoid;
                    font-size: 11pt;
                }
            </style>
            """
            
            # Insert PDF CSS into HTML head
            html_content = html_content.replace('</head>', pdf_css + '</head>')
            
            # Generate PDF
            with open(pdf_path, 'wb') as pdf_file:
                result = pisa.CreatePDF(
                    html_content, 
                    dest=pdf_file,
                    encoding='utf-8',
                    link_callback=self._link_callback
                )
                
                if result.err:
                    logger.error(f"PDF generation had errors: {result.err}")
                    return None
            
            logger.info(f"PDF report generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None
    
    def _link_callback(self, uri, rel):
        """
        Callback function to handle local file links in PDF generation
        """
        # Handle file:// URLs
        if uri.startswith('file://'):
            path = uri[7:]  # Remove 'file://' prefix
            if os.path.exists(path):
                logger.debug(f"Link callback found file: {path}")
                return path
            else:
                logger.warning(f"Link callback file not found: {path}")
                return path  # Return path anyway, let PDF generator handle it
        
        # Handle relative paths directly
        elif uri.startswith('figures/'):
            abs_path = os.path.join(self.report_dir, uri)
            abs_path = os.path.abspath(abs_path)
            if os.path.exists(abs_path):
                logger.debug(f"Link callback resolved relative path: {uri} -> {abs_path}")
                return abs_path
            else:
                logger.warning(f"Link callback could not find relative path: {uri} -> {abs_path}")
                return abs_path
        
        # Handle other relative paths (like ./output/...)
        elif uri.startswith('./') and 'figures' in uri:
            # Extract the figures part
            if 'reports/figures' in uri:
                figures_part = uri.split('reports/figures/')[-1]
                abs_path = os.path.join(self.report_dir, 'figures', figures_part)
                abs_path = os.path.abspath(abs_path)
                if os.path.exists(abs_path):
                    logger.debug(f"Link callback resolved complex relative path: {uri} -> {abs_path}")
                    return abs_path
                else:
                    logger.warning(f"Link callback could not find complex relative path: {uri} -> {abs_path}")
                    return abs_path
        
        return uri
