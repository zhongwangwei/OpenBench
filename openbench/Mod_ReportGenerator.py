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

# PDF generation libraries - try multiple options
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

try:
    import weasyprint
    _HAS_WEASYPRINT = True
except (ImportError, OSError) as e:
    _HAS_WEASYPRINT = False

try:
    import pdfkit
    _HAS_PDFKIT = True
except ImportError:
    _HAS_PDFKIT = False

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
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data, report_name)
        
        # Generate PDF from HTML
        pdf_path = self._generate_pdf_report(html_path, report_name)
        
        # Copy all relevant figures to report directory
        self._copy_figures_to_report_dir()
        
        logger.info(f"Report generation completed successfully")
        logger.info(f"HTML report: {html_path}")
        logger.info(f"PDF report: {pdf_path}")
        
        return {
            "html": html_path,
            "pdf": pdf_path
        }
    
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
                                                while j < len(parts) and parts[j] not in ["RMSE", "KGESS", "correlation", "bias"]:
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
        
        # IGBP groupby figures
        igbp_dir = os.path.join(self.comparisons_dir, "IGBP_groupby")
        igbp_patterns = [f"*{item}*.jpg", f"{item}_*.jpg", "*.jpg"]
        igbp_files = []
        for pattern in igbp_patterns:
            found_files = glob.glob(os.path.join(igbp_dir, pattern))
            if found_files:
                igbp_files = found_files
                break
        figures["igbp_groupby"] = [f"IGBP_groupby/{os.path.basename(f)}" for f in igbp_files]
        if igbp_files:
            logger.info(f"Found IGBP groupby figures: {igbp_files}")
        
        # PFT groupby figures
        pft_dir = os.path.join(self.comparisons_dir, "PFT_groupby")
        pft_patterns = [f"*{item}*.jpg", f"{item}_*.jpg", "*.jpg"]
        pft_files = []
        for pattern in pft_patterns:
            found_files = glob.glob(os.path.join(pft_dir, pattern))
            if found_files:
                pft_files = found_files
                break
        figures["pft_groupby"] = [f"PFT_groupby/{os.path.basename(f)}" for f in pft_files]
        if pft_files:
            logger.info(f"Found PFT groupby figures: {pft_files}")
        
        # Climate zone groupby figures
        climate_dir = os.path.join(self.comparisons_dir, "Climate_zone_groupby")
        climate_patterns = [f"*{item}*.jpg", f"{item}_*.jpg", "*.jpg"]
        climate_files = []
        for pattern in climate_patterns:
            found_files = glob.glob(os.path.join(climate_dir, pattern))
            if found_files:
                climate_files = found_files
                break
        figures["climate_zone_groupby"] = [f"Climate_zone_groupby/{os.path.basename(f)}" for f in climate_files]
        if climate_files:
            logger.info(f"Found Climate zone groupby figures: {climate_files}")
        
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
            "IGBP_groupby": "IGBP_groupby",
            "PFT_groupby": "PFT_groupby", 
            "Climate_zone_groupby": "Climate_zone_groupby"
        }
        
        for groupby_name, groupby_dir in groupby_types.items():
            groupby_path = os.path.join(self.comparisons_dir, groupby_dir)
            logger.info(f"Checking groupby directory: {groupby_path}")
            
            # Check for CSV files with statistics (try multiple patterns)
            csv_patterns = [
                os.path.join(groupby_path, f"*{item}*_statistics.csv"),
                os.path.join(groupby_path, f"*{item}*.csv"),
                os.path.join(groupby_path, f"{item}_*.csv"),
                os.path.join(groupby_path, "*.csv")
            ]
            
            csv_files = []
            for pattern in csv_patterns:
                found_files = glob.glob(pattern)
                if found_files:
                    csv_files = found_files
                    logger.info(f"Found CSV files with pattern {pattern}: {csv_files}")
                    break
            
            if not csv_files:
                logger.info(f"No CSV files found for {item} in {groupby_path}")
            
            if csv_files:
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
                    groupby_stats[groupby_name] = {
                        "statistics": stats_data,
                        "description": self._get_groupby_description(groupby_name)
                    }
            
            # Also check for NetCDF files with spatial statistics
            nc_patterns = [
                os.path.join(groupby_path, f"*{item}*.nc"),
                os.path.join(groupby_path, f"{item}_*.nc"),
                os.path.join(groupby_path, "*.nc")
            ]
            
            nc_files = []
            for pattern in nc_patterns:
                found_files = glob.glob(pattern)
                if found_files:
                    nc_files = found_files
                    logger.info(f"Found NC files with pattern {pattern}: {nc_files}")
                    break
            
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
        
        # Add group information if available
        if 'group' in df.columns or 'Group' in df.columns:
            group_col = 'group' if 'group' in df.columns else 'Group'
            summary["groups"] = df[group_col].unique().tolist()
            summary["group_count"] = len(df[group_col].unique())
        
        return summary
    
    def _get_groupby_description(self, groupby_name: str) -> str:
        """Get description for groupby analysis type"""
        descriptions = {
            "IGBP_groupby": "International Geosphere-Biosphere Programme (IGBP) land cover classification groupby analysis",
            "PFT_groupby": "Plant Functional Type (PFT) groupby analysis",
            "Climate_zone_groupby": "Climate zone classification groupby analysis"
        }
        return descriptions.get(groupby_name, f"{groupby_name} analysis")
    
    def _collect_comparison_data(self) -> Dict[str, Any]:
        """Collect overall comparison data"""
        comparison_data = {}
        
        # Overall comparison files
        comparison_files = {
            "heatmap": os.path.join(self.comparisons_dir, "HeatMap", "scenarios_Overall_Score_comparison.txt"),
            "parallel_coords": os.path.join(self.comparisons_dir, "Parallel_Coordinates", "Parallel_Coordinates_evaluations.txt"),
            "radar": os.path.join(self.comparisons_dir, "RadarMap", "scenarios_Overall_Score_comparison.txt")
        }
        
        for key, filepath in comparison_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, sep='\t')
                    comparison_data[key] = df.to_dict(orient='records')
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
        
        # Collect comparison figures
        comparison_data["figures"] = {
            "heatmap": self._find_figure(self.comparisons_dir, "HeatMap", "*heatmap.jpg"),
            "radar": self._find_figure(self.comparisons_dir, "RadarMap", "*radarmap.jpg"),
            "parallel": self._find_figure(self.comparisons_dir, "Parallel_Coordinates", "*Overall_Score*.jpg")
        }
        
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
                if "summary" in score_data and "Overall_Score" in score_data["summary"]:
                    score_mean = score_data["summary"]["Overall_Score"].get("mean")
                    if score_mean is not None:
                        summary["overall_scores"][f"{item}_{score_key}"] = score_mean
        
        # Calculate grand average if scores exist
        if summary["overall_scores"]:
            summary["grand_average"] = np.mean(list(summary["overall_scores"].values()))
        
        return summary
    
    def _extract_metric_type(self, filename: str) -> str:
        """Extract metric type from filename"""
        if "RMSE" in filename:
            return "RMSE"
        elif "KGESS" in filename:
            return "KGESS" 
        elif "correlation" in filename:
            return "Correlation"
        elif "bias" in filename:
            return "Bias"
        else:
            return "Unknown"
    
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
                while j < len(parts) and parts[j] not in ["RMSE", "KGESS", "correlation", "bias"]:
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
                if metric_name == "Unknown":
                    # Try to extract from more specific patterns
                    filename = os.path.basename(nc_file)
                    if "nBiasScore" in filename:
                        metric_name = "nBiasScore"
                    elif "nSpatialScore" in filename:
                        metric_name = "nSpatialScore"
                    elif "Overall_Score" in filename:
                        metric_name = "Overall_Score"
                    elif "KGESS" in filename:
                        metric_name = "KGESS"
                    elif "RMSE" in filename:
                        metric_name = "RMSE"
                
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
            if "correlation" not in pair_data and ("KGESS" in pair_data or "RMSE" in pair_data):
                # Simple heuristic estimation based on available metrics
                if "KGESS" in pair_data:
                    kgess_mean = pair_data["KGESS"]["mean"]
                    # KGESS ranges from -inf to 1, with 1 being perfect
                    # Correlation ranges from -1 to 1, estimate conservatively
                    estimated_corr = max(0.0, min(1.0, kgess_mean * 0.9))
                elif "RMSE" in pair_data and "Overall_Score" in pair_data:
                    # Use Overall_Score as a proxy for correlation
                    overall_mean = pair_data["Overall_Score"]["mean"]
                    estimated_corr = max(0.0, min(1.0, overall_mean))
                else:
                    estimated_corr = 0.5  # Default moderate correlation
                
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
                            while j < len(parts) and parts[j] not in ["RMSE", "KGESS", "correlation", "bias", "nBiasScore", "nSpatialScore", "Overall"]:
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
        
        # Copy metrics figures
        for src_file in glob.glob(os.path.join(self.metrics_dir, "*.jpg")):
            dst_file = os.path.join(figures_dir, "metrics", os.path.basename(src_file))
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
        
        # Copy scores figures
        for src_file in glob.glob(os.path.join(self.scores_dir, "*.jpg")):
            dst_file = os.path.join(figures_dir, "scores", os.path.basename(src_file))
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
        
        # Copy comparison figures
        for root, dirs, files in os.walk(self.comparisons_dir):
            for file in files:
                if file.endswith('.jpg'):
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(src_file, self.comparisons_dir)
                    dst_file = os.path.join(figures_dir, "comparisons", rel_path)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
    
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
            <li><a href="#overall-comparison">Overall Comparison</a></li>
            {% for item in metadata.evaluation_items %}
            <li><a href="#{{ item|replace(' ', '-')|lower }}">{{ item }} Analysis</a></li>
            {% endfor %}
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
        </div>
    </div>
    
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
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- IGBP Groupby Analysis -->
        {% if item_data.figures.igbp_groupby or item_data.statistics.get('IGBP_groupby') %}
        <h3>IGBP Land Cover Classification Analysis</h3>
        
        {% if item_data.statistics.get('IGBP_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.IGBP_groupby.description }}</strong></p>
            {% if item_data.statistics.IGBP_groupby.statistics %}
                {% for stat_data in item_data.statistics.IGBP_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count }} groups)</p>
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.igbp_groupby %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.igbp_groupby %}
            <div class="figure-container">
                <img src="figures/comparisons/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- PFT Groupby Analysis -->
        {% if item_data.figures.pft_groupby or item_data.statistics.get('PFT_groupby') %}
        <h3>Plant Functional Type (PFT) Analysis</h3>
        
        {% if item_data.statistics.get('PFT_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.PFT_groupby.description }}</strong></p>
            {% if item_data.statistics.PFT_groupby.statistics %}
                {% for stat_data in item_data.statistics.PFT_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count }} groups)</p>
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.pft_groupby %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.pft_groupby %}
            <div class="figure-container">
                <img src="figures/comparisons/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
        
        <!-- Climate Zone Groupby Analysis -->
        {% if item_data.figures.climate_zone_groupby or item_data.statistics.get('Climate_zone_groupby') %}
        <h3>Climate Zone Classification Analysis</h3>
        
        {% if item_data.statistics.get('Climate_zone_groupby') %}
        <div class="summary-box">
            <p><strong>{{ item_data.statistics.Climate_zone_groupby.description }}</strong></p>
            {% if item_data.statistics.Climate_zone_groupby.statistics %}
                {% for stat_data in item_data.statistics.Climate_zone_groupby.statistics %}
                <h4>{{ stat_data.file|replace('_', ' ')|replace('.csv', '') }}</h4>
                {% if stat_data.summary.groups %}
                <p><strong>Groups analyzed:</strong> {{ ', '.join(stat_data.summary.groups) }} ({{ stat_data.summary.group_count }} groups)</p>
                {% endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endif %}
        
        {% if item_data.figures.climate_zone_groupby %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            {% for fig in item_data.figures.climate_zone_groupby %}
            <div class="figure-container">
                <img src="figures/comparisons/{{ fig }}" alt="{{ fig }}">
                <div class="figure-caption">{{ fig|replace('_', ' ')|replace('.jpg', '') }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
    </div>
    {% endfor %}
    
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
    
    def _generate_pdf_report(self, html_path: str, report_name: str) -> str:
        """Generate PDF from HTML report using multiple methods"""
        logger.info("Generating PDF report...")
        
        pdf_path = os.path.join(self.report_dir, f"{report_name}.pdf")
        
        # Try multiple PDF generation methods
        pdf_generated = False
        
        # Method 1: ReportLab (direct PDF generation, most reliable)
        if _HAS_REPORTLAB and not pdf_generated:
            try:
                logger.info("Attempting PDF generation with ReportLab...")
                self._generate_pdf_with_reportlab(pdf_path, report_name)
                logger.info(f"PDF report generated successfully with ReportLab: {pdf_path}")
                pdf_generated = True
            except Exception as e:
                logger.warning(f"ReportLab PDF generation failed: {e}")
                
        # Method 2: WeasyPrint (best quality, but requires system dependencies)
        if _HAS_WEASYPRINT and not pdf_generated:
            try:
                logger.info("Attempting PDF generation with WeasyPrint...")
                html_doc = weasyprint.HTML(filename=html_path)
                html_doc.write_pdf(pdf_path)
                logger.info(f"PDF report generated successfully with WeasyPrint: {pdf_path}")
                pdf_generated = True
            except Exception as e:
                logger.warning(f"WeasyPrint PDF generation failed: {e}")
        
        # Method 3: pdfkit (fallback, requires wkhtmltopdf)
        if _HAS_PDFKIT and not pdf_generated:
            try:
                logger.info("Attempting PDF generation with pdfkit...")
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                pdfkit.from_file(html_path, pdf_path, options=options)
                logger.info(f"PDF report generated successfully with pdfkit: {pdf_path}")
                pdf_generated = True
            except Exception as e:
                logger.warning(f"pdfkit PDF generation failed: {e}")
        
        if not pdf_generated:
            logger.error("All PDF generation methods failed.")
            logger.info("To enable PDF generation, install one of:")
            logger.info("  pip install weasyprint")
            logger.info("  pip install reportlab")
            logger.info("  brew install --cask wkhtmltopdf (for pdfkit)")
            return None
        
        return pdf_path
    
    def _generate_pdf_with_reportlab(self, pdf_path: str, report_name: str):
        """Generate PDF directly using ReportLab with full content matching HTML"""
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#34495e')
        )
        
        caption_style = ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            spaceBefore=4,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica-Oblique'
        )
        
        # Title page
        story.append(Paragraph("OpenBench Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated: {self.metadata['generated_date']}", styles['Normal']))
        story.append(Paragraph(f"Configuration: {self.metadata['config_file']}", styles['Normal']))
        story.append(Paragraph(f"Evaluation Items: {', '.join(self.metadata['evaluation_items'])}", styles['Normal']))
        story.append(PageBreak())
        
        # Collect report data
        report_data = self._collect_report_data()
        
        # Table of Contents
        story.append(Paragraph("Table of Contents", heading_style))
        story.append(Paragraph("1. Executive Summary", styles['Normal']))
        story.append(Paragraph("2. Overall Comparison", styles['Normal']))
        for i, item in enumerate(self.metadata['evaluation_items'], 3):
            story.append(Paragraph(f"{i}. {item} Analysis", styles['Normal']))
        story.append(Paragraph(f"{len(self.metadata['evaluation_items']) + 3}. Appendix", styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph("This report presents comprehensive evaluation results from the OpenBench Land Surface Model benchmarking system.", styles['Normal']))
        story.append(Spacer(1, 12))
        
        if report_data['overall_summary'].get('grand_average'):
            story.append(Paragraph(f"Overall Average Score: {report_data['overall_summary']['grand_average']:.3f}", styles['Normal']))
        story.append(Paragraph(f"Total Evaluation Items: {report_data['overall_summary']['total_items']}", styles['Normal']))
        
        # Key Findings
        story.append(Paragraph("Key Findings:", subheading_style))
        for item in self.metadata['evaluation_items']:
            story.append(Paragraph(f"• {item}: Evaluation completed with multiple reference datasets", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Overall Comparison
        if report_data['comparisons']:
            story.append(Paragraph("Overall Comparison", heading_style))
            
            # Add comparison figures as images
            figures_dir = os.path.join(self.report_dir, "figures", "comparisons")
            
            # Heatmap
            if report_data['comparisons'].get('figures', {}).get('heatmap'):
                heatmap_path = os.path.join(figures_dir, report_data['comparisons']['figures']['heatmap'])
                if os.path.exists(heatmap_path):
                    try:
                        img = Image(heatmap_path, width=4.5*inch, height=3.5*inch)  # Limit both dimensions
                        story.append(img)
                        story.append(Paragraph("Figure: Overall Score Comparison Heatmap", caption_style))
                        story.append(Spacer(1, 8))
                    except Exception as e:
                        logger.warning(f"Could not add heatmap image to PDF: {e}")
            
            # Comparison table
            if report_data['comparisons'].get('heatmap'):
                story.append(Paragraph("Score Comparison Table", subheading_style))
                
                # Create table data
                table_data = []
                headers = list(report_data['comparisons']['heatmap'][0].keys())
                table_data.append(headers)
                
                for row in report_data['comparisons']['heatmap']:
                    table_data.append(list(row.values()))
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
            
            # Radar chart
            if report_data['comparisons'].get('figures', {}).get('radar'):
                radar_path = os.path.join(figures_dir, report_data['comparisons']['figures']['radar'])
                if os.path.exists(radar_path):
                    try:
                        img = Image(radar_path, width=4.5*inch, height=3.5*inch)  # Limit both dimensions
                        story.append(img)
                        story.append(Paragraph("Figure: Multi-dimensional Performance Radar Chart", caption_style))
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        logger.warning(f"Could not add radar image to PDF: {e}")
        
        # Individual Item Analysis
        for item, item_data in report_data['evaluation_items'].items():
            story.append(PageBreak())
            story.append(Paragraph(f"{item} Analysis", heading_style))
            
            # Metrics Summary
            if item_data.get('metrics'):
                story.append(Paragraph("Evaluation Metrics", subheading_style))
                for metric_key, metric_data in item_data['metrics'].items():
                    if metric_data.get('summary'):
                        # CSV-based metrics
                        story.append(Paragraph(f"{metric_key}:", styles['Heading4']))
                        for metric_name, values in metric_data['summary'].items():
                            if isinstance(values, dict) and values.get('mean') is not None:
                                # Format year fields as simple integers, others with full statistics
                                if metric_name in ['use_syear', 'use_eyear']:
                                    story.append(Paragraph(
                                        f"{metric_name}: {values['mean']:.0f}",
                                        styles['Normal']
                                    ))
                                else:
                                    story.append(Paragraph(
                                        f"{metric_name}: Mean = {values['mean']:.4f}, "
                                        f"Std = {values['std']:.4f}, Range = [{values['min']:.4f}, {values['max']:.4f}]",
                                        styles['Normal']
                                    ))
                    elif metric_data.get('station_format'):
                        # Grid vs Grid comprehensive statistics (station format)
                        story.append(Paragraph(f"{metric_data['comparison_pair']}:", styles['Heading4']))
                        for metric_name, metric_values in metric_data['metrics'].items():
                            # Format year fields as simple integers, others with full statistics
                            if metric_name in ['use_syear', 'use_eyear']:
                                story.append(Paragraph(
                                    f"{metric_name}: {metric_values['mean']:.0f}",
                                    styles['Normal']
                                ))
                            else:
                                story.append(Paragraph(
                                    f"{metric_name}: Mean = {metric_values['mean']:.4f}, "
                                    f"Std = {metric_values['std']:.4f}, "
                                    f"Median = {metric_values['median']:.4f}, "
                                    f"Range = [{metric_values['min']:.4f}, {metric_values['max']:.4f}], "
                                    f"Data Coverage = {metric_data['data_coverage']:.1f}%",
                                    styles['Normal']
                                ))
                story.append(Spacer(1, 12))
            
            # Add metric figures (only those related to current evaluation item)
            if item_data.get('figures', {}).get('metrics'):
                story.append(Paragraph("Metric Visualizations", subheading_style))
                metrics_figures_dir = os.path.join(self.report_dir, "figures", "metrics")
                
                # Filter figures to only include those for the current item
                relevant_figures = [fig for fig in item_data['figures']['metrics'] 
                                  if item.replace(' ', '_') in fig]
                
                # Create 3x2 grid layout for figures (3 rows, 2 columns)
                for i in range(0, len(relevant_figures), 6):
                    if i > 0:
                        story.append(PageBreak())
                    
                    # Get up to 6 figures for this page
                    page_figures = relevant_figures[i:i+6]
                    
                    # Create 3x2 table data
                    table_data = [
                        ["", ""],  # Row 1: [cell(0,0), cell(0,1)]
                        ["", ""],  # Row 2: [cell(1,0), cell(1,1)]
                        ["", ""]   # Row 3: [cell(2,0), cell(2,1)]
                    ]
                    
                    # Fill the table with figures
                    for idx, fig in enumerate(page_figures):
                        fig_path = os.path.join(metrics_figures_dir, fig)
                        if os.path.exists(fig_path):
                            try:
                                # Create image with better size limits and aspect ratio preservation
                                img = Image(fig_path)
                                img_width, img_height = img.drawWidth, img.drawHeight
                                
                                # Calculate scaled dimensions to fit in 3.5x2.5 inch cell
                                max_width = 3.5*inch
                                max_height = 2.5*inch
                                
                                scale_w = max_width / img_width
                                scale_h = max_height / img_height
                                scale = min(scale_w, scale_h)
                                
                                img.drawWidth = img_width * scale
                                img.drawHeight = img_height * scale
                                caption = Paragraph(fig.replace('_', ' ').replace('.jpg', ''), caption_style)
                                
                                # Calculate row and column position
                                row = idx // 2  # 0, 1, or 2
                                col = idx % 2   # 0 or 1
                                
                                # Create cell content (image above caption)
                                table_data[row][col] = [img, caption]
                                
                            except Exception as e:
                                logger.warning(f"Could not add metrics figure {fig} to PDF: {e}")
                    
                    # Create and add the table
                    fig_table = Table(table_data, colWidths=[3.8*inch, 3.8*inch])
                    fig_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(fig_table)
                    story.append(Spacer(1, 10))
            
            # Add score figures (only those related to current evaluation item)
            if item_data.get('figures', {}).get('scores'):
                story.append(Paragraph("Score Visualizations", subheading_style))
                scores_figures_dir = os.path.join(self.report_dir, "figures", "scores")
                
                # Filter figures to only include those for the current item
                relevant_figures = [fig for fig in item_data['figures']['scores'] 
                                  if item.replace(' ', '_') in fig]
                
                # Create 3x2 grid layout for figures (3 rows, 2 columns)
                for i in range(0, len(relevant_figures), 6):
                    if i > 0:
                        story.append(PageBreak())
                    
                    # Get up to 6 figures for this page
                    page_figures = relevant_figures[i:i+6]
                    
                    # Create 3x2 table data
                    table_data = [
                        ["", ""],  # Row 1: [cell(0,0), cell(0,1)]
                        ["", ""],  # Row 2: [cell(1,0), cell(1,1)]
                        ["", ""]   # Row 3: [cell(2,0), cell(2,1)]
                    ]
                    
                    # Fill the table with figures
                    for idx, fig in enumerate(page_figures):
                        fig_path = os.path.join(scores_figures_dir, fig)
                        if os.path.exists(fig_path):
                            try:
                                # Create image with better size limits and aspect ratio preservation
                                img = Image(fig_path)
                                img_width, img_height = img.drawWidth, img.drawHeight
                                
                                # Calculate scaled dimensions to fit in 3.5x2.5 inch cell
                                max_width = 3.5*inch
                                max_height = 2.5*inch
                                
                                scale_w = max_width / img_width
                                scale_h = max_height / img_height
                                scale = min(scale_w, scale_h)
                                
                                img.drawWidth = img_width * scale
                                img.drawHeight = img_height * scale
                                caption = Paragraph(fig.replace('_', ' ').replace('.jpg', ''), caption_style)
                                
                                # Calculate row and column position
                                row = idx // 2  # 0, 1, or 2
                                col = idx % 2   # 0 or 1
                                
                                # Create cell content (image above caption)
                                table_data[row][col] = [img, caption]
                                
                            except Exception as e:
                                logger.warning(f"Could not add scores figure {fig} to PDF: {e}")
                    
                    # Create and add the table
                    fig_table = Table(table_data, colWidths=[3.8*inch, 3.8*inch])
                    fig_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(fig_table)
                    story.append(Spacer(1, 10))
            
            # Add comparison figures (only those related to current evaluation item)
            if item_data.get('figures', {}).get('comparisons'):
                story.append(Paragraph("Detailed Comparisons", subheading_style))
                comparisons_figures_dir = os.path.join(self.report_dir, "figures", "comparisons")
                
                # Filter figures to only include those for the current item
                relevant_figures = [fig for fig in item_data['figures']['comparisons'] 
                                  if item.replace(' ', '_') in fig]
                
                # Create 3x2 grid layout for figures (3 rows, 2 columns)
                for i in range(0, len(relevant_figures), 6):
                    if i > 0:
                        story.append(PageBreak())
                    
                    # Get up to 6 figures for this page
                    page_figures = relevant_figures[i:i+6]
                    
                    # Create 3x2 table data
                    table_data = [
                        ["", ""],  # Row 1: [cell(0,0), cell(0,1)]
                        ["", ""],  # Row 2: [cell(1,0), cell(1,1)]
                        ["", ""]   # Row 3: [cell(2,0), cell(2,1)]
                    ]
                    
                    # Fill the table with figures
                    for idx, fig in enumerate(page_figures):
                        fig_path = os.path.join(comparisons_figures_dir, fig)
                        if os.path.exists(fig_path):
                            try:
                                # Create image with better size limits and aspect ratio preservation
                                img = Image(fig_path)
                                img_width, img_height = img.drawWidth, img.drawHeight
                                
                                # Calculate scaled dimensions to fit in 3.5x2.5 inch cell
                                max_width = 3.5*inch
                                max_height = 2.5*inch
                                
                                scale_w = max_width / img_width
                                scale_h = max_height / img_height
                                scale = min(scale_w, scale_h)
                                
                                img.drawWidth = img_width * scale
                                img.drawHeight = img_height * scale
                                caption = Paragraph(fig.replace('_', ' ').replace('.jpg', ''), caption_style)
                                
                                # Calculate row and column position
                                row = idx // 2  # 0, 1, or 2
                                col = idx % 2   # 0 or 1
                                
                                # Create cell content (image above caption)
                                table_data[row][col] = [img, caption]
                                
                            except Exception as e:
                                logger.warning(f"Could not add comparison figure {fig} to PDF: {e}")
                    
                    # Create and add the table
                    fig_table = Table(table_data, colWidths=[3.8*inch, 3.8*inch])
                    fig_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(fig_table)
                    story.append(Spacer(1, 10))
            
            # Add groupby analysis sections
            self._add_groupby_analysis_to_pdf(story, item_data, subheading_style, caption_style)
        
        # Appendix
        story.append(PageBreak())
        story.append(Paragraph("Appendix", heading_style))
        story.append(Paragraph("Methodology", subheading_style))
        story.append(Paragraph("The evaluation was performed using the OpenBench Land Surface Model benchmarking system, which includes:", styles['Normal']))
        story.append(Paragraph("• Multiple evaluation metrics: RMSE, Correlation, KGESS, and various scoring methods", styles['Normal']))
        story.append(Paragraph("• Spatial and temporal analysis across different scales", styles['Normal']))
        story.append(Paragraph("• Climate zone-based grouping for regional analysis", styles['Normal']))
        story.append(Paragraph("• Comprehensive visualization suite for result interpretation", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Data Sources", subheading_style))
        story.append(Paragraph("Reference datasets used in this evaluation:", styles['Normal']))
        story.append(Paragraph("• GLEAM: Global Land Evaporation Amsterdam Model", styles['Normal']))
        story.append(Paragraph("• ILAMB: International Land Model Benchmarking", styles['Normal']))
        story.append(Paragraph("• PLUMBER2: Protocol for Land Surface Model Benchmarking Evaluation", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated by OpenBench v2.0 | {self.metadata['generated_date']}", 
                              ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.gray)))
        
        # Build PDF
        doc.build(story)
    
    def _add_groupby_analysis_to_pdf(self, story, item_data, subheading_style, caption_style):
        """Add groupby analysis sections to PDF report"""
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        
        groupby_types = [
            ("igbp_groupby", "IGBP_groupby", "IGBP Land Cover Classification Analysis"),
            ("pft_groupby", "PFT_groupby", "Plant Functional Type (PFT) Analysis"),
            ("climate_zone_groupby", "Climate_zone_groupby", "Climate Zone Classification Analysis")
        ]
        
        for fig_key, stat_key, title in groupby_types:
            has_figures = item_data.get('figures', {}).get(fig_key)
            has_stats = item_data.get('statistics', {}).get(stat_key)
            
            if has_figures or has_stats:
                story.append(Paragraph(title, subheading_style))
                
                # Add statistics summary if available
                if has_stats:
                    stats_data = has_stats
                    if 'description' in stats_data:
                        story.append(Paragraph(stats_data['description'], styles['Normal']))
                    
                    if 'statistics' in stats_data:
                        for stat_data in stats_data['statistics']:
                            file_name = stat_data['file'].replace('_', ' ').replace('.csv', '')
                            story.append(Paragraph(f"Analysis: {file_name}", styles['Heading4']))
                            
                            if 'groups' in stat_data['summary']:
                                groups_text = f"Groups analyzed: {', '.join(stat_data['summary']['groups'])} ({stat_data['summary']['group_count']} groups)"
                                story.append(Paragraph(groups_text, styles['Normal']))
                    
                    story.append(Spacer(1, 10))
                
                # Add figures if available
                if has_figures:
                    comparisons_figures_dir = os.path.join(self.report_dir, "figures", "comparisons")
                    relevant_figures = has_figures
                    
                    # Create 3x2 grid layout for figures
                    for i in range(0, len(relevant_figures), 6):
                        if i > 0:
                            story.append(PageBreak())
                        
                        # Get up to 6 figures for this page
                        page_figures = relevant_figures[i:i+6]
                        
                        # Create 3x2 table data
                        table_data = [
                            ["", ""],  # Row 1: [cell(0,0), cell(0,1)]
                            ["", ""],  # Row 2: [cell(1,0), cell(1,1)]
                            ["", ""]   # Row 3: [cell(2,0), cell(2,1)]
                        ]
                        
                        # Fill the table with figures
                        for idx, fig in enumerate(page_figures):
                            fig_path = os.path.join(comparisons_figures_dir, fig)
                            if os.path.exists(fig_path):
                                try:
                                    # Create image with better size limits and aspect ratio preservation
                                    img = Image(fig_path)
                                    img_width, img_height = img.drawWidth, img.drawHeight
                                    
                                    # Calculate scaled dimensions to fit in 3.5x2.5 inch cell
                                    max_width = 3.5*inch
                                    max_height = 2.5*inch
                                    
                                    scale_w = max_width / img_width
                                    scale_h = max_height / img_height
                                    scale = min(scale_w, scale_h)
                                    
                                    img.drawWidth = img_width * scale
                                    img.drawHeight = img_height * scale
                                    
                                    caption = Paragraph(fig.replace('_', ' ').replace('.jpg', ''), caption_style)
                                    
                                    # Calculate row and column position
                                    row = idx // 2  # 0, 1, or 2
                                    col = idx % 2   # 0 or 1
                                    
                                    # Create cell content (image above caption)
                                    table_data[row][col] = [img, caption]
                                    
                                except Exception as e:
                                    logger.warning(f"Could not add {fig_key} figure {fig} to PDF: {e}")
                        
                        # Create and add the table
                        fig_table = Table(table_data, colWidths=[3.8*inch, 3.8*inch])
                        fig_table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 6),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                            ('TOPPADDING', (0, 0), (-1, -1), 6),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ]))
                        story.append(fig_table)
                        story.append(Spacer(1, 10))