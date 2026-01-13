import os
import re
import shutil
import sys
import warnings
import logging
import gc
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Check the platform
from openbench.core.metrics.Mod_Metrics import metrics
from openbench.core.scoring.Mod_Scores import scores
from openbench.core.statistic import statistics_calculate
from . import *
from openbench.util.Mod_Converttype import Convert_Type

# Configure logging
logging.getLogger('xarray').setLevel(logging.WARNING)  # Suppress INFO messages from xarray
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress numpy runtime warnings
logging.getLogger('dask').setLevel(logging.WARNING)  # Suppress INFO messages from dask
class Evaluation_grid_only_drawing(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_grid_only_drawing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'June 2025'
        self.author = "Xionghui Xu"
        self.__dict__.update(info)
        self.fig_nml = fig_nml

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                Drawing evaluation processes starting!         ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def make_Evaluation(self, **kwargs):
        try:
            make_plot_index_grid(self)
        finally:
            gc.collect()  # Final cleanup

class Evaluation_stn_only_drawing(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_point_only_drawing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'June 2025'
        self.author = "Xionghui Xu"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                  Evaluation processes starting!               ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def make_evaluation_P(self):
        try:
            make_plot_index_stn(self)
        finally:
            gc.collect()  # Final cleanup



class LC_groupby_only_drawing(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = 'StatisticsDataHandler_only_drawing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'June 2025'
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml['general']
        self._igbp_station_warning_shown = False  # Track if IGBP station data warning has been shown
        self._pft_station_warning_shown = False   # Track if PFT station data warning has been shown
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()
        self.casedir = os.path.join(self.main_nml['general']['basedir'], self.main_nml['general']['basename'])
        # Set default weight method to 'none'
        self.weight = self.main_nml['general'].get('weight', 'none')
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ['climatology-year', 'climatology-month']:
            logging.debug(f"LC_groupby_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion")
        else:
            match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
            if not match:
                logging.error(f"Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")
            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
        self.metrics = metrics
        self.scores = scores
    
    def scenarios_IGBP_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _scenarios_IGBP_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"Processing evaluation item: {evaluation_item}")
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for i, sim_source in enumerate(sim_sources):
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            if not self._igbp_station_warning_shown:
                                logging.warning(f"warning: station data is not supported for IGBP class comparison")
                                self._igbp_station_warning_shown = True
                            pass
                        else:
                            dir_path = os.path.join(f'{basedir}', 'comparisons', 'IGBP_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.csv')
                                # Also check for .txt extension (legacy files)
                                output_file_path_txt = output_file_path[:-4] + '.txt'

                                if os.path.exists(output_file_path) or os.path.exists(output_file_path_txt):
                                    selected_metrics = self.metrics
                                    option['path'] = f"{self.casedir}/comparisons/IGBP_groupby/{sim_source}___{ref_source}/"
                                    option['item'] = [evaluation_item, sim_source, ref_source]
                                    option['groupby'] = 'IGBP_groupby'
                                    make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                                else:
                                    logging.warning(f"Metrics file not found: {output_file_path}")
                            else:
                                logging.error('Error: No metrics for IGBP class comparison')

                            if len(self.scores) > 0:
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.csv')
                                # Also check for .txt extension (legacy files)
                                output_file_path2_txt = output_file_path2[:-4] + '.txt'

                                if os.path.exists(output_file_path2) or os.path.exists(output_file_path2_txt):
                                    selected_scores = self.scores
                                    option['path'] = f"{self.casedir}/comparisons/IGBP_groupby/{sim_source}___{ref_source}/"
                                    option['groupby'] = 'IGBP_groupby'
                                    make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                else:
                                    logging.warning(f"Scores file not found: {output_file_path2}")
                            else:
                                logging.error('Error: No scores for IGBP class comparison')
        _scenarios_IGBP_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _scenarios_PFT_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"now processing the evaluation item: {evaluation_item}" )
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            if not self._pft_station_warning_shown:
                                logging.warning(f"warning: station data is not supported for PFT class comparison")
                                self._pft_station_warning_shown = True
                        else:
                            dir_path = os.path.join(f'{basedir}', 'comparisons', 'PFT_groupby',
                                                    f'{sim_source}___{ref_source}')

                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.csv')
                                output_file_path_txt = output_file_path[:-4] + '.txt'

                                if os.path.exists(output_file_path) or os.path.exists(output_file_path_txt):
                                    selected_metrics = self.metrics
                                    option['path'] = f"{self.casedir}/comparisons/PFT_groupby/{sim_source}___{ref_source}/"
                                    option['item'] = [evaluation_item, sim_source, ref_source]
                                    option['groupby'] = 'PFT_groupby'
                                    make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                                else:
                                    logging.warning(f"Metrics file not found: {output_file_path}")
                            else:
                                logging.error('Error: No scores for PFT class comparison')

                            if len(self.scores) > 0:
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.csv')
                                output_file_path2_txt = output_file_path2[:-4] + '.txt'

                                if os.path.exists(output_file_path2) or os.path.exists(output_file_path2_txt):
                                    selected_scores = self.scores
                                    option['path'] = f"{self.casedir}/comparisons/PFT_groupby/{sim_source}___{ref_source}/"
                                    option['groupby'] = 'PFT_groupby'
                                    make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                else:
                                    logging.warning(f"Scores file not found: {output_file_path2}")
                            else:
                                logging.error('Error: No scores for PFT class comparison')
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)


class CZ_groupby_only_drawing(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = 'StatisticsDataHandler_only_drawing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'June 2025'
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml['general']
        self._station_warning_shown = False  # Track if station data warning has been shown
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()
        self.casedir = os.path.join(self.main_nml['general']['basedir'], self.main_nml['general']['basename'])
        # Set default weight method to 'none'
        self.weight = self.main_nml['general'].get('weight', 'none')
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ['climatology-year', 'climatology-month']:
            logging.debug(f"CZ_groupby_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion")
        else:
            match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
            if not match:
                logging.error(f"Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")
            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
        self.metrics = metrics
        self.scores = scores

    def scenarios_CZ_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _CZ_class_remap_cdo(self):
            """
            Compare the Climate zone class of the model output data and the reference data
            """
            from openbench.data.regrid import regridder_cdo

            # creat a text file, record the grid information
            nx = int(360. / self.compare_grid_res)
            ny = int(180. / self.compare_grid_res)
            grid_info = f'{self.casedir}/comparisons/CZ_groupby/CZ_info.txt'

            with open(grid_info, 'w') as f:
                f.write(f"gridtype = lonlat\n")
                f.write(f"xsize    =  {nx} \n")
                f.write(f"ysize    =  {ny}\n")
                f.write(f"xfirst   =  {self.min_lon + self.compare_grid_res / 2}\n")
                f.write(f"xinc     =  {self.compare_grid_res}\n")
                f.write(f"yfirst   =  {self.min_lat + self.compare_grid_res / 2}\n")
                f.write(f"yinc     =  {self.compare_grid_res}\n")
                f.close()
            self.target_grid = grid_info
            CZtype_orig = './dataset/Climate_zone.nc'
            CZtype_remap = f'{self.casedir}/comparisons/CZ_groupby/CZ_remap.nc'
            regridder_cdo.largest_area_fraction_remap_cdo(self, CZtype_orig, CZtype_remap, self.target_grid)
            self.CZ_dir = CZtype_remap

        def _CZ_class_remap(self):
            """
            Compare the Climate zone class of the model output data and the reference data using xarray
            """
            from openbench.data.regrid import Grid, create_regridding_dataset, Regridder
            ds = xr.open_dataset("./dataset/Climate_zone.nc", chunks={"lat": 2000, "lon": 2000})
            ds = ds["climate_zone"]
            ds = ds.sortby(["lat", "lon"])
            # ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            new_grid = Grid(
                north=self.max_lat - self.compare_grid_res / 2,
                south=self.min_lat + self.compare_grid_res / 2,
                west=self.min_lon + self.compare_grid_res / 2,
                east=self.max_lon - self.compare_grid_res / 2,
                resolution_lat=self.compare_grid_res,
                resolution_lon=self.compare_grid_res,
            )
            target_dataset = create_regridding_dataset(new_grid)
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(1, 31))
            CZtype_remap = f'{self.casedir}/comparisons/CZ_groupby/CZ_remap.nc'
            ds_regrid.to_netcdf(CZtype_remap)
            self.CZ_dir = CZtype_remap

        def _scenarios_CZ_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the Climate zone class of the model output data and the reference data
            """
            CZtype = xr.open_dataset(self.CZ_dir)['climate_zone']
            # convert CZ type to int
            CZtype = CZtype.astype(int)
            CZ_class_names = {
                1: "Af",
                2: "Am",
                3: "Aw",
                4: "BWh",
                5: "BWk",
                6: "BSh",
                7: "BSk",
                8: "Csa",
                9: "Csb",
                10: "Csc",
                11: "Cwa",
                12: "Cwb",
                13: "Cwc",
                14: "Cfa",
                15: "Cfb",
                16: "Cfc",
                17: "Dsa",
                18: "Dsb",
                19: "Dsc",
                20: "Dsd",
                21: "Dwa",
                22: "Dwb",
                23: "Dwc",
                24: "Dwd",
                25: "Dfa",
                26: "Dfb",
                27: "Dfc",
                28: "Dfd",
                29: "ET",
                30: "EF"
            }

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"now processing the evaluation item: {evaluation_item}")
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            if not self._station_warning_shown:
                                logging.warning(f"warning: station data is not supported for Climate zone class comparison")
                                self._station_warning_shown = True
                        else:
                            dir_path = os.path.join(f'{basedir}', 'comparisons', 'CZ_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.csv')
                                with open(output_file_path, "w") as output_file:
                                    # Print the table header with an additional column for the overall median
                                    output_file.write("ID\t")
                                    for i in range(1, 31):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for CZ_class_name in CZ_class_names.values():
                                        output_file.write(f"{CZ_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print median values

                                    for metric in self.metrics:
                                        metric_file = f'{self.casedir}/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc'
                                        if not os.path.exists(metric_file):
                                            logging.error(f"File not found in only_drawing mode: {metric_file}")
                                            logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                            continue
                                        ds = xr.open_dataset(metric_file)
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{metric}\t")

                                        # Calculate and write the overall median first
                                        ds = ds.where(np.isfinite(ds), np.nan)
                                        q_value = ds[metric].quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                                        ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)

                                        overall_median = ds[metric].median(skipna=True).values
                                        overall_median_str = f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"

                                        for i in range(1, 31):
                                            ds1 = ds.where(CZtype == i)
                                            CZ_class_name = CZ_class_names.get(i, f"CZ_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/comparisons/CZ_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_CZ_{CZ_class_name}.nc")
                                            median_value = ds1[metric].median(skipna=True).values
                                            median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                            output_file.write(f"{median_value_str}\t")
                                        output_file.write(f"{overall_median_str}\t")  # Write overall median
                                        output_file.write("\n")

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option['path'] = f"{self.casedir}/comparisons/CZ_groupby/{sim_source}___{ref_source}/"
                                option['item'] = [evaluation_item, sim_source, ref_source]
                                option['groupby'] = 'CZ_groupby'
                                make_CZ_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                                # print(f"CZ class metrics comparison results are saved to {output_file_path}")
                            else:
                                logging.error('Error: No scores for climate zone class comparison')

                            if len(self.scores) > 0:
                                dir_path = os.path.join(f'{basedir}', 'comparisons', 'CZ_groupby',
                                                        f'{sim_source}___{ref_source}')
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.csv')
                                with open(output_file_path2, "w") as output_file:
                                    # Print the table header with an additional column for the overall mean
                                    output_file.write("ID\t")
                                    for i in range(1, 31):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for CZ_class_name in CZ_class_names.values():
                                        output_file.write(f"{CZ_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print mean values

                                    for score in self.scores:
                                        score_file = f'{self.casedir}/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc'
                                        if not os.path.exists(score_file):
                                            logging.error(f"File not found in only_drawing mode: {score_file}")
                                            logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                            continue
                                        ds = xr.open_dataset(score_file)
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{score}\t")

                                        # Calculate and write the overall mean first

                                        if self.weight.lower() == 'area':
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == 'mass':
                                            # Get reference data for flux weighting
                                            ref_data_file = f'{self.casedir}/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc'
                                            if not os.path.exists(ref_data_file):
                                                logging.error(f"File not found in only_drawing mode: {ref_data_file}")
                                                logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                continue
                                            o = xr.open_dataset(ref_data_file)[f'{ref_varname}']

                                            # Calculate area weights (cosine of latitude)
                                            area_weights = np.cos(np.deg2rad(ds.lat))

                                            # Calculate absolute flux weights
                                            flux_weights = np.abs(o.mean('time'))

                                            # Combine area and flux weights
                                            combined_weights = area_weights * flux_weights

                                            # Normalize weights to sum to 1
                                            normalized_weights = combined_weights / combined_weights.sum()

                                            # Calculate weighted mean
                                            overall_mean = ds[score].weighted(normalized_weights.fillna(0)).mean(skipna=True).values
                                        else:
                                            overall_mean = ds[score].mean(skipna=True).values

                                        overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                        for i in range(1, 31):
                                            ds1 = ds.where(CZtype == i)
                                            CZ_class_name = CZ_class_names.get(i, f"CZ_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/comparisons/CZ_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_CZ_{CZ_class_name}.nc")
                                            # Calculate and write the overall mean first
                                            if self.weight.lower() == 'area':
                                                weights = np.cos(np.deg2rad(ds.lat))
                                                mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                            elif self.weight.lower() == 'mass':
                                                # Get reference data for flux weighting
                                                ref_data_file = f'{self.casedir}/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc'
                                                if not os.path.exists(ref_data_file):
                                                    logging.error(f"File not found in only_drawing mode: {ref_data_file}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                o = xr.open_dataset(ref_data_file)[f'{ref_varname}']

                                                # Calculate area weights (cosine of latitude)
                                                area_weights = np.cos(np.deg2rad(ds.lat))

                                                # Calculate absolute flux weights
                                                flux_weights = np.abs(o.mean('time'))

                                                # Combine area and flux weights
                                                combined_weights = area_weights * flux_weights

                                                # Normalize weights to sum to 1
                                                normalized_weights = combined_weights / combined_weights.sum()

                                                # Calculate weighted mean
                                                mean_value = ds1[score].weighted(normalized_weights.fillna(0)).mean(skipna=True).values
                                            else:
                                                mean_value = ds1[score].mean(skipna=True).values

                                                # mean_value = ds1[score].mean(skipna=True).values
                                            mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                            output_file.write(f"{mean_value_str}\t")
                                        output_file.write(f"{overall_mean_str}\t")  # Write overall mean
                                        output_file.write("\n")

                                selected_scores = self.scores
                                option['path'] = f"{self.casedir}/comparisons/CZ_groupby/{sim_source}___{ref_source}/"
                                option['groupby'] = 'CZ_groupby'
                                make_CZ_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                # print(f"CZ class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.error('Error: No scores for climate zone class comparison')

        metricsdir_path = os.path.join(f'{casedir}', 'comparisons', 'CZ_groupby')
        # if os.path.exists(metricsdir_path):
        #     shutil.rmtree(metricsdir_path)
        # print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f'{casedir}', 'comparisons', 'CZ_groupby')
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        _CZ_class_remap(self)
        _scenarios_CZ_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

class ComparisonProcessing_only_drawing(metrics, scores, statistics_calculate):
    def __init__(self, main_nml, scores, metrics):
        self.name = 'ComparisonDataHandler_only_drawing'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'June 2025'
        self.author = "Xionghui Xu"
        self.main_nml = main_nml
        self.general_config = self.main_nml['general']
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        self.compare_nml = {}
        # Add default weight attribute
        self.weight = self.main_nml['general'].get('weight', 'none')  # Default to 'none' if not specified
        self._igbp_station_warning_shown = False  # Track if IGBP station data warning has been shown

        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()
        self.casedir = os.path.join(self.main_nml['general']['basedir'], self.main_nml['general']['basename'])
        # this should be done in read_namelist
        # adjust the time frequency
        # Check if climatology mode - skip frequency parsing
        if self.compare_tim_res in ['climatology-year', 'climatology-month']:
            logging.debug(f"ComparisonProcessing_only_drawing: Climatology mode detected ({self.compare_tim_res}), skipping frequency conversion")
        else:
            match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
            if not match:
                logging.error(f"Invalid time resolution format. Use '3month', '6hr', etc.")
                raise ValueError("Invalid time resolution format. Use '3month', '6hr', etc.")

            value, unit = match.groups()
            if not value:
                value = 1
            else:
                value = int(value)  # Convert the numerical value to an integer
            # Get the corresponding pandas frequency
            freq = self.freq_map.get(unit.lower())
            if not freq:
                raise ValueError(f"Unsupported time unit: {unit}")
            self.compare_tim_res = f'{value}{freq}E'

        self.metrics = metrics
        self.scores = scores

        # self.ref_source              =  ref_source
        # self.sim_source              =  sim_source

    def scenarios_IGBP_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _scenarios_IGBP_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the IGBP class of the model output data and the reference data
            """
            try:
                # read the simulation source and reference source
                for evaluation_item in evaluation_items:
                    logging.info(f"now processing the evaluation item: {evaluation_item}")
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    for ref_source in ref_sources:
                        for i, sim_source in enumerate(sim_sources):
                            try:
                                ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                                if ref_data_type == 'stn' or sim_data_type == 'stn':
                                    if not self._igbp_station_warning_shown:
                                        logging.warning(f"warning: station data is not supported for IGBP class comparison")
                                        self._igbp_station_warning_shown = True
                                    pass
                                else:
                                    dir_path = os.path.join(basedir, 'comparisons', 'IGBP_groupby',
                                                            f'{sim_source}___{ref_source}')
                                    os.makedirs(dir_path, exist_ok=True)

                                    output_file_path = os.path.join(dir_path,
                                                                    f'{evaluation_item}_{sim_source}___{ref_source}_metrics.csv')
                                    selected_metrics = self.metrics
                                    option['path'] = os.path.join(self.casedir, 'comparisons', 'IGBP_groupby',
                                                                  f'{sim_source}___{ref_source}')
                                    option['item'] = [evaluation_item, sim_source, ref_source]
                                    option['groupby'] = 'IGBP_groupby'
                                    make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)

                                    output_file_path2 = os.path.join(basedir, 'comparisons', 'IGBP_groupby',
                                                                     f'{evaluation_item}_{sim_source}___{ref_source}_scores.csv')
                                    selected_scores = self.scores
                                    option['groupby'] = 'IGBP_groupby'
                                    make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                            finally:
                                gc.collect()  # Clean up memory after processing each simulation-reference pair
            finally:
                gc.collect()  # Final cleanup for the entire function

        _scenarios_IGBP_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)
        gc.collect()  # Final cleanup for the entire method

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _scenarios_PFT_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                logging.info(f"now processing the evaluation item: {evaluation_item}")
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            if not self._pft_station_warning_shown:
                                logging.warning(f"warning: station data is not supported for PFT class comparison")
                                self._pft_station_warning_shown = True
                        else:
                            dir_path = os.path.join(f'{basedir}', 'comparisons', 'PFT_groupby',
                                                    f'{sim_source}___{ref_source}')
                            os.makedirs(dir_path, exist_ok='True')

                            output_file_path = os.path.join(dir_path,
                                                            f'{evaluation_item}_{sim_source}___{ref_source}_metrics.csv')

                            selected_metrics = self.metrics
                            # selected_metrics = list(selected_metrics)
                            option['path'] = f"{self.casedir}/comparisons/PFT_groupby/{sim_source}___{ref_source}/"
                            option['item'] = [evaluation_item, sim_source, ref_source]
                            option['groupby'] = 'PFT_groupby'
                            make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)

                            output_file_path2 = os.path.join(dir_path,
                                                             f'{evaluation_item}_{sim_source}___{ref_source}_scores.csv')

                            selected_scores = self.scores
                            option['groupby'] = 'PFT_groupby'
                            make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_HeatMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, 'comparisons', 'HeatMap')
            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                make_scenarios_scores_comparison_heat_map(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Taylor_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, 'comparisons', 'Taylor_Diagram')
            os.makedirs(dir_path, exist_ok=True)

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                try:
                    # read the simulation source and reference source
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        try:
                            output_file_path = os.path.join(dir_path, f"taylor_diagram_{evaluation_item}_{ref_source}.csv")
                            # Fallback to .txt if .csv not found
                            if not os.path.exists(output_file_path):
                                output_file_path_txt = output_file_path[:-4] + '.txt'
                                if os.path.exists(output_file_path_txt):
                                    output_file_path = output_file_path_txt
                                else:
                                    logging.error(f"File not found in only_drawing mode: {output_file_path}")
                                    logging.error(f"Please run the comparison first (set only_drawing=False) to generate required data files.")
                                    continue
                            stds = np.zeros(len(sim_sources) + 1)
                            cors = np.zeros(len(sim_sources) + 1)
                            RMSs = np.zeros(len(sim_sources) + 1)
                            with open(output_file_path, 'r') as file:
                                lines = file.readlines()
                            second_row = lines[1].strip().split('\t')
                            list = [float(x) for x in second_row[2:] if x]
                            stds[0] = list[3]
                            for i, sim_source in enumerate(sim_sources):
                                stds[i+1] = list[i*4]
                                cors[i+1] = list[i*4+1]
                                RMSs[i+1] = list[i*4+2]
                            make_scenarios_comparison_Taylor_Diagram(casedir, evaluation_item, stds, RMSs, cors, ref_source, sim_sources,
                                                                        option)
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method
    
    def scenarios_Target_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, 'comparisons', 'Target_Diagram')
            os.makedirs(dir_path, exist_ok=True)

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                try:
                    # read the simulation source and reference source
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        try:
                            output_file_path = os.path.join(dir_path, f"target_diagram_{evaluation_item}_{ref_source}.csv")
                            # Fallback to .txt if .csv not found
                            if not os.path.exists(output_file_path):
                                output_file_path_txt = output_file_path[:-4] + '.txt'
                                if os.path.exists(output_file_path_txt):
                                    output_file_path = output_file_path_txt
                                else:
                                    logging.error(f"File not found in only_drawing mode: {output_file_path}")
                                    logging.error(f"Please run the comparison first (set only_drawing=False) to generate required data files.")
                                    continue
                            biases = np.zeros(len(sim_sources))
                            rmses = np.zeros(len(sim_sources))
                            crmsds = np.zeros(len(sim_sources))
                            with open(output_file_path, 'r') as file:
                                lines = file.readlines()
                            second_row = lines[1].strip().split('\t')
                            list = [float(x) for x in second_row[2:] if x]
                            # ill determine the number of simulation sources
                            for i, sim_source in enumerate(sim_sources):
                                biases = list[i*3]
                                rmses = list[i*3+1]
                                crmsds = list[i*3+2]

                            make_scenarios_comparison_Target_Diagram(dir_path, evaluation_item, biases, rmses, crmsds, ref_source,
                                                                        sim_sources, option)
                        finally:
                            gc.collect()  # Clean up memory after processing each reference source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method
    
    def scenarios_Kernel_Density_Estimate_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, 'comparisons', 'Kernel_Density_Estimate')
            os.makedirs(dir_path, exist_ok=True)

            # fixme: add the Kernel Density Estimate
            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == 'nSpatialScore':
                                logging.info(f"Skipping {score} for Kernel Density Estimate - it's a constant value")
                                continue
                            for ref_source in ref_sources:
                                try:
                                    file_paths = []
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                            sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                                            if ref_data_type == 'stn' or sim_data_type == 'stn':
                                                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                                                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                                                if sim_varname is None or sim_varname == '':
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == '':
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(basedir, 'scores',
                                                                         f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                # read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=',', header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(basedir, 'scores',
                                                                         f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc")
                                                ds = xr.open_dataset(file_path)
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[score].values
                                            datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Kernel_Density_Estimate(dir_path, evaluation_item, ref_source, sim_sources,
                                                                                          score, datasets_filtered, option)
                                    except:
                                        logging.error(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Kernel Density Estimate failed!")
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    file_paths = []
                                    datasets_filtered = []
                                    # create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                            sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                                            if ref_data_type == 'stn' or sim_data_type == 'stn':
                                                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                                                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                                                if sim_varname is None or sim_varname == '':
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == '':
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(basedir, 'metrics',
                                                                         f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                # read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=',', header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(basedir, 'metrics',
                                                                         f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc")
                                                ds = xr.open_dataset(file_path)
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[metric].values

                                            data = data[~np.isinf(data)]
                                            if metric == 'percent_bias':
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Kernel_Density_Estimate(dir_path, evaluation_item, ref_source, sim_sources,
                                                                                          metric, datasets_filtered, option)
                                    except:
                                        logging.error(f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed!")
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Parallel_Coordinates_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, 'comparisons', 'Parallel_Coordinates')
            os.makedirs(dir_path, exist_ok=True)
            output_file_path = os.path.join(dir_path, "Parallel_Coordinates_evaluations.csv")

            # Fallback to .txt if .csv not found
            if not os.path.exists(output_file_path):
                output_file_path_txt = output_file_path[:-4] + '.txt'
                if os.path.exists(output_file_path_txt):
                    output_file_path = output_file_path_txt
                else:
                    logging.error(f"File not found in only_drawing mode: {output_file_path}")
                    logging.error(f"Please run the comparison first (set only_drawing=False) to generate required data files.")
                    return
            df = pd.read_csv(output_file_path, sep='\t', header=0)
            df = df.dropna(axis=1, how='any')
            # If index in scores or metrics was dropped, then remove the corresponding scores or metrics
            scores = [score for score in scores if score in df.columns]
            metrics = [metric for metric in metrics if metric in df.columns]

            output_file_path1 = os.path.join(dir_path, "Parallel_Coordinates_evaluations_remove_nan.csv")
            df.to_csv(output_file_path1, sep='\t', index=False)

            make_scenarios_comparison_parallel_coordinates(output_file_path1, self.casedir, evaluation_items, scores, metrics, option)
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Portrait_Plot_seasonal_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, 'comparisons', 'Portrait_Plot_seasonal')
            os.makedirs(dir_path, exist_ok=True)
            output_file_path = os.path.join(dir_path, "Portrait_Plot_seasonal.csv")
            # Fallback to .txt if .csv not found
            if not os.path.exists(output_file_path):
                output_file_path_txt = output_file_path[:-4] + '.txt'
                if os.path.exists(output_file_path_txt):
                    output_file_path = output_file_path_txt
            make_scenarios_comparison_Portrait_Plot_seasonal(output_file_path, self.casedir, evaluation_items, scores, metrics,
                                                             option)
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Whisker_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(basedir, 'comparisons', 'Whisker_Plot')
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # If the sim_sources and ref_sources are not lists, convert them to lists
                    if isinstance(sim_sources, str):
                        sim_sources = [sim_sources]
                    if isinstance(ref_sources, str):
                        ref_sources = [ref_sources]

                    for score in scores:
                        try:
                            # Skip nSpatialScore since it's a constant value
                            if score == 'nSpatialScore':
                                logging.info(f"Skipping {score} for Whisker Plot - it's a constant value")
                                continue
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                            sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                                            if ref_data_type == 'stn' or sim_data_type == 'stn':
                                                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                                                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                                                if sim_varname is None or sim_varname == '':
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == '':
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(basedir, 'scores',
                                                                         f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                # Read the file_path data and select the score
                                                df = pd.read_csv(file_path, sep=',', header=0)
                                                df = Convert_Type.convert_Frame(df)
                                                data = df[score].values
                                            else:
                                                file_path = os.path.join(basedir, 'scores',
                                                                         f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                ds = xr.open_dataset(file_path)
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[score].values
                                            datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Whisker_Plot(dir_path, evaluation_item, ref_source, sim_sources, score,
                                                                               datasets_filtered, option)
                                    except:
                                        logging.error(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Whisker Plot failed!")
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each score

                    for metric in metrics:
                        try:
                            for ref_source in ref_sources:
                                try:
                                    datasets_filtered = []
                                    # Create a numpy matrix to store the data
                                    for sim_source in sim_sources:
                                        try:
                                            ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                            sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                                            if ref_data_type == 'stn' or sim_data_type == 'stn':
                                                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                                                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                                                if sim_varname is None or sim_varname == '':
                                                    sim_varname = evaluation_item
                                                if ref_varname is None or ref_varname == '':
                                                    ref_varname = evaluation_item

                                                file_path = os.path.join(basedir, 'metrics',
                                                                         f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                # Read the file_path data and select the metric
                                                df = pd.read_csv(file_path, sep=',', header=0)
                                                data = df[metric].values
                                            else:
                                                file_path = os.path.join(basedir, 'metrics',
                                                                         f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc")
                                                if not os.path.exists(file_path):
                                                    logging.error(f"File not found in only_drawing mode: {file_path}")
                                                    logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                                    continue
                                                ds = xr.open_dataset(file_path)
                                                ds = Convert_Type.convert_nc(ds)
                                                data = ds[metric].values

                                            data = data[~np.isinf(data)]
                                            if metric == 'percent_bias':
                                                data = data[(data >= -100) & (data <= 100)]
                                            datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append
                                        finally:
                                            gc.collect()  # Clean up memory after processing each simulation source

                                    try:
                                        make_scenarios_comparison_Whisker_Plot(dir_path, evaluation_item, ref_source, sim_sources, metric,
                                                                               datasets_filtered, option)
                                    except:
                                        logging.error(f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Whisker Plot failed!")
                                finally:
                                    gc.collect()  # Clean up memory after processing each reference source
                        finally:
                            gc.collect()  # Clean up memory after processing each metric
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Relative_Score_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, 'comparisons', 'Relative_Score')
            os.makedirs(dir_path, exist_ok=True)

            for evaluation_item in evaluation_items:
                try:
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        for sim_source in sim_sources:
                            try:
                                ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                                sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                                if ref_data_type == 'stn' or sim_data_type == 'stn':
                                    try:
                                        make_scenarios_comparison_Relative_Score(dir_path, evaluation_item, ref_source, sim_source,
                                                                                scores, 'stn', self.main_nml['general'], option)
                                    except:
                                        logging.info(f"No files found")
                                else:
                                    try:

                                        make_scenarios_comparison_Relative_Score(dir_path, evaluation_item, ref_source, sim_source, scores, 'grid',
                                                                                self.main_nml['general'], option)
                                    except:
                                        logging.info(f"No files found")
                            finally:
                                gc.collect()  # Clean up memory after processing each simulation source
                finally:
                    gc.collect()  # Clean up memory after processing each evaluation item
        finally:
            gc.collect()  # Final cleanup for the entire method

    def scenarios_Single_Model_Performance_Index_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics,
                                                            option):
        make_scenarios_comparison_Single_Model_Performance_Index(basedir, evaluation_items, ref_nml, sim_nml, option)

    def scenarios_Ridgeline_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{basedir}', 'comparisons', 'Ridgeline_Plot')
        # if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        # print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            for score in scores:
                # Skip nSpatialScore since it's a constant value
                if score == 'nSpatialScore':
                    logging.info(f"Skipping {score} for Ridgeline Plot - it's a constant value")
                    continue
                for ref_source in ref_sources:
                    file_paths = []
                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if isinstance(sim_sources, str): sim_sources = [sim_sources]
                        if isinstance(ref_sources, str): ref_sources = [ref_sources]
                        # create a numpy matrix to store the data

                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            file_path = os.path.join(basedir, 'scores',
                                                     f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                            if not os.path.exists(file_path):
                                logging.error(f"File not found in only_drawing mode: {file_path}")
                                logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                continue
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            df = Convert_Type.convert_Frame(df)
                            data = df[score].values
                        else:
                            file_path = os.path.join(basedir, 'scores',
                                                     f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc")
                            if not os.path.exists(file_path):
                                logging.error(f"File not found in only_drawing mode: {file_path}")
                                logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                continue
                            ds = xr.open_dataset(file_path)
                            ds = Convert_Type.convert_nc(ds)
                            data = ds[score].values
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(dir_path, evaluation_item, ref_source, sim_sources, score,
                                                                 datasets_filtered, option)
                    except:
                        logging.error(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Ridgeline_Plot failed!")

            for metric in metrics:
                for ref_source in ref_sources:
                    dir_path = os.path.join(f'{basedir}', 'comparisons', 'Ridgeline_Plot')
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    file_paths = []
                    datasets_filtered = []
                    # create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if isinstance(sim_sources, str): sim_sources = [sim_sources]
                        if isinstance(ref_sources, str): ref_sources = [ref_sources]
                        # create a numpy matrix to store the data
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            file_path = os.path.join(basedir, 'metrics',
                                                     f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv")
                            if not os.path.exists(file_path):
                                logging.error(f"File not found in only_drawing mode: {file_path}")
                                logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                continue
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[metric].values
                        else:
                            file_path = os.path.join(basedir, 'metrics',
                                                     f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc")
                            if not os.path.exists(file_path):
                                logging.error(f"File not found in only_drawing mode: {file_path}")
                                logging.error(f"Please run the evaluation first (set only_drawing=False) to generate required data files.")
                                continue
                            ds = xr.open_dataset(file_path)
                            ds = Convert_Type.convert_nc(ds)
                            data = ds[metric].values
                        data = data[~np.isinf(data)]
                        if metric == 'percent_bias':
                            data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(dir_path, evaluation_item, ref_source, sim_sources, metric,
                                                                 datasets_filtered, option)
                    except:
                        logging.error(
                            f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed!")

    def to_dict(self):
        return self.__dict__
    
    coordinate_map = {
        'longitude': 'lon', 'long': 'lon', 'lon_cama': 'lon', 'lon0': 'lon', 'x': 'lon',
        'latitude': 'lat', 'lat_cama': 'lat', 'lat0': 'lat', 'y': 'lat',
        'Time': 'time', 'TIME': 'time', 't': 'time', 'T': 'time',
        'elevation': 'elev', 'height': 'elev', 'z': 'elev', 'Z': 'elev',
        'h': 'elev', 'H': 'elev', 'ELEV': 'elev', 'HEIGHT': 'elev',
    }

    freq_map = {
        'month': 'M',
        'mon': 'M',
        'monthly': 'M',
        'day': 'D',
        'daily': 'D',
        'hour': 'H',
        'Hour': 'H',
        'hr': 'H',
        'Hr': 'H',
        'h': 'H',
        'hourly': 'H',
        'year': 'Y',
        'yr': 'Y',
        'yearly': 'Y',
        'week': 'W',
        'wk': 'W',
        'weekly': 'W',
    }

    def scenarios_Diff_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """
        Compare metrics and scores between different simulations:
        1. Calculate ensemble mean across all simulations
        2. Calculate anomalies from ensemble mean for each simulation
        3. Calculate pairwise differences between simulations
        4. Plot the results
        Parameters:
            basedir: base directory path
            sim_nml: simulation namelist
            ref_nml: reference namelist 
            evaluation_items: list of evaluation items
            scores: list of scores to compare
            metrics: list of metrics to compare
            option: additional options
        """
        dir_path = os.path.join(f'{basedir}', 'comparisons', 'Diff_Plot')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                # Skip if only one simulation source

                # Check data types for all simulation sources
                data_types = []
                for sim_source in sim_sources:
                    sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                    data_types.append(sim_data_type)

                # Check if both 'stn' and grid data exist
                if 'stn' in data_types and any(dt != 'stn' for dt in data_types):
                    logging.warning(f"Error: Cannot compare station and gridded data together for {evaluation_item}")
                    logging.warning("All simulation sources must be either station data or gridded data")
                    continue

                ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                # After calculating anomalies for metrics
                make_scenarios_comparison_Diff_Plot(dir_path, metrics, scores, evaluation_item, ref_source, sim_sources,
                                                    self.general_config, sim_nml,
                                                    ref_data_type, option)

    def scenarios_Basic_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        """
        Calculate all the data (including input data,metrics,scores):
        1. Calculate ensemble mean, median, min, max
        2. Calculate sum value for each input
        4. Plot the results
        """
        basic_method = option['key']
        dir_path = os.path.join(f'{basedir}', 'comparisons', basic_method)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for ref_source in ref_sources:
                ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']

                if ref_data_type == 'stn':
                    try:
                        for sim_source in sim_sources:
                            output_path = f'{dir_path}/{evaluation_item}_stn_{ref_source}_{sim_source}_{basic_method}.csv'
                            make_stn_plot_index(output_path, basic_method, self.main_nml['general'], (ref_source, sim_source), option)
                    except Exception as e:
                        logging.error(f"Error processing station {basic_method} calculations for {ref_source}: {e}")
                else:
                    try:
                        output_path = os.path.join(dir_path, f'{evaluation_item}_ref_{ref_source}_{ref_varname}_{basic_method}.nc')
                        # Skip global map plotting for nSpatialScore since it's constant globally
                        if basic_method != 'nSpatialScore':
                            make_geo_plot_index(output_path, basic_method, self.main_nml['general'], option)
                    except Exception as e:
                        logging.error(f"Error processing Grid {basic_method} calculations for {ref_source}: {e}")

            for sim_source in sim_sources:
                if len(sim_sources) < 2:
                    continue

                sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                if sim_data_type != 'stn':
                    try:
                        output_path = os.path.join(dir_path, f'{evaluation_item}_sim_{sim_source}_{sim_varname}_{basic_method}.nc')
                        # Skip global map plotting for nSpatialScore since it's constant globally
                        if basic_method != 'nSpatialScore':
                            make_geo_plot_index(output_path, basic_method, self.main_nml['general'], option)
                    except Exception as e:
                        logging.error(f"Error processing station {basic_method} calculations for {sim_source}: {e}")

    def scenarios_Mann_Kendall_Trend_Test_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        method_name = 'Mann_Kendall_Trend_Test'
        method_function = getattr(self, f"stat_{method_name.lower()}", None)
        dir_path = os.path.join(basedir, 'comparisons', 'Mann_Kendall_Trend_Test')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.compare_nml['Mann_Kendall_Trend_Test'] = {}
        self.compare_nml['Mann_Kendall_Trend_Test']['significance_level'] = option['significance_level']
        for evaluation_item in evaluation_items:
            # Get simulation sources
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

            # Convert to lists if needed
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for sim_source in sim_sources:
                # Skip if only one simulation source

                sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']

                if sim_data_type != 'stn':
                    try:
                        output_file = os.path.join(dir_path,
                                                   f'Mann_Kendall_Trend_Test_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')
                        make_Mann_Kendall_Trend_Test(output_file, method_name, sim_source, self.main_nml['general'], option)
                    except Exception as e:
                        logging.error(f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}")
            for ref_source in ref_sources:
                ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                if ref_data_type != 'stn':
                    try:
                        output_file = os.path.join(dir_path,
                                                   f'Mann_Kendall_Trend_Test_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')
                        make_Mann_Kendall_Trend_Test(output_file, method_name, ref_source, self.main_nml['general'], option)
                    except Exception as e:
                        logging.error(f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}")

    def scenarios_Standard_Deviation_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            method_name = 'Standard_Deviation'
            method_function = getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, 'comparisons', method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for sim_source in sim_sources:
                    try:
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']

                        if sim_data_type != 'stn':
                            output_file = os.path.join(dir_path,
                                                       f'{method_name}_{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')

                            make_Standard_Deviation(output_file, method_name, sim_source, self.main_nml['general'], option)
                        else:
                            logging.info(f"Skipping {method_name} drawing for {evaluation_item} {sim_source}: station data type.")
                    except Exception as e:
                        logging.error(f"Error processing {method_name} calculations for {evaluation_item} {sim_source}: {e}")
                    finally:
                        # Clean up memory after each simulation source
                        gc.collect()

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']

                        if ref_data_type != 'stn':
                            output_file = os.path.join(dir_path,
                                                       f'{method_name}_{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')

                            make_Standard_Deviation(output_file, method_name, ref_source, self.main_nml['general'], option)
                        else:
                            logging.info(f"Skipping {method_name} drawing for {evaluation_item} {ref_source}: station data type.")
                    except Exception as e:
                        logging.error(f"Error processing {method_name} calculations for {evaluation_item} {ref_source}: {e}")
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_Functional_Response_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        self.compare_nml['Functional_Response'] = {}
        self.compare_nml['Functional_Response']['nbins'] = option['nbins']
        try:
            method_name = 'Functional_Response'
            method_function = getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, 'comparisons', method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if isinstance(ref_sources, str):
                    ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    try:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']

                        if ref_data_type != 'stn':
                            for sim_source in sim_sources:
                                try:
                                    sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                                    sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                                    if sim_data_type != 'stn':

                                        output_file = os.path.join(dir_path,
                                                                   f'{method_name}_{evaluation_item}_ref_{ref_source}_sim_{sim_source}.nc')

                                        make_Functional_Response(output_file, method_name, sim_source, self.main_nml['general'], option)
                                except Exception as e:
                                    logging.error(
                                        f"Error processing {method_name} calculations for {evaluation_item} {ref_source} {sim_source}: {e}")
                                finally:
                                    # Clean up memory after each simulation source
                                    gc.collect()
                    finally:
                        # Clean up memory after each reference source
                        gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()

    def scenarios_RadarMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            dir_path = os.path.join(casedir, 'comparisons', 'RadarMap')
            for score in scores:
                output_file_path = os.path.join(dir_path, f"scenarios_{score}_comparison.csv")
                make_scenarios_comparison_radar_map(output_file_path, score, option)
        finally:
            gc.collect()  # Clean up memory after processing

    def scenarios_Correlation_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        try:
            method_name = 'Correlation'
            method_function = getattr(self, f"stat_{method_name.lower()}", None)
            dir_path = os.path.join(basedir, 'comparisons', method_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for evaluation_item in evaluation_items:
                # Get simulation sources
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                # Convert to lists if needed
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if len(sim_sources) < 2:
                    continue

                for i, sim1 in enumerate(sim_sources):
                    for j, sim2 in enumerate(sim_sources[i + 1:], i + 1):
                        sim_data_type1 = sim_nml[f'{evaluation_item}'][f'{sim1}_data_type']
                        sim_data_type2 = sim_nml[f'{evaluation_item}'][f'{sim2}_data_type']
                        if sim_data_type1 == 'stn' or sim_data_type2 == 'stn':
                            logging.warning(f"Error: Cannot compare station and gridded data together for {evaluation_item}")
                            logging.warning("All simulation sources must be gridded data")
                            continue

                        try:
                            output_file = os.path.join(dir_path,
                                                       f'{method_name}_{evaluation_item}_{sim1}_and_{sim2}.nc')
                            make_Correlation(output_file, method_name, self.main_nml['general'], option)
                        except Exception as e:
                            logging.error(f"Error processing {method_name} calculations for {evaluation_item} {sim1} and {sim2}: {e}")
                        finally:
                            # Clean up memory after each iteration
                            gc.collect()
        finally:
            # Ensure memory is cleaned up after the entire process
            gc.collect()
