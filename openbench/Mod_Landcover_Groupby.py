import os
import re
import shutil
import sys
import warnings
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Check the platform
from Mod_Metrics import metrics
from Mod_Scores import scores
from Mod_Converttype import Convert_Type
from visualization import *

class LC_groupby(metrics, scores):
    def __init__(self, main_nml, scores, metrics):
        self.name = 'StatisticsDataHandler'
        self.version = '0.3'
        self.release = '0.3'
        self.date = 'June 2024'
        self.author = "Zhongwang Wei"
        self.main_nml = main_nml
        self.general_config = self.main_nml['general']
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
        def _IGBP_class_remap_cdo():
            """
            Compare the IGBP class of the model output data and the reference data
            """
            from regrid import regridder_cdo
            # creat a text file, record the grid information
            nx = int(360. / self.compare_grid_res)
            ny = int(180. / self.compare_grid_res)
            grid_info = f'{self.casedir}/output/metrics/IGBP_groupby/grid_info.txt'
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
            IGBPtype_orig = './data/IGBP.nc'
            IGBPtype_remap = f'{self.casedir}/output/metrics/IGBP_groupby/IGBP_remap.nc'
            regridder_cdo.largest_area_fraction_remap_cdo(self, IGBPtype_orig, IGBPtype_remap, self.target_grid)
            self.IGBP_dir = IGBPtype_remap

        def _IGBP_class_remap(self):
            from regrid import Grid, create_regridding_dataset
            ds = xr.open_dataset(
                "./data/IGBP.nc",
                chunks={"lat": 2000, "lon": 2000},
            )
            ds = ds["IGBP"]  # Only take the class variable.
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
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(1, 18))
            IGBPtype_remap = f'{self.casedir}/output/metrics/IGBP_groupby/IGBP_remap.nc'
            ds_regrid.to_netcdf(IGBPtype_remap)
            self.IGBP_dir = IGBPtype_remap

        def _scenarios_IGBP_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the IGBP class of the model output data and the reference data
            """
            IGBPtype = xr.open_dataset(self.IGBP_dir)['IGBP']
            # convert IGBP type to int
            IGBPtype = IGBPtype.astype(int)

            igbp_class_names = {
                1: "evergreen_needleleaf_forest",
                2: "evergreen_broadleaf_forest",
                3: "deciduous_needleleaf_forest",
                4: "deciduous_broadleaf_forest",
                5: "mixed_forests",
                6: "closed_shrubland",
                7: "open_shrublands",
                8: "woody_savannas",
                9: "savannas",
                10: "grasslands",
                11: "permanent_wetlands",
                12: "croplands",
                13: "urban_and_built_up",
                14: "cropland_natural_vegetation_mosaic",
                15: "snow_and_ice",
                16: "barren_or_sparsely_vegetated",
                17: "water_bodies",
            }

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
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            logging.warning(f"warning: station data is not supported for IGBP class comparison")
                            pass
                        else:
                            dir_path = os.path.join(f'{basedir}', 'output', 'metrics', 'IGBP_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.txt')
                                with open(output_file_path, "w") as output_file:
                                    # Print the table header with an additional column for the overall mean
                                    output_file.write("ID\t")
                                    for i in range(1, 18):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for igbp_class_name in igbp_class_names.values():
                                        output_file.write(f"{igbp_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print mean values
                                    for metric in self.metrics:
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc')
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{metric}\t")

                                        # Calculate and write the overall mean first
                                        ds = ds.where(np.isfinite(ds), np.nan)
                                        q_value = ds[metric].quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                                        ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)

                                        overall_median = ds[metric].median(skipna=True).values
                                        overall_median_str = f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"

                                        for i in range(1, 18):
                                            ds1 = ds.where(IGBPtype == i)
                                            igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/output/metrics/IGBP_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_IGBP_{igbp_class_name}.nc")
                                            median_value = ds1[metric].median(skipna=True).values
                                            median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                            output_file.write(f"{median_value_str}\t")
                                        output_file.write(f"{overall_median_str}\t")  # Write overall median
                                        output_file.write("\n")

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option['path'] = f"{self.casedir}/output/metrics/IGBP_groupby/{sim_source}___{ref_source}/"
                                option['item'] = [evaluation_item, sim_source, ref_source]
                                option['groupby'] = 'IGBP_groupby'
                                make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                            else:
                                logging.error('Error: No metrics for IGBP class comparison')

                            if len(self.scores) > 0:
                                dir_path = os.path.join(f'{basedir}', 'output', 'scores', 'IGBP_groupby',
                                                        f'{sim_source}___{ref_source}')
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.txt')

                                with open(output_file_path2, "w") as output_file:
                                    # Print the table header with an additional column for the overall mean
                                    output_file.write("ID\t")
                                    for i in range(1, 18):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for igbp_class_name in igbp_class_names.values():
                                        output_file.write(f"{igbp_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print mean values

                                    for score in self.scores:
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc')
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{score}\t")
                                       
                                        if self.weight.lower() == 'area':
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == 'mass':
                                            # Get reference data for flux weighting
                                            o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                f'{ref_varname}']
                                            
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

                                        # Calculate and write the overall mean first
                                        # overall_mean = ds[score].mean(skipna=True).values
                                        overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                        for i in range(1, 18):
                                            ds1 = ds.where(IGBPtype == i)
                                            igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/output/scores/IGBP_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_IGBP_{igbp_class_name}.nc")
                                            
                                            if self.weight.lower() == 'area':
                                                weights = np.cos(np.deg2rad(ds.lat))
                                                mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                            elif self.weight.lower() == 'mass':
                                                # Get reference data for flux weighting
                                                o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                    f'{ref_varname}']
                                                
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
                                            
                                            mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                            output_file.write(f"{mean_value_str}\t")
                                        output_file.write(f"{overall_mean_str}\t")  # Write overall mean
                                        output_file.write("\n")

                                selected_scores = self.scores
                                option['path'] = f"{self.casedir}/output/scores/IGBP_groupby/{sim_source}___{ref_source}/"
                                option['groupby'] = 'IGBP_groupby'
                                make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                # print(f"IGBP class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.error('Error: No scores for IGBP class comparison')

        metricsdir_path = os.path.join(f'{casedir}', 'output', 'metrics', 'IGBP_groupby')
        #if os.path.exists(metricsdir_path):
        #    shutil.rmtree(metricsdir_path)
        #print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f'{casedir}', 'output', 'scores', 'IGBP_groupby')
        #if os.path.exists(scoresdir_path):
        #    shutil.rmtree(scoresdir_path)
        #print(f"Re-creating output directory: {scoresdir_path}")
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        try:
            _IGBP_class_remap_cdo()
        except Exception as e:
            logging.error(f"CDO remapping failed: {e}")
            logging.info("Falling back to xarray-regrid remapping...")
            _IGBP_class_remap(self)
        _scenarios_IGBP_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_PFT_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _PFT_class_remap_cdo(self):
            """
            Compare the PFT class of the model output data and the reference data
            """
            from regrid import regridder_cdo

            # creat a text file, record the grid information
            nx = int(360. / self.compare_grid_res)
            ny = int(180. / self.compare_grid_res)
            grid_info = f'{self.casedir}/output/metrics/PFT_groupby/PFT_info.txt'

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
            PFTtype_orig = './data/PFT.nc'
            PFTtype_remap = f'{self.casedir}/output/metrics/PFT_groupby/PFT_remap.nc'
            regridder_cdo.largest_area_fraction_remap_cdo(self, PFTtype_orig, PFTtype_remap, self.target_grid)
            self.PFT_dir = PFTtype_remap

        def _PFT_class_remap(self):
            """
            Compare the PFT class of the model output data and the reference data using xarray
            """
            from regrid import Grid, create_regridding_dataset
            ds = xr.open_dataset("./data/PFT.nc", chunks={"lat": 2000, "lon": 2000})
            ds = ds["PFT"]
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
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(0, 16))
            PFTtype_remap = f'{self.casedir}/output/metrics/PFT_groupby/PFT_remap.nc'
            ds_regrid.to_netcdf(PFTtype_remap)
            self.PFT_dir = PFTtype_remap

        def _scenarios_PFT_groupby(basedir, scores, metrics, sim_nml, ref_nml, evaluation_items):
            """
            Compare the PFT class of the model output data and the reference data
            """
            PFTtype = xr.open_dataset(self.PFT_dir)['PFT']
            # convert PFT type to int
            PFTtype = PFTtype.astype(int)
            PFT_class_names = {
                0: "bare_soil",
                1: "needleleaf_evergreen_temperate_tree",
                2: "needleleaf_evergreen_boreal_tree",
                3: "needleleaf_deciduous_boreal_tree",
                4: "broadleaf_evergreen_tropical_tree",
                5: "broadleaf_evergreen_temperate_tree",
                6: "broadleaf_deciduous_tropical_tree",
                7: "broadleaf_deciduous_temperate_tree",
                8: "broadleaf_deciduous_boreal_tree",
                9: "broadleaf_evergreen_shrub",
                10: "broadleaf_deciduous_temperate_shrub",
                11: "broadleaf_deciduous_boreal_shrub",
                12: "c3_arctic_grass",
                13: "c3_non-arctic_grass",
                14: "c4_grass",
                15: "c3_crop",
            }

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
                            logging.warning(f"warning: station data is not supported for PFT class comparison")
                        else:
                            dir_path = os.path.join(f'{basedir}', 'output', 'metrics', 'PFT_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.txt')
                                with open(output_file_path, "w") as output_file:
                                    # Print the table header with an additional column for the overall median
                                    output_file.write("ID\t")
                                    for i in range(0, 16):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for PFT_class_name in PFT_class_names.values():
                                        output_file.write(f"{PFT_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print median values

                                    for metric in self.metrics:
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc')
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{metric}\t")

                                        # Calculate and write the overall median first
                                        ds = ds.where(np.isfinite(ds), np.nan)
                                        q_value = ds[metric].quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                                        ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)

                                        overall_median = ds[metric].median(skipna=True).values
                                        overall_median_str = f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"

                                        for i in range(0, 16):
                                            ds1 = ds.where(PFTtype == i)
                                            PFT_class_name = PFT_class_names.get(i, f"PFT_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/output/metrics/PFT_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_PFT_{PFT_class_name}.nc")
                                            median_value = ds1[metric].median(skipna=True).values
                                            median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                            output_file.write(f"{median_value_str}\t")
                                        output_file.write(f"{overall_median_str}\t")  # Write overall median
                                        output_file.write("\n")

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option['path'] = f"{self.casedir}/output/metrics/PFT_groupby/{sim_source}___{ref_source}/"
                                option['item'] = [evaluation_item, sim_source, ref_source]
                                option['groupby'] = 'PFT_groupby'
                                make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                                # print(f"PFT class metrics comparison results are saved to {output_file_path}")
                            else:
                                logging.error('Error: No scores for PFT class comparison')

                            if len(self.scores) > 0:
                                dir_path = os.path.join(f'{basedir}', 'output', 'scores', 'PFT_groupby',
                                                        f'{sim_source}___{ref_source}')
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.txt')
                                with open(output_file_path2, "w") as output_file:
                                    # Print the table header with an additional column for the overall mean
                                    output_file.write("ID\t")
                                    for i in range(0, 16):
                                        output_file.write(f"{i}\t")
                                    output_file.write("All\n")  # Move "All" to the first line
                                    output_file.write("FullName\t")
                                    for PFT_class_name in PFT_class_names.values():
                                        output_file.write(f"{PFT_class_name}\t")
                                    output_file.write("Overall\n")  # Write "Overall" on the second line

                                    # Calculate and print mean values

                                    for score in self.scores:
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc')
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{score}\t")

                                        # Calculate and write the overall mean first
                                       
                                        if self.weight.lower() == 'area':
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == 'mass':
                                            # Get reference data for flux weighting
                                            o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                f'{ref_varname}']
                                            
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

                                        for i in range(0, 16):
                                            ds1 = ds.where(PFTtype == i)
                                            PFT_class_name = PFT_class_names.get(i, f"PFT_{i}")
                                            ds1.to_netcdf(
                                                f"{self.casedir}/output/scores/PFT_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_PFT_{PFT_class_name}.nc")
                                            # Calculate and write the overall mean first
                                            if self.weight.lower() == 'area':
                                                weights = np.cos(np.deg2rad(ds.lat))
                                                mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                            elif self.weight.lower() == 'mass':
                                                # Get reference data for flux weighting
                                                o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                    f'{ref_varname}']
                                                
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
                                            
                                            #mean_value = ds1[score].mean(skipna=True).values
                                            mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                            output_file.write(f"{mean_value_str}\t")
                                        output_file.write(f"{overall_mean_str}\t")  # Write overall mean
                                        output_file.write("\n")

                                selected_scores = self.scores
                                option['path'] = f"{self.casedir}/output/scores/PFT_groupby/{sim_source}___{ref_source}/"
                                option['groupby'] = 'PFT_groupby'
                                make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                # print(f"PFT class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.error('Error: No scores for PFT class comparison')

        metricsdir_path = os.path.join(f'{casedir}', 'output', 'metrics', 'PFT_groupby')
        #if os.path.exists(metricsdir_path):
       #     shutil.rmtree(metricsdir_path)
        #print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f'{casedir}', 'output', 'scores', 'PFT_groupby')
        #if os.path.exists(scoresdir_path):
       #     shutil.rmtree(scoresdir_path)
        #print(f"Re-creating output directory: {scoresdir_path}")
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        try:
            _PFT_class_remap_cdo(self)
        except Exception as e:
            logging.error(f"CDO remapping failed: {e}")
            logging.info("Falling back to xarray-regrid remapping...")
            _PFT_class_remap(self)
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_CZ_groupby_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        def _CZ_class_remap_cdo(self):
            """
            Compare the Climate zone class of the model output data and the reference data
            """
            from regrid import regridder_cdo

            # creat a text file, record the grid information
            nx = int(360. / self.compare_grid_res)
            ny = int(180. / self.compare_grid_res)
            grid_info = f'{self.casedir}/output/metrics/CZ_groupby/CZ_info.txt'

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
            CZtype_orig = './data/Climate_zone.nc'
            CZtype_remap = f'{self.casedir}/output/metrics/CZ_groupby/CZ_remap.nc'
            regridder_cdo.largest_area_fraction_remap_cdo(self, CZtype_orig, CZtype_remap, self.target_grid)
            self.CZ_dir = CZtype_remap

        def _CZ_class_remap(self):
            """
            Compare the Climate zone class of the model output data and the reference data using xarray
            """
            from regrid import Grid, create_regridding_dataset
            ds = xr.open_dataset("./data/CClimate_zone.nc", chunks={"lat": 2000, "lon": 2000})
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
            ds_regrid = ds.astype(int).regrid.most_common(target_dataset, values=np.arange(0, 16))
            CZtype_remap = f'{self.casedir}/output/metrics/CZ_groupby/CZ_remap.nc'
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
                            logging.warning(f"warning: station data is not supported for Climate zone class comparison")
                        else:
                            dir_path = os.path.join(f'{basedir}', 'output', 'metrics', 'CZ_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            if len(self.metrics) > 0:
                                output_file_path = os.path.join(dir_path,
                                                                f'{evaluation_item}_{sim_source}___{ref_source}_metrics.txt')
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
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc')
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
                                                f"{self.casedir}/output/metrics/CZ_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_CZ_{CZ_class_name}.nc")
                                            median_value = ds1[metric].median(skipna=True).values
                                            median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                            output_file.write(f"{median_value_str}\t")
                                        output_file.write(f"{overall_median_str}\t")  # Write overall median
                                        output_file.write("\n")

                                selected_metrics = self.metrics
                                # selected_metrics = list(selected_metrics)
                                option['path'] = f"{self.casedir}/output/metrics/CZ_groupby/{sim_source}___{ref_source}/"
                                option['item'] = [evaluation_item, sim_source, ref_source]
                                option['groupby'] = 'CZ_groupby'
                                make_CZ_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                                # print(f"CZ class metrics comparison results are saved to {output_file_path}")
                            else:
                                logging.error('Error: No scores for climate zone class comparison')

                            if len(self.scores) > 0:
                                dir_path = os.path.join(f'{basedir}', 'output', 'scores', 'CZ_groupby',
                                                        f'{sim_source}___{ref_source}')
                                if not os.path.exists(dir_path):
                                    os.makedirs(dir_path)
                                output_file_path2 = os.path.join(dir_path,
                                                                 f'{evaluation_item}_{sim_source}___{ref_source}_scores.txt')
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
                                        ds = xr.open_dataset(
                                            f'{self.casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc')
                                        ds = Convert_Type.convert_nc(ds)
                                        output_file.write(f"{score}\t")

                                        # Calculate and write the overall mean first

                                        if self.weight.lower() == 'area':
                                            weights = np.cos(np.deg2rad(ds.lat))
                                            overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                        elif self.weight.lower() == 'mass':
                                            # Get reference data for flux weighting
                                            o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                f'{ref_varname}']

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
                                                f"{self.casedir}/output/scores/CZ_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_CZ_{CZ_class_name}.nc")
                                            # Calculate and write the overall mean first
                                            if self.weight.lower() == 'area':
                                                weights = np.cos(np.deg2rad(ds.lat))
                                                mean_value = ds1[score].weighted(weights).mean(skipna=True).values
                                            elif self.weight.lower() == 'mass':
                                                # Get reference data for flux weighting
                                                o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                                    f'{ref_varname}']

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
                                option['path'] = f"{self.casedir}/output/scores/CZ_groupby/{sim_source}___{ref_source}/"
                                option['groupby'] = 'CZ_groupby'
                                make_CZ_based_heat_map(output_file_path2, selected_scores, 'score', option)
                                # print(f"CZ class scores comparison results are saved to {output_file_path2}")
                            else:
                                logging.error('Error: No scores for climate zone class comparison')

        metricsdir_path = os.path.join(f'{casedir}', 'output', 'metrics', 'CZ_groupby')
        # if os.path.exists(metricsdir_path):
        #     shutil.rmtree(metricsdir_path)
        # print(f"Re-creating output directory: {metricsdir_path}")
        if not os.path.exists(metricsdir_path):
            os.makedirs(metricsdir_path)

        scoresdir_path = os.path.join(f'{casedir}', 'output', 'scores', 'CZ_groupby')
        if not os.path.exists(scoresdir_path):
            os.makedirs(scoresdir_path)

        try:
            _CZ_class_remap_cdo(self)
        except Exception as e:
            logging.error(f"CDO remapping failed: {e}")
            logging.info("Falling back to xarray-regrid remapping...")
            _CZ_class_remap(self)
        _scenarios_CZ_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)
