# -*- coding: utf-8 -*-
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from Mod_Metrics import metrics
from Mod_Scores import scores
from Mod_Statistics import statistics_calculate
from figlib import *


class ComparisonProcessing(metrics, scores, statistics_calculate):
    def __init__(self, main_nml, scores, metrics):
        self.name = 'ComparisonDataHandler'
        self.version = '0.3'
        self.release = '0.3'
        self.date = 'June 2024'
        self.author = "Zhongwang Wei"
        self.main_nml = main_nml
        self.general_config = self.main_nml['general']
        # update self based on self.general_config
        self.__dict__.update(self.general_config)
        # self.min_lon        = self.main_nml['general']['min_lon']
        # self.max_lon        = self.main_nml['general']['max_lon']
        # self.min_lat        = self.main_nml['general']['min_lat']
        # self.max_lat        = self.main_nml['general']['max_lat']
        # Extract remapping information from main namelist
        self.compare_grid_res = self.main_nml['general']['compare_grid_res']
        self.compare_tim_res = self.main_nml['general'].get('compare_tim_res', '1').lower()
        self.casedir = os.path.join(self.main_nml['general']['basedir'], self.main_nml['general']['basename'])
        # this should be done in read_namelist
        # adjust the time frequency
        match = re.match(r'(\d*)\s*([a-zA-Z]+)', self.compare_tim_res)
        if not match:
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
        def _IGBP_class_remap_cdo():
            """
            Compare the IGBP class of the model output data and the reference data
            """
            from regrid import regridder_cdo
            # creat a text file, record the grid information
            nx = int(360. / self.compare_grid_res)
            ny = int(180. / self.compare_grid_res)
            grid_info = f'{self.casedir}/output/comparisons/IGBP_groupby/grid_info.txt'
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
            IGBPtype_remap = f'{self.casedir}/output/comparisons/IGBP_groupby/IGBP_remap.nc'
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
            IGBPtype_remap = f'{self.casedir}/output/comparisons/IGBP_groupby/IGBP_remap.nc'
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
                print("now processing the evaluation item: ", evaluation_item)
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for i, sim_source in enumerate(sim_sources):
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        # if isinstance(ref_data_type, str): ref_data_type = [ref_data_type]
                        # if isinstance(sim_data_type, str): sim_data_type = [sim_data_type]

                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            print(f"warning: station data is not supported for IGBP class comparison")
                            pass
                        else:
                            dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'IGBP_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            output_file_path = os.path.join(dir_path,
                                                            f'{evaluation_item}_{sim_source}___{ref_source}_metrics.txt')
                            with open(output_file_path, "w") as output_file:
                                # Print the table header with an additional column for the overall median
                                output_file.write("ID\t")
                                for i in range(1, 18):
                                    output_file.write(f"{i}\t")
                                output_file.write("All\n")  # Move "All" to the first line
                                output_file.write("FullName\t")
                                for igbp_class_name in igbp_class_names.values():
                                    output_file.write(f"{igbp_class_name}\t")
                                output_file.write("Overall\n")  # Write "Overall" on the second line

                                # Calculate and print median values

                                for metric in self.metrics:
                                    ds = xr.open_dataset(
                                        f'{self.casedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc')
                                    output_file.write(f"{metric}\t")

                                    # Calculate and write the overall median first
                                    ds = ds.where(np.isfinite(ds), np.nan)
                                    q_value = ds[metric].quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                                    ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)

                                    overall_median = ds[metric].median(skipna=True).values
                                    overall_median_str = f"{overall_median:.3f}" if not np.isnan(overall_median) else "N/A"

                                    for i in range(1, 18):
                                        ds1 = ds.where(IGBPtype == i)
                                        igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                        ds1.to_netcdf(
                                            f"{self.casedir}/output/comparisons/IGBP_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_IGBP_{igbp_class_name}.nc")
                                        median_value = ds1[metric].median(skipna=True).values
                                        median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                        output_file.write(f"{median_value_str}\t")
                                    output_file.write(f"{overall_median_str}\t")  # Write overall median
                                    output_file.write("\n")

                            selected_metrics = self.metrics
                            # selected_metrics = list(selected_metrics)
                            option['path'] = f"{self.casedir}/output/comparisons/IGBP_groupby/{sim_source}___{ref_source}/"
                            option['item'] = [evaluation_item, sim_source, ref_source]
                            option['groupby'] = 'IGBP_groupby'
                            make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                            # print(f"IGBP class metrics comparison results are saved to {output_file_path}")
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
                                    output_file.write(f"{score}\t")

                                    # Calculate and write the overall mean first
                                    overall_mean = ds[score].mean(skipna=True).values
                                    overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                    for i in range(1, 18):
                                        ds1 = ds.where(IGBPtype == i)
                                        igbp_class_name = igbp_class_names.get(i, f"IGBP_{i}")
                                        ds1.to_netcdf(
                                            f"{self.casedir}/output/comparisons/IGBP_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_IGBP_{igbp_class_name}.nc")
                                        mean_value = ds1[score].mean(skipna=True).values
                                        mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                        output_file.write(f"{mean_value_str}\t")
                                    output_file.write(f"{overall_mean_str}\t")  # Write overall mean
                                    output_file.write("\n")

                            selected_scores = self.scores
                            # selected_metrics = list(selected_metrics)
                            option['groupby'] = 'IGBP_groupby'
                            make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                            # print(f"IGBP class scores comparison results are saved to {output_file_path2}")

        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'IGBP_groupby')

        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        try:
            _IGBP_class_remap_cdo()
        except Exception as e:
            print(f"CDO remapping failed: {e}")
            print("Falling back to xarray-regrid remapping...")
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
            grid_info = f'{self.casedir}/output/comparisons/PFT_groupby/PFT_info.txt'

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
            PFTtype_remap = f'{self.casedir}/output/comparisons/PFT_groupby/PFT_remap.nc'
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
            PFTtype_remap = f'{self.casedir}/output/comparisons/PFT_groupby/PFT_remap.nc'
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
                print("now processing the evaluation item: ", evaluation_item)
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
                            print(f"warning: station data is not supported for PFT class comparison")
                        else:
                            dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'PFT_groupby',
                                                    f'{sim_source}___{ref_source}')
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            output_file_path = os.path.join(dir_path,
                                                            f'{evaluation_item}_{sim_source}___{ref_source}_metrics.txt')
                            with open(output_file_path, "w") as output_file:
                                # Print the table header with an additional column for the overall mean
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
                                            f"{self.casedir}/output/comparisons/PFT_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}_PFT_{PFT_class_name}.nc")
                                        median_value = ds1[metric].median(skipna=True).values
                                        median_value_str = f"{median_value:.3f}" if not np.isnan(median_value) else "N/A"
                                        output_file.write(f"{median_value_str}\t")
                                    output_file.write(f"{overall_median_str}\t")  # Write overall median
                                    output_file.write("\n")

                            selected_metrics = self.metrics
                            # selected_metrics = list(selected_metrics)
                            option['path'] = f"{self.casedir}/output/comparisons/PFT_groupby/{sim_source}___{ref_source}/"
                            option['item'] = [evaluation_item, sim_source, ref_source]
                            option['groupby'] = 'PFT_groupby'
                            make_LC_based_heat_map(output_file_path, selected_metrics, 'metric', option)
                            # print(f"PFT class metrics comparison results are saved to {output_file_path}")

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
                                    output_file.write(f"{score}\t")

                                    # Calculate and write the overall mean first
                                    overall_mean = ds[score].mean(skipna=True).values
                                    overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"

                                    for i in range(0, 16):
                                        ds1 = ds.where(PFTtype == i)
                                        PFT_class_name = PFT_class_names.get(i, f"PFT_{i}")
                                        ds1.to_netcdf(
                                            f"{self.casedir}/output/comparisons/PFT_groupby/{sim_source}___{ref_source}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}_PFT_{PFT_class_name}.nc")
                                        mean_value = ds1[score].mean(skipna=True).values
                                        mean_value_str = f"{mean_value:.3f}" if not np.isnan(mean_value) else "N/A"
                                        output_file.write(f"{mean_value_str}\t")
                                    output_file.write(f"{overall_mean_str}\t")  # Write overall mean
                                    output_file.write("\n")

                            selected_scores = self.scores
                            option['groupby'] = 'PFT_groupby'
                            make_LC_based_heat_map(output_file_path2, selected_scores, 'score', option)
                            # print(f"PFT class scores comparison results are saved to {output_file_path2}")

        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'PFT_groupby')

        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        try:
            _PFT_class_remap_cdo(self)
        except Exception as e:
            print(f"CDO remapping failed: {e}")
            print("Falling back to xarray-regrid remapping...")
            _PFT_class_remap(self)
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)

    def scenarios_HeatMap_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'HeatMap')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for score in scores:
            output_file_path = f"{dir_path}/scenarios_{score}_comparison.txt"
            with open(output_file_path, "w") as output_file:
                output_file.write(f"Item\t")
                output_file.write("Reference\t")
                # fixme: ugly code, need to be improved
                for evaluation_item in evaluation_items:
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                for sim_source in sim_sources:
                    output_file.write(f"{sim_source}\t")
                output_file.write("\n")  # Move "All" to the first line

                for evaluation_item in evaluation_items:
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if the sim_sources and ref_sources are not list, then convert them to list
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    if isinstance(ref_sources, str): ref_sources = [ref_sources]
                    # ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                    # if isinstance(ref_sources, str): ref_sources = [ref_sources]

                    for ref_source in ref_sources:
                        output_file.write(f"{evaluation_item}\t")
                        output_file.write(f"{ref_source}\t")
                        sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                        if isinstance(sim_sources, str): sim_sources = [sim_sources]

                        for sim_source in sim_sources:
                            ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                            sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                            if ref_data_type == 'stn' or sim_data_type == 'stn':
                                file = f"{casedir}/output/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                                df = pd.read_csv(file, sep=',', header=0)
                                overall_mean = df[f'{score}'].mean(skipna=True)
                            else:
                                ds = xr.open_dataset(
                                    f'{casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc')
                                                                   
                                if self.weight.lower() == 'area':
                                    weights = np.cos(np.deg2rad(ds.lat))
                                    overall_mean = ds[score].weighted(weights).mean(skipna=True).values
                                elif self.weight.lower() == 'mass':
                                    # Get reference data for flux weighting
                                    o = xr.open_dataset(f'{self.casedir}/output/data/{evaluation_item}_ref_{ref_source}_{self.ref_varname}.nc')[
                                        f'{self.ref_varname}']
                                    
                                    # Calculate area weights (cosine of latitude)
                                    area_weights = np.cos(np.deg2rad(ds.lat))
                                    
                                    # Calculate absolute flux weights
                                    flux_weights = np.abs(o.mean('time'))
                                    
                                    # Combine area and flux weights
                                    combined_weights = area_weights * flux_weights
                                    
                                    # Normalize weights to sum to 1
                                    normalized_weights = combined_weights / combined_weights.sum()
                                    
                                    # Calculate weighted mean
                                    overall_mean = ds[score].weighted(normalized_weights).mean(skipna=True).values
                                else:
                                    overall_mean = ds[score].mean(skipna=True).values

                            overall_mean_str = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "N/A"
                            output_file.write(f"{overall_mean_str}\t")
                        output_file.write("\n")

            make_scenarios_scores_comparison_heat_map(output_file_path, score, option)

    def scenarios_Taylor_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'Taylor_Diagram')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # read the simulation source and reference source
        for evaluation_item in evaluation_items:
            # read the simulation source and reference source
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]

            for ref_source in ref_sources:
                output_file_path = f"{dir_path}/taylor_diagram_{evaluation_item}_{ref_source}.txt"
                with open(output_file_path, "w") as output_file:
                    output_file.write("Item\t")
                    output_file.write("Reference\t")
                    for sim_source in sim_sources:
                        output_file.write(f"{sim_source}_std\t")
                        output_file.write(f"{sim_source}_COR\t")
                        output_file.write(f"{sim_source}_RMS\t")

                    output_file.write("Reference_std\t")
                    output_file.write("\n")  # Move "All" to the first line
                    output_file.write(f"{evaluation_item}\t")
                    output_file.write(f"{ref_source}\t")
                    stds = np.zeros(len(sim_sources) + 1)
                    cors = np.zeros(len(sim_sources) + 1)
                    RMSs = np.zeros(len(sim_sources) + 1)
                    for i, sim_source in enumerate(sim_sources):
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        # ugly code, need to be improved
                        # if self.sim_varname is empty, then set it to item
                        if sim_varname is None or sim_varname == '':
                            sim_varname = evaluation_item
                        if ref_varname is None or ref_varname == '':
                            ref_varname = evaluation_item
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            stnlist = f"{casedir}/stn_list.txt"
                            station_list = pd.read_csv(stnlist, header=0)
                            # this should be moved to other place
                            if ref_source.lower() == 'grdc':
                                station_list['ref_lon'] = station_list['lon']
                                station_list['ref_lat'] = station_list['lat']

                            def _make_validation_parallel(casedir, ref_source, sim_source, item, sim_varname, ref_varname,
                                                          station_list, iik):
                                s = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    sim_varname].squeeze()
                                o = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    ref_varname].squeeze()

                                s['time'] = o['time']
                                mask1 = np.isnan(s) | np.isnan(o)
                                s.values[mask1] = np.nan
                                o.values[mask1] = np.nan
                                # remove the nan values
                                # s=s.dropna(dim='time').astype(np.float32)
                                # o=o.dropna(dim='time').astype(np.float32)
                                row = {}
                                # for based plot
                                try:
                                    row['std_s'] = self.stat_standard_deviation(s).values  # .values
                                except:
                                    row['std_s'] = np.nan
                                try:
                                    row['std_o'] = self.stat_standard_deviation(o).values  # .values
                                except:
                                    row['std_o'] = np.nan
                                try:
                                    row['CRMSD'] = self.CRMSD(s, o).values  # .values
                                except:
                                    row['CRMSD'] = np.nan
                                try:
                                    row['correlation'] = self.correlation(s, o).values  # .values
                                except:
                                    row['correlation'] = np.nan
                                return row

                            results = Parallel(n_jobs=-1)(
                                delayed(_make_validation_parallel)(casedir, ref_source, sim_source, evaluation_item,
                                                                   sim_varname, ref_varname, station_list, iik) for iik in
                                range(len(station_list['ID'])))

                            station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)
                            station_list.to_csv(f"{dir_path}/taylor_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.txt")

                            station_list = pd.read_csv(
                                f"{dir_path}/taylor_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.txt", header=0)
                            std_sim = station_list['std_s'].mean(skipna=True)
                            output_file.write(f"{std_sim}\t")
                            stds[i + 1] = std_sim
                            cor_sim = station_list['correlation'].mean(skipna=True)
                            output_file.write(f"{cor_sim}\t")
                            cors[i + 1] = cor_sim
                            RMS_sim = station_list['CRMSD'].mean(skipna=True)
                            output_file.write(f"{RMS_sim}\t")
                            RMSs[i + 1] = RMS_sim
                            std_ref = station_list['std_o'].mean(skipna=True)

                        else:
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            reffile = \
                                xr.open_dataset(f'{casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                    ref_varname]
                            simfile = \
                                xr.open_dataset(f'{casedir}/output/data/{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')[
                                    sim_varname]
                            
                            std_sim_result = self.stat_standard_deviation(simfile)
                            cor_result = self.correlation(simfile, reffile)
                            RMS_result = self.CRMSD(simfile, reffile)

                            if self.weight.lower() == 'area':
                                weights = np.cos(np.deg2rad(ds.lat))
                                std_sim = std_sim_result.where(np.isfinite(std_sim_result)).weighted(weights).mean(skipna=True).values
                                cor_sim = cor_result.where(np.isfinite(cor_result)).weighted(weights).mean(skipna=True).values
                                RMS_sim = RMS_result.where(np.isfinite(RMS_result)).weighted(weights).mean(skipna=True).values
                            elif self.weight.lower() == 'mass':
                                # Calculate area weights (cosine of latitude)
                                area_weights = np.cos(np.deg2rad(ds.lat))     
                                # Calculate absolute flux weights
                                flux_weights = np.abs(reffile.mean('time'))
                                # Combine area and flux weights
                                combined_weights = area_weights * flux_weights
                                # Normalize weights to sum to 1
                                normalized_weights = combined_weights / combined_weights.sum()
                                # Calculate weighted mean
                                std_sim = std_sim_result.where(np.isfinite(std_sim_result)).weighted(normalized_weights).mean(skipna=True).values
                                cor_sim = cor_result.where(np.isfinite(cor_result)).weighted(normalized_weights).mean(skipna=True).values
                                RMS_sim = RMS_result.where(np.isfinite(RMS_result)).weighted(normalized_weights).mean(skipna=True).values
                            else:
                                std_sim = std_sim_result.where(np.isfinite(std_sim_result)).mean(skipna=True).values
                                cor_sim = cor_result.where(np.isfinite(cor_result)).mean(skipna=True).values
                                RMS_sim = RMS_result.where(np.isfinite(RMS_result)).mean(skipna=True).values
                                

                            output_file.write(f"{std_sim}\t")
                            stds[i + 1] = std_sim

                            output_file.write(f"{cor_sim}\t")
                            cors[i + 1] = cor_sim

                            output_file.write(f"{RMS_sim}\t")
                            RMSs[i + 1] = RMS_sim

                            if self.weight.lower() == 'area':
                                weights = np.cos(np.deg2rad(reffile.lat))
                                std_ref = self.stat_standard_deviation(reffile).where(np.isfinite(self.stat_standard_deviation(reffile))).weighted(weights).mean(skipna=True).values
                            elif self.weight.lower() == 'mass':
                                # Calculate area weights (cosine of latitude)
                                area_weights = np.cos(np.deg2rad(reffile.lat))
                                # Calculate absolute flux weights
                                flux_weights = np.abs(reffile.mean('time'))
                                # Combine area and flux weights
                                combined_weights = area_weights * flux_weights
                                # Normalize weights to sum to 1
                                normalized_weights = combined_weights / combined_weights.sum()
                                # Calculate weighted mean
                                std_ref = self.stat_standard_deviation(reffile).where(np.isfinite(self.stat_standard_deviation(reffile))).weighted(normalized_weights).mean(skipna=True).values
                            else:
                                std_ref = self.stat_standard_deviation(reffile).mean(skipna=True).values
                        stds[0] = std_ref

                    output_file.write(f"{std_ref}\n")

                make_scenarios_comparison_Taylor_Diagram(casedir, evaluation_item, stds, RMSs, cors, ref_source, sim_sources,
                                                         option)

    def scenarios_Target_Diagram_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'Target_Diagram')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # read the simulation source and reference source
        for evaluation_item in evaluation_items:
            # read the simulation source and reference source
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]

            for ref_source in ref_sources:
                output_file_path = f"{dir_path}/target_diagram_{evaluation_item}_{ref_source}.txt"

                with open(output_file_path, "w") as output_file:
                    output_file.write("Item\t")
                    output_file.write("Reference\t")
                    # ill determine the number of simulation sources
                    sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                    if isinstance(sim_sources, str): sim_sources = [sim_sources]
                    for sim_source in sim_sources:
                        output_file.write(f"{sim_source}_bias\t")
                        output_file.write(f"{sim_source}_crmsd\t")
                        output_file.write(f"{sim_source}_rmsd\t")

                    output_file.write("\n")  # Move "All" to the first line
                    output_file.write(f"{evaluation_item}\t")
                    output_file.write(f"{ref_source}\t")
                    biases = np.zeros(len(sim_sources))
                    rmses = np.zeros(len(sim_sources))
                    crmsds = np.zeros(len(sim_sources))
                    for i, sim_source in enumerate(sim_sources):
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if isinstance(sim_sources, str): sim_sources = [sim_sources]
                        if isinstance(ref_sources, str): ref_sources = [ref_sources]
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            stnlist = f"{casedir}/stn_list.txt"
                            station_list = pd.read_csv(stnlist, header=0)

                            def _make_validation_parallel(casedir, ref_source, sim_source, item, sim_varname, ref_varname,
                                                          station_list, iik):
                                s = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    sim_varname].squeeze()
                                o = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    ref_varname].squeeze()

                                s['time'] = o['time']
                                mask1 = np.isnan(s) | np.isnan(o)
                                s.values[mask1] = np.nan
                                o.values[mask1] = np.nan
                                row = {}

                                try:
                                    row['CRMSD'] = self.CRMSD(s, o).values  # .values
                                except:
                                    row['CRMSD'] = np.nan
                                try:
                                    row['bias'] = self.bias(s, o).values  # .values
                                except:
                                    row['bias'] = np.nan
                                try:
                                    row['rmse'] = self.RMSE(s, o).values  # .values
                                except:
                                    row['rmse'] = np.nan
                                return row

                            results = Parallel(n_jobs=-1)(
                                delayed(_make_validation_parallel)(casedir, ref_source, sim_source, evaluation_item,
                                                                   sim_varname, ref_varname, station_list, iik) for iik in
                                range(len(station_list['ID'])))

                            station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

                            station_list.to_csv(f"{dir_path}/target_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.txt")

                            station_list = pd.read_csv(
                                f"{dir_path}/target_diagram_{evaluation_item}_stn_{ref_source}_{sim_source}.txt", header=0)

                            bias_sim = station_list['bias'].mean(skipna=True)
                            output_file.write(f"{bias_sim}\t")
                            biases[i] = bias_sim

                            rmse_sim = station_list['rmse'].mean(skipna=True)
                            output_file.write(f"{rmse_sim}\t")
                            rmses[i] = rmse_sim

                            crmsd_sim = station_list['CRMSD'].mean(skipna=True)
                            output_file.write(f"{crmsd_sim}\t")
                            crmsds[i] = crmsd_sim
                        else:
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            reffile = \
                                xr.open_dataset(f'{casedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                    ref_varname]
                            simfile = \
                                xr.open_dataset(f'{casedir}/output/data/{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')[
                                    sim_varname]

                            bias_sim_result = self.bias(simfile, reffile)
                            if self.weight.lower() == 'area':
                                weights = np.cos(np.deg2rad(reffile.lat))
                                bias_sim = bias_sim_result.where(np.isfinite(bias_sim_result)).weighted(weights).mean(skipna=True).values
                            elif self.weight.lower() == 'mass':
                                # Calculate area weights (cosine of latitude)
                                area_weights = np.cos(np.deg2rad(reffile.lat))
                                # Calculate absolute flux weights
                                flux_weights = np.abs(reffile.mean('time'))
                                # Combine area and flux weights
                                combined_weights = area_weights * flux_weights
                                # Normalize weights to sum to 1
                                normalized_weights = combined_weights / combined_weights.sum()
                                # Calculate weighted mean
                                bias_sim = bias_sim_result.where(np.isfinite(bias_sim_result)).weighted(normalized_weights).mean(skipna=True).values
                            else:
                                bias_sim = bias_sim_result.mean(skipna=True).values

                            output_file.write(f"{bias_sim}\t")
                            biases[i] = bias_sim
                            rmse_sim = self.RMSE(simfile, reffile).mean(skipna=True).values
                            output_file.write(f"{rmse_sim}\t")
                            rmses[i] = rmse_sim
                            crmsd_sim = self.CRMSD(simfile, reffile).mean(skipna=True).values
                            output_file.write(f"{crmsd_sim}\t")
                            crmsds[i] = crmsd_sim

                    output_file.write("\n")
                    make_scenarios_comparison_Target_Diagram(dir_path, evaluation_item, biases, rmses, crmsds, ref_source,
                                                             sim_sources, option)

    def scenarios_Kernel_Density_Estimate_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Kernel_Density_Estimate')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # fixme: add the Kernel Density Estimate
        for evaluation_item in evaluation_items:
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            for score in scores:
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
                            file_path = f"{basedir}/output/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[score].values
                        else:
                            file_path = f"{basedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                            ds = xr.open_dataset(file_path)
                            data = ds[score].values
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append
                    # try:
                    make_scenarios_comparison_Kernel_Density_Estimate(dir_path, evaluation_item, ref_source, sim_sources,
                                                                      score, datasets_filtered, option)
                    # except:
                    #     print(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Kernel Density Estimate failed!")

            for metric in metrics:
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
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[metric].values
                        else:
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc"
                            ds = xr.open_dataset(file_path)
                            data = ds[metric].values
                        data = data[~np.isinf(data)]
                        if metric == 'percent_bias':
                            data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    # try:
                    make_scenarios_comparison_Kernel_Density_Estimate(dir_path, evaluation_item, ref_source, sim_sources,
                                                                      metric, datasets_filtered, option)
                    # except:
                    #     print(f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed!")

    def scenarios_Parallel_Coordinates_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Parallel_Coordinates')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        output_file_path = f"{dir_path}/Parallel_Coordinates_evaluations.txt"
        with open(output_file_path, "w") as output_file:
            output_file.write("Item\t")
            output_file.write("Reference\t")
            output_file.write("Simulation\t")
            for score in scores:
                output_file.write(f"{score}\t")
            for metric in metrics:
                output_file.write(f"{metric}\t")
            output_file.write("\n")  # Move "All" to the first line

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                # read the simulation source and reference source
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for i, sim_source in enumerate(sim_sources):
                        output_file.write(f"{evaluation_item}\t")
                        output_file.write(f"{ref_source}\t")
                        output_file.write(f"{sim_source}\t")
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            file_path = f"{basedir}/output/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            df = pd.read_csv(file_path, sep=',', header=0)

                            for score in scores:
                                kk = df[score].mean(skipna=True)
                                kk_str = f"{kk:.2f}" if not np.isnan(kk) else "N/A"
                                output_file.write(f"{kk_str}\t")

                            for metric in metrics:
                                df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
                                q_low, q_high = df[metric].quantile([0.05, 0.95])
                                df[metric] = df[metric].where((df[metric] >= q_low) & (df[metric] <= q_high), np.nan)

                                kk = df[metric].median(skipna=True)
                                kk_str = f"{kk:.2f}" if not np.isnan(kk) else "N/A"
                                output_file.write(f"{kk_str}\t")

                            output_file.write("\n")
                        else:
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            reffile = \
                                xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                    ref_varname]
                            simfile = \
                                xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')[
                                    sim_varname]
                            for score in scores:
                                ds = xr.open_dataset(
                                    f'{self.casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc')
                                if self.weight.lower() == 'area':
                                    weights = np.cos(np.deg2rad(reffile.lat))
                                    kk = ds[score].where(np.isfinite(ds[score]), np.nan).weighted(weights).mean(skipna=True).values
                                elif self.weight.lower() == 'mass':
                                    # Calculate area weights (cosine of latitude)
                                    area_weights = np.cos(np.deg2rad(reffile.lat))
                                    # Calculate absolute flux weights
                                    flux_weights = np.abs(reffile.mean('time'))
                                    # Combine area and flux weights
                                    combined_weights = area_weights * flux_weights
                                    # Normalize weights to sum to 1
                                    normalized_weights = combined_weights / combined_weights.sum()
                                    # Calculate weighted mean
                                    kk = ds[score].where(np.isfinite(ds[score]), np.nan).weighted(normalized_weights).mean(skipna=True).values
                                else:
                                    kk = ds[score].mean(skipna=True).values
                                kk_str = f"{kk:.2f}" if not np.isnan(kk) else "N/A"
                                output_file.write(f"{kk_str}\t")
                            for metric in metrics:
                                ds = xr.open_dataset(
                                    f'{self.casedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc')

                                ds = ds.where(np.isfinite(ds), np.nan)
                                q_value = ds[metric].quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                                ds = ds.where((ds >= q_value[0]) & (ds <= q_value[1]), np.nan)
                                kk = ds[metric].median(skipna=True).values
                                # kk = ds[metric].where(np.isfinite(ds[metric]), np.nan).mean(skipna=True).values
                                kk_str = f"{kk:.2f}" if not np.isnan(kk) else "N/A"
                                output_file.write(f"{kk_str}\t")
                            output_file.write("\n")
                            del reffile, simfile, ds, kk, kk_str
        #
        # # #deal with output_file_path, remove the column and its index with any nan values
        df = pd.read_csv(output_file_path, sep='\t', header=0)
        # #find the columns with any nan values, drop the index
        df = df.dropna(axis=1, how='any')
        # if index in scroes or metrics was dropped, then remove the corresponding scores or metrics
        scores = [score for score in scores if score in df.columns]
        metrics = [metric for metric in metrics if metric in df.columns]
        output_file_path1 = f"{dir_path}/Parallel_Coordinates_evaluations_remove_nan.txt"
        df.to_csv(output_file_path1, sep='\t', index=False)
        make_scenarios_comparison_parallel_coordinates(output_file_path1, self.casedir, evaluation_items, scores, metrics, option)

    def scenarios_Portrait_Plot_seasonal_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        # scenarios_portrait_plot is special, need to recalculate the scores and metrics
        # read the simulation source and reference source
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Portrait_Plot_seasonal')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        def process_metric(casedir, item, ref_source, sim_source, metric, s, o, vkey=None):
            pb = getattr(self, metric)(s, o)
            pb = pb.where(np.isfinite(pb), np.nan)
            try:
                # q1, q3 = np.percentile(pb.values[~np.isnan(pb.values)], [5, 95])
                q_value = pb.quantile([0.05, 0.95], dim=['lat', 'lon'], skipna=True)
                pb = pb.where((pb >= q_value[0]) & (pb <= q_value[1]), np.nan)
            except:
                pass

            try:
                pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
                pb_da.to_netcdf(
                    f'{casedir}/output/comparisons/Portrait_Plot_seasonal/{item}_ref_{ref_source}_sim_{sim_source}_{metric}{vkey}.nc')
            except:
                pass
            return np.nanmedian(pb)

        def process_score(casedir, item, ref_source, sim_source, score, s, o, vkey=None):
            pb = getattr(self, score)(s, o)
            try:
                pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
                pb_da.to_netcdf(
                    f'{casedir}/output/comparisons/Portrait_Plot_seasonal/{item}_ref_{ref_source}_sim_{sim_source}_{score}{vkey}.nc')
            except:
                pass
            if self.weight.lower() == 'area':
                weights = np.cos(np.deg2rad(o.lat))
                pb = pb.where(np.isfinite(pb), np.nan).weighted(weights).mean(skipna=True)
            elif self.weight.lower() == 'mass':
                # Calculate area weights (cosine of latitude)
                area_weights = np.cos(np.deg2rad(o.lat))
                # Calculate absolute flux weights
                flux_weights = np.abs(o.mean('time'))
                # Combine area and flux weights
                combined_weights = area_weights * flux_weights
                # Normalize weights to sum to 1
                normalized_weights = combined_weights / combined_weights.sum()
                # Calculate weighted mean
                pb = pb.where(np.isfinite(pb), np.nan).weighted(normalized_weights).mean(skipna=True)
            else:
                pb = pb.mean(skipna=True)
            return pb

        output_file_path = f"{dir_path}/Portrait_Plot_seasonal.txt"

        with open(output_file_path, "w") as output_file:
            output_file.write("Item\t")
            output_file.write("Reference\t")
            output_file.write("Simulation\t")

            for metric in metrics:
                output_file.write(f"{metric}_DJF\t")
                output_file.write(f"{metric}_MAM\t")
                output_file.write(f"{metric}_JJA\t")
                output_file.write(f"{metric}_SON\t")

            for score in scores:
                output_file.write(f"{score}_DJF\t")
                output_file.write(f"{score}_MAM\t")
                output_file.write(f"{score}_JJA\t")
                output_file.write(f"{score}_SON\t")

            output_file.write("\n")  # Move "All" to the first line

            # read the simulation source and reference source
            for evaluation_item in evaluation_items:
                print("now processing the evaluation item: ", evaluation_item)
                # read the simulation source and reference source
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                # if the sim_sources and ref_sources are not list, then convert them to list
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]
                for ref_source in ref_sources:
                    for i, sim_source in enumerate(sim_sources):
                        output_file.write(f"{evaluation_item}\t")
                        output_file.write(f"{ref_source}\t")
                        output_file.write(f"{sim_source}\t")
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            stnlist = f"{basedir}/stn_list.txt"
                            station_list = pd.read_csv(stnlist, header=0)

                            def _process_station_data_parallel(casedir, ref_source, sim_source, item, sim_varname, ref_varname,
                                                               station_list, iik, metric_or_score, season, metric=None,
                                                               score=None):
                                s = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    sim_varname].squeeze()
                                o = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                                    ref_varname].squeeze()

                                s['time'] = o['time']
                                mask1 = np.isnan(s) | np.isnan(o)
                                s.values[mask1] = np.nan
                                o.values[mask1] = np.nan

                                s_season = s.sel(time=s['time.season'] == season)
                                o_season = o.sel(time=o['time.season'] == season)

                                if metric_or_score == 'metric':
                                    return process_metric(casedir, item, ref_source, sim_source, metric, s_season, o_season)
                                elif metric_or_score == 'score':
                                    return process_score(casedir, item, ref_source, sim_source, score, s_season, o_season)

                            seasons = ['DJF', 'MAM', 'JJA', 'SON']
                            for metric in metrics:
                                for season in seasons:
                                    results = Parallel(n_jobs=-1)(
                                        delayed(_process_station_data_parallel)(basedir, ref_source, sim_source, evaluation_item,
                                                                                sim_varname, ref_varname, station_list, iik,
                                                                                'metric', season, metric=metric)
                                        for iik in range(len(station_list['ID'])))
                                    results = np.array(results)
                                    q1, q3 = np.percentile(results[~np.isnan(results)], [5, 95])
                                    results = np.where((results >= q1) & (results <= q3), results, np.nan)

                                    mean_value = np.nanmedian(results)
                                    kk_str = f"{mean_value:.2f}" if not np.isnan(mean_value) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                            for score in scores:
                                for season in seasons:
                                    results = Parallel(n_jobs=-1)(
                                        delayed(_process_station_data_parallel)(basedir, ref_source, sim_source, evaluation_item,
                                                                                sim_varname, ref_varname, station_list, iik,
                                                                                'score', season, score=score)
                                        for iik in range(len(station_list['ID'])))
                                    mean_value = np.nanmean(results)
                                    kk_str = f"{mean_value:.2f}" if not np.isnan(mean_value) else "N/A"
                                    output_file.write(f"{kk_str}\t")
                        else:
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            o = xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                f'{ref_varname}']
                            s = xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')[
                                f'{sim_varname}']
                            o = o.where(np.isfinite(o), np.nan)
                            s = s.where(np.isfinite(s), np.nan)
                            s['time'] = o['time']

                            mask1 = np.isnan(s) | np.isnan(o)
                            s.values[mask1] = np.nan
                            o.values[mask1] = np.nan
                            s_DJF = s.sel(time=s['time.season'] == 'DJF')
                            o_DJF = o.sel(time=o['time.season'] == 'DJF')
                            s_MAM = s.sel(time=s['time.season'] == 'MAM')
                            o_MAM = o.sel(time=o['time.season'] == 'MAM')
                            s_JJA = s.sel(time=s['time.season'] == 'JJA')
                            o_JJA = o.sel(time=o['time.season'] == 'JJA')
                            s_SON = s.sel(time=s['time.season'] == 'SON')
                            o_SON = o.sel(time=o['time.season'] == 'SON')
                            for metric in metrics:
                                if hasattr(self, metric):
                                    k = process_metric(basedir, evaluation_item, ref_source, sim_source, metric, s_DJF, o_DJF,
                                                       vkey='_DJF')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_metric(basedir, evaluation_item, ref_source, sim_source, metric, s_MAM, o_MAM,
                                                       vkey='_MAM')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_metric(basedir, evaluation_item, ref_source, sim_source, metric, s_JJA, o_JJA,
                                                       vkey='_JJA')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_metric(basedir, evaluation_item, ref_source, sim_source, metric, s_SON, o_SON,
                                                       vkey='_SON')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                else:
                                    print('No such metric: ', metric)
                                    sys.exit(1)

                            for score in scores:
                                if hasattr(self, score):
                                    k = process_score(basedir, evaluation_item, ref_source, sim_source, score, s_DJF, o_DJF,
                                                      vkey=f'_DJF')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_score(basedir, evaluation_item, ref_source, sim_source, score, s_MAM, o_MAM,
                                                      vkey=f'_MAM')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_score(basedir, evaluation_item, ref_source, sim_source, score, s_JJA, o_JJA,
                                                      vkey=f'_JJA')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")

                                    k = process_score(basedir, evaluation_item, ref_source, sim_source, score, s_SON, o_SON,
                                                      vkey=f'_SON')
                                    kk_str = f"{k:.2f}" if not np.isnan(k) else "N/A"
                                    output_file.write(f"{kk_str}\t")
                                else:
                                    print('No such score: ', score)
                                    sys.exit(1)
                        output_file.write("\n")



        make_scenarios_comparison_Portrait_Plot_seasonal(output_file_path, self.casedir, evaluation_items, scores, metrics,
                                                         option)

    def scenarios_Whisker_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Whisker_Plot')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # If the sim_sources and ref_sources are not lists, convert them to lists
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]
            if isinstance(ref_sources, str):
                ref_sources = [ref_sources]

            for score in scores:
                for ref_source in ref_sources:
                    datasets_filtered = []
                    # Create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        # if isinstance(sim_sources, str):
                        #     sim_sources = [sim_sources]
                        # if isinstance(ref_sources, str):
                        #     ref_sources = [ref_sources]

                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            file_path = f"{basedir}/output/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # Read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[score].values
                        else:
                            file_path = f"{basedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                            ds = xr.open_dataset(file_path)
                            data = ds[score].values
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Whisker_Plot(dir_path, evaluation_item, ref_source, sim_sources, score,
                                                               datasets_filtered, option)
                    except:
                        print(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Whisker Plot failed!")

            for metric in metrics:
                for ref_source in ref_sources:
                    dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Whisker_Plot')
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    datasets_filtered = []
                    # Create a numpy matrix to store the data
                    for sim_source in sim_sources:
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']
                        if isinstance(sim_sources, str):
                            sim_sources = [sim_sources]
                        if isinstance(ref_sources, str):
                            ref_sources = [ref_sources]

                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # Read the file_path data and select the metric
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[metric].values
                        else:
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc"
                            ds = xr.open_dataset(file_path)
                            data = ds[metric].values
                        data = data[~np.isinf(data)]
                        # if metric == 'percent_bias':
                        #     data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Whisker_Plot(dir_path, evaluation_item, ref_source, sim_sources, metric,
                                                               datasets_filtered, option)
                    except:
                        print(f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Whisker Plot failed!")

    def scenarios_Relative_Score_comparison(self, casedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):

        dir_path = os.path.join(f'{casedir}', 'output', 'comparisons', 'Relative_Score')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']

            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]

            for ref_source in ref_sources:
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]

                for sim_source in sim_sources:
                    ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                    sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                    if ref_data_type == 'stn' or sim_data_type == 'stn':
                        file_pattern = f"{casedir}/output/scores/{evaluation_item}_stn_{ref_source}_*_evaluations.csv"
                        all_files = glob.glob(file_pattern)

                        if not all_files:
                            print(f"No files found for pattern: {file_pattern}")
                            continue

                        combined_relative_scores = pd.DataFrame()

                        # Read all files
                        for score in scores:
                            dfs = []
                            score_column = None
                            for file in all_files:
                                df = pd.read_csv(file, sep=',', header=0)
                                df1 = df[f'{score}']
                                dfs.append(df1)

                                if not dfs:
                                    print(f"No valid data found for {evaluation_item}, {ref_source}, {sim_source}, {score}")
                                    continue

                                # Combine all dataframes
                                combined_df = pd.concat(dfs, axis=1)
                                score_mean = combined_df.mean(axis=1)
                                score_std = combined_df.std(axis=1)

                            # Calculate relative scores for each file
                            for i, filex in enumerate(all_files):
                                df = pd.read_csv(filex, sep=',', header=0)
                                combined_relative_scores['ID'] = df['ID']
                                relative_scores = (df[f'{score}'] - score_mean) / score_std

                                # Add the relative scores as a new column to the combined dataframe
                                combined_relative_scores[f'relative_{score}_{sim_source}'] = relative_scores

                        # Check if any valid relative scores were calculated
                        if not combined_relative_scores.empty:
                            # Save the combined relative scores to a single file
                            combined_relative_scores.to_csv(
                                f"{dir_path}/{evaluation_item}_stn_{ref_source}_relative_scores.csv",
                                index=False  # Exclude the row index
                            )
                        else:
                            print(f"No valid data found for {evaluation_item}, {ref_source}")  # More specific message

                    else:
                        for score in scores:
                            file_pattern = f'{casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_*_{score}.nc'
                            all_files = glob.glob(file_pattern)
                            if not all_files:
                                print(f"No files found for pattern: {file_pattern}")
                                continue
                            # Read all files and combine into a single dataset
                            datasets = []
                            for file in all_files:
                                ds = xr.open_dataset(file)
                                datasets.append(ds)

                            if not datasets:
                                print(f"No valid data found for {evaluation_item}, {ref_source}, {sim_source}, {score}")
                                continue

                            combined_ds = xr.concat(datasets, dim='file')

                            # Calculate mean and standard deviation for each grid point
                            score_mean = combined_ds[score].mean(dim='file')
                            score_std = combined_ds[score].std(dim='file')

                            # Calculate relative scores for each file and grid point
                            for sim_source in sim_sources:
                                file = f'{casedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc'
                                ds = xr.open_dataset(file)
                                relative_score = (ds[score] - score_mean) / score_std

                                # Create a new dataset to store the relative score
                                result_ds = xr.Dataset()
                                result_ds[f'relative_{score}'] = relative_score

                                output_file = f'{dir_path}/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_Relative{score}.nc'
                                result_ds.to_netcdf(output_file)

                                # print(f"Relative scores calculated and saved for {evaluation_item}, {ref_source}, {sim_source}, {score}")

    def scenarios_Single_Model_Performance_Index_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics,
                                                            option):
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Single_Model_Performance_Index')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        def calculate_smpi(s, o):
            # Calculate observational variance
            obs_var = np.var(o, axis=0, ddof=1)
            s_climate = s.mean(dim='time')
            o_climate = o.mean(dim='time')

            # Calculate squared differences
            diff_squared = (s_climate - o_climate) ** 2

            # Normalize by observational variance
            normalized_diff = diff_squared / obs_var

            diff_squared0 = (s - o) ** 2
            normalized_diff0 = diff_squared0 / obs_var

            # Calculate SMPI
            smpi = np.nanmean(normalized_diff)
            smpi = np.nanmean(normalized_diff0)

            if smpi > 100:
                smpi = 100

            # Bootstrap for uncertainty estimation
            n_bootstrap = 100
            bootstrap_smpi = []
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(o), size=len(o), replace=True)
                bootstrap_sample = normalized_diff0[bootstrap_indices]
                bootstrap_smpi.append(np.nanmean(bootstrap_sample))

            bootstrap_smpi = np.array(bootstrap_smpi)
            smpi_lower, smpi_upper = np.percentile(bootstrap_smpi, [5, 95])

            return smpi, smpi_lower, smpi_upper

        def process_smpi(casedir, item, ref_source, sim_source, s, o):
            # Calculate SMPI for each grid point
            obs_var = np.var(o, axis=0, ddof=1)
            s_climate = s.mean(dim='time')
            o_climate = o.mean(dim='time')

            diff_squared = (s_climate - o_climate) ** 2
            normalized_diff = diff_squared / obs_var
            normalized_diff = normalized_diff.where(normalized_diff < 100)

            # Calculate overall SMPI
            smpi = normalized_diff.mean().values
            # Bootstrap for uncertainty estimation
            n_bootstrap = 1000
            bootstrap_smpi = []
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(s_climate), size=len(s_climate), replace=True)
                bootstrap_sample = normalized_diff[bootstrap_indices]
                bootstrap_smpi.append(np.nanmean(bootstrap_sample))

            bootstrap_smpi = np.array(bootstrap_smpi)
            smpi_lower, smpi_upper = np.percentile(bootstrap_smpi, [5, 95])

            # Save grid-based SMPI
            try:
                smpi_da = xr.DataArray(normalized_diff, coords={'lat': o.lat, 'lon': o.lon}, dims=['lat', 'lon'], name='SMPI')
                smpi_da.to_netcdf(
                    f'{casedir}/output/comparisons/Single_Model_Performance_Index/{item}_ref_{ref_source}_sim_{sim_source}_SMPI_grid.nc')
            except Exception as e:
                print(f"Error saving grid-based SMPI: {e}")

            return smpi, smpi_lower, smpi_upper

        output_file_path = f"{dir_path}/SMPI_comparison.txt"

        with open(output_file_path, "w") as output_file:
            output_file.write("Item\tReference\tSimulation\tSMPI\tLower_CI\tUpper_CI\n")
            for evaluation_item in evaluation_items:
                sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
                ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
                if isinstance(sim_sources, str): sim_sources = [sim_sources]
                if isinstance(ref_sources, str): ref_sources = [ref_sources]

                for ref_source in ref_sources:
                    for sim_source in sim_sources:
                        output_file.write(f"{evaluation_item}\t{ref_source}\t{sim_source}\t")
                        ref_data_type = ref_nml[f'{evaluation_item}'][f'{ref_source}_data_type']
                        sim_data_type = sim_nml[f'{evaluation_item}'][f'{sim_source}_data_type']

                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            stnlist = f"{basedir}/stn_list.txt"
                            station_list = pd.read_csv(stnlist, header=0)

                            def _process_station_data_parallel(casedir, ref_source, sim_source, item, sim_varname, ref_varname,
                                                               station_list, iik):
                                s = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[
                                    sim_varname].squeeze()
                                o = xr.open_dataset(
                                    f"{casedir}/output/data/stn_{ref_source}_{sim_source}/{item}_ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[
                                    ref_varname].squeeze()

                                s['time'] = o['time']
                                mask1 = np.isnan(s) | np.isnan(o)
                                s.values[mask1] = np.nan
                                o.values[mask1] = np.nan

                                return calculate_smpi(s, o)

                            results = Parallel(n_jobs=-1)(
                                delayed(_process_station_data_parallel)(basedir, ref_source, sim_source, evaluation_item,
                                                                        sim_varname, ref_varname, station_list, iik)
                                for iik in range(len(station_list['ID'])))
                            smpi_values, lower_values, upper_values = zip(*results)
                            mean_smpi = np.nanmean(smpi_values)
                            mean_lower = np.nanmean(lower_values)
                            mean_upper = np.nanmean(upper_values)
                            output_file.write(f"{mean_smpi:.4f}\t{mean_lower:.4f}\t{mean_upper:.4f}\n")

                        else:
                            ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                            sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                            if sim_varname is None or sim_varname == '':
                                sim_varname = evaluation_item
                            if ref_varname is None or ref_varname == '':
                                ref_varname = evaluation_item
                            o = xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_ref_{ref_source}_{ref_varname}.nc')[
                                f'{ref_varname}']
                            s = xr.open_dataset(f'{basedir}/output/data/{evaluation_item}_sim_{sim_source}_{sim_varname}.nc')[
                                f'{sim_varname}']

                            s['time'] = o['time']
                            mask1 = np.isnan(s) | np.isnan(o)
                            s.values[mask1] = np.nan
                            o.values[mask1] = np.nan

                            smpi, lower, upper = process_smpi(basedir, evaluation_item, ref_source, sim_source, s, o)
                            output_file.write(f"{smpi:.4f}\t{lower:.4f}\t{upper:.4f}\n")

                print(f"Completed SMPI calculation for {evaluation_item}")
                print("===============================================================================")
        # After all calculations are done, call the plotting function
        make_scenarios_comparison_Single_Model_Performance_Index(basedir, evaluation_items, ref_nml, sim_nml, option)

        return

    def scenarios_Ridgeline_Plot_comparison(self, basedir, sim_nml, ref_nml, evaluation_items, scores, metrics, option):
        dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Ridgeline_Plot')
        #if os.path.exists(dir_path):
        #    shutil.rmtree(dir_path)
        #print(f"Re-creating output directory: {dir_path}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for evaluation_item in evaluation_items:
            sim_sources = sim_nml['general'][f'{evaluation_item}_sim_source']
            ref_sources = ref_nml['general'][f'{evaluation_item}_ref_source']
            # if the sim_sources and ref_sources are not list, then convert them to list
            if isinstance(sim_sources, str): sim_sources = [sim_sources]
            if isinstance(ref_sources, str): ref_sources = [ref_sources]
            for score in scores:
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
                            file_path = f"{basedir}/output/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[score].values
                        else:
                            file_path = f"{basedir}/output/scores/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc"
                            ds = xr.open_dataset(file_path)
                            data = ds[score].values
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(dir_path, evaluation_item, ref_source, sim_sources, score,
                                                                 datasets_filtered, option)
                    except:
                        print(f"Error: {evaluation_item} {ref_source} {sim_sources} {score} Ridgeline_Plot failed!")

            for metric in metrics:
                for ref_source in ref_sources:
                    dir_path = os.path.join(f'{basedir}', 'output', 'comparisons', 'Ridgeline_Plot')
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
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                            # read the file_path data and select the score
                            df = pd.read_csv(file_path, sep=',', header=0)
                            data = df[metric].values
                        else:
                            file_path = f"{basedir}/output/metrics/{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc"

                            ds = xr.open_dataset(file_path)
                            data = ds[metric].values
                        data = data[~np.isinf(data)]
                        if metric == 'percent_bias':
                            data = data[(data >= -100) & (data <= 100)]
                        datasets_filtered.append(data[~np.isnan(data)])  # Filter out NaNs and append

                    try:
                        make_scenarios_comparison_Ridgeline_Plot(dir_path, evaluation_item, ref_source, sim_sources, metric,
                                                                 datasets_filtered, option)
                    except:
                        print(f"Error: {evaluation_item} {ref_source} {sim_sources} {metric} Kernel Density Estimate failed!")

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
