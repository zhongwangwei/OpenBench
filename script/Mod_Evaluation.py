import os
import re
import shutil
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Check the platform
from Mod_Metrics import metrics
from Mod_Scores import scores
from figlib import *



class Evaluation_grid(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_grid'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        os.makedirs(self.casedir + '/output/', exist_ok=True)

        print(" ")
        print("\033[1;32m╔═══════════════════════════════════════════════════════════════╗\033[0m")
        print("\033[1;32m║                Evaluation processes starting!                 ║\033[0m")
        print("\033[1;32m╚═══════════════════════════════════════════════════════════════╝\033[0m")
        print("\n")

    def process_metric(self, metric, s, o, vkey=''):
        pb = getattr(self, metric)(s, o)
        pb = pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
        pb_da.to_netcdf(
            f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc')

    def process_score(self, score, s, o, vkey=''):
        pb = getattr(self, score)(s, o)
        pb = pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
        pb_da.to_netcdf(f'{self.casedir}/output/scores/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc')


    def make_Evaluation(self, **kwargs):
        o = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc')[
            f'{self.ref_varname}']
        s = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc')[
            f'{self.sim_varname}']

        s['time'] = o['time'] 
        if self.item == 'Terrestrial_Water_Storage_Change':
            # Calculate time difference while preserving coordinates
            s_values = s.values
            s_values[1:,:,:] = s_values[1:,:,:] - s_values[:-1,:,:]
            s_values[0,:,:] = np.nan
            s.values = s_values
            # Save s to original file
            s.to_netcdf(f'{self.casedir}/output/data/{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc')

        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        print("\033[1;32m" + "=" * 80 + "\033[0m")
        for metric in self.metrics:
            if hasattr(self, metric):
                print(f'calculating metric: {metric}')
                self.process_metric(metric, s, o)
            else:
                print(f'No such metric: {metric}')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                print(f'calculating score: {score}')
                self.process_score(score, s, o)
            else:
                print(f'No such score: {score}')
                sys.exit(1)

        print("\033[1;32m" + "=" * 80 + "\033[0m")
        make_plot_index_grid(self)
        return

class Evaluation_stn(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_point'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.fig_nml = fig_nml
        self.__dict__.update(info)
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]

        print('Evaluation processes starting!')
        print("=======================================")
        print(" ")
        print(" ")

    def make_evaluation(self):
        # read station information
        stnlist = f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist, header=0)

        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            station_list[f'{metric}'] = [-9999.0] * len(station_list['ID'])
        for score in self.scores:
            station_list[f'{score}'] = [-9999.0] * len(station_list['ID'])
        for iik in range(len(station_list['ID'])):
            s = xr.open_dataset(
                f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                self.sim_varname]
            o = xr.open_dataset(
                f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
                self.ref_varname]
            s['time'] = o['time']
            mask1 = np.isnan(s) | np.isnan(o)
            s.values[mask1] = np.nan
            o.values[mask1] = np.nan

            for metric in self.metrics:
                if hasattr(self, metric):
                    pb = getattr(self, metric)(s, o)
                    station_list.loc[iik, f'{metric}'] = pb.values
                #  self.plot_stn(s.squeeze(),o.squeeze(),station_list['ID'][iik],self.ref_varname, float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),float(station_list['correlation'][iik]))
                else:
                    print('No such metric')
                    sys.exit(1)

            for score in self.scores:
                if hasattr(self, score):
                    pb = getattr(self, score)(s, o)

                else:
                    print('No such score')
                    sys.exit(1)

            print("=======================================")
            print(" ")
            print(" ")

        print('Comparison dataset prepared!')
        print("=======================================")
        print(" ")
        print(" ")
        print(f"send {self.ref_varname} evaluation to {self.ref_varname}_{self.sim_varname}_metrics.csv'")
        station_list.to_csv(
            f'{self.casedir}/output/metrics/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_metrics.csv',
            index=False)
        station_list.to_csv(
            f'{self.casedir}/output/scores/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_scores.csv',
            index=False)

    
    def make_evaluation_parallel(self, station_list, iik):
        s = xr.open_dataset(
            f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
            self.sim_varname].to_array().squeeze()
        o = xr.open_dataset(
            f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[
            self.ref_varname].to_array().squeeze()

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
            row['KGESS'] = self.KGESS(s, o).values
        except:
            row['KGESS'] = -9999.0
        try:
            row['RMSE'] = self.rmse(s, o).values
        except:
            row['RMSE'] = -9999.0
        try:
            row['correlation'] = self.correlation(s, o).values
        except:
            row['correlation'] = -9999.0

        for metric in self.metrics:
            if hasattr(self, metric):
                pb = getattr(self, metric)(s, o)
                if pb.values is not None:
                    row[f'{metric}'] = pb.values
                else:
                    row[f'{metric}'] = -9999.0
                    if 'ref_lat' in station_list:
                        lat_lon = [station_list['ref_lat'][iik], station_list['ref_lon'][iik]]
                    else:
                        lat_lon = [station_list['sim_lat'][iik], station_list['sim_lon'][iik]]
                    self.plot_stn(s.squeeze(), o.squeeze(), station_list['ID'][iik], self.ref_varname,
                                  float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),
                                  float(station_list['correlation'][iik]), lat_lon)
            else:
                print(f'No such metric: {metric}')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                pb2 = getattr(self, score)(s, o)
                # if pb2.values is not None:
                if pb2.values is not None:
                    row[f'{score}'] = pb2.values
                else:
                    row[f'{score}'] = -9999.0
            else:
                print('No such score')
                sys.exit(1)

        if 'ref_lat' in station_list:
            lat_lon = [station_list['ref_lat'][iik], station_list['ref_lon'][iik]]
        else:
            lat_lon = [station_list['sim_lat'][iik], station_list['sim_lon'][iik]]
        self.plot_stn(s, o, station_list['ID'][iik], self.ref_varname, float(row['RMSE']), float(row['KGESS']),
                      float(row['correlation']), lat_lon)
        return row
        # return station_list

    def make_evaluation_P(self):
        stnlist = f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist, header=0)
        num_cores = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count(),或者随意设定<=cpu核心数的数值
        # shutil.rmtree(f'{self.casedir}/output',ignore_errors=True)
        # creat tmp directory
        # os.makedirs(f'{self.casedir}/output', exist_ok=True)

        # loop the keys in self.variables
        # loop the keys in self.variables to get the metric output
        # for metric in self.metrics.keys():
        #    station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        # if self.ref_source.lower() == 'grdc':

        results = Parallel(n_jobs=-1)(
            delayed(self.make_evaluation_parallel)(station_list, iik) for iik in range(len(station_list['ID'])))
        station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

        print('Evaluation finished')
        print("=======================================")
        print(" ")
        print(" ")
        print(f"send evaluation to {self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")
        print(f"send evaluation to {self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")

        station_list.to_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                            index=False)
        station_list.to_csv(f'{self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',
                            index=False)
        make_plot_index_stn(self)

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
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            print(f"warning: station data is not supported for IGBP class comparison")
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
                                print('Error: No metrics for IGBP class comparison')

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
                                print('Error: No scores for IGBP class comparison')

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
                        ref_varname = ref_nml[f'{evaluation_item}'][f'{ref_source}_varname']
                        sim_varname = sim_nml[f'{evaluation_item}'][f'{sim_source}_varname']
                        if ref_data_type == 'stn' or sim_data_type == 'stn':
                            print(f"warning: station data is not supported for PFT class comparison")
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
                                print('Error: No scores for PFT class comparison')

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
                                print('Error: No scores for PFT class comparison')

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
            print(f"CDO remapping failed: {e}")
            print("Falling back to xarray-regrid remapping...")
            _PFT_class_remap(self)
        _scenarios_PFT_groupby(casedir, scores, metrics, sim_nml, ref_nml, evaluation_items)
