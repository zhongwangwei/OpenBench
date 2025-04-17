import os
import re
import shutil
import sys
import warnings
import logging
import gc

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
from Mod_Converttype import Convert_Type

# Configure logging
logging.getLogger('xarray').setLevel(logging.WARNING)  # Suppress INFO messages from xarray
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress numpy runtime warnings
logging.getLogger('dask').setLevel(logging.WARNING)  # Suppress INFO messages from dask
class Evaluation_grid(metrics, scores):
    def __init__(self, info, fig_nml):
        self.name = 'Evaluation_grid'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)
        self.fig_nml = fig_nml
        os.makedirs(os.path.join(self.casedir, 'output'), exist_ok=True)

        logging.info(" ")
        logging.info("╔═══════════════════════════════════════════════════════════════╗")
        logging.info("║                Evaluation processes starting!                 ║")
        logging.info("╚═══════════════════════════════════════════════════════════════╝")
        logging.info(" ")

    def process_metric(self, metric, s, o, vkey=''):
        try:
            pb = getattr(self, metric)(s, o)
            pb = pb.squeeze()
            pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
            output_path = os.path.join(self.casedir, 'output', 'metrics', 
                                     f'{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pb_da.to_netcdf(output_path)
            logging.info(f"Saved metric {metric} to {output_path}")
        finally:
            gc.collect()  # Clean up memory after processing each metric

    def process_score(self, score, s, o, vkey=''):
        try:
            pb = getattr(self, score)(s, o)
            pb = pb.squeeze()
            pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
            output_path = os.path.join(self.casedir, 'output', 'scores', 
                                     f'{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pb_da.to_netcdf(output_path)
            logging.info(f"Saved score {score} to {output_path}")
        finally:
            gc.collect()  # Clean up memory after processing each score

    def make_Evaluation(self, **kwargs):
        try:
            ref_path = os.path.join(self.casedir, 'output', 'data', 
                                  f'{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc')
            sim_path = os.path.join(self.casedir, 'output', 'data', 
                                  f'{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc')
            
            o = xr.open_dataset(ref_path)[f'{self.ref_varname}']
            s = xr.open_dataset(sim_path)[f'{self.sim_varname}']
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            s['time'] = o['time'] 
            if self.item == 'Terrestrial_Water_Storage_Change':
                logging.info("Processing Terrestrial Water Storage Change...")
                # Calculate time difference while preserving coordinates
                s_values = s.values
                s_values[1:,:,:] = s_values[1:,:,:] - s_values[:-1,:,:]
                s_values[0,:,:] = np.nan
                s.values = s_values
                # Save s to original file
                s.to_netcdf(sim_path)

            mask1 = np.isnan(s) | np.isnan(o)
            s.values[mask1] = np.nan
            o.values[mask1] = np.nan
            logging.info("=" * 80)
            
            for metric in self.metrics:
                if hasattr(self, metric):
                    logging.info(f'Calculating metric: {metric}')
                    self.process_metric(metric, s, o)
                else:
                    logging.error(f'No such metric: {metric}')
                    sys.exit(1)

            for score in self.scores:
                if hasattr(self, score):
                    logging.info(f'Calculating score: {score}')
                    self.process_score(score, s, o)
                else:
                    logging.error(f'No such score: {score}')
                    sys.exit(1)

            logging.info("=" * 80)
            make_plot_index_grid(self)
        finally:
            gc.collect()  # Final cleanup

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

        logging.info('Evaluation processes starting!')
        logging.info("=======================================")
        logging.info(" ")
        logging.info(" ")

    def make_evaluation(self):
        try:
            # read station information
            stnlist = os.path.join(self.casedir, "stn_list.txt")
            station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist, header=0))

            # loop the keys in self.variables to get the metric output
            for metric in self.metrics:
                station_list[f'{metric}'] = [-9999.0] * len(station_list['ID'])
            for score in self.scores:
                station_list[f'{score}'] = [-9999.0] * len(station_list['ID'])
            
            for iik in range(len(station_list['ID'])):
                try:
                    sim_path = os.path.join(self.casedir, "output", "data", f"stn_{self.ref_source}_{self.sim_source}",
                                          f"sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")
                    ref_path = os.path.join(self.casedir, "output", "data", f"stn_{self.ref_source}_{self.sim_source}",
                                          f"ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")
                    
                    s = xr.open_dataset(sim_path)[self.sim_varname]
                    o = xr.open_dataset(ref_path)[self.ref_varname]
                    s['time'] = o['time']
                    mask1 = np.isnan(s) | np.isnan(o)
                    s.values[mask1] = np.nan
                    o.values[mask1] = np.nan

                    for metric in self.metrics:
                        if hasattr(self, metric):
                            pb = getattr(self, metric)(s, o)
                            station_list.loc[iik, f'{metric}'] = pb.values
                        else:
                            logging.error('No such metric')
                            sys.exit(1)

                    for score in self.scores:
                        if hasattr(self, score):
                            pb = getattr(self, score)(s, o)
                        else:
                            logging.error('No such score')
                            sys.exit(1)
                finally:
                    gc.collect()  # Clean up memory after each station

            logging.info('Comparison dataset prepared!')
            logging.info("=======================================")
            
            station_list = Convert_Type.convert_Frame(station_list)
            
            # Save metrics
            metrics_dir = os.path.join(self.casedir, 'output', 'metrics', f'stn_{self.ref_source}_{self.sim_source}')
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, f'{self.ref_varname}_{self.sim_varname}_metrics.csv')
            logging.info(f"Saving metrics to {metrics_path}")
            station_list.to_csv(metrics_path, index=False)
            
            # Save scores
            scores_dir = os.path.join(self.casedir, 'output', 'scores', f'stn_{self.ref_source}_{self.sim_source}')
            os.makedirs(scores_dir, exist_ok=True)
            scores_path = os.path.join(scores_dir, f'{self.ref_varname}_{self.sim_varname}_scores.csv')
            station_list.to_csv(scores_path, index=False)
        finally:
            gc.collect()  # Final cleanup

    def make_evaluation_parallel(self, station_list, iik):
        try:
            sim_path = os.path.join(self.casedir, "output", "data", f"stn_{self.ref_source}_{self.sim_source}",
                                  f"{self.item}_sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")
            ref_path = os.path.join(self.casedir, "output", "data", f"stn_{self.ref_source}_{self.sim_source}",
                                  f"{self.item}_ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")
            
            s = xr.open_dataset(sim_path)[self.sim_varname].to_array().squeeze()
            o = xr.open_dataset(ref_path)[self.ref_varname].to_array().squeeze()
            o = Convert_Type.convert_nc(o)
            s = Convert_Type.convert_nc(s)

            s['time'] = o['time']
            mask1 = np.isnan(s) | np.isnan(o)
            s.values[mask1] = np.nan
            o.values[mask1] = np.nan

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
                        plot_stn(self, s.squeeze(), o.squeeze(), station_list['ID'][iik], self.ref_varname,
                                      float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),
                                      float(station_list['correlation'][iik]), lat_lon)
                else:
                    logging.error(f'No such metric: {metric}')
                    sys.exit(1)

            for score in self.scores:
                if hasattr(self, score):
                    pb2 = getattr(self, score)(s, o)
                    if pb2.values is not None:
                        row[f'{score}'] = pb2.values
                    else:
                        row[f'{score}'] = -9999.0
                else:
                    logging.error('No such score')
                    sys.exit(1)

            if 'ref_lat' in station_list:
                lat_lon = [station_list['ref_lat'][iik], station_list['ref_lon'][iik]]
            else:
                lat_lon = [station_list['sim_lat'][iik], station_list['sim_lon'][iik]]
            plot_stn(self, s, o, station_list['ID'][iik], self.ref_varname, float(row['RMSE']), float(row['KGESS']),
                          float(row['correlation']), lat_lon)
            return row
        finally:
            gc.collect()  # Clean up memory after processing each station

    def make_evaluation_P(self):
        try:
            stnlist = os.path.join(self.casedir, "stn_list.txt")
            station_list = Convert_Type.convert_Frame(pd.read_csv(stnlist, header=0))
            
            results = Parallel(n_jobs=-1)(
                delayed(self.make_evaluation_parallel)(station_list, iik) for iik in range(len(station_list['ID'])))
            station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

            logging.info('Evaluation finished')
            logging.info("=======================================")

            station_list = Convert_Type.convert_Frame(station_list)
            
            # Save scores
            scores_path = os.path.join(self.casedir, 'output', 'scores',
                                     f'{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv')
            logging.info(f"Saving scores to {scores_path}")
            os.makedirs(os.path.dirname(scores_path), exist_ok=True)
            station_list.to_csv(scores_path, index=False)
            
            # Save metrics
            metrics_path = os.path.join(self.casedir, 'output', 'metrics',
                                      f'{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv')
            logging.info(f"Saving metrics to {metrics_path}")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            station_list.to_csv(metrics_path, index=False)
            
            make_plot_index_stn(self)
        finally:
            gc.collect()  # Final cleanup

