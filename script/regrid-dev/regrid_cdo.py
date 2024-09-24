import xarray as xr
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from joblib import Parallel, delayed

class regridder_cdo:
   def __init__(self, info):
      self.name = 'regridder_cdo'
      self.version = '0.1'
      self.release = '0.1'
      self.date = 'Mar 2023'
      self.author = "Zhongwang Wei"

   def largest_area_fraction_remap_cdo(self,input_file, output_file, target_grid):
      """Performs largest area fraction remapping on a netCDF file.

      Args:
         input_file (str): Path to the input netCDF file.
         output_file (str): Path to save the remapped netCDF file.
         target_grid (str): Path to the target grid file (or grid description).
      """

      # Use subprocess to execute CDO command
      cmd = f"cdo remaplaf,{target_grid} {input_file} {output_file}"
      subprocess.run(cmd, shell=True, check=True)

      # Load remapped data with xarray for further analysis (optional)
      remapped_data = xr.open_dataset(output_file)

      return remapped_data
   def to_dict(self):

      return self.__dict__
