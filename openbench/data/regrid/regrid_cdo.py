import xarray as xr
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from openbench.util.Mod_Converttype import Convert_Type
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

      return Convert_Type.convert_nc(remapped_data)
   '''
   def remaplaf_with_3D_parallel(
      source_data, source_lon, source_lat, target_lon, target_lat, n_jobs=-1
      ):
      """Performs REMAPLAF remapping with time dimension and parallelization."""
      def remap_single_timestep(source_polygons, target_polygons, target_tree, source_data, t, k):
          """Remaps a single timestep of data."""
         remapped_values = np.zeros(len(target_polygons)) 
         for i, source_poly in enumerate(source_polygons):
            # Find potential target cell overlaps using the spatial index
            _, target_indices = target_tree.query(
                  source_poly.centroid.coords[0], k=k
         )  

         max_overlap_area = 0
         max_overlap_index = None
         for j in target_indices:
            target_poly = target_polygons[j]
            intersection_area = source_poly.intersection(target_poly).area
            if intersection_area > max_overlap_area:
               max_overlap_area = intersection_area
               max_overlap_index = j

         # Assign the source value to the target cell with the largest overlap
         remapped_values[max_overlap_index] = source_data.values[t, i]
         return remapped_values

      # Calculate average cell areas
      source_cell_area_avg = np.mean(
         np.abs(np.diff(source_lon, axis=1)) * np.abs(np.diff(source_lat, axis=0))
       )
      target_cell_area_avg = np.mean(
         np.abs(np.diff(target_lon, axis=1)) * np.abs(np.diff(target_lat, axis=0))
      )

      # Estimate k based on the area ratio
      k_estimate = int(np.ceil(source_cell_area_avg / target_cell_area_avg * 10))

      # Create polygons
      source_polygons = [
        Polygon(zip(lon, lat))
        for lon, lat in zip(source_lon.values, source_lat.values)
      ]
      target_polygons = [
        Polygon(zip(lon, lat))
        for lon, lat in zip(target_lon.values, target_lat.values)
      ]

      # Build a spatial index (k-d tree) for target cells
      target_coords = np.column_stack(
        (target_lon.values.ravel(), target_lat.values.ravel())
      )
      target_tree = cKDTree(target_coords)

      # Use joblib to parallelize remapping over time steps
      remapped_values = Parallel(n_jobs=n_jobs)(
        delayed(remap_single_timestep)(
            source_polygons, target_polygons, target_tree, source_data, t, k_estimate
        )
        for t in range(len(source_data["time"]))
      )

      # Convert the list of remapped values back to an xarray DataArray
      remapped_data = xr.DataArray(
        np.array(remapped_values),
        coords={"time": source_data["time"], "lat": target_lat, "lon": target_lon},
        dims=("time", "lat", "lon"),
      )

      return remapped_data 

   def remaplaf(source_data, source_lon, source_lat, target_lon, target_lat):
      """Performs REMAPLAF remapping on a 2D dataset.

      Args:
         source_data (xarray.DataArray): The 2D data to be remapped.
         source_lon (xarray.DataArray): Longitude coordinates of the source grid.
         source_lat (xarray.DataArray): Latitude coordinates of the source grid.
         target_lon (xarray.DataArray): Longitude coordinates of the target grid.
         target_lat (xarray.DataArray): Latitude coordinates of the target grid.

      Returns:
         xarray.DataArray: The remapped data on the target grid.
      """

      # Calculate average cell areas
      source_cell_area_avg = np.mean(
        np.abs(np.diff(source_lon, axis=1)) * np.abs(np.diff(source_lat, axis=0))
         )
      target_cell_area_avg = np.mean(
        np.abs(np.diff(target_lon, axis=1)) * np.abs(np.diff(target_lat, axis=0))
         )

      # Estimate k based on the area ratio
      k_estimate = int(np.ceil(source_cell_area_avg / target_cell_area_avg * 10))

      # Create polygons for source and target grid cells
      source_polygons = [
         Polygon(zip(lon, lat))
         for lon, lat in zip(source_lon.values, source_lat.values)
         ]
      target_polygons = [
         Polygon(zip(lon, lat))
         for lon, lat in zip(target_lon.values, target_lat.values)
         ]

      # Build a spatial index (k-d tree) for target cells
      target_coords = np.column_stack(
        (target_lon.values.ravel(), target_lat.values.ravel())
         )
      target_tree = cKDTree(target_coords)

      # Initialize the remapped data array
      remapped_data = xr.DataArray(
         np.zeros_like(target_lon),
         coords={"lat": target_lat, "lon": target_lon},
         dims=("lat", "lon"),
         )

      # Perform remapping
      for i, source_poly in enumerate(source_polygons):
         # Find potential target cell overlaps using the spatial index
         _, target_indices = target_tree.query(
            source_poly.centroid.coords[0], k=k_estimate
         )  

         max_overlap_area = 0
         max_overlap_index = None
         for j in target_indices:
            target_poly = target_polygons[j]
            intersection_area = source_poly.intersection(target_poly).area
            if intersection_area > max_overlap_area:
               max_overlap_area = intersection_area
               max_overlap_index = j

         # Assign the source value to the target cell with the largest overlap
         remapped_data.values.flat[max_overlap_index] = source_data.values.flat[i]

   
   return remapped_data
   '''
   def to_dict(self):

      return self.__dict__
