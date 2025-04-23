#!/usr/bin/env python3
"""
CAMA-Flood Water Surface Elevation (WSE) Evaluation dataset preparation Script
This script processes water surface elevation data from HydroWeb and other sources,
performs bias correction, and prepares data for CAMA-Flood model evaluation.
the original script is from AltiMaP, but I modified it to be more general and flexible.
"""

import os
import logging
import subprocess
import calendar
import re
import tarfile
from datetime import datetime
from typing import List, Dict, Optional, Union, Literal, Tuple
from multiprocessing import Pool, Process
from pathlib import Path
from enum import Enum
import sys

import numpy as np
import pandas as pd
import xarray as xr
import requests
from tqdm import tqdm

class Config:
    """Configuration class for storing processing parameters and paths."""
    
    def __init__(self):
        self.hydroweb_base_url = "https://hydroweb.theia-land.fr/hydroweb/api"
        self.output_dir = Path("output")
        self.cache_dir = Path("cache")
        self.log_dir = Path("logs")
        
        # Create necessary directories
        for directory in [self.output_dir, self.cache_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class Logger:
    """Custom logger class for tracking processing steps and errors."""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_logger()
    
    def setup_logger(self):
        """Set up logging configuration."""
        log_file = self.config.log_dir / f"processing_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("CAMA_WSE_Processor")

class EarthGravityModel1996:
    """Class for processing EGM96 geoid heights."""
    
    def __init__(self, grid_file: Path, logger: logging.Logger):
        self.grid_file = grid_file
        self.logger = logger
        self.data = None
        
        # These values were determined by inspecting the WW15MGH.DAC file
        self.minimum_height = -106.99
        self.maximum_height = 85.39
    
    def load_data(self) -> None:
        """Load and process the EGM96 grid data."""
        try:
            # Read the binary data
            with open(self.grid_file, 'rb') as f:
                data = f.read()
            
            # Convert to numpy array and swap byte order (file is big-endian)
            data_array = np.frombuffer(data, dtype='>i2')  # >i2 means big-endian 16-bit integer
            self.data = data_array.reshape(721, 1440)  # 721 rows (0° to 180°), 1440 columns (0° to 360°)
            
            self.logger.info("Successfully loaded EGM96 grid data")
        except Exception as e:
            self.logger.error(f"Failed to load EGM96 grid data: {str(e)}")
            raise
    
    def get_height(self, longitude: float, latitude: float) -> float:
        """
        Get the height of EGM96 above the surface of the ellipsoid.
        
        Args:
            longitude: Longitude in degrees
            latitude: Latitude in degrees
            
        Returns:
            Height in meters. Negative numbers indicate mean sea level is below the ellipsoid.
        """
        if self.data is None:
            self.load_data()
        
        # Convert degrees to radians
        lon_rad = np.radians(longitude)
        lat_rad = np.radians(latitude)
        
        # Calculate indices
        record_index = (720 * (np.pi * 0.5 - lat_rad)) / np.pi
        record_index = np.clip(record_index, 0, 720)
        
        # Put longitude in range 0 to 2π
        lon_rad = lon_rad % (2 * np.pi)
        height_index = (1440 * lon_rad) / (2 * np.pi)
        height_index = np.clip(height_index, 0, 1440)
        
        # Get integer indices for bilinear interpolation
        i = int(height_index)
        j = int(record_index)
        
        # Calculate interpolation weights
        x_minus_x1 = height_index - i
        y_minus_y1 = record_index - j
        x2_minus_x = 1.0 - x_minus_x1
        y2_minus_y = 1.0 - y_minus_y1
        
        # Get height values (handling edge cases)
        f11 = self.data[j, i % 1440]
        f21 = self.data[j, (i + 1) % 1440]
        f12 = self.data[min(j + 1, 720), i % 1440]
        f22 = self.data[min(j + 1, 720), (i + 1) % 1440]
        
        # Bilinear interpolation
        height = (f11 * x2_minus_x * y2_minus_y +
                 f21 * x_minus_x1 * y2_minus_y +
                 f12 * x2_minus_x * y_minus_y1 +
                 f22 * x_minus_x1 * y_minus_y1) / 100.0  # Convert from centimeters to meters
        
        return height
    
    def get_heights(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """
        Get EGM96 heights for multiple coordinates.
        
        Args:
            coordinates: List of (longitude, latitude) pairs in degrees
            
        Returns:
            List of heights in meters
        """
        return [self.get_height(lon, lat) for lon, lat in coordinates]

class EarthGravityModel2008:
    """Class for processing EGM08 geoid heights using bilinear interpolation."""
    
    def __init__(self, grid_file: Path, logger: logging.Logger):
        self.grid_file = grid_file
        self.logger = logger
        self.data = None
        # Grid specifications for 1x1 minute resolution
        self.nrows = 10801  # Number of latitude points (90° to -90° at 1 arcmin spacing)
        self.ncols = 21600  # Number of longitude points (0° to 360° at 1 arcmin spacing)
        self.dlat = 1.0/60.0  # Grid spacing in latitude (1 arcmin)
        self.dlon = 1.0/60.0  # Grid spacing in longitude (1 arcmin)
        self.top_lat = 90.0  # Starting latitude
        self.west_lon = 0.0  # Starting longitude
        
        # Statistics from the official documentation for validation
        self.min_height = -106.910  # meters
        self.max_height = 85.840    # meters
    
    def load_data(self) -> None:
        """Load and process the EGM08 grid data."""
        try:
            # Check if file exists
            if not self.grid_file.exists():
                raise FileNotFoundError(f"EGM2008 grid file not found at {self.grid_file}")
            
            # Get file size
            file_size = self.grid_file.stat().st_size
            expected_size = self.nrows * self.ncols * 4  # 4 bytes per value (REAL*4)
            
            if file_size != expected_size:
                self.logger.warning(f"File size mismatch. Expected {expected_size} bytes, got {file_size} bytes.")
            
            # Initialize data array
            self.data = np.zeros((self.nrows, self.ncols), dtype=np.float32)
            
            # Read the binary data row by row
            with open(self.grid_file, 'rb') as f:
                for i in range(self.nrows):
                    # Read record marker (4 bytes)
                    record_marker = f.read(4)
                    if not record_marker:
                        raise EOFError(f"Unexpected end of file at row {i}")
                    
                    # Read row data (ncols * 4 bytes)
                    row_data = f.read(self.ncols * 4)
                    if len(row_data) != self.ncols * 4:
                        raise EOFError(f"Incomplete row data at row {i}")
                    
                    # Convert row data to float32 array
                    row = np.frombuffer(row_data, dtype='>f4')  # big-endian float32
                    self.data[i] = row
                    
                    # Read end record marker (4 bytes)
                    end_marker = f.read(4)
                    if not end_marker:
                        raise EOFError(f"Missing end record marker at row {i}")
            
            # Validate data range
            data_min = np.min(self.data)
            data_max = np.max(self.data)
            if data_min < self.min_height or data_max > self.max_height:
                self.logger.warning(
                    f"Data range ({data_min:.3f} to {data_max:.3f}) "
                    f"outside expected range ({self.min_height} to {self.max_height})"
                )
            
            self.logger.info(f"Successfully loaded EGM08 grid data with shape {self.data.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load EGM08 grid data: {str(e)}")
            raise
    
    def bilinear_interpolation(self, ri: float, rj: float) -> float:
        """
        Perform bilinear interpolation at a specific grid position.
        
        Args:
            ri: Fractional row index
            rj: Fractional column index
            
        Returns:
            Interpolated value in meters
        """
        try:
            # Get integer indices
            i = int(ri)
            j = int(rj)
            
            # Ensure indices are within bounds
            i = max(0, min(i, self.nrows - 2))
            j = max(0, min(j, self.ncols - 2))
            
            # Calculate weights
            di = ri - i
            dj = rj - j
            
            # Handle longitude wrapping
            j1 = j % self.ncols
            j2 = (j + 1) % self.ncols
            
            # Get corner values
            v00 = float(self.data[i, j1])
            v10 = float(self.data[i + 1, j1])
            v01 = float(self.data[i, j2])
            v11 = float(self.data[i + 1, j2])
            
            # Perform bilinear interpolation
            value = (v00 * (1 - di) * (1 - dj) +
                    v10 * di * (1 - dj) +
                    v01 * (1 - di) * dj +
                    v11 * di * dj)
            
            return value  # Values are already in meters
            
        except Exception as e:
            self.logger.error(f"Error in bilinear interpolation: {str(e)}")
            raise
    
    def get_height(self, longitude: float, latitude: float) -> float:
        """
        Get the height of EGM08 above the surface of the ellipsoid.
        
        Args:
            longitude: Longitude in degrees (0 to 360 or -180 to 180)
            latitude: Latitude in degrees (-90 to 90)
            
        Returns:
            Height in meters. Negative numbers indicate mean sea level is below the ellipsoid.
        """
        try:
            if self.data is None:
                self.load_data()
            
            # Validate input coordinates
            if not (-90 <= latitude <= 90):
                raise ValueError(f"Latitude {latitude} is outside valid range [-90, 90]")
            
            # Normalize longitude to 0-360 range
            lon = longitude % 360.0
            
            # Convert lat/lon to grid indices
            # Note: Data starts at 90° N, so we need to flip the latitude index
            ri = (90.0 - latitude) / self.dlat
            rj = lon / self.dlon
            
            # Ensure indices are within bounds
            ri = max(0, min(ri, self.nrows - 1))
            rj = max(0, min(rj, self.ncols - 1))
            
            return self.bilinear_interpolation(ri, rj)
            
        except Exception as e:
            self.logger.error(f"Error getting height for lon={longitude}, lat={latitude}: {str(e)}")
            raise
    
    def get_heights(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """
        Get EGM08 heights for multiple coordinates.
        
        Args:
            coordinates: List of (longitude, latitude) pairs in degrees
            
        Returns:
            List of heights in meters
        """
        try:
            return [self.get_height(lon, lat) for lon, lat in coordinates]
        except Exception as e:
            self.logger.error(f"Error getting heights for multiple coordinates: {str(e)}")
            raise

class HydroWebDownloader:
    """Class for downloading and managing HydroWeb data."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
        self.base_url = "https://hydroweb.theia-land.fr/hydroweb"
        self.data_dir = Path("./data_for_wse/HydroWeb")
        self.datalist = "HydroWeb_VS"  # Type of data to download: rivers, lakes
        self.dataformat = "txt"   # Format of the data
        self.session = requests.Session()
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize EGM96 processor
        self.egm96 = None
    
    def initialize_egm96(self, egm96_file: Path) -> None:
        """Initialize the EGM96 processor with the grid file."""
        self.egm96 = EarthGravityModel1996(egm96_file, self.logger)
    
    def download_station_list(self) -> List[str]:
        """
        Download the list of available stations from HydroWeb.
        Returns a list of station IDs.
        """
        try:
            # Construct URL for station list
            url = f"{self.base_url}/authdownload?list={self.datalist}"
            self.logger.info("Downloading station list...")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse the response text
            stations = []
            for line in response.text.splitlines():
                if line.strip():
                    station = re.split(",", line)[0].strip()
                    stations.append(station)
            
            self.logger.info(f"Found {len(stations)} stations")
            return stations
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download station list: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing station list: {str(e)}")
            raise
    
    def download_station_data(self, station_id: str, credentials: Dict[str, str]) -> Path:
        """
        Download data for a specific HydroWeb station using requests.
        Returns the path to the downloaded file.
        """
        try:
            output_file = self.data_dir / f"{station_id}.{self.dataformat}"
            
            # Construct URL with authentication
            url = (f"{self.base_url}/authdownload?"
                  f"products={station_id}&format={self.dataformat}"
                  f"&user={credentials['username']}&pwd={credentials['password']}")
            
            self.logger.info(f"Downloading data for station {station_id}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_file, 'wb') as f, tqdm(
                desc=f"Downloading {station_id}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            return output_file
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download station {station_id}: {str(e)}")
            raise
    
    def download_all_stations(self, credentials: Dict[str, str], max_workers: int = 4) -> Dict[str, Path]:
        """
        Download data for all available stations using multiprocessing.
        Returns a dictionary mapping station IDs to their file paths.
        """
        try:
            # Get list of stations
            stations = self.download_station_list()
            self.logger.info(f"Preparing to download {len(stations)} stations")
            
            # Download stations in parallel
            with Pool(max_workers) as pool:
                results = []
                for station in stations:
                    result = pool.apply_async(self.download_station_data, 
                                           args=(station, credentials))
                    results.append((station, result))
                
                # Collect results with progress bar
                station_files = {}
                with tqdm(total=len(stations), desc="Downloading stations") as pbar:
                    for station, result in results:
                        try:
                            file_path = result.get()
                            station_files[station] = file_path
                        except Exception as e:
                            self.logger.error(f"Failed to download {station}: {str(e)}")
                        pbar.update(1)
            
            return station_files
            
        except Exception as e:
            self.logger.error(f"Error in batch download: {str(e)}")
            raise
    
    def process_raw_data(self, file_path: Path) -> pd.DataFrame:
        """
        Process raw HydroWeb data file into standardized format.
        """
        try:
            # Read the text file
            df = pd.read_csv(file_path, delimiter=r'\s+', comment='#')
            
            # Rename columns if needed and ensure proper format
            if 'date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'date'})
            if 'water_level' not in df.columns:
                df = df.rename(columns={df.columns[1]: 'water_level'})
            
            # Convert date format
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['date'])
            
            # Handle missing values
            df = df.dropna(subset=['water_level'])
            
            # Add EGM96 height if available
            if self.egm96 is not None and 'longitude' in df.columns and 'latitude' in df.columns:
                df['egm96_height'] = self.egm96.get_height(df['longitude'].iloc[0], df['latitude'].iloc[0])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

class ExternalDataProcessor:
    """Class for processing data from additional sources."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
        self.data_dir = Path("./data_for_wse")
        self.egm2008_filename = "Und_min1x1_egm2008_isw=82_WGS84_TideFree"  # Removed .gz extension
        self.egm2008_url = "https://grid-partner-share.s3.amazonaws.com/egm2008/Und_min1x1_egm2008_isw%3D82_WGS84_TideFree.gz"
        self.egm1996_filename = "WW15MGH.DAC"
        self.egm1996_url = "https://raw.githubusercontent.com/TerriaJS/egm1996/master/data/WW15MGH.DAC"
        
        # Initialize EGM processors
        self.egm2008 = None
        self.egm1996 = None
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_egm2008(self) -> None:
        """Initialize the EGM2008 processor with the grid file."""
        egm2008_path = self.data_dir / self.egm2008_filename
        if egm2008_path.exists():
            self.egm2008 = EarthGravityModel2008(egm2008_path, self.logger)
            self.logger.info("Initialized EGM2008 processor")
    
    def initialize_egm1996(self) -> None:
        """Initialize the EGM1996 processor with the grid file."""
        egm1996_path = self.data_dir / self.egm1996_filename
        if egm1996_path.exists():
            self.egm1996 = EarthGravityModel1996(egm1996_path, self.logger)
            self.logger.info("Initialized EGM1996 processor")
    
    def check_and_download_egm1996(self) -> Path:
        """
        Check if EGM1996 dataset exists, if not download it.
        Returns the path to the dataset.
        """
        egm1996_path = self.data_dir / self.egm1996_filename
        
        if egm1996_path.exists():
            self.logger.info(f"EGM1996 dataset already exists at {egm1996_path}")
            return egm1996_path
        
        self.logger.info("EGM1996 dataset not found. Downloading...")
        try:
            response = requests.get(self.egm1996_url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(egm1996_path, 'wb') as f, tqdm(
                desc="Downloading EGM1996",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            self.logger.info(f"Successfully downloaded EGM1996 dataset to {egm1996_path}")
            return egm1996_path
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download EGM1996 dataset: {str(e)}")
            raise
    
    def check_and_download_egm2008(self) -> Path:
        """
        Check if EGM2008 dataset exists, if not download it.
        Returns the path to the dataset.
        """
        egm2008_gz = self.data_dir / f"{self.egm2008_filename}.gz"
        egm2008_path = self.data_dir / self.egm2008_filename
        
        if egm2008_path.exists():
            self.logger.info(f"EGM2008 dataset already exists at {egm2008_path}")
            return egm2008_path
        
        self.logger.info("EGM2008 dataset not found. Downloading...")
        try:
            response = requests.get(self.egm2008_url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(egm2008_gz, 'wb') as f, tqdm(
                desc="Downloading EGM2008",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            # Extract the gzipped file
            self.logger.info("Extracting EGM2008 dataset...")
            import gzip
            with gzip.open(egm2008_gz, 'rb') as f_in:
                with open(egm2008_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Remove the gzipped file
            egm2008_gz.unlink()
            
            self.logger.info(f"Successfully downloaded and extracted EGM2008 dataset to {egm2008_path}")
            return egm2008_path
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download EGM2008 dataset: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing EGM2008 dataset: {str(e)}")
            raise
    
    def load_external_data(self, source_path: Path) -> pd.DataFrame:
        """Load and process data from external sources."""
        pass
    
    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize external data to match HydroWeb format."""
        pass

class BiasCorrector:
    """Class for performing bias correction on water surface elevation data."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
    
    def correct_bias(self, data: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
        """Apply bias correction using reference data."""
        pass
    
    def validate_correction(self, original_data: pd.DataFrame, 
                          corrected_data: pd.DataFrame) -> Dict:
        """Validate bias correction results."""
        pass

class DatasetPreparator:
    """Class for preparing final dataset for CAMA-Flood evaluation."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
    
    def merge_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple datasets into final format."""
        pass
    
    def export_dataset(self, data: pd.DataFrame, output_format: str = "csv") -> Path:
        """Export processed dataset in specified format."""
        pass

class CaMaResolution(Enum):
    """Enumeration of available CaMa-Flood map resolutions."""
    MIN_15 = "15min"
    MIN_6 = "6min"
    MIN_5 = "5min"
    MIN_3 = "3min"
    MIN_1 = "1min"

class CaMaFloodMapDownloader:
    """Class for downloading and managing CaMa-Flood river topography maps."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
        self.base_url = "https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/distribute/map_v400"
        self.data_dir = Path("./data_for_wse/cama_maps")
        self.session = requests.Session()
        
        # Authentication credentials for CaMa-Flood maps
        self.auth = {
            "username": "camav4",
            "password": "hydrodynamics"
        }
        
        # Map resolution to filename mapping
        self.resolution_files = {
            CaMaResolution.MIN_15: "glb_15min.tar.gz",
            CaMaResolution.MIN_6: "glb_06min.tar.gz",
            CaMaResolution.MIN_5: "glb_05min.tar.gz",
            CaMaResolution.MIN_3: "glb_03min.tar.gz",
            CaMaResolution.MIN_1: "glb_01min.tar.gz"
        }
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_map_url(self, resolution: CaMaResolution) -> str:
        """Get the download URL for a specific resolution."""
        filename = self.resolution_files.get(resolution)
        if not filename:
            raise ValueError(f"Invalid resolution: {resolution}")
        return f"{self.base_url}/{filename}"
    
    def check_and_download_map(self, resolution: CaMaResolution) -> Path:
        """
        Check if the CaMa-Flood map for specified resolution exists, if not download it.
        Returns the path to the extracted map directory.
        """
        filename = self.resolution_files[resolution]
        map_file = self.data_dir / filename
        extracted_dir = self.data_dir / filename.replace('.tar.gz', '')
        
        # Check if already extracted
        if extracted_dir.exists():
            self.logger.info(f"CaMa-Flood map for {resolution.value} already exists at {extracted_dir}")
            return extracted_dir
        
        # Download if not exists
        if not map_file.exists():
            self.logger.info(f"Downloading CaMa-Flood map for {resolution.value}...")
            try:
                url = self.get_map_url(resolution)
                
                # Make authenticated request
                response = self.session.get(
                    url,
                    auth=(self.auth["username"], self.auth["password"]),
                    stream=True
                )
                response.raise_for_status()
                
                # Get total file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                with open(map_file, 'wb') as f, tqdm(
                    desc=f"Downloading {resolution.value} map",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
                
                self.logger.info(f"Successfully downloaded map to {map_file}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to download map: {str(e)}")
                if response.status_code == 401:
                    self.logger.error("Authentication failed. Please check credentials.")
                elif response.status_code == 403:
                    self.logger.error("Access forbidden. Please verify your access permissions.")
                raise
        
        # Extract the tar.gz file
        if map_file.exists() and not extracted_dir.exists():
            self.logger.info(f"Extracting {map_file}...")
            try:
                with tarfile.open(map_file, 'r:gz') as tar:
                    # Get total number of members for progress bar
                    total_members = len(tar.getmembers())
                    
                    with tqdm(
                        desc=f"Extracting {resolution.value} map",
                        total=total_members,
                        unit='files'
                    ) as pbar:
                        for member in tar.getmembers():
                            tar.extract(member, path=self.data_dir)
                            pbar.update(1)
                
                self.logger.info(f"Successfully extracted map to {extracted_dir}")
                
                # Optionally remove the tar.gz file after extraction
                map_file.unlink()
                self.logger.info(f"Removed compressed file {map_file}")
                
            except tarfile.TarError as e:
                self.logger.error(f"Failed to extract map: {str(e)}")
                raise
        
        return extracted_dir
    
    def validate_map_files(self, map_dir: Path) -> bool:
        """
        Validate that all required map files are present in the extracted directory.
        """
        required_files = [
            'lonlat.txt',
            'nextxy.bin',
            'rivwth.bin',
            'elevtn.bin',
            'uparea.bin'
        ]
        
        missing_files = []
        for file in required_files:
            if not (map_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.warning(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        self.logger.info("All required map files are present")
        return True

class StationAllocator:
    """Class for allocating virtual stations to river networks."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger.logger
        self.data_dir = Path("./data_for_wse")
        
        # Flag definitions for allocation types
        self.FLAG_RIVER_CENTERLINE = 10  # on the river centerline
        self.FLAG_RIVER_CHANNEL = 11    # on the river channel
        self.FLAG_CATCHMENT_OUTLET = 12  # location was on the unit-catchment outlet
        self.FLAG_NEAREST_RIVER = 20    # found the nearest river
        self.FLAG_NEAREST_MAIN = 21     # found the nearest main river
        self.FLAG_PERPENDICULAR = 30    # found the nearest perpendicular main river
        self.FLAG_BIFURCATION = 31      # bifurcation location
        self.FLAG_OCEAN_CORRECTION = 40  # correction for ocean grids
        self.FLAG_ERROR = 90            # error in allocation
        
        # Grid parameters
        self.nx = 1440  # Number of longitude points
        self.ny = 720   # Number of latitude points
        self.dx = 0.25  # Grid spacing in longitude (degrees)
        self.dy = 0.25  # Grid spacing in latitude (degrees)
        
        # Data arrays
        self.visual = None      # Visual classification array
        self.flwdir = None      # Flow direction array
        self.catmxx = None      # Catchment X indices
        self.catmyy = None      # Catchment Y indices
        self.uparea = None      # Upstream area
        self.rivwth = None      # River width
        self.elevtn = None      # Elevation
        self.nextxx = None      # Next X grid
        self.nextyy = None      # Next Y grid
        self.biftag = None      # Bifurcation tag
    
    def initialize_grid_data(self, map_dir: Path, region_name: str) -> None:
        """Initialize grid data for processing."""
        self.logger.info(f"Initializing grid data from {map_dir}")
        self.load_grid_data(map_dir, region_name)
    
    def load_grid_data(self, map_dir: Path, region_name: str) -> None:
        """
        Load grid data for a specific region.
        All binary files use yrev and little_endian options.
        
        Args:
            map_dir: Path to the map directory
            region_name: Name of the region/tile
        """
        try:
            def load_and_validate(file_path: Path, dtype: np.dtype, var_name: str) -> np.ndarray:
                """Helper function to load and validate binary data."""
                if not file_path.exists():
                    raise FileNotFoundError(f"{var_name} file not found: {file_path}")
                
                # Read binary data with little endian
                data = np.fromfile(file_path, dtype=dtype)
                
                # Check if endianness conversion is needed
                if not sys.byteorder.startswith('little'):
                    data = data.byteswap()
                
                # Reshape to 2D array
                try:
                    data = data.reshape(self.ny, self.nx)
                except ValueError as e:
                    raise ValueError(f"Failed to reshape {var_name} data: {str(e)}")
                
                ## Reverse latitude (yrev)
                #data = data[::-1, :]
                
                # Validate data
                if not np.isfinite(data).all():
                    n_invalid = np.sum(~np.isfinite(data))
                    self.logger.warning(
                        f"Found {n_invalid} invalid values in {var_name} "
                        f"({n_invalid/data.size:.2%} of total)"
                    )
                
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                self.logger.info(f"{var_name} data range: {data_min:.3f} to {data_max:.3f}")
                
                return data
            
            # Load visual classification
            visual_file = map_dir / f"{region_name}.visual.bin"
            self.visual = load_and_validate(visual_file, np.int8, "visual")
            
            # Load flow direction
            flwdir_file = map_dir / f"{region_name}.flwdir.bin"
            self.flwdir = load_and_validate(flwdir_file, np.int8, "flow direction")
            
            # Load catchment mapping (special case due to double array)
            catm_file = map_dir / f"{region_name}.catmxy.bin"
            data = np.fromfile(catm_file, dtype=np.int16)
            if not sys.byteorder.startswith('little'):
                data = data.byteswap()
            
            # Split into X and Y components
            half_size = self.nx * self.ny
            if len(data) != 2 * half_size:
                raise ValueError(
                    f"Invalid catchment mapping data size. "
                    f"Expected {2 * half_size} elements, got {len(data)}"
                )
            
            self.catmxx = data[:half_size].reshape(self.ny, self.nx)[::-1, :]  # Apply yrev
            self.catmyy = data[half_size:].reshape(self.ny, self.nx)[::-1, :]  # Apply yrev
            
            # Load river width
            rivwth_file = map_dir / f"{region_name}.rivwth.bin"
            self.rivwth = load_and_validate(rivwth_file, np.float32, "river width")
            
            # Load upstream area
            uparea_file = map_dir / f"{region_name}.uparea.bin"
            self.uparea = load_and_validate(uparea_file, np.float32, "upstream area")
            
            # Load elevation
            elevtn_file = map_dir / f"{region_name}.elevtn.bin"
            self.elevtn = load_and_validate(elevtn_file, np.float32, "elevation")
            
            # Basic validation of loaded data
            expected_shape = (self.ny, self.nx)
            for name, arr in [
                ("visual", self.visual),
                ("flow direction", self.flwdir),
                ("catchment X", self.catmxx),
                ("catchment Y", self.catmyy),
                ("river width", self.rivwth),
                ("upstream area", self.uparea),
                ("elevation", self.elevtn)
            ]:
                if arr is None:
                    raise ValueError(f"{name} data failed to load")
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"Invalid {name} data shape: {arr.shape}, "
                        f"expected {expected_shape}"
                    )
            
            self.logger.info(
                f"Successfully loaded grid data for region {region_name}\n"
                f"Grid dimensions: {self.nx}x{self.ny}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load grid data: {str(e)}")
            raise
    
    def find_nearest_river(self, lat: float, lon: float, search_radius: int = 60) -> Tuple[int, int, int, float]:
        """
        Find the nearest river pixel to the given coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            search_radius: Search radius in pixels
            
        Returns:
            Tuple of (x, y, flag, distance)
        """
        # Convert lat/lon to grid indices
        ix = int((lon - self.west) / self.dx)
        iy = int((self.north - lat) / self.dy)
        
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return -9999, -9999, self.FLAG_ERROR, -9999.0
        
        # Initialize search
        min_dist = 1.0e20
        best_x = ix
        best_y = iy
        flag = self.FLAG_ERROR
        
        # Search in radius
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                jx = ix + dx
                jy = iy + dy
                
                if not (0 <= jx < self.nx and 0 <= jy < self.ny):
                    continue
                
                # Skip if not river
                if self.visual[jy, jx] not in [10, 20]:  # 10=river, 20=outlet
                    continue
                
                # Calculate distance
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    if self.visual[jy, jx] == 10:  # River centerline
                        best_x = jx
                        best_y = jy
                        min_dist = dist
                        flag = self.FLAG_RIVER_CENTERLINE
                    elif self.visual[jy, jx] == 20:  # Outlet
                        best_x = jx
                        best_y = jy
                        min_dist = dist
                        flag = self.FLAG_CATCHMENT_OUTLET
        
        return best_x, best_y, flag, min_dist
    
    def find_main_river(self, ix: int, iy: int, search_radius: int = 60) -> Tuple[int, int, int, float]:
        """
        Find the nearest main river (largest upstream area) from the given point.
        
        Args:
            ix: X grid index
            iy: Y grid index
            search_radius: Search radius in pixels
            
        Returns:
            Tuple of (x, y, flag, distance)
        """
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return -9999, -9999, self.FLAG_ERROR, -9999.0
        
        # Initialize search
        max_uparea = self.uparea[iy, ix]
        best_x = ix
        best_y = iy
        min_dist = 1.0e20
        flag = self.FLAG_ERROR
        
        # Search in radius
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                jx = ix + dx
                jy = iy + dy
                
                if not (0 <= jx < self.nx and 0 <= jy < self.ny):
                    continue
                
                # Skip if not river or smaller upstream area
                if self.visual[jy, jx] not in [10, 20]:
                    continue
                if self.uparea[jy, jx] <= max_uparea:
                    continue
                
                # Calculate distance
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    best_x = jx
                    best_y = jy
                    min_dist = dist
                    max_uparea = self.uparea[jy, jx]
                    flag = self.FLAG_NEAREST_MAIN
        
        return best_x, best_y, flag, min_dist
    
    def allocate_station(self, lat: float, lon: float) -> Dict[str, Union[int, float]]:
        """
        Allocate a virtual station to the river network.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Dictionary containing allocation results
        """
        try:
            # First find nearest river
            x1, y1, flag1, dist1 = self.find_nearest_river(lat, lon)
            if flag1 == self.FLAG_ERROR:
                return {
                    'flag': self.FLAG_ERROR,
                    'x1': -9999,
                    'y1': -9999,
                    'x2': -9999,
                    'y2': -9999,
                    'dist1': -9999.0,
                    'dist2': -9999.0,
                    'river_width': -9999.0
                }
            
            # If not on river centerline, try to find main river
            x2, y2, flag2, dist2 = -9999, -9999, -9999, -9999.0
            if flag1 not in [self.FLAG_RIVER_CENTERLINE, self.FLAG_CATCHMENT_OUTLET]:
                x2, y2, flag2, dist2 = self.find_main_river(x1, y1)
                
                # Update flags based on results
                if flag2 != self.FLAG_ERROR:
                    if self.biftag[y1, x1] == 1:
                        flag1 = self.FLAG_BIFURCATION
                    elif dist1 < dist2:
                        flag1 = self.FLAG_NEAREST_RIVER
                    else:
                        flag1 = self.FLAG_NEAREST_MAIN
                        x1, y1 = x2, y2
                        x2, y2 = -9999, -9999
            
            return {
                'flag': flag1,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'dist1': dist1,
                'dist2': dist2,
                'river_width': self.rivwth[y1, x1] if flag1 != self.FLAG_ERROR else -9999.0
            }
            
        except Exception as e:
            self.logger.error(f"Error allocating station at lat={lat}, lon={lon}: {str(e)}")
            return {
                'flag': self.FLAG_ERROR,
                'x1': -9999,
                'y1': -9999,
                'x2': -9999,
                'y2': -9999,
                'dist1': -9999.0,
                'dist2': -9999.0,
                'river_width': -9999.0
            }

class CAMAFloodWSEProcessor:
    """Main class for orchestrating the entire WSE processing workflow."""
    
    def __init__(self):
        self.config = Config()
        self.logger = Logger(self.config)
        self.cama_map_downloader = CaMaFloodMapDownloader(self.config, self.logger)
        self.hydroweb_downloader = HydroWebDownloader(self.config, self.logger)
        self.external_processor = ExternalDataProcessor(self.config, self.logger)
        self.bias_corrector = BiasCorrector(self.config, self.logger)
        self.dataset_preparator = DatasetPreparator(self.config, self.logger)
        self.station_allocator = StationAllocator(self.config, self.logger)
    
    def run_processing_pipeline(self, hydroweb_credentials: Dict[str, str],
                              external_data_paths: List[Path] = None,
                              max_workers: int = 4,
                              cama_resolution: CaMaResolution = CaMaResolution.MIN_15) -> Path:
        """Execute the complete processing pipeline."""
        try:
            # Step 1: Check and download required external datasets
            self.logger.logger.info("Checking required external datasets...")
            
            # Download EGM2008 dataset
            egm2008_path = self.external_processor.check_and_download_egm2008()
            self.logger.logger.info(f"EGM2008 dataset available at: {egm2008_path}")
            
            # Download EGM1996 dataset
            egm1996_path = self.external_processor.check_and_download_egm1996()
            self.logger.logger.info(f"EGM1996 dataset available at: {egm1996_path}")
            
            # Initialize EGM96 processor
            self.hydroweb_downloader.initialize_egm96(egm1996_path)
            self.logger.logger.info("Initialized EGM96 processor")
            
            # Download CaMa-Flood map for specified resolution
            self.logger.logger.info(f"Checking CaMa-Flood map for {cama_resolution.value}...")
            cama_map_dir = self.cama_map_downloader.check_and_download_map(cama_resolution)
            
            # Validate CaMa-Flood map files
            if not self.cama_map_downloader.validate_map_files(cama_map_dir):
                raise ValueError(f"CaMa-Flood map files for {cama_resolution.value} are incomplete")
            
            # Step 2: Download HydroWeb data
            self.logger.logger.info("Starting HydroWeb data download...")
            station_files = self.hydroweb_downloader.download_all_stations(
                credentials=hydroweb_credentials,
                max_workers=max_workers
            )
            self.logger.logger.info(f"Successfully downloaded {len(station_files)} station files")
            
            # Step 3: Process downloaded data
            station_data = {}
            for station_id, file_path in station_files.items():
                try:
                    df = self.hydroweb_downloader.process_raw_data(file_path)
                    station_data[station_id] = df
                except Exception as e:
                    self.logger.logger.error(f"Error processing station {station_id}: {str(e)}")
                    continue
            
            # Step 4: Process external data sources
            self.logger.logger.info("Processing external data sources...")
            
            # Step 5: Perform bias correction
            self.logger.logger.info("Performing bias correction...")
            
            # Step 6: Prepare final dataset
            self.logger.logger.info("Preparing final dataset...")
            
            # Step 7: Export results
            self.logger.logger.info("Exporting processed dataset...")
            
            return self.config.output_dir / "final_dataset.nc"
            
        except Exception as e:
            self.logger.logger.error(f"Error in processing pipeline: {str(e)}")
            raise

    def process_local_hydroweb_data(self, input_csv: Path, output_csv: Path) -> None:
        """
        Process local HydroWeb CSV file and add EGM96 and EGM08 heights.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file
        """
        try:
            # Read the CSV file
            self.logger.logger.info(f"Reading input CSV file: {input_csv}")
            df = pd.read_csv(input_csv)
            
            # Initialize EGM processors
            egm96_path = self.external_processor.check_and_download_egm1996()
            egm2008_path = self.external_processor.check_and_download_egm2008()
            
            egm96 = EarthGravityModel1996(egm96_path, self.logger.logger)
            egm2008 = EarthGravityModel2008(egm2008_path, self.logger.logger)
            
            # Initialize grid data for station allocation
            map_dir = Path("./data_for_wse/map")  # Update this path as needed
            region_name = "glb_15min"  # Update this as needed
            self.station_allocator.initialize_grid_data(map_dir, region_name)
            
            # Calculate EGM heights and allocate stations
            self.logger.logger.info("Processing stations...")
            
            # Create lists to store results
            egm96_heights = []
            egm2008_heights = []
            allocation_flags = []
            river_widths = []
            distances = []
            
            # Process each station
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing stations"):
                lon = row['longitude']
                lat = row['latitude']
                
                # Get EGM96 height
                try:
                    egm96_height = egm96.get_height(lon, lat)
                except Exception as e:
                    self.logger.logger.warning(f"Failed to get EGM96 height for station {row['identifier']}: {str(e)}")
                    egm96_height = None
                
                # Get EGM2008 height
                try:
                    egm2008_height = egm2008.get_height(lon, lat)
                except Exception as e:
                    self.logger.logger.warning(f"Failed to get EGM2008 height for station {row['identifier']}: {str(e)}")
                    egm2008_height = None
                
                # Allocate station
                try:
                    allocation = self.station_allocator.allocate_station(lat, lon)
                    allocation_flags.append(allocation['flag'])
                    river_widths.append(allocation['river_width'])
                    distances.append(allocation['dist1'])
                except Exception as e:
                    self.logger.logger.warning(f"Failed to allocate station {row['identifier']}: {str(e)}")
                    allocation_flags.append(-9999)
                    river_widths.append(-9999.0)
                    distances.append(-9999.0)
                
                egm96_heights.append(egm96_height)
                egm2008_heights.append(egm2008_height)
            
            # Add results to dataframe
            df['egm96_height'] = egm96_heights
            df['egm2008_height'] = egm2008_heights
            df['allocation_flag'] = allocation_flags
            df['river_width'] = river_widths
            df['distance_to_river'] = distances
            
            # Save to new CSV file
            self.logger.logger.info(f"Saving results to: {output_csv}")
            df.to_csv(output_csv, index=False)
            
            self.logger.logger.info("Successfully processed HydroWeb data and added EGM heights and river allocations")
            
        except Exception as e:
            self.logger.logger.error(f"Error processing HydroWeb data: {str(e)}")
            raise

def get_credentials() -> Dict[str, str]:
    """
    Get credentials either from environment variables or user input.
    Returns a dictionary with username and password.
    """
    # Try to get credentials from environment variables first
    username = os.getenv("HYDROWEB_USERNAME")
    password = os.getenv("HYDROWEB_PASSWORD")
    
    # If not in environment variables, ask for input
    if not username:
        username = input("Enter HydroWeb username: ")
    if not password:
        import getpass
        password = getpass.getpass("Enter HydroWeb password: ")
    
    return {
        "username": username,
        "password": password
    }

def main():
    """Main function to run the WSE processing pipeline."""
    try:
        processor = CAMAFloodWSEProcessor()
        
        # Process local HydroWeb data
        input_csv = Path("/Users/zhongwangwei/Desktop/Github/OpenBench/preprocessing/get_cama_ready/data_for_wse/HydroWeb/hydroprd_river.csv")
        output_csv = input_csv.parent / "hydroprd_river_with_egm.csv"
        
        processor.process_local_hydroweb_data(input_csv, output_csv)
        print(f"Processing completed successfully. Output saved to: {output_csv}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 