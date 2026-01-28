"""Step 4: Output merge and file generation."""
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ..core.station import Station, StationList
from ..utils.logger import get_logger

logger = get_logger(__name__)

RESOLUTIONS = ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min']


class Step4Merge:
    """Step 4: Generate output files."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './output'))
        self.resolutions = config.get('resolutions', RESOLUTIONS)

    def run(self, stations: StationList, merge: bool = False) -> List[str]:
        """Generate output files.

        Args:
            stations: StationList with all processed stations
            merge: If True, merge all sources into single file

        Returns:
            List of output file paths
        """
        logger.info("[Step 4] 生成输出文件...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if merge:
            return self._write_merged(stations)
        else:
            return self._write_separate(stations)

    def _write_separate(self, stations: StationList) -> List[str]:
        """Write separate files for each source."""
        output_files = []
        sources = stations.get_sources()

        for source in sources:
            source_stations = stations.filter_by_source(source)
            if len(source_stations) == 0:
                continue

            filename = f"{source}_stations.txt"
            filepath = self.output_dir / filename

            self._write_stations(filepath, source_stations, include_source=False)
            output_files.append(str(filepath))
            logger.info(f"  写入 {filename}: {len(source_stations)} 站点")

        return output_files

    def _write_merged(self, stations: StationList) -> List[str]:
        """Write all stations to single file."""
        filename = "all_stations.txt"
        filepath = self.output_dir / filename

        self._write_stations(filepath, stations, include_source=True)
        logger.info(f"  写入 {filename}: {len(stations)} 站点")

        return [str(filepath)]

    def _write_stations(self, filepath: Path, stations: StationList, include_source: bool):
        """Write stations to file."""
        # Build header
        header = ['id', 'name', 'lon', 'lat', 'elevation', 'num_obs', 'EGM08', 'EGM96']

        if include_source:
            header.insert(0, 'source')

        # Add CaMa columns for each resolution
        for res in self.resolutions:
            res_suffix = res.replace('glb_', '')
            header.extend([
                f'flag_{res_suffix}',
                f'ix_{res_suffix}', f'iy_{res_suffix}',
                f'kx1_{res_suffix}', f'ky1_{res_suffix}',
                f'kx2_{res_suffix}', f'ky2_{res_suffix}',
                f'dist1_{res_suffix}', f'dist2_{res_suffix}',
                f'rivwth_{res_suffix}',
                f'lon_cama_{res_suffix}', f'lat_cama_{res_suffix}',
            ])

        with open(filepath, 'w') as f:
            # Write header
            f.write('\t'.join(header) + '\n')

            # Write data
            for station in stations:
                row = self._format_station_row(station, include_source)
                f.write('\t'.join(str(v) for v in row) + '\n')

    def _format_station_row(self, station: Station, include_source: bool) -> List[Any]:
        """Format station data as row."""
        row = []

        if include_source:
            row.append(station.source)

        row.extend([
            station.id,
            station.name,
            f"{station.lon:.6f}",
            f"{station.lat:.6f}",
            f"{station.elevation:.2f}",
            station.num_observations,
            f"{station.egm08:.3f}" if station.egm08 else 'NA',
            f"{station.egm96:.3f}" if station.egm96 else 'NA',
        ])

        # Add CaMa results for each resolution
        for res in self.resolutions:
            cama = station.cama_results.get(res, {})
            row.extend([
                cama.get('flag', 0),
                cama.get('ix', -1),
                cama.get('iy', -1),
                cama.get('kx1', -1),
                cama.get('ky1', -1),
                cama.get('kx2', -1),
                cama.get('ky2', -1),
                f"{cama.get('dist1', 0):.1f}",
                f"{cama.get('dist2', 0):.1f}",
                f"{cama.get('rivwth', 0):.1f}",
                f"{cama.get('lon_cama', 0):.6f}",
                f"{cama.get('lat_cama', 0):.6f}",
            ])

        return row
