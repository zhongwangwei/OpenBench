#!/usr/bin/env python3
"""WSE Pipeline CLI Entry Point."""
import click
from pathlib import Path
from typing import List, Optional

from .utils.config_loader import load_config
from .utils.logger import setup_logger, get_logger


def _default_config() -> dict:
    """Return default configuration."""
    return {
        'data_root': '/Volumes/Data01/Altimetry',
        'output_dir': './output',
        'cama_root': '/Volumes/Data01/2025',
        'geoid_root': '/Volumes/Data01/AltiMaPpy-data/egm-geoids',
        'resolutions': ['glb_01min', 'glb_03min', 'glb_05min', 'glb_06min', 'glb_15min'],
        'validation': {
            'min_observations': 10,
            'check_duplicates': True,
        },
        'credentials': {},
    }


@click.command()
@click.option('--source', '-s', default='all',
              help='Data source(s): hydroweb,cgls,icesat,hydrosat or "all"')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Config file path')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory')
@click.option('--merge', is_flag=True,
              help='Merge all sources into single output')
@click.option('--skip-download', is_flag=True,
              help='Skip download check, use existing data')
@click.option('--step', type=click.Choice(['download', 'validate', 'cama', 'reserved', 'merge']),
              help='Run specific step only')
@click.option('--num-workers', '-j', type=int, default=5,
              help='Parallel download workers')
@click.option('--log-level', default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
@click.option('--dry-run', is_flag=True,
              help='Simulation mode, no file writes')
def main(source: str, config: Optional[str], output: Optional[str],
         merge: bool, skip_download: bool, step: Optional[str],
         num_workers: int, log_level: str, dry_run: bool):
    """WSE Pipeline - Water Surface Elevation Processing.

    Process satellite altimetry data from multiple sources and allocate
    to CaMa-Flood grid cells.

    Examples:

        wse --source hydroweb

        wse --source hydroweb,cgls --merge

        wse --source all --skip-download
    """
    logger = setup_logger('wse_pipeline', log_level=log_level)

    # Parse sources
    if source == 'all':
        sources = ['hydrosat', 'hydroweb', 'cgls', 'icesat']
    else:
        sources = [s.strip().lower() for s in source.split(',')]

    # Load config
    if config:
        cfg = load_config(config)
    else:
        cfg = _default_config()

    # Override with CLI options
    if output:
        cfg['output_dir'] = output
    cfg['merge'] = merge
    cfg['skip_download'] = skip_download
    cfg['num_workers'] = num_workers
    cfg['dry_run'] = dry_run

    logger.info(f"WSE Pipeline - Processing sources: {', '.join(sources)}")

    if dry_run:
        logger.info("[DRY RUN] Simulation mode, no file writes")
        return

    # Import here to avoid circular imports (Pipeline will be created in Task 6)
    try:
        from .pipeline import Pipeline

        # For now, use existing pipeline interface
        # Task 6 will create a new Pipeline class with run_step method
        if step:
            logger.info(f"Running step: {step}")
            # Future: pipeline.run_step(step, sources)
            logger.warning(f"Step-only mode not yet fully implemented")
        else:
            logger.info("Full pipeline execution requires config and dataset files")
            logger.info("Use: wse --config config/global_paths.yaml for full run")

    except ImportError as e:
        logger.warning(f"Pipeline not fully implemented: {e}")
        logger.info("Use --dry-run to test CLI options")


if __name__ == '__main__':
    main()
