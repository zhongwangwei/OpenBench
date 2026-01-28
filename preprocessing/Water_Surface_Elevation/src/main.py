#!/usr/bin/env python3
"""WSE Pipeline CLI Entry Point."""
import os
import click
from pathlib import Path
from typing import List, Optional

from .utils.config_loader import load_config
from .utils.logger import setup_logger, get_logger
from .constants import VALID_SOURCES, PIPELINE_STEPS, RESOLUTIONS


def _get_env_or_default(env_var: str, default: str) -> str:
    """Get environment variable value or default, treating empty/whitespace as unset."""
    value = os.environ.get(env_var, '').strip()
    return value if value else default


def _get_env_optional(env_var: str) -> Optional[str]:
    """Get optional environment variable, returning None if unset or empty."""
    value = os.environ.get(env_var, '').strip()
    return value if value else None


def _default_config() -> dict:
    """Return default configuration using environment variables.

    Environment Variables:
        WSE_DATA_ROOT: Root directory for data (default: ./data)
        WSE_OUTPUT_DIR: Output directory (default: ./output)
        WSE_CAMA_ROOT: CaMa-Flood data root (required for CaMa processing)
        WSE_GEOID_ROOT: Geoid data root (required for EGM calculations)
    """
    return {
        'data_root': _get_env_or_default('WSE_DATA_ROOT', './data'),
        'output_dir': _get_env_or_default('WSE_OUTPUT_DIR', './output'),
        'cama_root': _get_env_optional('WSE_CAMA_ROOT'),
        'geoid_root': _get_env_optional('WSE_GEOID_ROOT'),
        'resolutions': RESOLUTIONS,
        'validation': {
            'min_observations': 10,
            'check_duplicates': True,
        },
        'credentials': {},
    }


def _validate_config(cfg: dict) -> List[str]:
    """Validate configuration and return list of warnings.

    Args:
        cfg: Configuration dictionary

    Returns:
        List of warning messages for missing or invalid configuration
    """
    warnings = []

    # Check required paths for full pipeline operation
    required_paths = {
        'cama_root': 'WSE_CAMA_ROOT environment variable or config setting',
        'geoid_root': 'WSE_GEOID_ROOT environment variable or config setting',
    }

    for key, description in required_paths.items():
        value = cfg.get(key)
        if not value or (isinstance(value, str) and not value.strip()):
            warnings.append(
                f"Missing {key}: Set {description} for full pipeline functionality"
            )

    return warnings


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
@click.option('--step', type=click.Choice(PIPELINE_STEPS),
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
        sources = VALID_SOURCES.copy()
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

    # Validate configuration and warn about missing paths
    config_warnings = _validate_config(cfg)
    for warning in config_warnings:
        logger.warning(warning)

    logger.info(f"WSE Pipeline - Processing sources: {', '.join(sources)}")

    if dry_run:
        logger.info("[DRY RUN] Simulation mode, no file writes")
        return

    # Import and run the pipeline
    from .pipeline import Pipeline

    pipeline = Pipeline(cfg)

    if step:
        logger.info(f"Running step: {step}")
        result = pipeline.run_step(step, sources)
    else:
        logger.info("Running full pipeline")
        result = pipeline.run(sources)

    if result.success:
        logger.info("Pipeline completed successfully")
        logger.info(f"Stations processed: {result.stations_processed}")
        if result.output_files:
            logger.info(f"Output files: {', '.join(result.output_files)}")
    else:
        logger.error(f"Pipeline failed: {result.error}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
