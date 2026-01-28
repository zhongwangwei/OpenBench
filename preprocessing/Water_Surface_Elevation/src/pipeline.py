"""Pipeline Controller for WSE Processing."""
from pathlib import Path
from typing import List, Optional, Dict, Any

from .steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge
from .core.station import StationList
from .utils.logger import get_logger
from .utils.checkpoint import Checkpoint

logger = get_logger(__name__)


class Pipeline:
    """WSE Pipeline Controller."""

    steps = ['download', 'validate', 'cama', 'reserved', 'merge']

    def __init__(self, config: dict):
        self.config = config
        self.checkpoint = Checkpoint(config.get('output_dir', './output'))

        # Initialize steps
        self._step_handlers = {
            'download': Step0Download(config),
            'validate': Step1Validate(config),
            'cama': Step2CaMa(config),
            'reserved': Step3Reserved(config),
            'merge': Step4Merge(config),
        }

    def run(self, sources: List[str]) -> Dict[str, Any]:
        """Run full pipeline for specified sources."""
        logger.info("=" * 60)
        logger.info("WSE Pipeline 开始")
        logger.info("=" * 60)

        results = {}
        stations = StationList()

        for step_name in self.steps:
            logger.info(f"\n{'='*20} {step_name.upper()} {'='*20}")

            if step_name == 'download':
                skip = self.config.get('skip_download', False)
                results['download'] = self._step_handlers['download'].run(
                    sources, skip_download=skip
                )
            elif step_name == 'validate':
                stations = self._step_handlers['validate'].run(sources)
                results['validate'] = {'total': len(stations)}
            elif step_name == 'cama':
                stations = self._step_handlers['cama'].run(stations)
                results['cama'] = {'allocated': len(stations)}
            elif step_name == 'reserved':
                stations = self._step_handlers['reserved'].run(stations)
            elif step_name == 'merge':
                output_files = self._step_handlers['merge'].run(
                    stations,
                    merge=self.config.get('merge', False)
                )
                results['merge'] = {'files': output_files}

            self.checkpoint.save(step_name, results)

        logger.info("\n" + "=" * 60)
        logger.info("WSE Pipeline 完成")
        logger.info("=" * 60)

        return results

    def run_step(self, step_name: str, sources: List[str]) -> Any:
        """Run a specific step only."""
        if step_name not in self.steps:
            raise ValueError(f"Unknown step: {step_name}")

        logger.info(f"运行单步: {step_name}")

        handler = self._step_handlers[step_name]

        if step_name == 'download':
            return handler.run(sources)
        elif step_name == 'validate':
            return handler.run(sources)
        else:
            # Load from checkpoint
            stations = self.checkpoint.load_stations()
            if stations is None:
                stations = StationList()
            if step_name == 'cama':
                return handler.run(stations)
            elif step_name == 'reserved':
                return handler.run(stations)
            elif step_name == 'merge':
                return handler.run(stations, merge=self.config.get('merge', False))
