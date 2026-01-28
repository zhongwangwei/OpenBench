"""Pipeline Controller for WSE Processing."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from .steps import Step0Download, Step1Validate, Step2CaMa, Step3Reserved, Step4Merge
from .core.station import StationList
from .utils.logger import get_logger
from .utils.checkpoint import Checkpoint

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline execution.

    Attributes:
        success: Whether the pipeline completed successfully.
        stations_processed: Number of stations processed.
        output_files: List of output file paths created.
        error: Error message if pipeline failed.
        step_results: Results from each pipeline step.
    """
    success: bool
    stations_processed: int
    output_files: List[str]
    error: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)


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

    def run(self, sources: List[str]) -> PipelineResult:
        """Run full pipeline for specified sources.

        Args:
            sources: List of data sources to process.

        Returns:
            PipelineResult with success status, processed stations, and output files.
        """
        logger.info("=" * 60)
        logger.info("WSE Pipeline 开始")
        logger.info("=" * 60)

        results = {}
        stations = StationList()
        output_files = []

        try:
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

            return PipelineResult(
                success=True,
                stations_processed=len(stations),
                output_files=output_files if output_files else [],
                step_results=results
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                success=False,
                stations_processed=len(stations),
                output_files=output_files if output_files else [],
                error=str(e),
                step_results=results
            )

    def run_step(self, step_name: str, sources: List[str]) -> PipelineResult:
        """Run a specific step only.

        Args:
            step_name: Name of the step to run.
            sources: List of data sources.

        Returns:
            PipelineResult with success status and step results.

        Raises:
            ValueError: If step_name is not a valid step.
        """
        if step_name not in self.steps:
            raise ValueError(f"Unknown step: {step_name}")

        logger.info(f"运行单步: {step_name}")

        handler = self._step_handlers[step_name]
        stations_count = 0
        output_files = []
        step_result = {}

        try:
            if step_name == 'download':
                step_result = handler.run(sources)
            elif step_name == 'validate':
                stations = handler.run(sources)
                stations_count = len(stations)
                step_result = {'total': stations_count}
            else:
                # Load from checkpoint
                stations = self.checkpoint.load_stations()
                if stations is None:
                    stations = StationList()

                if step_name == 'cama':
                    stations = handler.run(stations)
                    stations_count = len(stations)
                    step_result = {'allocated': stations_count}
                elif step_name == 'reserved':
                    stations = handler.run(stations)
                    stations_count = len(stations)
                elif step_name == 'merge':
                    output_files = handler.run(stations, merge=self.config.get('merge', False))
                    step_result = {'files': output_files}

            return PipelineResult(
                success=True,
                stations_processed=stations_count,
                output_files=output_files if output_files else [],
                step_results={step_name: step_result}
            )

        except Exception as e:
            logger.error(f"Step {step_name} failed: {e}")
            return PipelineResult(
                success=False,
                stations_processed=stations_count,
                output_files=[],
                error=str(e),
                step_results={step_name: step_result}
            )
