"""Base dataset-processing mixin assembled from focused processing helpers."""

from __future__ import annotations

from openbench.data._processing_config import ProcessingConfigMixin
from openbench.data._processing_transforms import ProcessingTransformMixin
from openbench.data._processing_yearly import YearlyPreprocessingMixin
from openbench.data.coordinates import COORDINATE_MAP_WITH_VERTICAL


class BaseProcessingMixin(ProcessingConfigMixin, ProcessingTransformMixin, YearlyPreprocessingMixin):
    coordinate_map = dict(COORDINATE_MAP_WITH_VERTICAL)
