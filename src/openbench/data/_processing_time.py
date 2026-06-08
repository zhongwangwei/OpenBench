"""Time-coordinate validation helper facade for dataset processing."""

from __future__ import annotations

from openbench.data._processing_time_adjustments import TimeAdjustmentMixin
from openbench.data._processing_time_core import TimeCoreMixin
from openbench.data._processing_time_integrity import TimeIntegrityWorkflowMixin


class TimeIntegrityMixin(TimeCoreMixin, TimeIntegrityWorkflowMixin, TimeAdjustmentMixin):
    """Temporal normalization and integrity checks shared by grid/station processing."""

    pass
