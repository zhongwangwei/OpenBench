"""Taylor and target comparison diagram scenario facade."""

from __future__ import annotations

from openbench.core._comparison_target import TargetDiagramComparisonMixin
from openbench.core._comparison_taylor import TaylorDiagramComparisonMixin


class DiagramComparisonMixin(TaylorDiagramComparisonMixin, TargetDiagramComparisonMixin):
    """Bundle Taylor and target diagram comparison scenarios."""

    pass
