"""Filename helpers shared by producers and visualization consumers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unicodedata

_RESERVED_FILENAME_CHARS = '<>:"/\\|?*'
_SEPARATOR = "__"
_MAX_COMPONENT_LENGTH = 180
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def filename_component(value: object) -> str:
    """Return a safe, collision-resistant component for generated filenames."""
    text = unicodedata.normalize("NFC", str(value))
    if text == "":
        return "_"

    replacements = {"%": "%25", _SEPARATOR: "%5F%5F"}
    replacements.update({char: f"%{ord(char):02X}" for char in _RESERVED_FILENAME_CHARS})

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = "".join(f"%{ord(char):02X}" if ord(char) < 32 else char for char in text)

    while text.endswith((" ", ".")):
        text = f"{text[:-1]}%{ord(text[-1]):02X}"

    # Windows device names are reserved even with extensions (e.g. CON.txt).
    stem = text.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        text = f"%{ord(text[0]):02X}{text[1:]}"

    if len(text) > _MAX_COMPONENT_LENGTH:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        text = f"{text[:_MAX_COMPONENT_LENGTH]}__sha256-{digest}"

    return text or "_"


def join_filename_components(*values: object) -> str:
    """Join filename components without underscore-boundary collisions."""
    return _SEPARATOR.join(filename_component(value) for value in values)


def station_file_path(
    scratch_dir: str | Path,
    station_id: object,
    *,
    index: int | None = None,
    duplicate_ids: set[str] | None = None,
) -> Path:
    """Return a collision-resistant station NetCDF path under ``scratch_dir``.

    Station IDs can contain characters that are unsafe on common filesystems,
    and some consolidated station products contain duplicate station IDs.  When
    the normalized station ID is known to be duplicated, include the source
    index in the filename so per-station scratch NetCDFs never overwrite each
    other.
    """
    station_component = filename_component(station_id)
    normalized_id = str(station_id)
    if duplicate_ids and normalized_id in duplicate_ids and index is not None:
        station_component = join_filename_components(station_component, f"idx{index}")
    return Path(scratch_dir) / f"{station_component}.nc"


def groupby_pair_dirname(sim_source: object, ref_source: object) -> str:
    """Safe directory name for a groupby simulation/reference pair."""
    return join_filename_components(sim_source, ref_source)


def legacy_groupby_pair_dirname(sim_source: object, ref_source: object) -> str:
    """Legacy groupby pair directory name kept for only_drawing fallbacks."""
    return f"{sim_source}___{ref_source}"


def groupby_table_filename(evaluation_item: object, sim_source: object, ref_source: object, table_type: object) -> str:
    """Safe LC/CZ groupby table filename."""
    return f"{join_filename_components(evaluation_item, sim_source, ref_source, table_type)}.csv"


def legacy_groupby_table_filename(
    evaluation_item: object, sim_source: object, ref_source: object, table_type: object
) -> str:
    """Legacy LC/CZ groupby table filename kept for only_drawing fallbacks."""
    return f"{evaluation_item}_{sim_source}___{ref_source}_{table_type}.csv"


def groupby_class_netcdf_filename(
    evaluation_item: object,
    ref_source: object,
    sim_source: object,
    statistic: object,
    class_prefix: object,
    class_name: object,
) -> str:
    """Safe filename for per-class LC/CZ groupby NetCDF outputs."""
    return f"{join_filename_components(evaluation_item, 'ref', ref_source, 'sim', sim_source, statistic, class_prefix, class_name)}.nc"


def groupby_class_netcdf_stem(
    evaluation_item: object,
    ref_source: object,
    sim_source: object,
    statistic: object,
    class_prefix: object,
) -> str:
    """Safe filename stem prefix for per-class LC/CZ groupby NetCDF outputs."""
    return join_filename_components(evaluation_item, "ref", ref_source, "sim", sim_source, statistic, class_prefix)


def diff_station_anomaly_filename(
    evaluation_item: object, ref_source: object, sim_source: object, item_type: object
) -> str:
    """Filename for station Diff Plot anomaly intermediate CSVs."""
    return (
        f"{join_filename_components(evaluation_item, 'stn', ref_source, 'sim', sim_source, item_type, 'anomaly')}.csv"
    )


def diff_station_difference_filename(
    evaluation_item: object,
    ref_source: object,
    sim1: object,
    sim_varname_1: object,
    sim2: object,
    sim_varname_2: object,
    item_type: object,
) -> str:
    """Filename for station Diff Plot pairwise-difference intermediate CSVs."""
    return f"{join_filename_components(evaluation_item, 'stn', ref_source, sim1, sim_varname_1, 'vs', sim2, sim_varname_2, item_type, 'diff')}.csv"


def diff_grid_anomaly_filename(
    evaluation_item: object, ref_source: object, sim_source: object, item_type: object
) -> str:
    """Filename for gridded Diff Plot anomaly intermediate NetCDFs."""
    return f"{join_filename_components(evaluation_item, 'ref', ref_source, 'sim', sim_source, item_type, 'anomaly')}.nc"


def diff_grid_difference_filename(
    evaluation_item: object, ref_source: object, sim1: object, sim2: object, item_type: object
) -> str:
    """Filename for gridded Diff Plot pairwise-difference intermediate NetCDFs."""
    return f"{join_filename_components(evaluation_item, 'ref', ref_source, sim1, 'vs', sim2, item_type, 'diff')}.nc"


def relative_station_scores_filename(evaluation_item: object, ref_source: object, sim_source: object) -> str:
    """Filename for station relative-score intermediate CSVs."""
    return f"{join_filename_components(evaluation_item, 'stn', ref_source, sim_source, 'relative_scores')}.csv"


def relative_station_score_plot_stem(
    evaluation_item: object, ref_source: object, sim_source: object, score: object
) -> str:
    """Stem for station relative-score plot outputs."""
    return join_filename_components(evaluation_item, "stn", ref_source, sim_source, "relative", score)


def relative_grid_score_filename(evaluation_item: object, ref_source: object, sim_source: object, score: object) -> str:
    """Filename for gridded relative-score intermediate NetCDFs."""
    return f"{join_filename_components(evaluation_item, 'ref', ref_source, 'sim', sim_source, f'Relative{score}')}.nc"
