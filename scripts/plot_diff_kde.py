#!/usr/bin/env python3
"""Render OpenBench pairwise Diff_Plot NetCDF data as a KDE figure."""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import xarray as xr

from openbench.config.adapter import build_fig_nml
from openbench.util.filenames import join_filename_components
from openbench.visualization import make_scenarios_comparison_Kernel_Density_Estimate

LOG = logging.getLogger("plot_diff_kde")


@dataclass(frozen=True)
class DiffMetadata:
    evaluation_item: str
    ref_source: str
    sim1: str
    sim2: str
    item_type: str

    @property
    def variable(self) -> str:
        return f"{self.item_type}_diff"

    @property
    def curve_label(self) -> str:
        return f"{self.sim1} − {self.sim2}"


def parse_diff_filename(path: Path) -> DiffMetadata:
    """Parse an OpenBench gridded pairwise-difference filename."""
    if path.suffix.lower() != ".nc":
        raise ValueError(f"Expected a .nc Diff_Plot file: {path}")

    parts = path.stem.split("__")
    if len(parts) != 8 or parts[1] != "ref" or parts[4] != "vs" or parts[7] != "diff":
        raise ValueError(
            f"Not an OpenBench pairwise diff filename: {path.name}; expected "
            "<item>__ref__<reference>__<sim1>__vs__<sim2>__<metric>__diff.nc"
        )

    item, _ref, ref_source, sim1, _vs, sim2, item_type, _diff = (unquote(part) for part in parts)
    if not all((item, ref_source, sim1, sim2, item_type)):
        raise ValueError(f"Diff_Plot filename contains an empty component: {path.name}")
    return DiffMetadata(item, ref_source, sim1, sim2, item_type)


def _validate_metadata(ds: xr.Dataset, metadata: DiffMetadata, path: Path) -> None:
    data_array = ds.get(metadata.variable)
    if data_array is None:
        available = ", ".join(ds.data_vars) or "none"
        raise ValueError(f"{path}: missing variable {metadata.variable!r}; available variables: {available}")

    expected = {"item": metadata.evaluation_item, "ref_source": metadata.ref_source}
    for key, value in expected.items():
        for attributes in (ds.attrs, data_array.attrs):
            actual = attributes.get(key)
            if actual not in (None, "") and str(actual) != value:
                raise ValueError(f"{path}: attribute {key}={actual!r} conflicts with filename value {value!r}")

    declared_type = data_array.attrs.get("score") or data_array.attrs.get("metric")
    if declared_type not in (None, "") and str(declared_type) != metadata.item_type:
        raise ValueError(
            f"{path}: variable metadata declares {declared_type!r}, "
            f"but filename declares {metadata.item_type!r}"
        )


def read_diff_values(path: Path, metadata: DiffMetadata) -> np.ndarray:
    """Load and validate the finite, equally weighted samples for one diff file."""
    if not path.is_file():
        raise FileNotFoundError(f"Diff_Plot NetCDF does not exist: {path}")
    with xr.open_dataset(path) as ds:
        _validate_metadata(ds, metadata, path)
        values = np.asarray(ds[metadata.variable].values).reshape(-1)

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError(f"{path}: KDE requires at least two finite values; found {finite.size}")
    if np.unique(finite).size < 2:
        raise ValueError(f"{path}: KDE requires at least two distinct finite values")
    return finite


def default_output_dir(path: Path) -> Path:
    if path.parent.name != "Diff_Plot":
        raise ValueError(f"Cannot infer output directory because {path} is not inside a Diff_Plot directory")
    return path.parent.parent / "Kernel_Density_Estimate"


def expected_output_path(output_dir: Path, metadata: DiffMetadata, option: dict) -> Path:
    sim_token = hashlib.sha1(metadata.curve_label.encode("utf-8")).hexdigest()[:8]
    filename = join_filename_components(
        "Kernel_Density_Estimate",
        metadata.evaluation_item,
        metadata.ref_source,
        metadata.variable,
        sim_token,
    )
    return output_dir / f"{filename}.{option['saving_format']}"


def plot_diff_file(path: Path, output_dir: Path | None = None) -> Path:
    """Create one KDE figure from a gridded OpenBench pairwise diff file."""
    path = path.expanduser().resolve()
    metadata = parse_diff_filename(path)
    values = read_diff_values(path, metadata)
    destination = output_dir.expanduser().resolve() if output_dir is not None else default_output_dir(path)

    option = build_fig_nml()["Comparison"]["Kernel_Density_Estimate"].copy()
    item_label = metadata.evaluation_item.replace("_", " ")
    type_label = metadata.item_type.replace("_", " ")
    option["title"] = f"Kernel Density Estimate of {item_label} Difference"
    option["xticklabel"] = f"{type_label} Difference"

    make_scenarios_comparison_Kernel_Density_Estimate(
        str(destination),
        metadata.evaluation_item,
        metadata.ref_source,
        [metadata.curve_label],
        metadata.variable,
        [values],
        option,
    )
    return expected_output_path(destination, metadata, option)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more OpenBench gridded __diff.nc files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for all figures (default: sibling comparisons/Kernel_Density_Estimate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures: list[tuple[Path, Exception]] = []
    for input_path in args.inputs:
        try:
            output_path = plot_diff_file(input_path, args.output_dir)
            print(f"Created {output_path}")
        except Exception as exc:
            LOG.error("Failed to plot %s: %s", input_path, exc)
            failures.append((input_path, exc))

    if failures:
        LOG.error("%d of %d input files failed", len(failures), len(args.inputs))
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
