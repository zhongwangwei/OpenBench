"""Bundled smoke-test command for installed OpenBench packages."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from importlib.resources import as_file, files
from pathlib import Path

import click
import yaml

SMOKE_REFERENCE_NAME = "OpenBench_Smoke_GLEAM4_2a"
SMOKE_GLEAM_HYBRID_NAME = "OpenBench_Smoke_GLEAM_hybrid_PLUMBER2"
SMOKE_ILAMB_NAME = "OpenBench_Smoke_ILAMB_monthly"
SMOKE_PLUMBER2_NAME = "OpenBench_Smoke_PLUMBER2"
SMOKE_RESOURCE_PACKAGE = "openbench"
SMOKE_RESOURCE_DIR = "dataset/smoke"
SMOKE_ARCHIVE = "initial_test.tar.gz"
SMOKE_TEMPLATE = "openbench-smoke.yaml"


def _resource_path(filename: str):
    resource = files(SMOKE_RESOURCE_PACKAGE)
    for part in SMOKE_RESOURCE_DIR.split("/"):
        resource = resource / part
    return resource / filename


def _fixture_path_from_legacy_dir(raw_dir: str, sample_root: Path) -> Path:
    raw = raw_dir.strip()
    marker = "dataset/Reference/Initial_test/"
    if marker in raw:
        return sample_root / "Reference" / "Initial_test" / raw.split(marker, 1)[1]
    marker = "dataset/Simulation/Initial_test/"
    if marker in raw:
        return sample_root / "Simulation" / "Initial_test" / raw.split(marker, 1)[1]
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (sample_root.parent / path).resolve()


def _localize_station_list(source: Path, destination: Path, sample_root: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as src_handle:
        reader = csv.DictReader(src_handle)
        if reader.fieldnames is None:
            raise click.ClickException(f"Station list has no header: {source}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if "DIR" not in fieldnames:
        raise click.ClickException(f"Station list is missing DIR column: {source}")
    for row in rows:
        row["DIR"] = str(_fixture_path_from_legacy_dir(row["DIR"], sample_root))

    with destination.open("w", newline="", encoding="utf-8") as dst_handle:
        writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _localize_station_lists(work_dir: Path, sample_root: Path) -> dict[str, Path]:
    reference_root = sample_root / "Reference" / "Initial_test"
    simulation_root = sample_root / "Simulation" / "Initial_test"
    list_dir = work_dir / "lists"
    return {
        "station_case": _localize_station_list(
            simulation_root / "station_case.csv",
            list_dir / "station_case.csv",
            sample_root,
        ),
        "GLEAM_hybrid_PLUMBER2": _localize_station_list(
            reference_root / "GLEAM_hybrid_PLUMBER2.csv",
            list_dir / "GLEAM_hybrid_PLUMBER2.csv",
            sample_root,
        ),
        "PLUMBER2": _localize_station_list(
            reference_root / "PLUMBER2.csv",
            list_dir / "PLUMBER2.csv",
            sample_root,
        ),
    }


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise click.ClickException(f"Unsafe path in smoke archive: {member.name}") from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise click.ClickException(f"Unsupported member in smoke archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise click.ClickException(f"Cannot extract smoke archive member: {member.name}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            target.chmod(member.mode & 0o777)


def _write_reference_catalog(home: Path, reference_root: Path, station_lists: dict[str, Path]) -> Path:
    from openbench.config.user_settings import USER_CONFIG_DIR_NAME
    from openbench.data.registry.manager import user_reference_catalog_path

    # The smoke subprocess resolves this same path via OPENBENCH_HOME=<home>,
    # so derive it from the registry helpers instead of duplicating the layout.
    catalog_path = user_reference_catalog_path(home / USER_CONFIG_DIR_NAME)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        "\n".join(
            [
                f"{SMOKE_REFERENCE_NAME}:",
                f"  name: {SMOKE_REFERENCE_NAME}",
                "  category: Water",
                "  data_type: grid",
                "  tim_res: Month",
                "  grid_res: 2.0",
                "  years: [2004, 2005]",
                f"  root_dir: {reference_root.as_posix()}",
                "  variables:",
                "    Evapotranspiration:",
                "      varname: E",
                "      varunit: mm day-1",
                "      prefix: E_",
                "      suffix: _GLEAM_v4.2a_MO",
                "      sub_dir: GLEAM4.2a_monthly",
                f"{SMOKE_GLEAM_HYBRID_NAME}:",
                f"  name: {SMOKE_GLEAM_HYBRID_NAME}",
                "  category: Water",
                "  data_type: stn",
                "  tim_res: Day",
                "  data_groupby: single",
                "  years: [2004, 2005]",
                f"  root_dir: {(reference_root / 'GLEAM_hybrid_PLUMBER2').as_posix()}",
                f"  fulllist: {station_lists['GLEAM_hybrid_PLUMBER2'].as_posix()}",
                "  variables:",
                "    Evapotranspiration:",
                "      varname: et",
                "      varunit: mm day-1",
                f"{SMOKE_ILAMB_NAME}:",
                f"  name: {SMOKE_ILAMB_NAME}",
                "  category: Energy",
                "  data_type: grid",
                "  tim_res: Month",
                "  data_groupby: Single",
                "  grid_res: 2.0",
                "  years: [2004, 2005]",
                f"  root_dir: {(reference_root / 'ILAMB').as_posix()}",
                "  variables:",
                "    Latent_Heat:",
                "      varname: le",
                "      varunit: W m-2",
                "      prefix: le",
                "      suffix: ''",
                "      sub_dir: ILAMB/Latent_Heat/FLUXCOM",
                "    Sensible_Heat:",
                "      varname: sh",
                "      varunit: W m-2",
                "      prefix: sh",
                "      suffix: ''",
                "      sub_dir: ILAMB/Sensible_Heat/FLUXCOM",
                f"{SMOKE_PLUMBER2_NAME}:",
                f"  name: {SMOKE_PLUMBER2_NAME}",
                "  category: Energy",
                "  data_type: stn",
                "  tim_res: Day",
                "  data_groupby: single",
                "  years: [2004, 2005]",
                f"  root_dir: {(reference_root / 'PLUMBER2').as_posix()}",
                f"  fulllist: {station_lists['PLUMBER2'].as_posix()}",
                "  variables:",
                "    Latent_Heat:",
                "      varname: Qle_cor",
                "      varunit: W m-2",
                "    Sensible_Heat:",
                "      varname: Qh_cor",
                "      varunit: W m-2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return catalog_path


def _write_smoke_config(work_dir: Path, sample_root: Path, station_lists: dict[str, Path]) -> Path:
    output_dir = work_dir / "output"
    reference_root = sample_root / "Reference" / "Initial_test"
    simulation_root = sample_root / "Simulation" / "Initial_test" / "grid"
    station_simulation_root = sample_root / "Simulation" / "Initial_test" / "stn"
    template = _resource_path(SMOKE_TEMPLATE).read_text(encoding="utf-8")
    config_text = template.format(
        output_dir=output_dir.as_posix(),
        reference_root=reference_root.as_posix(),
        simulation_root=simulation_root.as_posix(),
        station_simulation_root=station_simulation_root.as_posix(),
        station_simulation_list=station_lists["station_case"].as_posix(),
    )
    config_path = work_dir / "openbench-smoke.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _prepare_work_dir(work_dir: Path) -> tuple[Path, Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    with as_file(_resource_path(SMOKE_ARCHIVE)) as archive_path:
        _safe_extract_tar(Path(archive_path), work_dir)

    sample_root = work_dir / "Initial_test"
    reference_root = sample_root / "Reference" / "Initial_test"
    if not reference_root.exists():
        raise click.ClickException(f"Smoke fixture is incomplete: missing {reference_root}")
    simulation_root = sample_root / "Simulation" / "Initial_test" / "grid"
    if not simulation_root.exists():
        raise click.ClickException(f"Smoke fixture is incomplete: missing {simulation_root}")
    station_simulation_root = sample_root / "Simulation" / "Initial_test" / "stn"
    if not station_simulation_root.exists():
        raise click.ClickException(f"Smoke fixture is incomplete: missing {station_simulation_root}")

    station_lists = _localize_station_lists(work_dir, sample_root)
    home = work_dir / "home"
    _write_reference_catalog(home, reference_root, station_lists)
    config_path = _write_smoke_config(work_dir, sample_root, station_lists)
    return sample_root, home, config_path


def _run_openbench_subcommand(config_path: Path, home: Path, run_evaluation: bool) -> int:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["OPENBENCH_HOME"] = str(home)
    env["OPENBENCH_REF_ROOT"] = str(config_path.parent / "Initial_test" / "Reference" / "Initial_test")
    command = [sys.executable, "-m", "openbench", "run" if run_evaluation else "check", str(config_path)]
    if run_evaluation:
        command.append("--force")
    result = subprocess.run(command, env=env, check=False)
    return result.returncode


def _smoke_case_dir(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config.get("project", {})
    output_dir = Path(project["output_dir"])
    return output_dir / project["name"]


def _relative_artifact(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _collect_total_score_artifacts(case_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    heatmap_dir = case_dir / "comparisons" / "HeatMap"
    heatmap_figures = sorted(heatmap_dir.glob("scenarios_Overall_Score_comparison_heatmap.*"))
    heatmap_tables = sorted(heatmap_dir.glob("scenarios_Overall_Score_comparison.*"))
    heatmap_tables = [path for path in heatmap_tables if "_heatmap." not in path.name]

    score_dir = case_dir / "scores"
    figure_suffixes = {".jpg", ".jpeg", ".png", ".pdf", ".svg"}
    spatial_maps = sorted(
        path for path in score_dir.rglob("*Overall_Score*") if path.is_file() and path.suffix.lower() in figure_suffixes
    )
    return heatmap_figures, heatmap_tables, spatial_maps


def _report_total_score_artifacts(config_path: Path) -> None:
    case_dir = _smoke_case_dir(config_path)
    heatmap_figures, heatmap_tables, spatial_maps = _collect_total_score_artifacts(case_dir)

    missing = []
    if not heatmap_figures:
        missing.append("Total score heatmap figure")
    if not spatial_maps:
        missing.append("Overall score spatial maps")
    if missing:
        raise click.ClickException(
            "Smoke run did not produce required result artifact(s): " + ", ".join(missing) + f". Checked: {case_dir}"
        )

    click.echo("Smoke result artifacts:")
    click.echo("  Total score heatmap:")
    for path in heatmap_figures:
        click.echo(f"    - {_relative_artifact(path, case_dir)}")
    if heatmap_tables:
        click.echo("  Total score heatmap data:")
        for path in heatmap_tables:
            click.echo(f"    - {_relative_artifact(path, case_dir)}")
    click.echo("  Overall score spatial maps:")
    for path in spatial_maps:
        click.echo(f"    - {_relative_artifact(path, case_dir)}")


@click.command("smoke-test")
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory used to unpack the bundled Initial_test fixture.",
)
@click.option("--keep", is_flag=True, help="Keep the unpacked fixture and generated config after the command exits.")
@click.option(
    "--run",
    "run_evaluation",
    is_flag=True,
    help="Run the full evaluation instead of the default config/data check.",
)
def smoke_test(work_dir: Path | None, keep: bool, run_evaluation: bool):
    """Run the bundled Initial_test smoke fixture."""
    created_temp_dir = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="openbench-smoke-"))
        created_temp_dir = True
    else:
        work_dir = work_dir.expanduser().resolve()
        if work_dir.exists() and any(work_dir.iterdir()):
            raise click.ClickException(f"Smoke work directory already exists and is not empty: {work_dir}")

    try:
        sample_root, home, config_path = _prepare_work_dir(work_dir)
        click.echo(f"Smoke fixture: {sample_root}")
        click.echo(f"Config: {config_path}")

        exit_code = _run_openbench_subcommand(config_path, home, run_evaluation)
        if exit_code != 0:
            raise click.exceptions.Exit(exit_code)

        if run_evaluation:
            _report_total_score_artifacts(config_path)
            click.echo("Smoke run passed.")
        else:
            click.echo("Smoke check passed.")
    finally:
        if created_temp_dir and keep:
            click.echo(f"Kept smoke work directory: {work_dir}")
        elif created_temp_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
