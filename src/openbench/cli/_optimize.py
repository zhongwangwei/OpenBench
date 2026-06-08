"""Implementation for ``openbench ref optimize``."""

from __future__ import annotations

from pathlib import Path

import click


def optimize_reference(name: str) -> None:
    """Convert dataset to zarr for faster reads."""
    from openbench.data.registry import RegistryManager

    mgr = RegistryManager()
    ref = mgr.get_reference(name)
    if ref is None:
        raise click.ClickException(f"Dataset not found: {name}")

    if not ref.root_dir:
        raise click.ClickException(
            f"No local path configured for {name}. Set --root-dir via "
            f"`openbench ref register {name} --root-dir ...` or run "
            "`openbench ref scan <ref_root>`."
        )

    if getattr(ref, "data_type", None) == "stn":
        raise click.ClickException("ref optimize is not supported for station datasets.")

    root = Path(ref.root_dir)
    if not root.exists():
        raise click.ClickException(f"Data directory not found: {root}")

    zarr_dir = root.parent / f"{root.name}.zarr"
    if zarr_dir.exists():
        click.echo(f"Zarr store already exists: {zarr_dir}")
        if not click.confirm("Overwrite?"):
            return

    from openbench.data.coordinates import glob_nc

    nc_files = glob_nc(root, recursive=True)

    if not nc_files:
        click.secho(f"No NetCDF files found in {root}", fg="yellow")
        return

    click.echo(f"Found {len(nc_files)} NetCDF files")
    click.echo(f"Converting to zarr: {zarr_dir}")

    try:
        import os
        import shutil
        import uuid

        from openbench.util.dataset_loader import write_mfdataset_zarr

        tmp_zarr_dir = zarr_dir.parent / f".{zarr_dir.name}.tmp-{uuid.uuid4().hex}"
        backup_zarr_dir = None
        try:
            write_mfdataset_zarr(
                [str(path) for path in nc_files],
                tmp_zarr_dir,
                combine="by_coords",
                engine="netcdf4",
                batch_dir=zarr_dir.parent,
            )
            if zarr_dir.exists():
                backup_zarr_dir = zarr_dir.parent / f".{zarr_dir.name}.bak-{uuid.uuid4().hex}"
                zarr_dir.rename(backup_zarr_dir)
            tmp_zarr_dir.rename(zarr_dir)
            if backup_zarr_dir is not None:
                shutil.rmtree(backup_zarr_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_zarr_dir, ignore_errors=True)
            if backup_zarr_dir is not None and backup_zarr_dir.exists() and not zarr_dir.exists():
                backup_zarr_dir.rename(zarr_dir)
            raise

        nc_size = sum(os.path.getsize(f) for f in nc_files)
        zarr_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(zarr_dir) for f in fns)
        click.secho("✓ Converted to zarr", fg="green")
        click.echo(f"  NetCDF: {nc_size / 1e9:.1f} GB")
        click.echo(f"  Zarr:   {zarr_size / 1e9:.1f} GB")
    except Exception as e:
        raise click.ClickException(f"Conversion failed: {e}") from e
