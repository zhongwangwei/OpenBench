"""Variable selection and data-file lookup helpers for dataset processing."""

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import List

import xarray as xr

from openbench.data.time_utils import decode_nonstandard_time
from openbench.data.file_lookup import compute_input_files, select_data_files
from openbench.util.converttype import Convert_Type
from openbench.util.names import get_mapping_key_case_insensitive, get_xarray_key_case_insensitive
from openbench.data.compute import (
    ComputeError,
    MissingComputeVariable,
    compute_dependency_names,
    compute_inputs_known,
)

try:
    from openbench.util.dataset_loader import (
        cached_glob,
        open_dataset as open_dataset_chunked,
        open_mfdataset as open_mfdataset_chunked,
    )
except ImportError:  # pragma: no cover - mirrors processing.py fallback
    cached_glob = lambda pattern, **kwargs: sorted(glob.glob(pattern))
    open_dataset_chunked = xr.open_dataset

    def open_mfdataset_chunked(paths, *args, **kwargs):
        return xr.open_mfdataset(paths, *args, **kwargs)


def _processing_attr(name, fallback):
    processing = sys.modules.get("openbench.data.processing")
    return getattr(processing, name, fallback) if processing is not None else fallback


def _xr():
    return _processing_attr("xr", xr)


def _convert_type():
    return _processing_attr("Convert_Type", Convert_Type)


class SelectionMixin:
    """Variable extraction plus prefix/fallback file discovery."""

    def select_var(
        self,
        syear: int,
        eyear: int,
        tim_res: str,
        VarFile,
        varname: List[str],
        datasource: str,
        *,
        load: bool = True,
        return_source: bool = False,
    ) -> xr.Dataset:
        if not load and not return_source:
            raise ValueError("select_var(load=False) requires return_source=True so the caller can close the dataset")
        # Track the original file-backed dataset separately from the derived
        # `ds` so we can close the source handle before returning. Without
        # this, every call leaks an open NetCDF/HDF5 handle — which under
        # joblib.Parallel turns into HDF5 lock errors on the next call.
        src_ds = None
        original_convert = getattr(self, f"_fb_convert_{datasource}", None)
        filter_succeeded = False
        ds = None
        try:
            if isinstance(VarFile, list):
                try:
                    src_ds = open_mfdataset_chunked(VarFile, combine="by_coords", decode_timedelta=False)
                except (ValueError, OSError):
                    src_ds = open_mfdataset_chunked(
                        VarFile, combine="by_coords", decode_times=False, decode_timedelta=False
                    )
                    source_path = str(VarFile[0]) if VarFile else None
                    src_ds = decode_nonstandard_time(src_ds, source_path=source_path)
            else:
                try:
                    src_ds = (
                        _xr().open_dataset(VarFile, decode_timedelta=False)
                        if load
                        else open_dataset_chunked(VarFile, decode_timedelta=False)
                    )
                except (ValueError, OSError):
                    src_ds = (
                        _xr().open_dataset(VarFile, decode_times=False, decode_timedelta=False)
                        if load
                        else open_dataset_chunked(VarFile, decode_times=False, decode_timedelta=False)
                    )
                    src_ds = decode_nonstandard_time(src_ds, source_path=str(VarFile))
            ds = src_ds
        except Exception as e:
            logging.error(f"Failed to open dataset: {VarFile}")
            logging.error(f"Error: {str(e)}")
            if src_ds is not None:
                src_ds.close()
            raise

        # NOTE: This block can raise ValueError/KeyError when the requested
        # variable is missing AND no fallback resolves. We intentionally do
        # NOT swallow that exception — but we MUST close `src_ds` so the
        # underlying file handle isn't leaked across call sites (joblib
        # workers re-opening the same NC file then hit HDF5 lock errors).
        try:
            full_ds = ds
            try:
                ds = self.apply_custom_filter(datasource, ds, varname)
                ds = _convert_type().convert_nc(ds)
                filter_succeeded = True
            except Exception as error:
                if isinstance(error, ComputeError) and not isinstance(error, MissingComputeVariable):
                    raise
                if not varname or len(varname) == 0:
                    logging.error("Variable name list is empty")
                    raise ValueError("Variable name list cannot be empty")

                # Check if variable exists in dataset; if not, try normalized fallback varnames from model profile
                target_var = varname[0]
                actual_target_var = get_xarray_key_case_insensitive(ds, target_var)
                if actual_target_var is None:
                    fallback_found = False
                    try:
                        source = getattr(self, f"{datasource}_source", "")
                        runtime_fallbacks = getattr(self, f"{source}_fallbacks", None) or []
                        primary_unit = getattr(self, f"{datasource}_varunit", "")
                        for fb in runtime_fallbacks:
                            fb_var = fb.get("varname") if isinstance(fb, dict) else getattr(fb, "varname", "")
                            actual_fb_var = get_xarray_key_case_insensitive(ds, fb_var) if fb_var else None
                            if actual_fb_var is not None:
                                logging.warning(
                                    "Variable '%s' not found, using fallback '%s'", target_var, actual_fb_var
                                )
                                target_var = actual_fb_var
                                actual_target_var = actual_fb_var
                                setattr(self, f"{datasource}_varname", [target_var])
                                fb_convert = (
                                    fb.get("convert", "") if isinstance(fb, dict) else getattr(fb, "convert", "")
                                )
                                fb_unit = fb.get("varunit", "") if isinstance(fb, dict) else getattr(fb, "varunit", "")
                                if fb_convert:
                                    setattr(self, f"_fb_convert_{datasource}", fb_convert)
                                    setattr(self, f"{datasource}_varunit", primary_unit or fb_unit)
                                elif fb_unit:
                                    setattr(self, f"{datasource}_varunit", fb_unit)
                                fallback_found = True
                                break

                        if not fallback_found:
                            model = getattr(self, f"{source}_model", source)
                            from openbench.data.registry.manager import get_registry

                            mgr = get_registry()
                            profile = mgr.get_model(model)
                            item = getattr(self, "item", "")
                            profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
                            if profile and profile_key is not None:
                                var_mapping = profile.variables[profile_key]
                                # Try fallbacks
                                if var_mapping.fallbacks:
                                    for fb in var_mapping.fallbacks:
                                        actual_fb_var = get_xarray_key_case_insensitive(ds, fb.varname)
                                        if actual_fb_var is not None:
                                            logging.warning(
                                                "Variable '%s' not found, using fallback '%s'",
                                                target_var,
                                                actual_fb_var,
                                            )
                                            target_var = actual_fb_var
                                            actual_target_var = actual_fb_var
                                            setattr(self, f"{datasource}_varname", [target_var])
                                            if fb.convert:
                                                setattr(self, f"_fb_convert_{datasource}", fb.convert)
                                                setattr(
                                                    self,
                                                    f"{datasource}_varunit",
                                                    var_mapping.varunit or fb.varunit,
                                                )
                                            elif fb.varunit:
                                                setattr(self, f"{datasource}_varunit", fb.varunit)
                                            fallback_found = True
                                            break
                    except Exception as e:
                        logging.debug("Fallback lookup failed: %s", e)

                    # Final fallback: the data file may already carry the OpenBench
                    # standard variable name (e.g. 'Net_Ecosystem_Exchange') instead
                    # of the model's native name (f_nee/f_respc). This is common when
                    # users pre-process model output to standard names — the same way
                    # CoLM's Surface_Albedo profile uses 'Surface_Albedo' as its
                    # primary varname. If the standard item-named variable is present,
                    # use it directly and DROP any adapter-resolved convert expression:
                    # the data is already the final derived quantity, so re-applying a
                    # native-variable convert (e.g. f_respc → NEE) would corrupt it.
                    if not fallback_found:
                        item = getattr(self, "item", "")
                        actual_item_var = get_xarray_key_case_insensitive(ds, item) if item else None
                        if actual_item_var is not None:
                            logging.warning(
                                "Variable '%s' not found, using standard item-named "
                                "variable '%s' already present in the data file",
                                target_var,
                                actual_item_var,
                            )
                            target_var = actual_item_var
                            actual_target_var = actual_item_var
                            setattr(self, f"{datasource}_varname", [target_var])
                            if hasattr(self, f"_fb_convert_{datasource}"):
                                delattr(self, f"_fb_convert_{datasource}")
                            fallback_found = True

                    if not fallback_found:
                        available_vars = list(ds.data_vars) + list(ds.coords)
                        logging.error(f"Variable '{varname[0]}' not found in dataset")
                        logging.error(f"Available variables: {available_vars}")
                        raise KeyError(f"Variable '{varname[0]}' not in dataset")
                else:
                    target_var = actual_target_var

                if self._may_be_compute_input(datasource, error, target_var):
                    # An input of a failed derivation is not the derived quantity.
                    raise error
                ds = _convert_type().convert_nc(ds[target_var])
        except Exception as error:
            if isinstance(error, ComputeError):
                error.args = (f"{VarFile} ({syear}–{eyear}): {error}", *error.args[1:])
            # Bubble up after closing the file handle.
            if src_ds is not None:
                try:
                    src_ds.close()
                except Exception:
                    pass
            raise

        # Apply fallback conversion expressions. The expression can reference 'value' (current variable) and any other
        # variable in the NC file by name (e.g., 'f_assim', 'f_respc').
        # NOTE: This must be outside the except block so it runs even when the
        # primary varname is found without error (adapter-resolved fallbacks).
        fb_convert = getattr(self, f"_fb_convert_{datasource}", None)
        # A successful compute clears conversion only for this file; later files
        # may need the adapter-selected independent raw fallback again.
        if filter_succeeded and original_convert is not None and not hasattr(self, f"_fb_convert_{datasource}"):
            setattr(self, f"_fb_convert_{datasource}", original_convert)
        if fb_convert:
            try:
                from openbench.data.compute import _validate_expression
                import numpy as np

                value = ds.values if load else ds.data
                ns = {"value": value, "np": np}
                if "full_ds" in locals() and full_ds is not None:
                    for name, data_var in getattr(full_ds, "data_vars", {}).items():
                        if name not in ns:
                            ns[name] = data_var.values if load else data_var.data
                _validate_expression(fb_convert, allowed_names=ns.keys())
                converted = eval(fb_convert, {"__builtins__": {}}, ns)  # noqa: S307
                if load:
                    ds.values = converted
                else:
                    ds.data = converted
                logging.info("Applied fallback conversion: %s", fb_convert)
                # The expression yields a DERIVED quantity (e.g. NEE from
                # f_respc and f_assim), but `ds` still carries the source
                # variable's name and long_name (e.g. "respiration
                # (plant+soil)"), which then mislabels the derived field in
                # output files and plots. Relabel to the evaluation item,
                # mirroring the compute path (which sets result.name = item).
                fb_item = getattr(self, "item", "")
                if fb_item and hasattr(ds, "attrs"):
                    try:
                        ds.name = fb_item
                    except Exception:
                        pass
                    for _stale_attr in ("long_name", "standard_name", "original_name"):
                        ds.attrs.pop(_stale_attr, None)
                    ds.attrs["long_name"] = fb_item.replace("_", " ")
            except Exception as e:
                raise RuntimeError(
                    f"Fallback conversion {fb_convert!r} failed; refusing to continue with unconverted units"
                ) from e

        if load:
            # Materialise data into memory so we can close the source file
            # handle. Returning a lazy, file-backed Dataset would cause every
            # caller (preprocess_*_files, Mod_Statistics.process_*) to hold an
            # open NetCDF descriptor for the lifetime of the result.
            try:
                if hasattr(ds, "load"):
                    ds = ds.load()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to materialize selected variable from {VarFile}; "
                    "refusing to return a lazy file-backed object"
                ) from e
            finally:
                if src_ds is not None:
                    try:
                        src_ds.close()
                    except Exception:
                        pass

        return (ds, src_ds) if return_source else ds

    def _candidate_varnames_for_file_lookup(self, varname: List[str] | None, datasource: str) -> list[str]:
        """Return concrete variables that can satisfy this item in a candidate file."""
        candidates: list[str] = []
        for name in varname or []:
            if name:
                candidates.append(str(name))

        try:
            candidates.extend(self._compute_dependency_varnames_for_file_lookup(datasource))
            source = getattr(self, f"{datasource}_source", "")
            for fb in getattr(self, f"{source}_fallbacks", None) or []:
                fb_var = fb.get("varname") if isinstance(fb, dict) else getattr(fb, "varname", "")
                if fb_var:
                    candidates.append(str(fb_var))
            model = getattr(self, f"{source}_model", source)
            from openbench.data.registry.manager import get_registry

            profile = get_registry().get_model(model)
            item = getattr(self, "item", "")
            profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
            if profile and profile_key is not None:
                mapping = profile.variables[profile_key]
                raw_varname = mapping.varname
                if isinstance(raw_varname, str):
                    if raw_varname:
                        candidates.append(raw_varname)
                elif raw_varname:
                    candidates.extend(str(name) for name in raw_varname if name)
                for fallback in mapping.fallbacks or []:
                    if fallback.varname:
                        candidates.append(fallback.varname)
        except Exception as exc:
            logging.debug("Variable-aware prefix fallback lookup skipped: %s", exc)

        return list(dict.fromkeys(candidates))

    def _compute_expressions_for_file_lookup(self, datasource: str) -> list[str]:
        """Return only the expression selected by ``_try_compute_from_profile``."""
        source = getattr(self, f"{datasource}_source", "")
        expression = getattr(self, f"{datasource}_compute", "") or (
            getattr(self, f"{source}_compute", "") if source else ""
        )
        if expression:
            return [str(expression)]
        try:
            model = getattr(self, f"{source}_model", source)
            from openbench.data.registry.manager import get_registry

            registry = get_registry()
            item = getattr(self, "item", "")
            for lookup in ("get_model", "get_reference"):
                profile = getattr(registry, lookup)(model)
                profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
                if profile and profile_key is not None and profile.variables[profile_key].compute:
                    return [str(profile.variables[profile_key].compute)]
        except Exception as exc:
            logging.debug("Compute dependency lookup skipped: %s", exc)
        return []

    def _compute_dependency_varnames_for_file_lookup(self, datasource: str) -> list[str]:
        deps = []
        for expr in self._compute_expressions_for_file_lookup(datasource):
            deps.extend(compute_dependency_names(expr))
        return list(dict.fromkeys(deps))

    def _may_be_compute_input(self, datasource: str, error: BaseException, name: str) -> bool:
        """Whether ``name`` may be an input of the failed compute or of the compute that applies.

        Whatever made the derivation fail, one of its inputs read raw is not the
        derived quantity, so it must never stand in as a fallback. An expression
        whose inputs cannot be listed (``ds[key]``) may read any variable. Only
        the expression that applies counts (an inline compute replaces the
        profile's, as in ``_try_compute_from_profile``), so an overridden
        expression does not block an independent raw variable.
        """
        may_read = getattr(error, "may_read", None)
        if may_read is not None and may_read(name):
            return True
        try:
            expressions = self._compute_expressions_for_file_lookup(datasource)
        except Exception as exc:
            logging.debug("Could not read compute inputs for the fallback check: %s", exc)
            expressions = []
        if not expressions:
            return False
        expression = expressions[0]
        return not compute_inputs_known(expression) or name.casefold() in {
            dependency.casefold() for dependency in compute_dependency_names(expression)
        }

    def _find_compute_dependency_files(self, dirx: str, year: int | None, datasource: str) -> list[str]:
        deps = self._compute_dependency_varnames_for_file_lookup(datasource)
        # Shared with ``openbench check``; the candidates follow the lookup's
        # branch rule, so inputs of one compute never come from two branches.
        selected = compute_input_files(dirx, year, deps)
        if selected:
            logging.info(
                "Using %d files containing compute dependencies%s", len(selected), f" for year {year}" if year else ""
            )
        return selected

    def _select_files(self, dirx, prefix, suffix, year, datasource, varname):
        source = getattr(self, f"{datasource}_source", "") or (
            getattr(self, "sim_source", "") or getattr(self, "ref_source", "")
        )
        files, used_compute = select_data_files(
            dirx,
            prefix,
            suffix,
            year,
            prefix_fallback=getattr(self, f"{source}_prefix_fallback", None),
            candidate_varnames=self._candidate_varnames_for_file_lookup(varname, datasource),
            dependencies=self._compute_dependency_varnames_for_file_lookup(datasource),
        )
        return files, used_compute

    def _find_single_file(
        self,
        dirx: str,
        prefix: str,
        suffix: str,
        datasource: str = "sim",
        varname: List[str] | None = None,
    ) -> str | list[str]:
        files, used_compute = self._select_files(dirx, prefix, suffix, None, datasource, varname)
        if files:
            return files if used_compute else files[0]
        raise FileNotFoundError(f"Data file not found: {os.path.join(dirx, f'{prefix}{suffix}.nc[4]')}")

    def _find_data_files(
        self,
        dirx: str,
        prefix: str,
        year: int,
        suffix: str,
        datasource: str = "sim",
        varname: List[str] | None = None,
    ) -> list:
        return self._select_files(dirx, prefix, suffix, year, datasource, varname)[0]
