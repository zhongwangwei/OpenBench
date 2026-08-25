"""Variable selection and data-file lookup helpers for dataset processing."""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
from typing import List

import xarray as xr

from openbench.data.time_utils import decode_nonstandard_time
from openbench.data.coordinates import NC_SUFFIXES, glob_nc_pattern
from openbench.util.converttype import Convert_Type
from openbench.util.names import get_mapping_key_case_insensitive, get_xarray_key_case_insensitive

try:
    from openbench.util.dataset_loader import cached_glob, open_mfdataset as open_mfdataset_chunked
except ImportError:  # pragma: no cover - mirrors processing.py fallback
    cached_glob = lambda pattern, **kwargs: sorted(glob.glob(pattern))

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
        self, syear: int, eyear: int, tim_res: str, VarFile, varname: List[str], datasource: str
    ) -> xr.Dataset:
        # Track the original file-backed dataset separately from the derived
        # `ds` so we can close the source handle before returning. Without
        # this, every call leaks an open NetCDF/HDF5 handle — which under
        # joblib.Parallel turns into HDF5 lock errors on the next call.
        src_ds = None
        ds = None
        try:
            if isinstance(VarFile, list):
                try:
                    src_ds = open_mfdataset_chunked(VarFile, combine="by_coords")
                except (ValueError, OSError):
                    src_ds = open_mfdataset_chunked(VarFile, combine="by_coords", decode_times=False)
                    source_path = str(VarFile[0]) if VarFile else None
                    src_ds = decode_nonstandard_time(src_ds, source_path=source_path)
            else:
                try:
                    src_ds = _xr().open_dataset(VarFile)  # .squeeze()
                except (ValueError, OSError):
                    src_ds = _xr().open_dataset(VarFile, decode_times=False)  # .squeeze()
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
            except Exception:
                # Check if varname list is empty
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

                ds = _convert_type().convert_nc(ds[target_var])
        except Exception:
            # Bubble up after closing the file handle.
            if src_ds is not None:
                try:
                    src_ds.close()
                except Exception:
                    pass
            raise

        # Apply fallback conversion expression if set (from adapter or runtime fallback)
        # The expression can reference 'value' (current variable) and any other
        # variable in the NC file by name (e.g., 'f_assim', 'f_respc').
        # NOTE: This must be outside the except block so it runs even when the
        # primary varname is found without error (adapter-resolved fallbacks).
        fb_convert = getattr(self, f"_fb_convert_{datasource}", None)
        if fb_convert:
            try:
                from openbench.data.compute import _validate_expression
                import numpy as np

                value = ds.values
                ns = {"value": value, "np": np}
                if "full_ds" in locals() and full_ds is not None:
                    for name, data_var in getattr(full_ds, "data_vars", {}).items():
                        if name not in ns:
                            ns[name] = data_var.values
                _validate_expression(fb_convert, allowed_names=ns.keys())
                ds.values = eval(fb_convert, {"__builtins__": {}}, ns)  # noqa: S307
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

        # Materialise data into memory so we can close the source file
        # handle. Returning a lazy, file-backed Dataset would cause every
        # caller (preprocess_*_files, Mod_Statistics.process_*) to hold an
        # open NetCDF descriptor for the lifetime of the result.
        try:
            if hasattr(ds, "load"):
                ds = ds.load()
        except Exception as e:
            raise RuntimeError(
                f"Failed to materialize selected variable from {VarFile}; refusing to return a lazy file-backed object"
            ) from e
        finally:
            if src_ds is not None:
                try:
                    src_ds.close()
                except Exception:
                    pass

        return ds

    def _get_prefix_fallback_list(self, prefix: str, datasource: str = "sim") -> list:
        """Build list of prefixes to try: primary + fallbacks.

        Args:
            prefix: Primary file prefix (e.g., "Case01_hist_")
            datasource: "sim" or "ref" — determines which source's fallback to use
        """
        prefixes = [prefix]
        # Use the correct source based on datasource context
        source = getattr(self, f"{datasource}_source", "")
        if not source:
            source = getattr(self, "sim_source", "") or getattr(self, "ref_source", "")
            if source:
                logging.debug(
                    "prefix_fallback: using fallback source '%s' for datasource='%s'",
                    source,
                    datasource,
                )
        pf_list = getattr(self, f"{source}_prefix_fallback", None)
        if pf_list:
            for fb in pf_list:
                if prefix.endswith("_"):
                    prefixes.append(prefix[:-1] + fb)
                else:
                    prefixes.append(prefix + fb)
        return prefixes

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
                if mapping.compute:
                    candidates.extend(re.findall(r"ds\[['\"]([^'\"]+)['\"]\]", mapping.compute))
        except Exception as exc:
            logging.debug("Variable-aware prefix fallback lookup skipped: %s", exc)

        return list(dict.fromkeys(candidates))

    def _compute_dependency_varnames_for_file_lookup(self, datasource: str) -> list[str]:
        expressions = [getattr(self, f"{datasource}_compute", "")]
        try:
            source = getattr(self, f"{datasource}_source", "")
            model = getattr(self, f"{source}_model", source)
            from openbench.data.registry.manager import get_registry

            profile = get_registry().get_model(model)
            item = getattr(self, "item", "")
            profile_key = get_mapping_key_case_insensitive(profile.variables, item) if profile else None
            if profile and profile_key is not None:
                expressions.append(profile.variables[profile_key].compute or "")
        except Exception as exc:
            logging.debug("Compute dependency lookup skipped: %s", exc)
        deps = []
        for expr in expressions:
            deps.extend(re.findall(r"ds\[['\"]([^'\"]+)['\"]\]", str(expr)))
        return list(dict.fromkeys(dep for dep in deps if dep))

    def _find_compute_dependency_files(self, dirx: str, year: int | None, datasource: str) -> list[str]:
        deps = self._compute_dependency_varnames_for_file_lookup(datasource)
        if not deps:
            return []
        year_token = f"*{year}*" if year is not None else "*"
        pattern = os.path.join(dirx, "**", f"{year_token}.nc")
        files = sorted(
            set(
                glob.glob(pattern, recursive=True)
                + glob.glob(pattern + "4", recursive=True)
                + glob.glob(pattern[:-3] + ".NC", recursive=True)
                + glob.glob(pattern[:-3] + ".NC4", recursive=True)
            )
        )
        selected = []
        found = set()
        for file_path in files:
            try:
                with _xr().open_dataset(file_path, decode_times=False) as ds:
                    available = {str(name).lower(): str(name) for name in ds.variables}
                    hits = [dep for dep in deps if dep.lower() in available]
                    if hits:
                        selected.append(file_path)
                        found.update(hits)
            except Exception as exc:
                logging.debug("Could not inspect compute dependency file %s: %s", file_path, exc)
        if selected and set(deps).issubset(found):
            logging.info(
                "Using %d files containing compute dependencies%s", len(selected), f" for year {year}" if year else ""
            )
            return selected
        return []

    def _files_contain_any_var(self, files: list, varnames: list[str]) -> bool:
        if not varnames:
            return True

        inspected = False
        for file_path in files[:3]:
            try:
                with _xr().open_dataset(file_path, decode_times=False) as ds:
                    inspected = True
                    available = {str(name).lower() for name in ds.variables}
                    if any(str(name).lower() in available for name in varnames):
                        return True
            except Exception as exc:
                logging.debug("Could not inspect variables in %s: %s", file_path, exc)

        return not inspected

    def _prefixes_for_variable_lookup(
        self,
        prefix: str,
        datasource: str,
        candidate_varnames: list[str],
    ) -> list:
        prefixes = self._get_prefix_fallback_list(prefix, datasource)
        if candidate_varnames and len(prefixes) > 1:
            return [*prefixes[1:], prefixes[0]]
        return prefixes

    def _find_single_file(
        self,
        dirx: str,
        prefix: str,
        suffix: str,
        datasource: str = "sim",
        varname: List[str] | None = None,
    ) -> str | list[str]:
        """Find a single data file, trying prefix fallbacks and .nc/.nc4 extensions."""
        candidate_varnames = self._candidate_varnames_for_file_lookup(varname, datasource)
        first_existing_path = None
        for try_prefix in self._prefixes_for_variable_lookup(prefix, datasource, candidate_varnames):
            for ext in NC_SUFFIXES:
                path = os.path.join(dirx, f"{try_prefix}{suffix}{ext}")
                if os.path.exists(path):
                    if first_existing_path is None:
                        first_existing_path = path
                    if not self._files_contain_any_var([path], candidate_varnames):
                        continue
                    if try_prefix != prefix:
                        logging.info(f"Using fallback prefix '{try_prefix}' for single file")
                    return path
        compute_files = self._find_compute_dependency_files(dirx, None, datasource)
        if compute_files:
            return compute_files
        if first_existing_path is not None:
            return first_existing_path
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
        """Find data files, trying prefix_fallback if primary prefix has no matches.

        Search order for each prefix:
            1. dirx/prefix_year*suffix.nc
            2. dirx/year/prefix_year*suffix.nc

        Prefix order:
            1. Primary prefix (e.g., "Case01_hist_")
            2. Fallback prefixes (e.g., "Case01_hist_cama_", "Case01_hist_unitcat_")
        """
        candidate_varnames = self._candidate_varnames_for_file_lookup(varname, datasource)
        first_matching_pattern_files = None
        for try_prefix in self._prefixes_for_variable_lookup(prefix, datasource, candidate_varnames):
            # Escape glob metacharacters in user-supplied prefix/suffix. Without
            # this, a prefix or suffix containing '[', '?', or '*' would be
            # interpreted as a wildcard and either match the wrong files or fail
            # entirely. The intentional wildcard is between {year} and {suffix}.
            escaped_prefix = glob.escape(try_prefix)
            escaped_suffix = glob.escape(suffix)
            # Try primary path, then year subdir, then scanner-style nested date dirs.
            patterns = [
                os.path.join(dirx, f"{escaped_prefix}{year}*{escaped_suffix}.nc"),
                os.path.join(dirx, str(year), f"{escaped_prefix}{year}*{escaped_suffix}.nc"),
                os.path.join(dirx, "**", f"{escaped_prefix}{year}*{escaped_suffix}.nc"),
            ]
            var_files = []
            for pattern_path in patterns:
                var_files = glob_nc_pattern(pattern_path)
                if "**" in pattern_path:
                    var_files = sorted(set(var_files + glob.glob(pattern_path, recursive=True)))
                    if pattern_path.endswith(".nc"):
                        var_files = sorted(set(var_files + glob.glob(pattern_path + "4", recursive=True)))
                        var_files = sorted(set(var_files + glob.glob(pattern_path[:-3] + ".NC", recursive=True)))
                        var_files = sorted(set(var_files + glob.glob(pattern_path[:-3] + ".NC4", recursive=True)))
                if var_files:
                    break

            # Filter: only keep files where part between prefix+year and suffix has no letters
            if var_files:
                filtered = []
                prefix_escaped = re.escape(try_prefix)
                suffix_escaped = re.escape(suffix) if suffix else ""
                pattern = re.compile(rf"^{prefix_escaped}{year}[^a-zA-Z]*{suffix_escaped}\.nc4?$", re.IGNORECASE)
                for f in var_files:
                    if pattern.match(os.path.basename(f)):
                        filtered.append(f)
                var_files = filtered

            if not var_files and (try_prefix or suffix):
                exact = os.path.join(dirx, "**", f"{escaped_prefix}{escaped_suffix}.nc")
                nested = set(glob.glob(exact, recursive=True))
                nested.update(glob.glob(exact + "4", recursive=True))
                nested.update(glob.glob(exact[:-3] + ".NC", recursive=True))
                nested.update(glob.glob(exact[:-3] + ".NC4", recursive=True))
                year_dir = re.compile(rf"^{year}(?:[-_](?:0[1-9]|1[0-2]))?(?:[-_](?:0[1-9]|[12]\d|3[01]))?$")
                var_files = sorted(
                    path
                    for path in nested
                    if any(year_dir.fullmatch(part) for part in os.path.relpath(path, dirx).split(os.sep)[:-1])
                )

            if var_files:
                if first_matching_pattern_files is None:
                    first_matching_pattern_files = var_files
                if not self._files_contain_any_var(var_files, candidate_varnames):
                    logging.debug(
                        "Files for prefix '%s' do not contain any of %s; trying prefix fallback",
                        try_prefix,
                        candidate_varnames,
                    )
                    continue
                if try_prefix != prefix:
                    logging.info(f"Using fallback prefix '{try_prefix}' for year {year} (primary '{prefix}' not found)")
                return var_files

        compute_files = self._find_compute_dependency_files(dirx, year, datasource)
        if compute_files:
            return compute_files
        return first_matching_pattern_files or []
