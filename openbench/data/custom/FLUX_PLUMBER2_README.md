# FLUX_PLUMBER2 Custom Filter

## Overview

This filter handles FLUXNET PLUMBER2 reference data with automatic fallback logic for corrected vs. raw flux variables.

## Features

### Sensible Heat (Qh)
- **Primary**: `Qh_cor` (corrected sensible heat flux)
- **Fallback**: `Qh` (raw sensible heat flux)
- **Unit**: W m⁻²

### Latent Heat (Qle)
- **Primary**: `Qle_cor` (corrected latent heat flux)
- **Fallback**: `Qle` (raw latent heat flux)
- **Unit**: W m⁻²

## How It Works

1. **Priority**: The filter first attempts to use the corrected version (`*_cor`)
2. **Fallback**: If corrected version is not available, it automatically uses the raw version
3. **Logging**:
   - `INFO`: When using corrected version
   - `WARNING`: When falling back to raw version
   - `ERROR`: When neither version is available

## Usage

The filter is automatically invoked by OpenBench when:
- `ref_source` is set to `FLUX_PLUMBER2` in the configuration
- Processing Sensible_Heat or Latent_Heat evaluation items

### Example Configuration

In `ref-PLUMBER2.nml`:
```fortran
&general
  Sensible_Heat_ref_source=FLUX_PLUMBER2
/

&def_nml
  FLUX_PLUMBER2=./nml/nml-Fortran/Ref_variables_definition_station/FLUX_PLUMBER2.nml
/
```

In `FLUX_PLUMBER2.nml`:
```fortran
&Sensible_Heat
   varname =  Qh_cor  ! Will fallback to Qh if not found
   varunit =  w m-2
/

&Latent_Heat
   varname =  Qle_cor  ! Will fallback to Qle if not found
   varunit =  w m-2
/
```

## Test Results

```
✓ Test 1 (with Qh_cor): Uses Qh_cor when available
✓ Test 2 (fallback to Qh): Falls back to Qh when Qh_cor is missing
✅ All tests passed!
```

## Implementation Details

- **File**: `openbench/data/custom/FLUX_PLUMBER2_filter.py`
- **Function**: `filter_FLUX_PLUMBER2(info, ds)`
- **Returns**: `(updated_info, processed_data)`

## Benefits

1. **Flexibility**: Works with both corrected and raw FLUXNET data
2. **Robustness**: Handles missing corrected variables gracefully
3. **Transparency**: Clear logging of which variable is being used
4. **Automatic**: No manual intervention needed
