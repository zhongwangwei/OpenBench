# Station water-surface elevation: EGM2008 to EGM96

This opt-in correction applies **only to `Water_Surface_Elevation`** station
observations in EGM2008. The model/evaluation heights are already EGM96.
Other variables—including `River_Water_Level`, `Lake_Water_Level`,
`Dam_Water_Elevation` and dam-structure `Dam_Elevation`—are unchanged and must not
use this option. No model-side datum declaration or physical-description
inference is required or supported.

```yaml
reference:
  Water_Surface_Elevation: MyStations
  overrides:
    MyStations:
      variables:
        Water_Surface_Elevation:
          vertical_datum:
            source_geoid: /data/geoids/egm2008-2_5.pgm
            target_geoid: /data/geoids/egm96-5.pgm
```

Merge this fragment into a complete OpenBench project. `source: EGM2008`,
`target: EGM96` and `quantity: orthometric_height` are optional fixed assertions;
other values are rejected. Do not put `vertical_datum` in simulation mappings.
The simulation must actually represent absolute water-surface elevation in
EGM96, not water depth or a local-gauge anomaly; OpenBench does not infer this
from the model's name or generic descriptions.

GeographicLib defines ellipsoidal height `h = H + N`, hence:

```
H96 = H08 + N08 - N96
```

The correction uses each observation station's WGS84 `ref_lat` / `ref_lon`,
not its matched model-grid center. Both configured series and the processed
arrays must be in **metres (`m`)**. Raw station observations are also checked for
`units=m`; convert other units to metres before using this option. Explicit
observation datum metadata conflicting with EGM2008, a prior OpenBench
correction marker, missing coordinates or missing/invalid geoid grids fail
clearly. Free-text global descriptions are not interpreted as variable physics.

Normal evaluation applies one in-memory offset before metrics and station
plots; it never rewrites source or preprocessed NetCDF files. Repeated runs
start from the same uncorrected cached series. **When this option is enabled,
`only_drawing` reuses normal station evaluation on existing processed NetCDFs,
refreshing metric/score tables and plots rather than redrawing possibly stale
uncorrected tables.** It does not rerun preprocessing. Without this option,
`only_drawing` retains its ordinary read-only rendering behavior.

With this correction enabled, **`--comparison-only`, comparison redraws and report
generation in `only_drawing` mode are rejected** because cached aggregate products
cannot prove which datum produced them. Run normal evaluation with
`only_drawing: false` and without `--comparison-only` to regenerate products, or omit
`--comparison-only` and set
both `comparison.enabled: false` and `project.generate_report: false` for
station-only metric/plot refreshes (report generation defaults to enabled).
The runner rejects this combination before evaluation or reporting;
existing comparison products are not read or rewritten. Unselected references do
not trigger this restriction.

Supply the two global unsigned 16-bit
[GeographicLib PGM geoid grids](https://geographiclib.sourceforge.io/C++/doc/geoid.html).
OpenBench neither bundles nor automatically downloads them. Bilinear
interpolation is approximate and does not guarantee centimetre accuracy at
every location. Relative paths follow the reference source root; `~` and
environment variables are expanded. GUI import/export preserves these manual
YAML fields and maps the paths for remote execution; no new GUI editor is added.
Set `vertical_datum: null` to clear an inherited correction. Changes to geoid
files participate in the input cache signature.

Synthetic-grid tests cover sign, coordinates, malformed grids, repeated runs,
legacy opt-out behavior and metric/plot agreement. Real station observations
still require validation before scientific use.
