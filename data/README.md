# Data directory

This folder is an empty skeleton. No datasets are bundled with the release.

The pipeline expects you to provide a synthetic population (and optionally a geo polygon and raw surveillance data). Everything written into this folder at runtime (snapshots, logs, figures) is generated locally and is gitignored.

See `../docs/DATA.md` for:

- the synthetic-population file format (tab-separated, one row per agent),
- where to obtain a compatible population dataset,
- the optional geo file used by the Streamlit dashboard,
- the layout of the `results/` subdirectory written by every run.

## Expected layout once populated

```
data/
  population/
    households.txt          (you provide this)
  geo/
    vasileostrovsky.geojson (optional, you provide this)
  raw/
    influenza_*.csv         (optional, for ABC calibration)
  results/                  (created at runtime, gitignored)
    <run-id>/
      metrics.json
      snapshots/
      logs/
      figures/
```
