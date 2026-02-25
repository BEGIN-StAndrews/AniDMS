# AniDMS Package

## Installation

```bash
pip install -e .
```

## Core Features

1. Query DMS monthly NetCDF files by date range.
2. Annotate tracking data with daily DMS values from monthly files using LAEA-based IDW.

## Data Model

- File naming: `DMS_YYYYMM_4km_NA.nc`
- Required variables/coords in each file:
  - `DMS`
  - `time` (daily axis)
  - latitude/longitude coords (`latitude`/`longitude` or `lat`/`lon`)

## Usage

```python
from anidms import AniDMS

# Zenodo mode (default): dynamically discover record versions from concept DOI.
db = AniDMS(
    zenodo_doi="10.5281/zenodo.18615736",
    cache_dir="./dms_cache",
)

# Download month files overlapping the date range.
files = db.query_date_range("2010-01-05", "2010-03-10", output_dir="./downloads")
print(files)
```

```python
import pandas as pd

tracking = pd.DataFrame(
    {
        "DateTime": ["2010-01-07 06:00:00", "2010-01-07 12:00:00"],
        "Latitude": [42.1, 42.5],
        "Longitude": [-60.2, -59.8],
    }
)

annotated = db.annotate_tracking_data(tracking)
print(annotated.head())
```

## API Reference

### `AniDMS.__init__(...)`

Key parameters:

- `data_path`: optional local month-file directory. If set, local mode is used.
- `zenodo_doi`: concept DOI used for dynamic Zenodo version discovery.
- `cache_dir`: local cache for downloaded month files.
- `request_timeout`: HTTP timeout in seconds.
- `request_retries`: retry attempts for Zenodo requests/downloads.
- `retry_backoff`: exponential backoff factor between retries.

### `query_date_range(start_date, end_date, output_dir)`

- Input dates are inclusive.
- Returns local paths to downloaded/copied month files.
- In Zenodo mode, version routing is resolved automatically via concept DOI metadata.

### `annotate_tracking_data(input_data, ...)`

- `input_data` accepts either CSV path (`str`) or `pandas.DataFrame`.
- Data are grouped by tracking date.
- Default behavior (`temporal_interp=True`):
  - First locate previous/next daily DMS snapshots at 12:00.
  - Perform LAEA-IDW spatial interpolation on each reference snapshot.
  - Then apply linear temporal interpolation based on trajectory timestamp.
- Optional `temporal_interp=False` keeps spatial-only annotation on the same day snapshot.
- Missing DMS dates keep original rows and set `DMS = NaN`.

## Dependencies

- `pandas`, `numpy`, `xarray`, `scipy`, `netCDF4`
- `requests` (Zenodo API + file download)
- `pyproj` (LAEA projection)

## Conda Packaging

- A starter recipe is included at `conda-recipe/meta.yaml`.
- Build locally:

```bash
conda build conda-recipe
```

- Upload (after `anaconda login`):

```bash
anaconda upload <path-to-built-package>
```
