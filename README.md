# AniDMS

AniDMS is a Python package for querying a monthly DMS NetCDF database and annotating trajectory data with DMS values.

## What AniDMS does

- Query DMS files by a date range.
- Work in two modes:
  - Local mode: read monthly files from a local folder.
  - Zenodo mode: resolve and download required monthly files from Zenodo automatically.
- Annotate trajectory records with DMS using:
  - LAEA-projected spatial IDW interpolation.
  - Optional temporal linear interpolation between previous/next daily 12:00 snapshots.
- Normalize all input datetimes to UTC and store output datetime as UTC-naive (no timezone suffix).

## Data model

AniDMS expects monthly files named like:

`DMS_YYYYMM_4km_NA.nc`

Dataset DOI from Zenodo: https://doi.org/10.5281/zenodo.18615736

Required datasets/coordinates inside each DMS file:

- `DMS`
- `time` (daily axis)
- Latitude/longitude coordinates:
  - `latitude` / `longitude`, or
  - `lat` / `lon`    
    
        
If you found any high-lat DMS is missing, this might due to the missing data from satellites. Please refer to [version 4](https://zenodo.org/records/18776459). 

## Installation

### Recommended: create a clean conda environment

```bash
conda create -n anidms-test python=3.11 -y
conda activate anidms-test
```

### Install from source

*Download this repository* and change directory to your folder:

```bash
cd c:\...\"your folder"
```

```bash
python -m pip install -e .
```

Dependencies are installed automatically from `setup.py`.

### Recommended: Install from Package Index/Anaconda/Conda forge

From PyPI: 
```bash
pip install anidms
```
Or:  
From conda-forge: 
```bash
conda install -c conda-forge anidms
```
From Anaconda: 
```bash
conda install -c meixuanliu -y anidms

conda install meixuanliu::anidms # same
```

## Quick start & Download data

### 1) Import

```python
from anidms import AniDMS
```

### 2) Local mode  
If you already have DMS data:

```python
db = AniDMS(
    data_path=r"c:\Data\Your DMS Path"
)

# Enter a date range & define where it will be saved
files = db.query_date_range("2010-01-05", "2010-03-10", output_dir=r".\downloads")
print(files)
```

### 3) Zenodo mode (no pre-downloaded DB needed)
Multi-strategy crawling:
```python
db = AniDMS(
    zenodo_doi="10.5281/zenodo.18615736",
    cache_dir=r".\dms_cache",
    request_timeout=60,
    request_retries=3,
    retry_backoff=1.5,
)

files = db.query_date_range("2010-01-05", "2010-03-10", output_dir=r".\downloads")
print(files)
```

### 4) Annotation from DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "DateTime": ["2010-01-07 06:00:00", "2010-01-07 12:00:00", "2010-01-07 18:00:00"],
    "Latitude": [42.1, 42.1, 42.1],
    "Longitude": [-60.2, -60.2, -60.2],
})

annotated = db.annotate_tracking_data(
    input_data=df,
    output_path=r".\annotated.csv",
    datetime_col="DateTime", # Designate the columns
    lat_col="Latitude",
    lon_col="Longitude",
    temporal_interp=True, # between 12:00, can be False
    min_valid=4, # params for spatial interpolation
    max_k=36,
    power=2.0,
    eps=1e-6,
)

print(annotated[["DateTime", "Latitude", "Longitude", "DMS"]].head())
```

### 5) Annotation from CSV path

```python
annotated = db.annotate_tracking_data(
    input_data=r"C:\path\to\trajectory.csv",
    output_path=r".\annotated.csv",
)
```  

### 6) Whole testing workflow (no pre-download needed)
```python
from pathlib import Path
import shutil
import pandas as pd
from anidms import AniDMS

# 1) Prepare a cache folder
cache_dir = Path(r".\dms_cache_zenodo")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

out_dir = Path(r".\test_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 2) Initialize Zenodo mode
db = AniDMS(
    zenodo_doi="10.5281/zenodo.18615736",
    cache_dir=str(cache_dir),
    request_timeout=60,
    request_retries=3,
    retry_backoff=1.5,
)

# 3) Your tracking data (taking 40 rows for test)
traj_path = Path(r"C:\Data\path\to your traj\df.csv")
df_all = pd.read_csv(traj_path)
df_test = pd.concat([df_all.head(20), df_all.tail(20)], ignore_index=True)

# 4) Annotation (will download monthly files when needed)
annotated = db.annotate_tracking_data(
    input_data=df_test,
    output_path=str(out_dir / "df_annotated_zenodo.csv"),
    datetime_col="DateTime",
    lat_col="Latitude",
    lon_col="Longitude",
    temporal_interp=True,
    min_valid=4,
    max_k=36,
    power=2.0,
    eps=1e-6,
)

# 5) Check result
print("\n=== Annotation Summary ===")
print("Output rows:", len(annotated))
print("Valid DMS:", int(annotated["DMS"].notna().sum()))
print("NaN DMS:", int(annotated["DMS"].isna().sum()))
print("Coverage %:", round(annotated["DMS"].notna().mean() * 100, 2))

display(annotated[["DateTime", "Latitude", "Longitude", "DMS"]].head(10))
display(annotated[["DateTime", "Latitude", "Longitude", "DMS"]].tail(10))

# 6) Check cached files
downloaded = sorted(cache_dir.glob("DMS_*_4km_NA.nc"))
print(f"\nCached monthly files: {len(downloaded)}")
for f in downloaded[:10]:
    print(" -", f.name)
``` 

## API summary

### `AniDMS.__init__(...)`

Important parameters:

- `data_path`: local monthly DB folder. If set, local mode is used.
- `zenodo_doi`: Zenodo concept DOI used in Zenodo mode.
- `cache_dir`: local cache folder for downloaded month files.
- `request_timeout`: request timeout in seconds.
- `request_retries`: retry count for request/download.
- `retry_backoff`: exponential retry backoff base.

### `query_date_range(start_date, end_date, output_dir)`

- Input date range is inclusive.
- Returns local paths to monthly files covering the range.
- In Zenodo mode, required month files are downloaded automatically.

### `annotate_tracking_data(...)`

- `input_data` supports CSV path (`str`) or `pandas.DataFrame`.
- Required columns (default names):
  - `DateTime`
  - `Latitude`
  - `Longitude`
- Adds a `DMS` column to output.

## Time and interpolation behaviour

### Datetime normalization

AniDMS uses a minimal UTC strategy:

- Parse all timestamps with `utc=True`.
- Convert to UTC.
- Drop timezone marker in output (`datetime64[ns]`, UTC-naive representation).

Practical effect:

- `"2019-07-01 06:00:00+02:00"` becomes UTC-equivalent time.
- `"2019-07-01 06:00:00"` (naive) is treated as UTC.

### Spatial interpolation

- Coordinate system for distance: LAEA projection (meters).
- Method: IDW with progressive neighbor search (`[4, 16, max_k]`).

### Temporal interpolation

- `temporal_interp=True` (default):
  - Spatial interpolation is computed on previous/next daily 12:00 snapshots.
  - Final DMS is linear interpolation in time between those two values.
- `temporal_interp=False`:
  - Spatial-only interpolation on same-day snapshot.

## Output and cache folders

- `output_dir` in `query_date_range`: destination for downloaded/copied monthly files.
- `cache_dir` in Zenodo mode: internal cache used during annotation/query.
- `output_path` in `annotate_tracking_data`: optional CSV output path.

## Folder explanations in this package

- `__pycache__`:
  - Python bytecode cache (`.pyc`).
  - Safe to delete; auto-regenerated.
- `anidms.egg-info`:
  - Metadata generated by editable install (`pip install -e .`).
  - Safe to delete; recreated on reinstall.
- `conda-recipe`:
  - Conda build recipe (`meta.yaml`) for packaging/release.


## Troubleshooting

### 1. `ModuleNotFoundError: No module named anidms` in Jupyter

Usually kernel/env mismatch.  
Confirm notebook kernel uses the same env where you installed AniDMS:

```python
import sys
print(sys.executable)
```

Then reinstall in that env:

```bash
python -m pip install -e .
```

### 2. `The environment is inconsistent, please check the package plan carefully` when installing via conda-forge

Please check the error message. If the packages that causing inconsistency include:  
```anaconda-*```  
That is because your environment has already mixed with ```_anaconda_depends```. This locks the solver. 

Check with: ```conda config --show create_default_packages```  
If there is any defaut packages: `conda config --remove-key create_default_packages`  

Or you can create the environment with:  
``` bash 
conda create -n anidms-test -y --override-channels -c conda-forge --no-default-packages python=3.11 anidms
```

Check after activating the environment: 
``` bash
python -c "from anidms import AniDMS; print(AniDMS)"
```  
If this still can't fix, please try alternative sources. 

### 3. Zenodo query returns missing month files

- Check network/proxy access.
- Keep default DOI unless you are testing another record:
  - `10.5281/zenodo.18615736`
- Retry with a small date range first.

## License

MIT.
