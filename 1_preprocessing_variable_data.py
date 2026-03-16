'''
This script processes netCDF files containing environmental data, 
extracts relevant variables, 
and saves the processed data as Parquet files for efficient storage and analysis.

Here we use CHL (Chlorophyll) as an example variable, but the script can be adapted to process other variables as needed.
'''

import netCDF4 as nc
import dask.array as da
import dask.dataframe as dd
from dask import delayed
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import fastparquet
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# Step 1: Load and Prepare Data
# ==============================
def load_and_prepare(nc_path):
    """Load NetCDF file and prepare basic data"""
    print(f"\n{' STEP 1: LOAD AND PREPARE DATA ':=^80}")
    with ProgressBar():
        with tqdm(total=4, desc="Preparing data") as pbar:
            # Open NetCDF file
            ds = nc.Dataset(nc_path, 'r')
            pbar.update(1)
            
            # Convert time data
            time_data = nc.num2date(
                ds.variables['time'][:], 
                'Days Since 1900-01-01',
                only_use_cftime_datetimes=False
            ).data
            pbar.update(1)
            
            # Convert to pandas datetime
            time_dt = pd.to_datetime(time_data).to_pydatetime()
            pbar.update(1)
            
            # Get latitude/longitude data
            lats = ds.variables['latitude'][:].data
            lons = ds.variables['longitude'][:].data
            pbar.update(1)
            
            return ds, time_dt, lats, lons


# =============================
# Step 2: Process Single Chunk
# =============================
def process_single_chunk(ds, time_dt, lats, lons, start_idx, end_idx):
    """Process single time chunk and return DataFrame"""
    with tqdm(total=6, desc=f"Processing chunk {start_idx}-{end_idx}") as pbar:
        # Get current time chunk
        current_times = time_dt[start_idx:end_idx]
        pbar.update(1)
        
        # Get MLD data
        mld_data = ds.variables["CHL"][start_idx:end_idx]
        pbar.update(1)
        
        # Create empty DataFrame
        df = pd.DataFrame(columns=['time', 'latitude', 'longitude', 'CHL'])
        pbar.update(1)
        
        # Process time column (convert to ms precision)
        time_values = []
        for t in current_times:
            # Add 12-hour offset and convert to ms
            adjusted_time = pd.to_datetime(t) + pd.Timedelta(hours=12)
            ms_time = adjusted_time.value // 10**6
            time_values.extend([pd.to_datetime(ms_time, unit='ms')] * (len(lats)*len(lons)))
        df['time'] = time_values
        df['time'] = pd.to_datetime(df['time']).astype('datetime64[ms]')
        pbar.update(1)
        
        # Process coordinates
        lon_grid, lat_grid = np.meshgrid(
            lons.astype('float32'), 
            lats.astype('float32')
        )
        df['latitude'] = np.tile(lat_grid.ravel(), len(current_times))
        df['longitude'] = np.tile(lon_grid.ravel(), len(current_times))
        pbar.update(1)
        
        # Process MLD data (handle fill values)
        fill_value = getattr(ds.variables["CHL"], '_FillValue', np.nan)
        fill_value = fill_value.astype('float32')
        mld_values = mld_data.reshape(-1)
        df['CHL'] = np.where(
            mld_values == fill_value, 
            np.nan, 
            mld_values.astype('float32')  
        )
        pbar.update(1)

        print(f"Chunk {start_idx}-{end_idx} completes df and ready for saving")
        return df


# =============================
# Step 3: Save Chunk to Parquet
# =============================
def save_to_parquet_chunkwise(df, output_dir, part_idx):
    """Append a single chunk df to parquet part file, sorted by time"""
    filename = os.path.join(output_dir, f'part_{part_idx:05d}.parquet')
    df = df.sort_values("time")
    df.to_parquet(filename, compression = 'gzip')


# =============================
# Step 4: Plot
# =============================
def plot_mld_single_snapshot(df_sample, date_str, output_dir, fill_value=-999):
    """Plot a single snapshot of CHL, replacing fill_value with NaN"""

    df_sample = df_sample.copy()
    df_sample.loc[:, 'CHL'] = df_sample['CHL'].replace(fill_value, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    hb = ax.hexbin(
        df_sample["longitude"], df_sample["latitude"], C=df_sample["CHL"],
        gridsize=100, cmap="viridis", reduce_C_function=np.nanmean, transform=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    cbar = plt.colorbar(hb, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Chlorophyll-a (mg/m$^3$)")

    plt.title(f"North Atlantic Chlorophyll-a (Hexbin) - {date_str}", pad=20)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"CHL_{date_str}.png")
    plt.savefig(output_path, dpi=700, bbox_inches="tight")
    plt.close()


# =====================================================
# Step 5： Call plot function by finding specific date
# =====================================================
def plot_first_day_from_all_parts(parts_dir, output_dir, fill_value=-999):
    """Loop through all parquet parts and plot the first day's snapshot in each"""
    files = sorted([
        f for f in os.listdir(parts_dir) 
        if f.endswith(".parquet")
    ])

    for fname in files:
        fpath = os.path.join(parts_dir, fname)
        try:
            df = pd.read_parquet(fpath)

            if 'time' not in df.columns or df.empty:
                print(f"Skipped {fname} (no data)")
                continue

            min_time = df['time'].min()
            df_sample = df[df['time'] == min_time]

            date_str = pd.to_datetime(min_time).strftime("%Y-%m-%d")
            print(f"Plotting {fname} → {date_str}")

            plot_mld_single_snapshot(df_sample, date_str, output_dir, fill_value)

        except Exception as e:
            print(f"Error processing {fname}: {e}")


# =============================
# Main Pipeline
# =============================
def process_netcdf_file(nc_path, output_path, time_chunk=30):
    try:
        ds, time_dt, lats, lons = load_and_prepare(nc_path)

        print(f"\n{' STEP 2: PROCESS DATA IN CHUNKS ':=^80}")
        output_dir = output_path.replace('.parquet', '_parts')
        os.makedirs(output_dir, exist_ok=True)

        for part_idx, start_idx in enumerate(
            tqdm(range(0, len(time_dt), time_chunk), desc="Overall progress", unit="chunk")
        ):
            end_idx = min(start_idx + time_chunk, len(time_dt))

            df = process_single_chunk(ds, time_dt, lats, lons, start_idx, end_idx)
            save_to_parquet_chunkwise(df, output_dir, part_idx)

        print(f"\n{' STEP 3: VERIFY OUTPUT ':=^80}")
        check_df = dd.read_parquet(output_dir)
        print(f"\n✓ Successfully saved to: {output_dir}")
        print(f"• Part files: {len(os.listdir(output_dir))}")
        '''
        # Step 4: Merge into single parquet file
        print(f"\n{' STEP 4: MERGE TO SINGLE FILE ':=^80}")
        merged_path = output_path.replace('.parquet', '_merged.parquet')
        merge_parquet_parts(output_dir, merged_path)
        '''
        print(f"\n{' STEP 4: PLOT SNAPSHOTS ':=^80}")
        plot_basename = os.path.basename(output_path).replace(".parquet", "").lower()
        plot_output_dir = os.path.join("plots", plot_basename)
        plot_first_day_from_all_parts(output_dir, plot_output_dir)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise
    finally:
        ds.close() if 'ds' in locals() else None


# Call the main function
# Here we save the chunks at 2 months
process_netcdf_file(
    r'CHL\cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_CHL_79.98W-14.98E_0.02N-74.98N_2002-07-04-2024-11-30.nc',
    r"processed_data/CHL_20020704_20241130.parquet",
    time_chunk = 60
)


# Extra: if you want to check the file structure before processing
# You can run them separately in Jupyter
chl = nc.Dataset(r'CHL\cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_CHL_79.98W-14.98E_0.02N-74.98N_2002-07-04-2024-11-30.nc', 'r')
chl.variables
# {'CHL': <class 'netCDF4._netCDF4.Variable'>
#  float32 CHL(time, latitude, longitude)
#      _FillValue: -999.0
#      valid_min: 0.0
#      units: milligram m-3
#      standard_name: mass_concentration_of_chlorophyll_a_in_sea_water
#      long_name: Chlorophyll-a concentration - Mean of the binned pixels
#      valid_max: 1000.0
#      coordinates: time lat lon
#  unlimited dimensions: 
#  current shape = (8186, 1800, 2280)
#  filling on,
#  'latitude': <class 'netCDF4._netCDF4.Variable'>
#  float32 latitude(latitude)
#      units: degrees_north
#      standard_name: latitude
#      long_name: latitude
#      axis: Y
#  unlimited dimensions: 
#  current shape = (1800,)
#  filling off,
#  'longitude': <class 'netCDF4._netCDF4.Variable'>
#  float32 longitude(longitude)
#      units: degrees_east
#      standard_name: longitude
#      long_name: longitude
#      axis: X
#  unlimited dimensions: 
#  current shape = (2280,)
#  filling off,
#  'time': <class 'netCDF4._netCDF4.Variable'>
#  float32 time(time)
#      standard_name: time
#      long_name: Time
#      unit_long: Days Since 1900-01-01
#      axis: T
#      units: days since 1900-01-01
#      calendar: Gregorian
#  unlimited dimensions: 
#  current shape = (8186,)
#  filling off}

# check parquet output
ddf = pd.read_parquet("processed_data/CHL_20020704_20241130_parts/part_00003.parquet")
ddf.dtypes
# time         datetime64[ns]
# latitude            float32
# longitude           float32
# CHL                 float32
# dtype: object
print(ddf["CHL"].describe())
# count    2.462400e+08
# mean    -3.173320e+02
# std      4.654305e+02
# min     -9.990000e+02
# 25%     -9.990000e+02
# 50%      8.587623e-02
# 75%      3.053899e-01
# max      5.996628e+01
# Name: CHL, dtype: float64

# Check on the last date in this file
df = ddf[ddf['time'] == ddf['time'].max()]
df
#       time  latitude  longitude       CHL
# ......
# 4104000 rows × 4 columns
print(df['CHL'].describe())
# count    4.104000e+06
# mean    -3.827938e+02
# std      4.859124e+02
# min     -9.990000e+02
# 25%     -9.990000e+02
# 50%      1.062911e-01
# 75%      1.880394e-01
# max      6.475559e+01
# Name: CHL, dtype: float64