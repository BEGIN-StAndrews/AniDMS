'''This script performs IDW interpolation of DMS values from ocean grid points to bird tracking points for the month of July 2019. 
It uses a KDTree for efficient neighbor search and projects coordinates to a suitable metric system (LAEA) for accurate distance calculations. 
The output is a CSV file with the original bird tracking data plus an additional column for the interpolated DMS values.

This script is a test for the functions and workflow of the annotation process, and is not optimised for performance or edge cases.
The contents now have been included in the core.py of AniDMS. '''

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Transformer
from tqdm import tqdm


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
bird_data_path = r"C:\Your path\trajectory.csv"
output_dir = r"C:\Your path\annotated"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Date range (set to 2019-07-01 to 2019-07-31)
# NOTE: Make end exclusive to include the full day of 2019-07-31.
# ---------------------------------------------------------
start_date = datetime(2019, 7, 1)
end_date = datetime(2019, 7, 31)
end_exclusive = end_date + timedelta(days=1)

# ---------------------------------------------------------
# Projection setup (LAEA, meters) for distance calculations
# ---------------------------------------------------------
proj_string = (
    "+proj=laea +lat_0=37.5 +lon_0=-32.5 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs"
)

transformer_to_proj = Transformer.from_crs(
    "EPSG:4326",  # WGS84 lon/lat
    proj_string,  # LAEA meters
    always_xy=True
)

transformer_to_geo = Transformer.from_crs(
    proj_string,
    "EPSG:4326",
    always_xy=True
)

print("Projection: Lambert Azimuthal Equal Area")
print("Center: 37.5°N, 32.5°W")
print("Units: meters")
print("Datum: WGS84")


# ---------------------------------------------------------
# IDW: project coords -> KDTree in meters
# ---------------------------------------------------------
def interpolate_to_dms(
    ocean_df,
    var_name,
    dms_points,
    transformer_to_proj,
    min_valid=4,
    max_k=36,
    power=1.0,      # 1/d^2
    eps=1e-12
):
    """
    IDW interpolation with progressive neighbor search.
    Only change vs original: distances are computed in projected meters (LAEA).
    """

    # --- Build KDTree in projected coordinates (meters) ---
    ocean_lon = ocean_df["longitude"].to_numpy(dtype=float)
    ocean_lat = ocean_df["latitude"].to_numpy(dtype=float)
    ocean_x, ocean_y = transformer_to_proj.transform(ocean_lon, ocean_lat)

    coords = np.c_[ocean_x, ocean_y]
    values = ocean_df[var_name].to_numpy()
    tree = cKDTree(coords)

    interpolated = []

    # iterate target points (keep your iterrows style for minimal diff)
    for _, row in dms_points.iterrows():
        # --- project target point ---
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        tx, ty = transformer_to_proj.transform(lon, lat)
        target = [tx, ty]

        k_list = [4, 16, max_k] if max_k > 16 else [4, max_k]
        interpolated_val = np.nan

        for k in k_list:
            k = min(k, len(ocean_df))
            dists, idxs = tree.query(target, k=k)

            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            valid_mask = ~np.isnan(values[idxs])
            valid_dists = dists[valid_mask]
            valid_vals = values[idxs][valid_mask]

            if len(valid_vals) >= min_valid:
                vd = valid_dists[:min_valid]
                vv = valid_vals[:min_valid]
                weights = 1.0 / (np.power(vd, power) + eps)
                interpolated_val = np.sum(weights * vv) / np.sum(weights)
                break

            elif len(valid_vals) > 0:
                weights = 1.0 / (np.power(valid_dists, power) + eps)
                interpolated_val = np.sum(weights * valid_vals) / np.sum(weights)
                print(f"Warning: Only {len(valid_vals)} valid neighbors for point at {[lat, lon]}")
                break

        interpolated.append(interpolated_val)

    return interpolated


# ---------------------------------------------------------
# Load bird tracking data and filter to selected date range
# ---------------------------------------------------------
print("Loading bird tracking data...")
bird_df = pd.read_csv(bird_data_path)

bird_df["DateTime"] = pd.to_datetime(bird_df["DateTime"], utc=False, errors="coerce")
bird_df = bird_df.dropna(subset=["DateTime"])

mask = (bird_df["DateTime"] >= start_date) & (bird_df["DateTime"] < end_exclusive)
filtered_bird_df = bird_df.loc[mask].copy()

print(f"Original bird data points: {len(bird_df)}")
print(f"Filtered bird data points (2019-07-01 to 2019-07-31): {len(filtered_bird_df)}")

# Group by date
date_groups = filtered_bird_df.groupby(filtered_bird_df["DateTime"].dt.date)

daily_dfs = []

# ---------------------------------------------------------
# Process each day
# ---------------------------------------------------------
for date, daily_bird_df in tqdm(date_groups, desc="Processing daily data"):
    date_str = pd.Timestamp(date).strftime("%Y%m%d")
    dms_file = fr"C:\Data\DMS Estimation\DMS Env Data\predictions\DMS_{date_str}_4km_NA.nc"

    # Load DMS data for this day
    try:
        ds = xr.open_dataset(dms_file)
        dms_df = (
            ds[["DMS"]]
            .to_dataframe()
            .reset_index()[["latitude", "longitude", "DMS"]]
            .dropna(subset=["DMS"])
        )
        ds.close()
    except FileNotFoundError:
        print(f"DMS file not found for date {date_str}, skipping...")
        continue

    # Extract coordinates from bird data (keep your original logic, minimal changes)
    if "geometry" in daily_bird_df.columns:
        gdf = gpd.GeoDataFrame(
            daily_bird_df,
            geometry=gpd.points_from_xy(daily_bird_df["Longitude"], daily_bird_df["Latitude"]),
            crs="EPSG:4326",
        )
        daily_bird_df["latitude"] = gdf.geometry.y
        daily_bird_df["longitude"] = gdf.geometry.x
    else:
        daily_bird_df["latitude"] = daily_bird_df["Latitude"]
        daily_bird_df["longitude"] = daily_bird_df["Longitude"]

    # Perform IDW interpolation (projected meter distances)
    print(f"Interpolating DMS values for {date_str}...")
    interpolated_dms = interpolate_to_dms(
        ocean_df=dms_df,
        var_name="DMS",
        dms_points=daily_bird_df[["latitude", "longitude"]],
        transformer_to_proj=transformer_to_proj,
        min_valid=4,
        max_k=36,
        power=2.0,   # keep accurate IDW: 1/d^2
        eps=1e-6
    )

    daily_bird_df["DMS"] = interpolated_dms
    daily_dfs.append(daily_bird_df)

# ---------------------------------------------------------
# Combine and save
# ---------------------------------------------------------
if daily_dfs:
    annotated_df = pd.concat(daily_dfs, ignore_index=True)

    # Remove temporary columns if they were added
    if "latitude" in annotated_df.columns and "longitude" in annotated_df.columns:
        annotated_df.drop(columns=["latitude", "longitude"], inplace=True)
else:
    print("No data was processed - check if DMS files exist for the date range")
    annotated_df = pd.DataFrame()

if not annotated_df.empty:
    output_path = os.path.join(
        output_dir,
        f"annotated_bird_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    )
    annotated_df.to_csv(output_path, index=False)
    print(f"Successfully saved annotated data to {output_path}")
    print(f"Total annotated points: {len(annotated_df)}")
    if "DMS" in annotated_df.columns:
        print(f"Points with valid DMS: {annotated_df['DMS'].notna().sum()}")
else:
    print("No data to save - processing failed or no matching dates found")
