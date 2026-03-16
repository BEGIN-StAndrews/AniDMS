import multiprocessing as mp
import threading
from functools import partial
import pyarrow
import fastparquet
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datetime import datetime, timedelta, date
import xarray as xr
import dask.dataframe as dd
from dask import delayed
from functools import lru_cache
from tqdm import tqdm  
import warnings
import json
warnings.filterwarnings('ignore')
import time
from pathlib import Path
import gc
import psutil
from scipy.spatial import cKDTree
import regionmask
from joblib import Parallel, delayed
from pyproj import Transformer

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor


class Efficient4kmDMSPredictor:
    def __init__(self, model_path, scaler_path, output_dir='predictions/redo', n_processes=None):
        """
        Efficient 4km DMS predictor with PROJECTED COORDINATES
        
        Key improvements:
        - Uses Lambert Azimuthal Equal Area projection for North Atlantic
        - One-shot IDW interpolation (no chunking)
        - Accurate distance calculations in meters
        
        Args:
            n_processes: Number of processes to use (None = auto-detect)
        """
        print("Initializing 4km DMS Predictor with Projected Coordinates...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Conservative process count to avoid memory issues
        if n_processes is None:
            self.n_processes = min(mp.cpu_count(), 23)  # max 23
        else:
            self.n_processes = min(n_processes, 23)
        print(f"Using {self.n_processes} processes")
        
        self.features = ["CHL", "SST", "MLD", "PAR", "NO3"]
        self.special_values = {
            'CHL': -999, 'NO3': 9.96921e+36, 'SST': -32768, 'MLD': -32767
        }
        
        # Data source transitions
        self.date_thresholds = {
            'MLD': datetime(2021, 7, 1).date(),
            'NO3': datetime(2023, 1, 1).date(),
            'SST': datetime(2023, 1, 1).date()
        }
        
        # Updated paths with transitions
        self.var_paths = {
            'CHL': {'primary': 'processed_data_redo/CHL_20020704_20241130_parts'},
            'MLD': {
                'primary': 'processed_data_redo/MLD_20020704_20210630_parts',
                'secondary': 'processed_data_redo/MLD_20210701_20241130_parts'
            },
            'NO3': {
                'primary': 'processed_data_redo/NO3_20020704_20221231_parts',
                'secondary': 'processed_data_redo/NO3_20230101_20241130_parts'
            },
            'SST': {
                'primary': 'processed_data_redo/SST_CDSv2.1_parts',
                'secondary': 'processed_data_redo/SST_CDRv3.0_parts'
            },
            'PAR': {'primary': 'processed_data_redo/PAR'}
        }
        
        # Setup projection system FIRST
        self.setup_projection()
        
        # Create standard 4km grid with projection
        self.create_standard_4km_grid()
        
        # Gap detection log
        self.gap_log = []
        self.sst_mapping_cache = {}
        
        print("4km Predictor initialised with projection")

    def setup_projection(self):
        """
        Setup Lambert Azimuthal Equal Area projection for North Atlantic
        
        This projection:
        - Preserves area (important for oceanographic data)
        - Minimizes distortion in the North Atlantic region
        - Uses meters as units (easy distance calculations)
        """
        print("\n=== Setting up coordinate projection ===")
        
        # Lambert Azimuthal Equal Area centered on North Atlantic
        # Center point: middle of domain (37.5°N, 32.5°W)
        proj_string = "+proj=laea +lat_0=37.5 +lon_0=-32.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        
        # Create transformers for both directions
        self.transformer_to_proj = Transformer.from_crs(
            "EPSG:4326",      # WGS84 (lat/lon)
            proj_string,      # LAEA projection (x/y in meters)
            always_xy=True    # Input/output order: (lon, lat) or (x, y)
        )
        
        self.transformer_to_geo = Transformer.from_crs(
            proj_string,
            "EPSG:4326",
            always_xy=True
        )
        
        print("  Projection: Lambert Azimuthal Equal Area")
        print("  Center: 37.5°N, 32.5°W")
        print("  Units: meters")
        print("  Datum: WGS84")

    def create_standard_4km_grid(self):
        """
        Create standard 4km North Atlantic grid with BOTH geographic and projected coordinates
        
        Geographic coordinates: for NetCDF output
        Projected coordinates: for accurate distance calculations in IDW
        """
        print("\n=== Creating standard 4km North Atlantic grid ===")
        
        # 4km resolution (~0.036 degrees)
        resolution = 0.036
        lats = np.arange(0, 60 + resolution, resolution, dtype=np.float64)
        lons = np.arange(-80, 15 + resolution, resolution, dtype=np.float64)
        print(f"Geographic grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
        
        # Create coordinate mesh
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Compute ocean mask
        print("Computing ocean mask...")
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        ocean_mask_2d = np.isnan(land.mask(lons, lats).values)
        ocean_mask_flat = ocean_mask_2d.flatten()
        
        # Store GEOGRAPHIC grid (for NetCDF output)
        self.standard_lats_1d = lats
        self.standard_lons_1d = lons
        self.standard_grid_shape = lat_grid.shape
        self.standard_grid_lats_flat = lat_grid.flatten().astype(np.float64)
        self.standard_grid_lons_flat = lon_grid.flatten().astype(np.float64)
        self.standard_ocean_mask = ocean_mask_flat
        
        # Ocean coordinates (geographic)
        self.standard_ocean_lats = self.standard_grid_lats_flat[ocean_mask_flat]
        self.standard_ocean_lons = self.standard_grid_lons_flat[ocean_mask_flat]
        
        print(f"Ocean points: {len(self.standard_ocean_lats):,} ({len(self.standard_ocean_lats)/len(self.standard_grid_lats_flat)*100:.1f}%)")
        
        # PROJECT ocean coordinates to planar system
        print("Projecting ocean coordinates to LAEA...")
        self.standard_ocean_x, self.standard_ocean_y = self.transformer_to_proj.transform(
            self.standard_ocean_lons,
            self.standard_ocean_lats
        )
        self.standard_ocean_x = self.standard_ocean_x.astype(np.float64)
        self.standard_ocean_y = self.standard_ocean_y.astype(np.float64)
        
        # Print projection statistics
        print(f"  Projected X range: {self.standard_ocean_x.min()/1000:.0f} to {self.standard_ocean_x.max()/1000:.0f} km")
        print(f"  Projected Y range: {self.standard_ocean_y.min()/1000:.0f} to {self.standard_ocean_y.max()/1000:.0f} km")
        print(f"  Domain width: {(self.standard_ocean_x.max() - self.standard_ocean_x.min())/1000:.0f} km")
        print(f"  Domain height: {(self.standard_ocean_y.max() - self.standard_ocean_y.min())/1000:.0f} km")
        
        print("Standard 4km grid created with projection\n")
    
    def get_data_path(self, var_name, target_date):
        """Get appropriate data path based on date"""
        paths = self.var_paths[var_name]
        
        if var_name in self.date_thresholds:
            threshold = self.date_thresholds[var_name]
            if target_date >= threshold and 'secondary' in paths:
                return paths['secondary']
        
        return paths['primary']

    def get_sst_file_date_mapping(self, base_path):
        """Use cached SST file mappings or build if not available"""
        base_path_obj = Path(base_path)
        mapping_file = base_path_obj / "sst_date_mapping.json"
        
        # If mapping file exists, load it
        if mapping_file.exists():
            print(f"    Loading pre-generated SST mapping from {mapping_file}")
            try:
                with open(mapping_file, 'r') as f:
                    raw_mapping = json.load(f)
                    mapping = {str(k): v for k, v in raw_mapping.items()}
                    self.sst_mapping_cache[str(base_path)] = mapping
                    if mapping:
                        sample_dates = list(mapping.keys())[:5]
                        print(f"    Sample dates in mapping: {sample_dates}")
                    return mapping
            except:
                print("    Failed to load mapping file, regenerating...")
        
        # Check cache
        base_path_str = str(base_path)
        if base_path_str in self.sst_mapping_cache:
            return self.sst_mapping_cache[base_path_str]
        
        # If not cached, build mapping
        return self._build_sst_mapping(base_path_obj)

    def _build_sst_mapping(self, base_path_obj):
        """Build SST date to file mapping"""
        print(f"    Building SST mapping for {base_path_obj}...")
        mapping = {}
        sst_files = list(base_path_obj.glob("part_*.parquet"))
        
        for i, f in enumerate(sst_files):
            if i % 500 == 0:
                print(f"    Processing {i+1}/{len(sst_files)} files")
            
            try:
                df_sample = pd.read_parquet(
                    f, 
                    columns=['time'], 
                    engine='pyarrow',
                    use_threads=False  
                )
                
                if not df_sample.empty:
                    file_date = pd.to_datetime(df_sample['time'].iloc[0]).date()
                    mapping[file_date.isoformat()] = str(f)
            
            except Exception as e:
                print(f"    Error with {f.name}: {e}")
                continue
        
        # Save mapping to file
        mapping_file = base_path_obj / "sst_date_mapping.json"
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            print(f"    SST mapping saved to {mapping_file}")
        except Exception as e:
            print(f"    Failed to save mapping file: {e}")
        
        # Update cache
        self.sst_mapping_cache[str(base_path_obj)] = mapping
        return mapping
    
    def get_parquet_path_for_date(self, var_name, target_date):
        """Get exact parquet file path with debug info"""
        base_path = self.get_data_path(var_name, target_date)
        print(f"    DEBUG: Base path for {var_name}: {base_path}")
        
        START_DATES = {
            'CHL': datetime(2002, 7, 4).date(),
            'MLD': datetime(2002, 7, 4).date(),
            'NO3': datetime(2002, 7, 4).date(),
            'SST': datetime(2002, 7, 1).date(),
            'PAR': None
        }
        
        if var_name == 'PAR':
            base_dir = Path(base_path)
            par_files = list(base_dir.glob("PAR_*.parquet"))
            
            for f in par_files:
                name_parts = f.stem.split('_')
                if len(name_parts) >= 3:
                    start_str, end_str = name_parts[1], name_parts[2]
                    start_date = datetime.strptime(start_str, "%Y%m%d").date()
                    end_date = datetime.strptime(end_str, "%Y%m%d").date()
                    if start_date <= target_date <= end_date:
                        return str(f)
            print(f"    DEBUG: No PAR file found for {target_date}")
            return None
            
        elif var_name == 'SST':
            mapping = self.get_sst_file_date_mapping(base_path)

            if isinstance(target_date, datetime):
                date_key = target_date.date().isoformat()
            elif isinstance(target_date, date):
                date_key = target_date.isoformat()
            else:
                date_key = str(target_date)
            
            file_path = mapping.get(date_key)
            if file_path is None:
                print(f"    DEBUG: No mapping found for date {date_key}")
                print(f"    DEBUG: Available dates sample: {list(mapping.keys())[:5] if mapping else 'No dates available'}")
            return file_path
            
        else:
            # Handle dataset transitions
            if var_name == 'MLD' and target_date >= self.date_thresholds['MLD']:
                start_date = self.date_thresholds['MLD']
                print(f"    DEBUG: Using MLD secondary dataset, start: {start_date}")
            elif var_name == 'NO3' and target_date >= self.date_thresholds['NO3']:
                start_date = self.date_thresholds['NO3']
                print(f"    DEBUG: Using NO3 secondary dataset, start: {start_date}")
            else:
                start_date = START_DATES[var_name]
            
            days_since_start = (target_date - start_date).days
            
            if days_since_start < 0:
                return None
                
            file_num = days_since_start // 60  # 60-day chunks
            file_path = Path(base_path) / f"part_{file_num:05d}.parquet"
            
            print(f"    DEBUG: File exists: {file_path.exists()}")
            
            return str(file_path) if file_path.exists() else None
    
    def load_variable_data(self, var_name, target_date):
        """Load variable data for a specific date with debug info"""
        print(f"    DEBUG: Loading {var_name} for {target_date}")
        
        file_path = self.get_parquet_path_for_date(var_name, target_date)
        if not file_path:
            print(f"    DEBUG: No file path found for {var_name} on {target_date}")
            return None
        
        try:
            # Try with pyarrow first
            target_ts_utc = pd.Timestamp(target_date).tz_localize('UTC').replace(hour=12)
            filters = [('time', '=', target_ts_utc)]
            
            df = pd.read_parquet(
                file_path,
                columns=['time', 'latitude', 'longitude', var_name],
                filters=filters,
                engine='pyarrow'  
            )
            print(f"    DEBUG: PyArrow loaded {len(df)} rows")
            
        except Exception as e:
            print(f"    DEBUG: PyArrow failed: {e}")
            try:
                # Fallback to fastparquet
                print(f"    DEBUG: Trying fastparquet...")
                df = pd.read_parquet(
                    file_path,
                    columns=['time', 'latitude', 'longitude', var_name],
                    engine='fastparquet'
                )
                print(f"    DEBUG: FastParquet loaded {len(df)} rows")
                
                # Filter after loading
                if not df.empty:
                    original_len = len(df)
                    df = df[df['time'].dt.date == target_date]
                
            except Exception as e2:
                print(f"    DEBUG: FastParquet also failed: {e2}")
                return None

        if df.empty:
            print(f"    DEBUG: DataFrame is empty after loading/filtering")
            return None
        
        # Special value handling
        if var_name in self.special_values:
            special_val = self.special_values[var_name]
            
            special_count = (df[var_name] == special_val).sum()
            print(f"    DEBUG: Found {special_count} exact matches for special value {special_val}")
            
            # Check for values very close to special value
            if var_name == 'NO3':
                close_special = df[var_name] > 1e30
                df.loc[close_special, var_name] = np.nan
            elif var_name == 'CHL':
                close_special = df[var_name] < -900
                df.loc[close_special, var_name] = np.nan
            elif var_name == 'SST':
                close_special = df[var_name] < -1000
                df.loc[close_special, var_name] = np.nan
            elif var_name == 'MLD':
                close_special = df[var_name] < -1000
                df.loc[close_special, var_name] = np.nan
            
            # Original exact replacement
            df[var_name] = df[var_name].replace(special_val, np.nan)
        
        print(f"    DEBUG: After special value replacement - {var_name} range: {df[var_name].min():.6f} to {df[var_name].max():.6f}")
        print(f"    DEBUG: {var_name} NaN count: {df[var_name].isna().sum()}/{len(df)}")
        
        df['latitude'] = df['latitude'].astype(np.float32)
        df['longitude'] = df['longitude'].astype(np.float32)
        df[var_name] = df[var_name].astype(np.float32)

        return df.dropna(subset=['latitude', 'longitude', var_name]).copy()
    
    def align_to_standard_grid(self, df, var_name):
        """Align variable data to 4km grid points using projected IDW"""
        if df is None or df.empty:
            return np.full(len(self.standard_ocean_lats), np.nan, dtype=np.float32)
        
        return self.interpolate_to_4km(df, var_name)

    def interpolate_to_4km(self, df, var_name):
        """
        ONE-SHOT IDW interpolation using PROJECTED COORDINATES
        
        Key improvements:
        - Uses projected coordinates (meters) for accurate distances
        - Processes ALL grid points in one batch (no chunking)
        - Avoids batch boundary
        
        Distance calculations are now accurate across all latitudes!
        """
        print(f"    Interpolating {var_name} to 4km grid (PROJECTED)")
        
        valid_df = df.dropna(subset=[var_name])
        if valid_df.empty:
            return np.full(len(self.standard_ocean_lats), np.nan, dtype=np.float32)
        
        # PROJECT source coordinates to planar system
        print(f"      Projecting {len(valid_df):,} source points...")
        src_x, src_y = self.transformer_to_proj.transform(
            valid_df['longitude'].values,
            valid_df['latitude'].values
        )
        src_coords = np.column_stack([src_x, src_y]).astype(np.float64)
        src_values = valid_df[var_name].values.astype(np.float32)
        
        # Handle infinite values
        if np.any(np.isinf(src_values)):
            print("      WARNING: Infinite values detected, replacing with NaN")
            src_values = np.where(np.isinf(src_values), np.nan, src_values)
        
        # Build KDTree in PROJECTED space (units: meters)
        print(f"      Building KDTree with {len(src_coords):,} source points...")
        tree = cKDTree(src_coords)
        
        # Target coordinates (projected ocean grid points)
        tgt_coords = np.column_stack([self.standard_ocean_x, self.standard_ocean_y])
        
        # ========================================
        # ONE-SHOT QUERY
        # ========================================
        print(f"      Querying {len(tgt_coords):,} grid points in ONE BATCH...")
        k_neighbors = min(16, len(src_values))
        
        t_start = time.time()
        distances, indices = tree.query(
            tgt_coords,
            k=k_neighbors,
            workers=self.n_processes,
            distance_upper_bound=50000  # 50km search radius (meters)
        )
        t_query = time.time() - t_start
        print(f"      Query completed in {t_query:.2f} seconds")
        
        # IDW weight calculation
        print(f"      Computing IDW weights...")
        
        # Handle the case where k=1 (distances/indices are 1D)
        if k_neighbors == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        # Initialize result array
        result = np.full(len(tgt_coords), np.nan, dtype=np.float32)
        
        # Handle invalid indices from distance_upper_bound
        # When a point is beyond search radius, KDTree returns index = len(array)
        # This causes index out of bounds errors
        invalid_indices = indices >= len(src_values)

        # Clip indices to valid range (replace invalid with 0, we'll mask them)
        indices_safe = np.clip(indices, 0, len(src_values) - 1)

        # Vectorized weight calculation
        epsilon = 1e-10
        valid_mask = np.isfinite(distances) & (~invalid_indices)
        
        # Replace infinite distances (beyond search radius) with NaN
        distances = np.where(np.isinf(distances), np.nan, distances)
        
        # Compute weights: 1/d^2 (IDW with power=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (distances**2 + epsilon)
            weights = np.where(valid_mask, weights, 0)
        
        # Get values for all neighbors
        neighbor_values = src_values[indices_safe]
        neighbor_values = np.where(valid_mask, neighbor_values, np.nan)
        
        # Weighted sum
        weights_sum = np.nansum(weights, axis=1)
        weighted_values = np.nansum(weights * neighbor_values, axis=1)
        
        # Only compute result where we have valid weights
        valid_result = weights_sum > 0
        result[valid_result] = (weighted_values[valid_result] / weights_sum[valid_result]).astype(np.float32)
        
        valid_count = (~np.isnan(result)).sum()
        coverage = valid_count / len(tgt_coords) * 100
        print(f"      Interpolated: {valid_count:,}/{len(tgt_coords):,} points ({coverage:.1f}%)")
        
        return result

    def temporal_interpolate(self, var_name, target_date):
        """Improved temporal interpolation with 7-day window"""
        print(f"    Temporal interpolation for {var_name} (7-day window)")
        
        # Try different window sizes in order
        for window_size in [1, 2, 3]:
            days_to_try = []
            weights = []
            
            # Generate days to try with weights
            for offset in range(-window_size, window_size + 1):
                if offset == 0:  # Skip target date itself
                    continue
                days_to_try.append(target_date + timedelta(days=offset))
                weights.append(1.0 / (abs(offset) + 0.1))
            
            # Load data for all days in window
            aligned_arrays = []
            valid_weights = []
            valid_days = 0
            
            for i, day in enumerate(days_to_try):
                df = self.load_variable_data(var_name, day)
                if df is not None and not df.empty:
                    aligned = self.align_to_standard_grid(df, var_name)
                    aligned_arrays.append(aligned)
                    valid_weights.append(weights[i])
                    valid_days += 1
            
            # If we have at least one valid day in this window
            if valid_days > 0:
                print(f"      Using {window_size}-day window ({valid_days} valid days)")
                
                # Weighted average
                stacked = np.stack(aligned_arrays, axis=0)
                weights_arr = np.array(valid_weights).reshape(-1, 1)
                result = np.nansum(stacked * weights_arr, axis=0) / np.nansum(weights_arr, axis=0)
                
                valid_count = (~np.isnan(result)).sum()
                print(f"      Interpolated: {valid_count} points")
                return result.astype(np.float32)
        
        # If all windows failed
        print("      All windows failed - returning NaN")
        return np.full(len(self.standard_ocean_lats), np.nan, dtype=np.float32)
        
    def process_variable(self, var_name, date):
        """Process single variable for the date"""
        print(f"Processing {var_name}...")
        
        try:
            # Load data
            df = self.load_variable_data(var_name, date)
            
            # If no data, try temporal interpolation
            if df is None or df.empty:
                result = self.temporal_interpolate(var_name, date)
            else:
                # Align to standard grid
                result = self.align_to_standard_grid(df, var_name)
            
            result = result.astype(np.float16)
            # Clean up
            del df
            gc.collect()
            self.sst_mapping_cache.clear()
            return result
            
        except Exception as e:
            print(f"Error processing {var_name}: {e}")
            return np.full(len(self.standard_ocean_lats), np.nan, dtype=np.float16)

    def predict_daily(self, date):
        """Predict DMS for a single date"""
        print(f"\n=== Predicting DMS for {date} ===")
        
        # Process variables one at a time
        features = {}
        
        for i, var in enumerate(self.features):
            print(f"  Processing {var} ({i+1}/{len(self.features)})")
            features[var] = self.process_variable(var, date).astype(np.float16)
            gc.collect() 
    
        # Feature matrix
        X = np.empty((len(self.standard_ocean_lats), len(self.features)), dtype=np.float16)
        for i, var in enumerate(self.features):
            X[:, i] = features[var]
        
        # Clean up features dict
        del features
        gc.collect()
        
        # Only predict for complete rows
        valid_mask = ~np.isnan(X).any(axis=1)
        predictions = np.full(len(self.standard_ocean_lats), np.nan, dtype=np.float16)

        if valid_mask.any():
            X_valid = X[valid_mask]
            predictions[valid_mask] = self._parallel_predict(X_valid)
        
        # Clean up
        del X, X_valid
        gc.collect()
        self.detect_prediction_gaps(predictions, date)
        return predictions                       
    
    def _parallel_predict(self, X_valid):
        """Parallel prediction in batches"""
        n_samples = len(X_valid)
    
        batch_size = 10000
        
        batches = []
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batches.append(X_valid[i:end_idx])
        
        print(f"  Predicting {n_samples} points in {len(batches)} batches...")
        
        # Thread-based parallelism
        results = Parallel(
            n_jobs=min(23, self.n_processes),
            prefer="threads",
            verbose=23
        )(
            delayed(self._predict_batch)(batch) for batch in batches
        )
        
        return np.concatenate(results)

    def _predict_batch(self, batch):
        """Predict a single batch"""
        try:
            batch_scaled = self.scaler.transform(batch)
            return self.model.predict(batch_scaled).astype(np.float16)
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return np.full(len(batch), np.nan, dtype=np.float16)
    
    def detect_prediction_gaps(self, predictions, date):
        """Detect large latitudinal gaps"""
        if np.all(np.isnan(predictions)):
            self.gap_log.append({
                'date': date.isoformat(),
                'gap_type': 'complete_failure',
                'description': 'No predictions for entire domain'
            })
            return
        
        # Check by latitude bands
        pred_df = pd.DataFrame({
            'latitude': self.standard_ocean_lats,
            'prediction': predictions
        })
        
        pred_df['lat_band'] = np.round(pred_df['latitude']).astype(int)
        lat_coverage = pred_df.groupby('lat_band')['prediction'].apply(lambda x: (~x.isna()).any())
        
        missing_bands = sorted(lat_coverage[~lat_coverage].index.values)
        
        # Find continuous gaps >= 5 degrees
        if missing_bands:
            gaps = []
            gap_start = missing_bands[0]
            gap_end = missing_bands[0]
            
            for i in range(1, len(missing_bands)):
                if missing_bands[i] - missing_bands[i-1] <= 2:
                    gap_end = missing_bands[i]
                else:
                    if gap_end - gap_start >= 4:
                        gaps.append((gap_start, gap_end))
                    gap_start = missing_bands[i]
                    gap_end = missing_bands[i]
            
            if gap_end - gap_start >= 4:
                gaps.append((gap_start, gap_end))
            
            # Log gaps
            for gap_start, gap_end in gaps:
                self.gap_log.append({
                    'date': date.isoformat(),
                    'gap_type': 'latitudinal_gap',
                    'lat_start': int(gap_start),
                    'lat_end': int(gap_end),
                    'gap_size': int(gap_end - gap_start + 1)
                })
                print(f"Gap detected: {gap_start}°N - {gap_end}°N")
    
    def save_daily_nc(self, date, predictions):
        """
        Save to NetCDF with GEOGRAPHIC coordinates
        
        Note: Predictions are computed in projected space,
        but output uses original lat/lon grid for compatibility
        """
        # Map back to full grid
        full_grid = np.full(len(self.standard_grid_lats_flat), np.nan, dtype=np.float32)
        full_grid[self.standard_ocean_mask] = predictions.astype(np.float32)
        grid_2d = full_grid.reshape(self.standard_grid_shape)
        
        # Create dataset with GEOGRAPHIC coordinates
        ds = xr.Dataset(
            {'DMS': (['latitude', 'longitude'], grid_2d.astype(np.float32))},
            coords={
                'latitude': self.standard_lats_1d.astype(np.float32), 
                'longitude': self.standard_lons_1d.astype(np.float32)
            },
            attrs={
                'title': f'DMS prediction for {date}',
                'domain': 'North Atlantic [0-60°N, 80°W-15°E]',
                'resolution': '4km (~0.036°)',
                'projection': 'Data processed in LAEA projection, output in WGS84',
                'created': datetime.now().isoformat(),
            }
        )
        
        ds['DMS'].attrs = {'long_name': 'Dimethyl Sulphide', 'units': 'μmol m-3'}
        
        filename = self.output_dir / f"DMS_{date.strftime('%Y%m%d')}_4km_NA.nc"
        ds.to_netcdf(filename)
        print(f"Saved: {filename}")
        return filename
    
    def save_metadata(self, start_date=None, end_date=None):
        """Save gap metadata"""
        if self.gap_log:
            metadata = {
                'gap_detection': {'threshold': '5+ degrees latitude'},
                'gaps': self.gap_log,
                'created': datetime.now().isoformat(), 
                'date_range': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                }
            }
            
            if start_date and end_date:
                if start_date == end_date:
                    date_str = start_date.strftime('%Y%m%d')
                else:
                    date_str = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
            else:
                date_str = datetime.now().strftime('%Y%m%d')
            
            meta_file = self.output_dir / f'gap_metadata_{date_str}.json'
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved: {meta_file}")

    def run_prediction_for_date(self, target_date):
        """Full pipeline for a single date"""
        print(f"Starting prediction for {target_date}")
        
        result = self.predict_daily(target_date)
        
        if result is not None:
            self.save_daily_nc(target_date, result)
            print(f"Prediction for {target_date} saved successfully.")
            
            # Log valid prediction count
            valid_count = (~np.isnan(result)).sum()
            print(f"Valid predictions: {valid_count}/{len(result)}")
            
            self.print_memory_usage()
            return True
        else:
            print(f"Prediction failed for {target_date}")
            return False
        
    @staticmethod
    def print_memory_usage():
        """Print current memory usage"""
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
