"""
Test script for DMS estimation on three specific dates with visualisation

For producing data, we use HPC with a SLURM job manager. Please refer to the bash files in the same directory for how to run the predictions on HPC. 
This script is meant for local testing and visualization of results.

predictor_projected.py contains the main logic for loading data, running predictions, and saving outputs. We use the same predictor class on HPC. 
"""

from datetime import datetime, date
import gc
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# Change working directory
os.chdir(r'C:\Data\DMS Estimation\DMS Env Data')
from predictor_projected import Efficient4kmDMSPredictor

# ========================================
# PART 1: Run predictions for test dates
# ========================================

# Initialize predictor
model_path = r"C:\Data\DMS Estimation\Models\best_stacking_model_2_elasticnet.joblib"
scaler_path = r"C:\Data\DMS Estimation\Models\standard_scaler.joblib"
pred = Efficient4kmDMSPredictor(
    model_path, 
    scaler_path, 
    output_dir='test_predictions'  # Separate directory for test outputs
)

# Test dates
test_dates = [
    date(2019, 1, 15),   # Winter
    date(2019, 7, 15),   # Summer
    date(2019, 12, 25)   # Late winter
]

print("\n" + "="*60)
print("TESTING DMS PREDICTION WITH PROJECTED COORDINATES")
print("="*60)

# Run predictions
nc_files = []
for test_date in test_dates:
    print(f"\n{'#'*60}")
    print(f"Processing: {test_date}")
    print(f"{'#'*60}\n")
    
    try:
        success = pred.run_prediction_for_date(test_date)
        if success:
            print(f"Successfully processed {test_date}")
            # Store the output filename
            nc_file = pred.output_dir / f"DMS_{test_date.strftime('%Y%m%d')}_4km_NA.nc"
        else:
            print(f"Failed to process {test_date}")
            
    except Exception as e:
        print(f"Error processing {test_date}: {e}")
        import traceback
        traceback.print_exc()
    
    gc.collect()

# Save metadata
pred.save_metadata()

print("\n" + "="*60)
print("PREDICTION COMPLETE")
print("="*60)

# ========================================
# PART 2: Visualize results
# ========================================

def plot_dms_map(nc_file, test_date, ax, vmin=None, vmax=None):
    """
    Plot DMS distribution for a single date
    
    Args:
        nc_file: Path to NetCDF file
        test_date: Date object
        ax: Matplotlib axis
        vmin, vmax: Color scale limits
    """
    # Load data
    ds = xr.open_dataset(nc_file)
    dms = ds['DMS'].values
    lons = ds['longitude'].values
    lats = ds['latitude'].values
    ds.close()
    
    # Replace NaN for better visualization
    dms_masked = np.ma.masked_invalid(dms)
    
    # Plot
    projection = ccrs.PlateCarree()
    
    # Pcolormesh
    im = ax.pcolormesh(
        lons, lats, dms_masked,
        transform=projection,
        cmap='viridis',
        vmin=vmin, vmax=vmax,
        shading='auto'
    )
    
    # Add coastlines and features
    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    
    # Grid
    gl = ax.gridlines(
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=0.3, 
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    
    # Title
    season = {1: 'Late Winter', 7: 'Summer', 12: 'Early Winter'}[test_date.month]
    ax.set_title(
        f'{test_date.strftime("%Y-%m-%d")} ({season})',
        fontsize=12,
        fontweight='bold'
    )
    
    # Calculate statistics
    valid_data = dms_masked.compressed()
    if len(valid_data) > 0:
        stats_text = f"Mean: {np.mean(valid_data):.2f}\nStd: {np.std(valid_data):.2f}\nN: {len(valid_data):,}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    return im


print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Filter valid files
valid_files = [(f, d) for f, d in zip(nc_files, test_dates) if f is not None and f.exists()]

if len(valid_files) == 0:
    print("No valid NetCDF files found for visualization!")
else:
    print(f"Found {len(valid_files)} valid files")
    
    # Determine global color scale for consistency
    all_values = []
    for nc_file, _ in valid_files:
        ds = xr.open_dataset(nc_file)
        valid_vals = ds['DMS'].values[~np.isnan(ds['DMS'].values)]
        all_values.extend(valid_vals)
        ds.close()
    
    if len(all_values) > 0:
        vmin = np.percentile(all_values, 2)   # 2nd percentile
        vmax = np.percentile(all_values, 98)  # 98th percentile
        print(f"  Color scale: {vmin:.2f} to {vmax:.2f} μmol m⁻³")
    else:
        vmin, vmax = None, None
    
    # Create figure
    n_plots = len(valid_files)
    fig = plt.figure(figsize=(18, 10 * n_plots))
    
    # Plot each date
    for idx, (nc_file, test_date) in enumerate(valid_files, 1):
        print(f"  Plotting {test_date}...")
        
        ax = fig.add_subplot(n_plots, 1, idx, projection=ccrs.PlateCarree())
        im = plot_dms_map(nc_file, test_date, ax, vmin=vmin, vmax=vmax)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Dimethyl Sulphide [μmol m⁻³]', fontsize=11, fontweight='bold')
    
    # Overall title
    fig.suptitle(
        'DMS Predictions - North Atlantic (4km resolution, PROJECTED COORDINATES)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Save figure
    output_fig = Path('test_predictions') / 'DMS_test.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(output_fig, dpi=600, bbox_inches='tight')
    print(f"\n Figure saved: {output_fig}")
    
    # Show
    plt.show()

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)