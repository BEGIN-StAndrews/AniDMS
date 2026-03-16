'''
This script reads the original parquet files for each dataset, filters out rows with latitude > 60, and writes the filtered data back to new parquet files in a new folder. 
It also generates head/tail and summary CSVs for the first and last parquet files of each dataset for quick inspection.

After filering high-lat, and before producing DMS estimation, please remember to go back and test the model performance on clipped data, and compare with the original performance. 
This is to check if the generalisation performance is significantly affected by the clipping, and to ensure that the model can still capture the patterns in the data after clipping.'''

import os
import gc
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------
# paths and settings
# ---------------------------------------
base_dir = Path(r"C:\Your Path")
output_base = Path(r"C:\Your Renewed Path")
# skip_folder = "PAR_complementary"

output_base.mkdir(exist_ok=True)


# ---------------------------------------
# read / write utils
# ---------------------------------------
def head_tail(df: dd.DataFrame, n=20):
    return df.head(n), df.tail(n)

def write_single_parquet(df_pandas, out_fp):
    """
    use pyarrow to write a single parquet file from pandas DataFrame.
     - df_pandas: pandas DataFrame to write
    """
    table = pa.Table.from_pandas(df_pandas, preserve_index=False)
    pq.write_table(table, out_fp)

# ---------------------------------------
# main
# ---------------------------------------
for folder in base_dir.iterdir():

    if not folder.is_dir():
        continue
    if folder.name == skip_folder:
        continue

    print(f"Processing folder: {folder.name}")

    # find all parquet files in the folder
    parquet_files = sorted(folder.glob("*.parquet"))
    if len(parquet_files) == 0:
        print(f"  No parquet files in {folder}")
        continue

    # create output folder for this dataset
    out_folder = output_base / folder.name
    out_folder.mkdir(exist_ok=True)

    # -----------------------------
    # loop through each parquet file, filter, and write out
    # -----------------------------
    for fp in parquet_files:
        print(f"  Filtering {fp.name}")

        df = dd.read_parquet(fp)

        # filter out rows with latitude > 60 
        df_filt = df[df["latitude"] <= 60]

        # compute into pandas DataFrame
        df_pd = df_filt.compute()

        out_fp = out_folder / fp.name
        write_single_parquet(df_pd, out_fp)

        del df, df_filt, df_pd
        gc.collect()

    # -----------------------------
    # do head/tail and summary for the first and last parquet files, and save as CSV
    # -----------------------------
    first_fp = parquet_files[0]
    last_fp = parquet_files[-1]

    print(f"  Generating summary CSVs for: {first_fp.name} and {last_fp.name}")

    # read first parquet
    df_first = dd.read_parquet(first_fp)
    head_first, tail_first = head_tail(df_first, n=20)

    # save head/tail as CSV
    head_first.to_csv(out_folder / f"{folder.name}_first_head20.csv", index=False)
    tail_first.to_csv(out_folder / f"{folder.name}_first_tail20.csv", index=False)

    # summary（Dask describe into pandas）
    summary_first = df_first.describe().compute()
    summary_first.to_csv(out_folder / f"{folder.name}_first_summary.csv")

    del df_first
    gc.collect()

    # read last parquet
    df_last = dd.read_parquet(last_fp)
    head_last, tail_last = head_tail(df_last, n=20)

    head_last.to_csv(out_folder / f"{folder.name}_last_head20.csv", index=False)
    tail_last.to_csv(out_folder / f"{folder.name}_last_tail20.csv", index=False)

    summary_last = df_last.describe().compute()
    summary_last.to_csv(out_folder / f"{folder.name}_last_summary.csv")
    
    del df_last
    gc.collect()

print("Done.")
