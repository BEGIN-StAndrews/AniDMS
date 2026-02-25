"""Example usage for AniDMS."""

from __future__ import annotations

import pandas as pd

from anidms import AniDMS


if __name__ == "__main__":
    # Example 1: Zenodo mode (default) - download monthly files by date range.
    db = AniDMS(
        zenodo_doi="10.5281/zenodo.18615736",
        cache_dir="./dms_cache",
        request_timeout=60,
    )

    monthly_files = db.query_date_range(
        start_date="2010-01-05",
        end_date="2010-03-10",
        output_dir="./dms_downloads",
    )
    print("Downloaded month files:")
    for fp in monthly_files:
        print(f"  - {fp}")

    # Example 2: Annotate a tracking DataFrame directly.
    tracking_df = pd.DataFrame(
        {
            "DateTime": [
                "2010-01-07 06:00:00",
                "2010-01-07 12:00:00",
                "2010-02-18 18:00:00",
            ],
            "Latitude": [42.1, 42.5, 41.8],
            "Longitude": [-60.2, -59.8, -61.0],
            "BirdID": ["BirdA", "BirdA", "BirdB"],
        }
    )

    annotated_df = db.annotate_tracking_data(
        input_data=tracking_df,
        output_path="./annotated_tracking.csv",
    )

    print("\nAnnotated preview:")
    print(annotated_df.head())
