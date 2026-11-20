# Dataset Documentation: New York City Airbnb Open Data (2019)

## Overview
This directory is designated for the Airbnb New York City 2019 dataset. Due to file size constraints and GitHub best practices, the raw dataset file (`AB_NYC_2019.csv`) is excluded from version control via `.gitignore`.

## Dataset Source
- **Platform:** Kaggle
- **Dataset:** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **File Name:** `AB_NYC_2019.csv`
- **File Size:** ~7.08 MB
- **Dimensions:** 48,895 rows × 16 columns

## How to Obtain and Set Up
1. Visit the [Kaggle Dataset Link](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).
2. Download `AB_NYC_2019.csv`.
3. Place `AB_NYC_2019.csv` directly into this `data/` folder:
   ```text
   data/
   ├── AB_NYC_2019.csv
   └── README.md
   ```

## Schema & Column Dictionary

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `int64` | Unique listing identifier |
| `name` | `object` | Title / name of the listing |
| `host_id` | `int64` | Unique host identifier |
| `host_name` | `object` | Name of the host |
| `neighbourhood_group` | `object` | NYC Borough (`Manhattan`, `Brooklyn`, `Queens`, `Bronx`, `Staten Island`) |
| `neighbourhood` | `object` | Specific neighborhood name (221 unique) |
| `latitude` | `float64` | GPS Latitude coordinates |
| `longitude` | `float64` | GPS Longitude coordinates |
| `room_type` | `object` | `Entire home/apt`, `Private room`, `Shared room` |
| `price` | `int64` | Nightly rental price in USD (\$) |
| `minimum_nights` | `int64` | Minimum reservation duration in nights |
| `number_of_reviews` | `int64` | Total reviews received |
| `last_review` | `object` | Date of the latest review (`YYYY-MM-DD`) |
| `reviews_per_month` | `float64` | Average reviews generated per month |
| `calculated_host_listings_count` | `int64` | Total listings managed by this host |
| `availability_365` | `int64` | Number of available booking days per year |

## Processed & Cleaned Data Outputs
When running `src/analysis.py` or `notebooks/Airbnb_NYC_2019_EDA.ipynb`, data quality investigation and validation scripts generate the analytical outputs saved under the `outputs/` and `visualizations/` directories.
