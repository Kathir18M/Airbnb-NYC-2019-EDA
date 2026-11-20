"""
Airbnb NYC 2019 - Comprehensive Exploratory Data Analysis & Production Pipeline
Author: Kathir Dhivaghar
Repository: https://github.com/Kathir18M/Airbnb-NYC-2019-EDA
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath="data/AB_NYC_2019.csv"):
    """
    Loads and validates the raw Airbnb NYC 2019 dataset.
    """
    if not os.path.exists(filepath):
        # Fallback check in parent or local directory
        alternatives = ["AB_NYC_2019.csv", "../data/AB_NYC_2019.csv"]
        found = False
        for alt in alternatives:
            if os.path.exists(alt):
                filepath = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. Please download AB_NYC_2019.csv from Kaggle "
                "and place it in the 'data/' directory."
            )

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded. Dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def investigate_data_quality(df, output_dir="outputs"):
    """
    Performs comprehensive data quality audit, missing value analysis,
    duplicate check, and anomaly detection.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("PHASE 4: DATA QUALITY INVESTIGATION")
    print("=" * 50)

    # 1. Missing Values
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing_Count": null_counts,
        "Missing_Percentage": null_pct
    })
    missing_report_path = os.path.join(output_dir, "missing_values_report.csv")
    missing_report.to_csv(missing_report_path, index_label="Column")
    print(f"[+] Missing values report saved to '{missing_report_path}'.")

    # 2. Duplicate Detection
    exact_duplicates = df.duplicated().sum()
    id_duplicates = df["id"].duplicated().sum()
    print(f"[+] Exact duplicate rows: {exact_duplicates}")
    print(f"[+] Duplicate listing IDs: {id_duplicates}")

    # 3. Price Anomalies
    zero_prices = (df["price"] <= 0).sum()
    print(f"[+] Zero or negative prices: {zero_prices}")

    # 4. Minimum Nights Anomalies
    unusual_min_nights = (df["minimum_nights"] > 365).sum()
    print(f"[+] Listings with minimum nights > 365: {unusual_min_nights}")

    # 5. Coordinate Bounds (NYC Latitude ~40.4 to 40.95, Longitude ~ -74.3 to -73.65)
    invalid_lat = (~df["latitude"].between(40.4, 41.0)).sum()
    invalid_lon = (~df["longitude"].between(-74.3, -73.6)).sum()
    print(f"[+] Listings with out-of-bounds coordinates: Latitude={invalid_lat}, Longitude={invalid_lon}")

    # 6. Categorical Validation
    print(f"[+] Neighbourhood Groups: {list(df['neighbourhood_group'].unique())}")
    print(f"[+] Room Types: {list(df['room_type'].unique())}")

    return missing_report


def clean_data(df):
    """
    Cleans the raw dataset and produces:
    - df_clean: Full dataset with imputed missing values and standardized types.
    - df_analysis: Subset refined for price and spatial analysis (price > 0, valid coords).
    """
    print("\n" + "=" * 50)
    print("PHASE 5: DATA CLEANING")
    print("=" * 50)

    # Work on a copy to preserve raw data
    df_clean = df.copy()

    # 1. Remove exact duplicate rows if any exist
    df_clean.drop_duplicates(inplace=True)

    # 2. Impute text columns
    df_clean["name"] = df_clean["name"].fillna("Unknown Listing")
    df_clean["host_name"] = df_clean["host_name"].fillna("Unknown Host")

    # 3. Impute reviews_per_month with 0 for listings with no reviews
    df_clean["reviews_per_month"] = df_clean["reviews_per_month"].fillna(0.0)

    # 4. Standardize datetime for last_review, preserving missing dates as NaT
    df_clean["last_review"] = pd.to_datetime(df_clean["last_review"], errors="coerce")

    # 5. Create df_analysis by filtering out unlisted zero/negative price records and invalid coordinates
    df_analysis = df_clean[
        (df_clean["price"] > 0) &
        (df_clean["latitude"].between(40.4, 41.0)) &
        (df_clean["longitude"].between(-74.3, -73.6))
    ].copy()

    # 6. Flag unusual minimum nights without deleting valid commercial/extended-stay listings
    df_analysis["unusual_min_nights_flag"] = df_analysis["minimum_nights"] > 365

    print(f"[+] Raw records: {len(df)}")
    print(f"[+] Cleaned master records (df_clean): {len(df_clean)}")
    print(f"[+] Validated analysis records (df_analysis): {len(df_analysis)}")

    return df_clean, df_analysis


def engineer_features(df_analysis):
    """
    Constructs business and analytical features:
    - review_year: Year of the most recent review
    - price_category: Budget, Standard, Premium, Luxury
    - long_stay_flag: True for stays requiring 30+ nights
    - host_activity: Single Listing, Multi-Listing, Commercial Operator
    """
    print("\n" + "=" * 50)
    print("PHASE 6: FEATURE ENGINEERING")
    print("=" * 50)

    df_featured = df_analysis.copy()

    # 1. Review Year
    df_featured["review_year"] = df_featured["last_review"].dt.year

    # 2. Price Category (Tiers based on NYC market distribution)
    # Budget: < $75, Standard: $75-$150, Premium: $151-$300, Luxury: > $300
    bins = [0, 75, 150, 300, np.inf]
    labels = ["Budget", "Standard", "Premium", "Luxury"]
    df_featured["price_category"] = pd.cut(df_featured["price"], bins=bins, labels=labels, right=True)

    # 3. Long Stay Flag (NYC Short-Term Rental Regulations threshold is 30 days)
    df_featured["long_stay_flag"] = df_featured["minimum_nights"] >= 30

    # 4. Host Activity Portfolio
    def classify_host(count):
        if count == 1:
            return "Single Listing"
        elif 2 <= count <= 4:
            return "Multi-Listing"
        else:
            return "Commercial Operator"

    df_featured["host_activity"] = df_featured["calculated_host_listings_count"].apply(classify_host)

    print("[+] Engineered features: 'review_year', 'price_category', 'long_stay_flag', 'host_activity'")
    return df_featured


def generate_visualizations(df_analysis, output_dir="visualizations"):
    """
    Generates all 9 high-resolution visualizations for the EDA report.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    print("\n" + "=" * 50)
    print("PHASE 7 & 8: GENERATING VISUALIZATIONS")
    print("=" * 50)

    # 1. Borough Listings Bar Chart
    plt.figure(figsize=(10, 6))
    borough_counts = df_analysis["neighbourhood_group"].value_counts()
    ax = sns.barplot(x=borough_counts.index, y=borough_counts.values, hue=borough_counts.index, palette="Blues_r", legend=False)
    plt.title("Airbnb Listings by Neighbourhood Group (NYC 2019)", fontsize=14, fontweight="bold")
    plt.xlabel("Borough", fontweight="bold")
    plt.ylabel("Number of Listings", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="baseline", fontsize=10, xytext=(0, 5), textcoords="offset points")
    plt.savefig(os.path.join(output_dir, "borough_listings.png"), dpi=300)
    plt.close()
    print("[+] Saved 'borough_listings.png'")

    # 2. Room Type Distribution
    plt.figure(figsize=(8, 8))
    room_counts = df_analysis["room_type"].value_counts()
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    plt.pie(room_counts.values, labels=room_counts.index, autopct="%1.1f%%", startangle=140,
            colors=colors, explode=[0.03, 0.03, 0.08], wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    plt.title("Distribution of Airbnb Room Types in NYC", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "room_type_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved 'room_type_distribution.png'")

    # 3. Price Distribution (Up to 99th percentile)
    price_99 = df_analysis["price"].quantile(0.99)
    price_filtered = df_analysis[df_analysis["price"] <= price_99]
    plt.figure(figsize=(11, 6))
    sns.histplot(price_filtered["price"], bins=50, kde=True, color="#2b5c8f")
    plt.title(f"Price Distribution up to 99th Percentile (${int(price_99)})", fontsize=14, fontweight="bold")
    plt.xlabel("Nightly Price (USD)", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.axvline(df_analysis["price"].median(), color="red", linestyle="--", label=f"Median (${df_analysis['price'].median():.0f})")
    plt.axvline(df_analysis["price"].mean(), color="orange", linestyle="-", label=f"Mean (${df_analysis['price'].mean():.2f})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "price_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved 'price_distribution.png'")

    # 4. Correlation Heatmap
    num_cols = [
        "price", "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365", "latitude", "longitude"
    ]
    corr = df_analysis[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print("[+] Saved 'correlation_heatmap.png'")

    # 5. Top 10 Neighbourhoods by Listing Volume
    plt.figure(figsize=(12, 6))
    top_neighbourhoods = df_analysis["neighbourhood"].value_counts().head(10)
    ax = sns.barplot(x=top_neighbourhoods.values, y=top_neighbourhoods.index, hue=top_neighbourhoods.index, palette="viridis", legend=False)
    plt.title("Top 10 Most Popular NYC Neighbourhoods by Listings", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Listings", fontweight="bold")
    plt.ylabel("Neighbourhood", fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
    plt.savefig(os.path.join(output_dir, "neighbourhood_analysis.png"), dpi=300)
    plt.close()
    print("[+] Saved 'neighbourhood_analysis.png'")

    # 6. Price by Room Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="room_type", y="price", data=price_filtered, hue="room_type", palette="Set2", legend=False, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
    plt.title("Price Comparison by Room Type (Filtered to ≤ 99th Percentile)", fontsize=14, fontweight="bold")
    plt.xlabel("Room Type", fontweight="bold")
    plt.ylabel("Nightly Price (USD)", fontweight="bold")
    plt.savefig(os.path.join(output_dir, "price_by_room_type.png"), dpi=300)
    plt.close()
    print("[+] Saved 'price_by_room_type.png'")

    # 7. Minimum Nights Distribution (Zoomed to 35 nights)
    plt.figure(figsize=(11, 5))
    min_nights_sub = df_analysis[df_analysis["minimum_nights"] <= 35]
    sns.countplot(x="minimum_nights", data=min_nights_sub, color="#4878d0")
    plt.title("Distribution of Minimum Nights (Up to 35 Nights)", fontsize=14, fontweight="bold")
    plt.xlabel("Minimum Nights Required", fontweight="bold")
    plt.ylabel("Count of Listings", fontweight="bold")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "minimum_nights.png"), dpi=300)
    plt.close()
    print("[+] Saved 'minimum_nights.png'")

    # 8. Reviews Analysis (Price vs. Number of Reviews)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="number_of_reviews", y="price", hue="room_type", data=price_filtered,
                    alpha=0.4, palette="deep")
    plt.title("Listing Price vs. Total Customer Reviews (≤ 99th Percentile)", fontsize=14, fontweight="bold")
    plt.xlabel("Total Number of Reviews", fontweight="bold")
    plt.ylabel("Nightly Price (USD)", fontweight="bold")
    plt.savefig(os.path.join(output_dir, "reviews_analysis.png"), dpi=300)
    plt.close()
    print("[+] Saved 'reviews_analysis.png'")

    # 9. Geographical Distribution
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        x="longitude", y="latitude", hue="neighbourhood_group",
        data=df_analysis, alpha=0.5, s=15, palette="tab10"
    )
    plt.title("Geographical Distribution of Airbnb Listings Across NYC Boroughs", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude", fontweight="bold")
    plt.ylabel("Latitude", fontweight="bold")
    plt.legend(title="Borough", loc="upper left")
    plt.savefig(os.path.join(output_dir, "geographical_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved 'geographical_distribution.png'")

    return corr


def compute_kpis_and_insights(df_analysis, corr_matrix, output_dir="outputs"):
    """
    Computes key performance indicators, summary tables, and exports analytical reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("PHASE 9: KPI & BUSINESS INSIGHTS DERIVATION")
    print("=" * 50)

    # 1. KPIs
    total_listings = len(df_analysis)
    unique_hosts = df_analysis["host_id"].nunique()
    avg_price = round(df_analysis["price"].mean(), 2)
    median_price = round(df_analysis["price"].median(), 2)
    avg_min_nights = round(df_analysis["minimum_nights"].mean(), 2)
    total_reviews = int(df_analysis["number_of_reviews"].sum())
    avg_availability = round(df_analysis["availability_365"].mean(), 2)
    unique_neighbourhoods = df_analysis["neighbourhood"].nunique()

    kpi_dict = {
        "Total Listings": total_listings,
        "Unique Hosts": unique_hosts,
        "Average Price (USD)": avg_price,
        "Median Price (USD)": median_price,
        "Average Minimum Nights": avg_min_nights,
        "Total Reviews": total_reviews,
        "Average Availability (Days/Year)": avg_availability,
        "Unique Neighbourhoods": unique_neighbourhoods
    }
    kpi_df = pd.DataFrame(list(kpi_dict.items()), columns=["Metric", "Value"])
    kpi_df.to_csv(os.path.join(output_dir, "airbnb_kpis.csv"), index=False)
    print("[+] Saved 'airbnb_kpis.csv'")

    # 2. Correlation Matrix
    corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    print("[+] Saved 'correlation_matrix.csv'")

    # 3. Top Hosts
    top_hosts = (
        df_analysis.groupby("host_id")
        .agg(
            host_name=("host_name", "first"),
            listing_count=("id", "nunique"),
            average_price=("price", "mean")
        )
        .sort_values(by="listing_count", ascending=False)
        .head(10)
    )
    top_hosts["average_price"] = top_hosts["average_price"].round(2)
    top_hosts.to_csv(os.path.join(output_dir, "top_hosts.csv"))
    print("[+] Saved 'top_hosts.csv'")

    # 4. Business Insights & Summary Text
    top_borough = df_analysis["neighbourhood_group"].value_counts().idxmax()
    top_room = df_analysis["room_type"].value_counts().idxmax()
    most_expensive_borough = df_analysis.groupby("neighbourhood_group")["price"].mean().idxmax()
    highest_avg_price = round(df_analysis.groupby("neighbourhood_group")["price"].mean().max(), 2)
    top_neighbourhood = df_analysis["neighbourhood"].value_counts().idxmax()
    zero_review_listings = int((df_analysis["number_of_reviews"] == 0).sum())

    summary_text = (
        "AIRBNB NYC 2019 - EXECUTIVE BUSINESS INSIGHTS\n"
        "============================================================\n"
        f"1. Total Analyzed Listings: {total_listings:,} across {unique_hosts:,} unique hosts.\n"
        f"2. Primary Market Concentration: {top_borough} leads with "
        f"{df_analysis['neighbourhood_group'].value_counts()[top_borough]:,} listings "
        f"({df_analysis['neighbourhood_group'].value_counts(normalize=True)[top_borough]*100:.1f}% share).\n"
        f"3. Dominant Room Type: {top_room} accounting for "
        f"{df_analysis['room_type'].value_counts(normalize=True)[top_room]*100:.1f}% of total supply.\n"
        f"4. Pricing Benchmark: Overall median price is ${median_price:.2f}/night. Mean price is ${avg_price:.2f}/night.\n"
        f"5. Highest Average Price Borough: {most_expensive_borough} at ${highest_avg_price:.2f}/night.\n"
        f"6. Top Neighbourhood by Volume: {top_neighbourhood} with "
        f"{df_analysis['neighbourhood'].value_counts()[top_neighbourhood]:,} active listings.\n"
        f"7. Review Engagement: {total_reviews:,} cumulative reviews recorded. "
        f"{zero_review_listings:,} listings ({zero_review_listings/total_listings*100:.1f}%) have zero reviews.\n"
        f"8. Commercial Operator Impact: Top commercial host (ID {top_hosts.index[0]} - '{top_hosts.iloc[0]['host_name']}') "
        f"operates {top_hosts.iloc[0]['listing_count']} properties with an average rate of ${top_hosts.iloc[0]['average_price']:.2f}.\n"
        "============================================================\n"
    )

    with open(os.path.join(output_dir, "project_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("[+] Saved 'project_summary.txt'")
    print("\n" + summary_text)

    return kpi_dict


def run_validation_tests(df, df_clean, df_analysis):
    """
    Automated assertions and quality assurance checks.
    """
    print("\n" + "=" * 50)
    print("PHASE 10: VALIDATION AND TESTING (ASSERTIONS)")
    print("=" * 50)

    # Test 1: No duplicate IDs in cleaned dataset
    assert df_clean["id"].nunique() == len(df_clean), "FAIL: Duplicate listing IDs found in df_clean!"
    print("[PASS] Test 1: Listing IDs are unique in df_clean.")

    # Test 2: No exact duplicate rows in cleaned dataset
    assert df_clean.duplicated().sum() == 0, "FAIL: Duplicate rows detected in df_clean!"
    print("[PASS] Test 2: Zero exact duplicate rows in df_clean.")

    # Test 3: No missing listing names or host names in cleaned dataset
    assert df_clean["name"].isnull().sum() == 0, "FAIL: Missing listing names remain!"
    assert df_clean["host_name"].isnull().sum() == 0, "FAIL: Missing host names remain!"
    print("[PASS] Test 3: Listing and host names fully imputed with 'Unknown Listing' / 'Unknown Host'.")

    # Test 4: reviews_per_month is non-null
    assert df_clean["reviews_per_month"].isnull().sum() == 0, "FAIL: Null values in reviews_per_month!"
    print("[PASS] Test 4: reviews_per_month completely non-null.")

    # Test 5: Price in df_analysis is strictly positive
    assert (df_analysis["price"] <= 0).sum() == 0, "FAIL: Non-positive prices exist in df_analysis!"
    print("[PASS] Test 5: All prices in df_analysis are strictly greater than zero.")

    # Test 6: Geographic coordinates within valid NYC boundary box
    assert df_analysis["latitude"].between(40.4, 41.0).all(), "FAIL: Latitude out of NYC bounds!"
    assert df_analysis["longitude"].between(-74.3, -73.6).all(), "FAIL: Longitude out of NYC bounds!"
    print("[PASS] Test 6: All geographic coordinates conform to valid NYC geographical bounds.")

    # Test 7: Correct row count reduction logic (exactly 11 zero-price rows excluded)
    expected_rows = len(df_clean) - (df_clean["price"] <= 0).sum()
    assert len(df_analysis) == expected_rows, f"FAIL: Expected {expected_rows} rows, got {len(df_analysis)}"
    print(f"[PASS] Test 7: Dimension assertion passed ({len(df_analysis)} validated rows).")

    # Test 8: No infinite values in numerical features
    num_cols = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"]
    assert not np.isinf(df_analysis[num_cols]).any().any(), "FAIL: Infinite values detected in analytical fields!"
    print("[PASS] Test 8: No infinite values detected in analytical fields.")

    print("\nALL 8 VALIDATION CHECKS PASSED SUCCESSFULLY (100% PASS RATE).")


def run_pipeline(data_path="data/AB_NYC_2019.csv"):
    """
    Executes the entire end-to-end analytical workflow.
    """
    df_raw = load_dataset(data_path)
    investigate_data_quality(df_raw, output_dir="outputs")
    df_clean, df_analysis = clean_data(df_raw)
    df_analysis = engineer_features(df_analysis)
    corr_matrix = generate_visualizations(df_analysis, output_dir="visualizations")
    compute_kpis_and_insights(df_analysis, corr_matrix, output_dir="outputs")
    run_validation_tests(df_raw, df_clean, df_analysis)
    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    return df_clean, df_analysis


if __name__ == "__main__":
    run_pipeline()
