# logistic_script_data.py
# Author: Adrian Caballero
# This script aggregates business birth and death data quarterly by industry to increase row count.

import pandas as pd
import os

# Load filtered dataset
df = pd.read_csv("../processed/filtered_bds_data.csv")

# Keep only relevant columns
df = df[["year", "period", "value", "dataclass_name", "industry_name"]]

# Clean invalid or missing values
df = df.replace("-", pd.NA).dropna()
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Remove rows with any missing numerics
df = df.dropna(subset=["value", "year"])

# Group by year, period (quarter), and industry
pivot = df.pivot_table(
    index=["year", "period", "industry_name"],
    columns="dataclass_name",
    values="value",
    aggfunc="sum"
).reset_index()

# Rename and clean up columns
pivot.columns.name = None
pivot = pivot.rename(columns={
    "Establishment Births": "births",
    "Establishment Deaths": "deaths"
})
pivot["births"] = pd.to_numeric(pivot["births"], errors="coerce")
pivot["deaths"] = pd.to_numeric(pivot["deaths"], errors="coerce")
pivot = pivot.dropna(subset=["births", "deaths"])

# Feature engineering
pivot["net_jobs"] = pivot["births"] - pivot["deaths"]
pivot["survival_rate"] = pivot["births"] / (pivot["births"] + pivot["deaths"])

# Save output
output_path = "../processed/aggregated_bds_data.csv"
pivot.to_csv(output_path, index=False)
print(f"✅ Aggregated data saved to: {output_path}")
