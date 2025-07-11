# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:03:33 2025

@author: User
"""

import pandas as pd

# File paths
csv_file1 = r"C:\Users\User\Documents\DELETE\20240110 pls last time\timestamps_output.csv"
csv_file2 = r"C:\Users\User\Documents\timestamps_output.csv"
output_csv = r"C:\Users\User\Documents\merged_timestamps.csv"

# Read and merge CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save merged CSV
merged_df.to_csv(output_csv, index=False)
print(f"Merged CSV saved to {output_csv}")

import json

# File paths
json_file1 = r"C:\Users\User\Documents\DELETE\20240110 pls last time\timestamps_output.json"
json_file2 = r"C:\Users\User\Documents\timestamps_output.json"
output_json = r"C:\Users\User\Documents\merged_timestamps.json"

# Read and merge JSON files
with open(json_file1, 'r') as f1, open(json_file2, 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)
    
    # Handle different JSON structures:
    if isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2
    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = {**data1, **data2}  # For dictionaries, this may overwrite duplicate keys
    else:
        merged_data = [data1, data2]  # Mixed types case

# Save merged JSON
with open(output_json, 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)
print(f"Merged JSON saved to {output_json}")