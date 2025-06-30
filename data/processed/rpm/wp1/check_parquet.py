#!/usr/bin/env python3
"""
Check parquet file structure
"""

import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

parquet_file = Path("data/processed/rpm/wp1/output_wp1/afternoon/011_Static_stbd_1/proc_IMU_Sensor_3.parquet")

# Read metadata
parquet_file_obj = pq.ParquetFile(parquet_file)
metadata = parquet_file_obj.metadata
schema = parquet_file_obj.schema_arrow

print("Parquet File Info:")
print(f"  Num row groups: {metadata.num_row_groups}")
print(f"  Num rows: {metadata.num_rows}")
print(f"  Schema: {schema}")

# Check custom metadata
if schema.metadata:
    print("\nCustom Metadata:")
    for key, value in schema.metadata.items():
        print(f"  {key.decode('utf-8')}: {value.decode('utf-8')}")

# Read sample data
df = pd.read_parquet(parquet_file)
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())