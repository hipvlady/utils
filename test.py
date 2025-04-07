import pyarrow.parquet as pq

file_path = "/Users/vlad/projects/utils/test_data/to_snowflake_TEST_ga4_hits_20250331_incremental_batch_id=3c1ca5ca-d6cb-4530-b82b-66d4ece14580_000000000000.parquet"


parquet_schema = pq.read_schema(file_path)
print("Schema information:")
print(parquet_schema)