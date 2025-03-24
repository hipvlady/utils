#!/usr/bin/env python3
"""
Schema and Table Comparison Tool

This script compares data between production and development schemas to validate
consistency after refactoring. It supports comparing BigQuery schemas and tables,
performing schema validation, row count checks, aggregate checks, and sample testing.

Usage:
    python automated_testing.py --config config.json --output results.log
"""

import os
import sys
import json
import random
import argparse
import logging
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BigQuery
from google.cloud import bigquery
from google.oauth2 import service_account

# Create a class to duplicate output
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush()  # Ensure output is written immediately
    
    def flush(self):
        for file in self.files:
            file.flush()


class SchemaComparisonTool:
    def __init__(self, config_path: str = None, 
                 log_file: str = None, 
                 log_level: str = "INFO",
                 time_window_type: str = "fixed",
                 days_back: int = 30):
        """
        Initialize the SchemaComparisonTool class.
        
        Args:
            config_path: Path to configuration file
            log_file: Path to log file
            log_level: Logging level
            time_window_type: Type of time window (fixed or floating)
            days_back: Number of days to look back for floating time window
        """
        # Set up logging
        self.setup_logging(log_file, log_level)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Time window settings
        self.time_window_type = time_window_type
        self.days_back = days_back
        
        # Initialize BigQuery client
        self.init_connection()
        
        # Test results storage
        self.test_results = {
            "schema_checks": [],
            "table_comparisons": []
        }

    def setup_logging(self, log_file: str, log_level: str) -> None:
        """Set up logging configuration."""
        log_level_dict = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        numeric_level = log_level_dict.get(log_level.upper(), logging.INFO)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=numeric_level, format=log_format)
        
        # Create logger
        self.logger = logging.getLogger("SchemaComparisonTool")
        
        # Add file handler if log_file is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)
            
            # Redirect stdout to log file as well
            sys.stdout = Tee(sys.stdout, open(log_file, 'a'))
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        if not config_path:
            # Default configuration
            self.logger.info("Using default configuration")
            return {
                "project_id": "okta-ga-rollup",
                "schema_comparisons": [
                    {
                        "prod_schema": "dbt_prod_ga4_reporting",
                        "dev_schema": "dev_admind_joe_ga4_reporting",
                        "tables": [
                            "ga4__sessions",
                            "ga4__pageviews",
                            "ga4__clicks",
                            "ga4__content_okta",
                            "ga4__traffic_okta",
                            "ga4__flattened_hits",
                            "ga4__traffic_with_ua_union",
                            "ga4__content_with_ua_union"
                        ]
                    },
                    {
                        "prod_schema": "dbt_prod_reporting",
                        "dev_schema": "dev_admind_joe_reporting",
                        "tables": [
                            "search_console_url",
                            "search_console_site",
                            "page_report_combined"
                        ]
                    }
                ],
                "fixed_time_windows": [
                    ["2024-02-01", "2024-02-28"],
                    ["2024-03-01", "2024-03-15"]
                ],
                "metrics_to_compare": ["sessions", "pageviews", "unique_pageviews", "clicks", "users"],
                "sample_size": 5
            }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def init_connection(self) -> None:
        """Initialize BigQuery client."""
        try:
            self.project_id = self.config["project_id"]
            
            # Get the service account path from configuration or environment variable
            service_account_path = self.config.get("service_account_path")
            
            # If service account path is specified as an environment variable reference
            if service_account_path and service_account_path.startswith("${") and service_account_path.endswith("}"):
                env_var = service_account_path[2:-1]
                service_account_path = os.environ.get(env_var)
                if not service_account_path:
                    self.logger.warning(f"Environment variable {env_var} not found. Falling back to application default credentials.")
            
            # Initialize BigQuery client with or without service account
            if service_account_path and os.path.exists(service_account_path):
                self.logger.info(f"Using service account credentials from: {service_account_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.bq_client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                # Use application default credentials
                self.logger.info("Using application default credentials")
                self.bq_client = bigquery.Client(project=self.project_id)
            
            self.timeout_seconds = 300
            self.logger.info(f"Connected to BigQuery project: {self.project_id}")
        except Exception as e:
            self.logger.error(f"Error connecting to BigQuery: {e}")
            sys.exit(1)
    
    def get_time_windows(self) -> List[Tuple[str, str]]:
        """Get time windows for testing based on the selected strategy."""
        if self.time_window_type == "floating":
            # Generate time windows based on current date - days_back
            today = datetime.now().date()
            start_date = (today - timedelta(days=self.days_back)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            self.logger.info(f"Using floating time window: {start_date} to {end_date}")
            return [(start_date, end_date)]
        else:
            # Use fixed time windows from configuration
            self.logger.info("Using fixed time windows from configuration")
            return self.config["fixed_time_windows"]
    
    def get_table_schema(self, dataset: str, table: str) -> pd.DataFrame:
        """Get table schema from BigQuery."""
        query = f"""
        SELECT column_name, data_type
        FROM `{self.project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
        """
        try:
            df = self.bq_client.query(query).to_dataframe()
            return df
        except Exception as e:
            self.logger.error(f"Error getting schema for {dataset}.{table}: {e}")
            return pd.DataFrame()
    
    def compare_schemas(self, prod_schema_df: pd.DataFrame, dev_schema_df: pd.DataFrame) -> Tuple[set, set, List[Tuple[str, str, str]]]:
        """
        Compare production and development schemas for missing, extra, or mismatched columns.
        """
        if prod_schema_df.empty or dev_schema_df.empty:
            return set(), set(), []
            
        # Convert to dict: {column_name.lower(): data_type}
        prod_cols = {
            row["column_name"].lower(): row["data_type"].lower()
            for _, row in prod_schema_df.iterrows()
        }
        dev_cols = {
            row["column_name"].lower(): row["data_type"].lower()
            for _, row in dev_schema_df.iterrows()
        }
        
        prod_set = set(prod_cols.keys())
        dev_set = set(dev_cols.keys())
        
        missing_in_dev = prod_set - dev_set
        extra_in_dev = dev_set - prod_set
        
        # For columns that exist in both, compare data types
        type_mismatches = []
        for col in (prod_set & dev_set):
            if prod_cols[col] != dev_cols[col]:
                type_mismatches.append((col, prod_cols[col], dev_cols[col]))
        
        return missing_in_dev, extra_in_dev, type_mismatches
    
    def get_row_count(self, dataset: str, table: str, start_date: str, end_date: str) -> int:
        """Get row count from a table within a date range."""
        try:
            # First check if 'date' column exists in the table
            schema_df = self.get_table_schema(dataset, table)
            date_column = None
            
            # Look for date-related columns
            potential_date_columns = ['date', 'report_date', 'event_date', 'hit_date', 'page_date']
            for col in potential_date_columns:
                if col.lower() in [c.lower() for c in schema_df['column_name'].tolist()]:
                    date_column = col
                    break
            
            if date_column:
                self.logger.info(f"  Using date column '{date_column}' for filtering {dataset}.{table}")
                query = f"""
                SELECT COUNT(*) AS row_count
                FROM `{self.project_id}.{dataset}.{table}`
                WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
                """
            else:
                # If no date column found, count all rows
                self.logger.warning(f"  No date column found in {dataset}.{table}. Counting all rows (be patient, this could take longer).")
                query = f"""
                SELECT COUNT(*) AS row_count
                FROM `{self.project_id}.{dataset}.{table}`
                """
            
            df = self.bq_client.query(query).to_dataframe()
            count = int(df["row_count"].iloc[0])
            self.logger.info(f"  Found {count:,} rows in {dataset}.{table}")
            return count
        except Exception as e:
            self.logger.error(f"Error getting row count for {dataset}.{table}: {e}")
            return 0
    
    def get_aggregate(self, dataset: str, table: str, column: str, start_date: str, end_date: str) -> float:
        """Get aggregate value from a table within a date range."""
        try:
            # First check if 'date' column exists in the table
            schema_df = self.get_table_schema(dataset, table)
            date_column = None
            
            # Look for date-related columns
            potential_date_columns = ['date', 'report_date', 'event_date', 'hit_date', 'page_date']
            for col in potential_date_columns:
                if col.lower() in [c.lower() for c in schema_df['column_name'].tolist()]:
                    date_column = col
                    break
            
            if date_column:
                query = f"""
                SELECT SUM({column}) AS total_value
                FROM `{self.project_id}.{dataset}.{table}`
                WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
                """
            else:
                # If no date column found, aggregate all rows
                self.logger.warning(f"No date column found in {dataset}.{table}. Aggregating all rows.")
                query = f"""
                SELECT SUM({column}) AS total_value
                FROM `{self.project_id}.{dataset}.{table}`
                """
            
            df = self.bq_client.query(query).to_dataframe()
            value = df["total_value"].iloc[0]
            # Convert numpy types to Python native types for JSON serialization
            if pd.isna(value):
                return 0
            elif hasattr(value, 'item'):
                return value.item()  # Convert numpy types to native Python types
            else:
                return float(value)
        except Exception as e:
            self.logger.error(f"Error getting aggregate for {dataset}.{table}.{column}: {e}")
            return 0
    
    def get_available_metrics(self, dataset: str, table: str) -> List[str]:
        """Get list of numeric columns that can be used as metrics."""
        schema_df = self.get_table_schema(dataset, table)
        
        # Filter for numeric types
        numeric_types = ['int64', 'float64', 'numeric', 'integer', 'float', 'number']
        numeric_columns = schema_df[schema_df['data_type'].str.lower().isin(numeric_types)]['column_name'].tolist()
        
        # Filter out ID columns and other non-metric columns
        excluded_terms = ['id', 'key', 'num', 'count', 'index', 'position', 'order']
        metrics = [col for col in numeric_columns if not any(term in col.lower() for term in excluded_terms)]
        
        # Add back specific count metrics that are actually valuable
        count_metrics = [col for col in numeric_columns if 'count' in col.lower()]
        return metrics + count_metrics
    
    def check_column_exists(self, dataset: str, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        schema_df = self.get_table_schema(dataset, table)
        return column.lower() in [col.lower() for col in schema_df['column_name'].tolist()]
    
    def random_sample_check(self, prod_dataset: str, dev_dataset: str, table: str, 
                           start_date: str, end_date: str, sample_size: int = 5) -> Dict:
        """
        Perform a random sample check between production and development tables.
        """
        # Get schemas to determine available columns
        prod_schema = self.get_table_schema(prod_dataset, table)
        
        if prod_schema.empty:
            return {"error": f"Could not retrieve schema for {prod_dataset}.{table}"}
        
        # Get all column names
        columns = prod_schema['column_name'].tolist()
        
        # Find date column if it exists
        date_column = None
        potential_date_columns = ['date', 'report_date', 'event_date', 'hit_date', 'page_date']
        for col in potential_date_columns:
            if col.lower() in [c.lower() for c in columns]:
                date_column = col
                break
        
        # Prepare a query to get sample data
        columns_str = ", ".join(f"`{col}`" for col in columns)
        
        if date_column:
            sample_query = f"""
            SELECT {columns_str}
            FROM `{self.project_id}.{prod_dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
            LIMIT 1000
            """
        else:
            # If no date column found, sample without date filter
            self.logger.warning(f"No date column found in {prod_dataset}.{table}. Sampling without date filter.")
            sample_query = f"""
            SELECT {columns_str}
            FROM `{self.project_id}.{prod_dataset}.{table}`
            LIMIT 1000
            """
        
        try:
            # Get sample data
            df_sample = self.bq_client.query(sample_query).to_dataframe()
            
            if df_sample.empty:
                return {"error": f"No data found in {prod_dataset}.{table} for the given date range"}
            
            # Randomly select rows
            df_random = df_sample.sample(min(sample_size, len(df_sample)))
            
            # Results
            results = []
            
            # For each sampled row, check if it exists in dev
            for _, row in df_random.iterrows():
                # Build WHERE conditions
                conditions = []
                key_columns = []
                
                # Add date condition if date column exists
                if date_column and date_column in row and not pd.isna(row[date_column]):
                    conditions.append(f"`{date_column}` = '{row[date_column]}'")
                    key_columns.append(date_column)
                
                # Add other good identifier columns
                potential_key_cols = ['page_path', 'url', 'event_name', 'content_group', 'device_category', 'country']
                for col in potential_key_cols:
                    if col in columns and col in row and not pd.isna(row.get(col)):
                        if isinstance(row[col], str):
                            conditions.append(f"`{col}` = '{row[col].replace("'", "''")}'")
                        else:
                            conditions.append(f"`{col}` = {row[col]}")
                        key_columns.append(col)
                
                # If we don't have enough conditions, add some numeric columns
                if len(conditions) < 3:
                    numeric_cols = [col for col in columns if 
                                   col not in key_columns and 
                                   col in row and
                                   isinstance(row.get(col), (int, float)) and 
                                   not pd.isna(row.get(col))]
                    
                    for col in numeric_cols[:2]:  # Add up to 2 numeric columns
                        conditions.append(f"`{col}` = {row[col]}")
                        key_columns.append(col)
                
                # If we still don't have enough conditions, skip this row
                if len(conditions) < 2:
                    results.append({
                        "matched": None,
                        "reason": "Insufficient unique identifiers in row"
                    })
                    continue
                
                # Construct and execute the query
                where_clause = " AND ".join(conditions)
                check_query = f"""
                SELECT COUNT(*) AS match_count
                FROM `{self.project_id}.{dev_dataset}.{table}`
                WHERE {where_clause}
                """
                
                try:
                    result = self.bq_client.query(check_query).to_dataframe()
                    count = result["match_count"].iloc[0]
                    
                    # Convert numpy int64 to Python int for JSON serialization
                    if hasattr(count, 'item'):
                        count = count.item()
                    
                    results.append({
                        "matched": count > 0,
                        "match_count": int(count),
                        "key_columns": key_columns,
                        "where_clause": where_clause
                    })
                    
                except Exception as e:
                    results.append({
                        "matched": None,
                        "error": str(e),
                        "where_clause": where_clause
                    })
            
            # Calculate match rate
            matches = [r for r in results if r.get("matched") is True]
            match_rate = len(matches) / len(results) if results else 0
            
            return {
                "sample_size": len(results),
                "matches": len(matches),
                "match_rate": match_rate,
                "details": results
            }
            
        except Exception as e:
            self.logger.error(f"Error in random sample check for {table}: {e}")
            return {"error": str(e)}
    
    def compare_table(self, prod_schema: str, dev_schema: str, table: str, time_windows: List[Tuple[str, str]]) -> Dict:
        """Compare a table between production and development schemas."""
        self.logger.info(f"\n--- Comparing Table: {prod_schema}.{table} vs {dev_schema}.{table} ---")
        
        # Schema comparison
        self.logger.info(f"PHASE 1: Retrieving and comparing schema definitions...")
        prod_table_schema = self.get_table_schema(prod_schema, table)
        dev_table_schema = self.get_table_schema(dev_schema, table)
        
        if prod_table_schema.empty:
            self.logger.error(f"Could not retrieve schema for {prod_schema}.{table}")
            return {
                "prod_table": f"{prod_schema}.{table}",
                "dev_table": f"{dev_schema}.{table}",
                "error": f"Could not retrieve schema for {prod_schema}.{table}"
            }
            
        if dev_table_schema.empty:
            self.logger.error(f"Could not retrieve schema for {dev_schema}.{table}")
            return {
                "prod_table": f"{prod_schema}.{table}",
                "dev_table": f"{dev_schema}.{table}",
                "error": f"Could not retrieve schema for {dev_schema}.{table}"
            }
        
        self.logger.info(f"Retrieved schemas for both tables. Production has {len(prod_table_schema)} columns, Development has {len(dev_table_schema)} columns.")
        
        missing_in_dev, extra_in_dev, type_mismatches = self.compare_schemas(prod_table_schema, dev_table_schema)
        
        schema_results = {
            "missing_in_dev": list(missing_in_dev),
            "extra_in_dev": list(extra_in_dev),
            "type_mismatches": type_mismatches
        }
        
        self.logger.info("PHASE 1 RESULTS - Schema Comparison Results:")
        self.logger.info(f"  Missing in dev: {missing_in_dev}" if missing_in_dev else "  No missing columns.")
        self.logger.info(f"  Extra in dev: {extra_in_dev}" if extra_in_dev else "  No extra columns.")
        if type_mismatches:
            for col, prod_type, dev_type in type_mismatches:
                self.logger.info(f"  Type mismatch: {col} (PROD={prod_type}, DEV={dev_type})")
        else:
            self.logger.info("  No data type mismatches.")
        
        # Get available metrics for this table
        self.logger.info(f"PHASE 2: Identifying metrics to compare...")
        metrics = [m for m in self.config.get("metrics_to_compare", []) 
                  if self.check_column_exists(prod_schema, table, m) and 
                  self.check_column_exists(dev_schema, table, m)]
        
        if not metrics:
            potential_metrics = self.get_available_metrics(prod_schema, table)
            metrics = potential_metrics[:5]  # Take up to 5 metrics
            self.logger.info(f"No specified metrics found. Using discovered metrics: {metrics}")
        else:
            self.logger.info(f"Found {len(metrics)} metrics to compare: {metrics}")
        
        # Row counts and metrics by time window
        window_results = []
        for start_date, end_date in time_windows:
            self.logger.info(f"\nPHASE 3: Comparing data for time window {start_date} to {end_date}...")
            
            try:
                self.logger.info(f"  3.1: Counting rows in both environments...")
                prod_count = self.get_row_count(prod_schema, table, start_date, end_date)
                dev_count = self.get_row_count(dev_schema, table, start_date, end_date)
                
                self.logger.info(f"  Date Range: {start_date} to {end_date}")
                self.logger.info(f"  PROD row count: {prod_count}")
                self.logger.info(f"  DEV row count: {dev_count}")
                
                row_count_diff_pct = 0
                if prod_count > 0:
                    row_count_diff_pct = abs(prod_count - dev_count) / prod_count * 100
                
                self.logger.info(f"  Row count difference: {row_count_diff_pct:.2f}%")
                
                window_result = {
                    "time_window": (start_date, end_date),
                    "prod_row_count": prod_count,
                    "dev_row_count": dev_count,
                    "row_count_diff_pct": round(row_count_diff_pct, 2),
                    "metrics": []
                }
                
                # Compare metrics
                self.logger.info(f"  3.2: Comparing {len(metrics)} metrics between environments...")
                for idx, metric in enumerate(metrics):
                    try:
                        self.logger.info(f"  Comparing metric {idx+1}/{len(metrics)}: {metric}...")
                        prod_value = self.get_aggregate(prod_schema, table, metric, start_date, end_date)
                        dev_value = self.get_aggregate(dev_schema, table, metric, start_date, end_date)
                        
                        metric_diff_pct = 0
                        if prod_value != 0:
                            metric_diff_pct = abs(prod_value - dev_value) / abs(prod_value) * 100
                        
                        self.logger.info(f"  {metric}: PROD={prod_value}, DEV={dev_value}, Diff={metric_diff_pct:.2f}%")
                        
                        window_result["metrics"].append({
                            "name": metric,
                            "prod_value": float(prod_value) if pd.notna(prod_value) else 0,
                            "dev_value": float(dev_value) if pd.notna(dev_value) else 0,
                            "diff_pct": round(metric_diff_pct, 2)
                        })
                    except Exception as e:
                        self.logger.error(f"  Error comparing metric {metric}: {e}")
                
                window_results.append(window_result)
                
            except Exception as e:
                self.logger.error(f"Error comparing time window {start_date} to {end_date}: {e}")
        
        # Random sample check for the most recent time window
        sample_check_result = None
        if time_windows:
            recent_window = time_windows[0]  # Use the first time window
            self.logger.info(f"\nPHASE 4: Performing random sample checks...")
            self.logger.info(f"  Taking {self.config.get('sample_size', 5)} random samples from production and checking if they exist in development...")
            
            sample_check_result = self.random_sample_check(
                prod_dataset=prod_schema,
                dev_dataset=dev_schema,
                table=table,
                start_date=recent_window[0],
                end_date=recent_window[1],
                sample_size=self.config.get("sample_size", 5)
            )
            
            if "error" in sample_check_result:
                self.logger.error(f"Sample check error: {sample_check_result['error']}")
            else:
                self.logger.info("\nPHASE 4 RESULTS - Random Sample Check Results:")
                self.logger.info(f"  Match rate: {sample_check_result['match_rate'] * 100:.1f}% ({sample_check_result['matches']} of {sample_check_result['sample_size']} rows matched)")
                if sample_check_result['match_rate'] < 1.0:
                    self.logger.warning("  ⚠️ Some sample rows from production were not found in development")
                else:
                    self.logger.info("  ✓ All sample rows from production were found in development")
        
        self.logger.info(f"\nTesting completed for {prod_schema}.{table} vs {dev_schema}.{table}\n")
        
        return {
            "prod_table": f"{prod_schema}.{table}",
            "dev_table": f"{dev_schema}.{table}",
            "schema_comparison": schema_results,
            "time_windows": window_results,
            "sample_check": sample_check_result
        }
    
    def run_comparison(self) -> Dict:
        """Run comparisons for all configured schema pairs and tables."""
        self.logger.info("\n\n=============================================")
        self.logger.info("STARTING SCHEMA AND TABLE COMPARISON PROCESS")
        self.logger.info("=============================================\n")
        
        results = []
        time_windows = self.get_time_windows()
        
        self.logger.info(f"Testing will use the following time windows:")
        for idx, (start_date, end_date) in enumerate(time_windows):
            self.logger.info(f"  Window {idx+1}: {start_date} to {end_date}")
        
        total_schema_pairs = len(self.config["schema_comparisons"])
        for schema_idx, schema_pair in enumerate(self.config["schema_comparisons"]):
            prod_schema = schema_pair["prod_schema"]
            dev_schema = schema_pair["dev_schema"]
            tables = schema_pair["tables"]
            
            self.logger.info(f"\n\n=== Comparing Schema Pair {schema_idx+1}/{total_schema_pairs}: {prod_schema} vs {dev_schema} ===\n")
            
            total_tables = len(tables)
            for table_idx, table in enumerate(tables):
                try:
                    self.logger.info(f"Starting table comparison {table_idx+1}/{total_tables}: {table}")
                    table_result = self.compare_table(prod_schema, dev_schema, table, time_windows)
                    results.append(table_result)
                except Exception as e:
                    self.logger.error(f"Error comparing table {table}: {e}")
                    results.append({
                        "prod_table": f"{prod_schema}.{table}",
                        "dev_table": f"{dev_schema}.{table}",
                        "error": str(e)
                    })
        
        self.logger.info("\n\n=============================================")
        self.logger.info("COMPARISON PROCESS COMPLETED")
        self.logger.info(f"Total schema pairs tested: {total_schema_pairs}")
        self.logger.info(f"Total tables tested: {len(results)}")
        self.logger.info("=============================================\n")
        
        return {
            "comparison_time": datetime.now().isoformat(),
            "time_window_type": self.time_window_type,
            "time_windows": time_windows,
            "results": results
        }
    
    def save_results(self, results: Dict, output_file: str) -> None:
        """Save comparison results to a JSON file."""
        try:
            # Convert any NumPy or pandas types to native Python types for JSON serialization
            def json_serialize(obj):
                if isinstance(obj, dict):
                    return {k: json_serialize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [json_serialize(i) for i in obj]
                elif hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            self.logger.info("\nSaving comparison results to JSON file...")
            serializable_results = json_serialize(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            self.logger.info(f"Comparison results successfully saved to {output_file}")
            
            # Calculate and log some summary statistics
            tables_count = len(results.get("results", []))
            
            # Count schemas with issues
            schemas_with_column_mismatches = set()
            schemas_with_type_mismatches = set()
            schemas_with_low_match_rate = set()
            
            for table_result in results.get("results", []):
                prod_table = table_result.get("prod_table", "")
                schema = prod_table.split(".")[0] if "." in prod_table else ""
                
                # Check for schema issues
                schema_comparison = table_result.get("schema_comparison", {})
                if schema_comparison.get("missing_in_dev") or schema_comparison.get("extra_in_dev"):
                    schemas_with_column_mismatches.add(schema)
                if schema_comparison.get("type_mismatches"):
                    schemas_with_type_mismatches.add(schema)
                
                # Check for data issues
                sample_check = table_result.get("sample_check", {})
                match_rate = sample_check.get("match_rate", 1.0)
                if match_rate < 0.9:  # Less than 90% match
                    schemas_with_low_match_rate.add(schema)
            
            self.logger.info("\nSUMMARY STATISTICS:")
            self.logger.info(f"- Total tables compared: {tables_count}")
            self.logger.info(f"- Schemas with column mismatches: {len(schemas_with_column_mismatches)}")
            self.logger.info(f"- Schemas with data type mismatches: {len(schemas_with_type_mismatches)}")
            self.logger.info(f"- Schemas with low sample match rates: {len(schemas_with_low_match_rate)}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.logger.error("The comparison was completed, but results could not be saved to file.")
    
    def close_connection(self) -> None:
        """Close BigQuery client."""
        try:
            if hasattr(self, 'bq_client') and self.bq_client:
                self.bq_client.close()
                self.logger.info("Closed BigQuery connection")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Schema and Table Comparison Tool')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='schema_comparison_results.json',
                        help='Output file for comparison results')
    parser.add_argument('--log', type=str, default='schema_comparison.log',
                        help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--time-window', type=str, default='fixed',
                        choices=['fixed', 'floating'],
                        help='Time window type (fixed or floating)')
    parser.add_argument('--days-back', type=int, default=30,
                        help='Number of days to look back for floating time window')
    return parser.parse_args()


def main():
    """Main entry point."""
    start_time = datetime.now()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create SchemaComparisonTool instance
    comparison_tool = SchemaComparisonTool(
        config_path=args.config,
        log_file=args.log,
        log_level=args.log_level,
        time_window_type=args.time_window,
        days_back=args.days_back
    )
    
    try:
        comparison_tool.logger.info("======================================================")
        comparison_tool.logger.info("PROD VS DEV SCHEMA COMPARISON TOOL")
        comparison_tool.logger.info("======================================================")
        comparison_tool.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_tool.logger.info(f"Comparing schemas using {args.time_window} time window")
        comparison_tool.logger.info(f"Output file: {args.output}")
        comparison_tool.logger.info(f"Log file: {args.log if args.log else 'console only'}")
        comparison_tool.logger.info("======================================================\n")
        
        # Run comparison
        results = comparison_tool.run_comparison()
        
        # Save results
        comparison_tool.save_results(results, args.output)
        
        # Close connection
        comparison_tool.close_connection()
        
        # Calculate and display execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        comparison_tool.logger.info("\n======================================================")
        comparison_tool.logger.info(f"COMPARISON COMPLETED SUCCESSFULLY")
        comparison_tool.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_tool.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_tool.logger.info(f"Total execution time: {execution_time}")
        comparison_tool.logger.info("======================================================")
        
        # Exit with success
        sys.exit(0)
    except Exception as e:
        comparison_tool.logger.critical(f"CRITICAL ERROR: {e}")
        comparison_tool.logger.critical("The comparison process was terminated due to an error.")
        comparison_tool.close_connection()
        
        # Calculate and display execution time even on failure
        end_time = datetime.now()
        execution_time = end_time - start_time
        comparison_tool.logger.critical(f"Execution time before failure: {execution_time}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()