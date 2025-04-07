"""
Core functionality for schema comparison.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime, timedelta

# Internal imports
from . import utils
from . import query_builder
from . import analysis

# BigQuery
from google.cloud import bigquery
from google.oauth2 import service_account


class SchemaComparisonTool:
    """
    Tool for comparing schemas and data between production and development tables.
    """
    
    def __init__(self, config_path: str = None, 
                 log_file: str = None, 
                 log_level: str = "INFO",
                 time_window_type: str = "fixed",
                 days_back: int = 30,
                 specific_date: str = None,
                 override_sample_size: int = None):
        """
        Initialize the SchemaComparisonTool class.
        
        Args:
            config_path: Path to configuration file
            log_file: Path to log file
            log_level: Logging level
            time_window_type: Type of time window (fixed, floating, or specific_day)
            days_back: Number of days to look back for floating time window
            specific_date: Specific date to test (for time_window_type='specific_day')
            override_sample_size: Override sample size for random sample checks
        """
        # Set up logging
        self.setup_logging(log_file, log_level)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Time window settings
        self.time_window_type = time_window_type
        self.days_back = days_back
        self.specific_date = specific_date
        
        # Sample size override
        self.override_sample_size = override_sample_size
        
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
            sys.stdout = utils.Tee(sys.stdout, open(log_file, 'a'))
    
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
                "dimensions_to_compare": ["device_category", "country", "source_medium", "platform", "channel"],
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
        if self.time_window_type == "floating" or self.time_window_type == "specific_day":
            return utils.get_formatted_date_range(
                self.time_window_type, 
                self.days_back, 
                self.specific_date
            )
        else:
            # Use fixed time windows from configuration
            self.logger.info("Using fixed time windows from configuration")
            return self.config["fixed_time_windows"]
    
    def get_table_schema(self, dataset: str, table: str) -> pd.DataFrame:
        """Get table schema from BigQuery."""
        query = query_builder.build_schema_query(self.project_id, dataset, table)
        try:
            df = self.bq_client.query(query).to_dataframe()
            return df
        except Exception as e:
            self.logger.error(f"Error getting schema for {dataset}.{table}: {e}")
            return pd.DataFrame()
    
    def check_column_exists(self, dataset: str, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        schema_df = self.get_table_schema(dataset, table)
        return column.lower() in [col.lower() for col in schema_df['column_name'].tolist()]
    
    def get_row_count(self, dataset: str, table: str, start_date: str, end_date: str) -> int:
        """Get row count from a table within a date range."""
        try:
            # First check if 'date' column exists in the table
            schema_df = self.get_table_schema(dataset, table)
            date_column = analysis.find_date_column(schema_df, f"{dataset}.{table}")
            
            if date_column:
                # Special handling for shard column
                if date_column.lower() == 'shard':
                    self.logger.info(f"  Using shard column for filtering {dataset}.{table}")
                else:
                    self.logger.info(f"  Using date column '{date_column}' for filtering {dataset}.{table}")
            else:
                self.logger.warning(f"  No date column found in {dataset}.{table}. Counting all rows (be patient, this could take longer).")
            
            # Build the appropriate query
            query = query_builder.build_row_count_query(
                self.project_id, dataset, table, 
                date_column, start_date, end_date
            )
            
            df = self.bq_client.query(query).to_dataframe()
            count = int(df["row_count"].iloc[0])
            self.logger.info(f"  Found {count:,} rows in {dataset}.{table}")
            return count
        except Exception as e:
            self.logger.error(f"Error getting row count for {dataset}.{table}: {e}")
            return 0
    
    def get_aggregate(self, dataset: str, table: str, column: str, 
                      start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get aggregate value from a table within a date range, using appropriate aggregation 
        based on data type.
        
        Args:
            dataset: BigQuery dataset name
            table: BigQuery table name
            column: Column to aggregate
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with aggregation results
        """
        try:
            # Get table schema to determine column data type
            schema_df = self.get_table_schema(dataset, table)
            date_column = analysis.find_date_column(schema_df, f"{dataset}.{table}")
            
            # Determine column type and appropriate aggregation
            column_type = analysis.determine_column_type(schema_df, column)
            agg_function, description = analysis.get_appropriate_aggregation(column_type)
            
            self.logger.info(f"  Column '{column}' identified as {column_type} type, using {agg_function} aggregation")
            
            # Build the appropriate query
            query = query_builder.build_aggregate_query(
                self.project_id, dataset, table, column,
                date_column, start_date, end_date,
                agg_function
            )
            
            df = self.bq_client.query(query).to_dataframe()
            value = df["total_value"].iloc[0]
            
            # Convert numpy types to Python native types for JSON serialization
            if pd.isna(value):
                final_value = 0
            elif hasattr(value, 'item'):
                final_value = value.item()  # Convert numpy types to native Python types
            else:
                final_value = float(value) if column_type == 'numeric' else int(value)
            
            return {
                "value": final_value, 
                "type": column_type,
                "aggregation": agg_function,
                "description": description
            }
        except Exception as e:
            self.logger.error(f"Error getting aggregate for {dataset}.{table}.{column}: {e}")
            return {
                "value": 0, 
                "type": "unknown", 
                "aggregation": "none",
                "description": "error",
                "error": str(e)
            }
    
    def get_distribution(self, dataset: str, table: str, column: str, 
                         start_date: str, end_date: str, 
                         top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Get value distribution for a categorical column.
        
        Args:
            dataset: BigQuery dataset name
            table: BigQuery table name
            column: Column to analyze distribution
            start_date: Start date for filtering
            end_date: End date for filtering
            top_n: Number of top values to return
            
        Returns:
            DataFrame with distribution or None on error
        """
        try:
            # Get table schema to find the date column (if any)
            schema_df = self.get_table_schema(dataset, table)
            date_column = analysis.find_date_column(schema_df, f"{dataset}.{table}")
            
            # Build the query
            query = query_builder.build_distribution_query(
                self.project_id, dataset, table, column,
                date_column, start_date, end_date,
                top_n
            )
            
            # Execute the query
            df = self.bq_client.query(query).to_dataframe()
            return df
        except Exception as e:
            self.logger.error(f"Error getting distribution for {dataset}.{table}.{column}: {e}")
            return None
    
    def get_cardinality(self, dataset: str, table: str, column: str, 
                        start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get cardinality statistics for a column.
        
        Args:
            dataset: BigQuery dataset name
            table: BigQuery table name
            column: Column to check cardinality
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with cardinality statistics
        """
        try:
            # Get table schema to find the date column (if any)
            schema_df = self.get_table_schema(dataset, table)
            date_column = analysis.find_date_column(schema_df, f"{dataset}.{table}")
            
            # Build the query
            query = query_builder.build_cardinality_query(
                self.project_id, dataset, table, column,
                date_column, start_date, end_date
            )
            
            # Execute the query
            df = self.bq_client.query(query).to_dataframe()
            
            # Extract results
            distinct_count = int(df["distinct_count"].iloc[0])
            total_count = int(df["total_count"].iloc[0])
            cardinality_ratio = float(df["cardinality_ratio"].iloc[0])
            
            return {
                "distinct_count": distinct_count,
                "total_count": total_count,
                "cardinality_ratio": cardinality_ratio
            }
        except Exception as e:
            self.logger.error(f"Error getting cardinality for {dataset}.{table}.{column}: {e}")
            return {
                "distinct_count": 0,
                "total_count": 0,
                "cardinality_ratio": 0,
                "error": str(e)
            }
    
    def get_available_metrics(self, dataset: str, table: str) -> List[str]:
        """Get list of numeric columns that can be used as metrics."""
        schema_df = self.get_table_schema(dataset, table)
        return analysis.get_available_metrics(schema_df)
    
    def get_available_dimensions(self, dataset: str, table: str) -> List[str]:
        """Get list of categorical columns that can be used as dimensions."""
        schema_df = self.get_table_schema(dataset, table)
        return analysis.get_available_dimensions(schema_df)
    
    def random_sample_check(self, prod_dataset: str, dev_dataset: str, table: str, 
                           start_date: str, end_date: str, sample_size: int = 5) -> Dict:
        """
        Perform a random sample check between production and development tables.
        """
        # Use override sample size if provided
        if self.override_sample_size:
            sample_size = self.override_sample_size
            self.logger.info(f"Using override sample size: {sample_size}")
        
        # Get schemas to determine available columns
        prod_schema = self.get_table_schema(prod_dataset, table)
        
        if prod_schema.empty:
            return {"error": f"Could not retrieve schema for {prod_dataset}.{table}"}
        
        # Get all column names
        columns = prod_schema['column_name'].tolist()
        
        # Find date column if it exists
        date_column = analysis.find_date_column(prod_schema, f"{prod_dataset}.{table}")
        
        # Prepare a query to get sample data
        sample_query = query_builder.build_sample_query(
            self.project_id, prod_dataset, table, columns,
            date_column, start_date, end_date, 1000  # Get 1000 rows to sample from
        )
        
        try:
            # Get sample data
            df_sample = self.bq_client.query(sample_query).to_dataframe()
            
            if df_sample.empty:
                return {"error": f"No data found in {prod_dataset}.{table} for the given date range"}
            
            # Randomly select rows
            df_random = df_sample.sample(min(sample_size, len(df_sample)))
            
            self.logger.info(f"Selected {len(df_random)} random rows for comparison testing")
            
            # Results
            results = []
            
            # For each sampled row, check if it exists in dev
            for _, row in df_random.iterrows():
                # Build WHERE conditions
                conditions, key_columns = analysis.build_row_conditions(row, columns, date_column)
                
                # If we don't have enough conditions, skip this row
                if len(conditions) < 2:
                    results.append({
                        "matched": None,
                        "reason": "Insufficient unique identifiers in row"
                    })
                    continue
                
                # Construct and execute the query
                check_query = query_builder.build_match_query(
                    self.project_id, dev_dataset, table, conditions
                )
                
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
                        "where_clause": " AND ".join(conditions)
                    })
                    
                except Exception as e:
                    results.append({
                        "matched": None,
                        "error": str(e),
                        "where_clause": " AND ".join(conditions)
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
    
    def compare_dimension(self, prod_schema: str, dev_schema: str, table: str, 
                         dimension: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Compare distribution of a dimension between prod and dev.
        
        Args:
            prod_schema: Production schema name
            dev_schema: Development schema name
            table: Table name
            dimension: Dimension column to compare
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"  Comparing dimension distribution: {dimension}")
            
            # Get distributions
            prod_dist = self.get_distribution(prod_schema, table, dimension, start_date, end_date)
            dev_dist = self.get_distribution(dev_schema, table, dimension, start_date, end_date)
            
            if prod_dist is None or dev_dist is None:
                return {
                    "dimension": dimension,
                    "error": "Failed to get distribution from one or both environments"
                }
            
            # Compare distributions
            comparison = analysis.compare_distributions(prod_dist, dev_dist)
            
            # Get cardinality stats
            prod_cardinality = self.get_cardinality(prod_schema, table, dimension, start_date, end_date)
            dev_cardinality = self.get_cardinality(dev_schema, table, dimension, start_date, end_date)
            
            # Combine results
            result = {
                "dimension": dimension,
                "overlap_rate": comparison["overlap_rate"],
                "prod_cardinality": prod_cardinality.get("distinct_count", 0),
                "dev_cardinality": dev_cardinality.get("distinct_count", 0),
                "cardinality_diff_pct": utils.calculate_diff_percentage(
                    prod_cardinality.get("distinct_count", 0), 
                    dev_cardinality.get("distinct_count", 0)
                ),
                "top_diffs": comparison["percentage_diffs"],
                "missing_values": {
                    "in_dev": comparison["missing_in_dev"][:5] if len(comparison["missing_in_dev"]) > 5 else comparison["missing_in_dev"],
                    "in_prod": comparison["missing_in_prod"][:5] if len(comparison["missing_in_prod"]) > 5 else comparison["missing_in_prod"]
                }
            }
            
            # Log results
            self.logger.info(f"    Overlap rate: {result['overlap_rate'] * 100:.1f}%")
            self.logger.info(f"    Cardinality: PROD={result['prod_cardinality']}, DEV={result['dev_cardinality']}, Diff={result['cardinality_diff_pct']:.1f}%")
            if result['top_diffs']:
                self.logger.info(f"    Largest distribution differences:")
                for diff in result['top_diffs'][:3]:
                    self.logger.info(f"      '{diff['value']}': PROD={diff['prod_pct']:.1f}%, DEV={diff['dev_pct']:.1f}%, Diff={diff['diff']:.1f}%")
            
            return result
        except Exception as e:
            self.logger.error(f"Error comparing dimension {dimension}: {e}")
            return {
                "dimension": dimension,
                "error": str(e)
            }
    
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
        
        missing_in_dev, extra_in_dev, type_mismatches = analysis.compare_schemas(prod_table_schema, dev_table_schema)
        
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
        
        # Identify metrics and dimensions to compare
        self.logger.info(f"PHASE 2: Identifying metrics and dimensions to compare...")
        
        # For metrics - first check the metrics_to_compare from config, then discover additional metrics
        configured_metrics = self.config.get("metrics_to_compare", [])
        metrics = [m for m in configured_metrics 
                  if self.check_column_exists(prod_schema, table, m) and 
                  self.check_column_exists(dev_schema, table, m)]
        
        if not metrics:
            self.logger.info(f"No configured metrics found. Discovering numeric metrics...")
            potential_metrics = self.get_available_metrics(prod_schema, table)
            metrics = potential_metrics[:5]  # Take up to 5 metrics
        
        # Then check schema to filter metrics to only numeric columns
        numeric_metrics = []
        for metric in metrics:
            column_type = analysis.determine_column_type(prod_table_schema, metric)
            if column_type == 'numeric':
                numeric_metrics.append(metric)
            else:
                self.logger.info(f"  Skipping non-numeric metric: {metric} (type: {column_type})")
        
        self.logger.info(f"Found {len(numeric_metrics)} numeric metrics to compare: {numeric_metrics}")
        
        # For dimensions - first check dimensions_to_compare from config, then discover
        configured_dimensions = self.config.get("dimensions_to_compare", [])
        if not configured_dimensions and "metrics_to_compare" in self.config:
            # For backwards compatibility, check if metrics_to_compare contains dimensions
            configured_dimensions = [m for m in self.config["metrics_to_compare"] 
                                   if analysis.determine_column_type(prod_table_schema, m) == 'categorical']
        
        dimensions = [d for d in configured_dimensions
                     if self.check_column_exists(prod_schema, table, d) and 
                     self.check_column_exists(dev_schema, table, d)]
        
        if not dimensions:
            self.logger.info(f"No configured dimensions found. Discovering categorical dimensions...")
            potential_dimensions = self.get_available_dimensions(prod_schema, table)
            dimensions = potential_dimensions[:5]  # Take up to 5 dimensions
        
        self.logger.info(f"Found {len(dimensions)} dimensions to compare: {dimensions}")
        
        # Row counts and metrics by time window
        window_results = []
        dimension_comparisons = []
        
        for start_date, end_date in time_windows:
            self.logger.info(f"\nPHASE 3: Comparing data for time window {start_date} to {end_date}...")
            
            try:
                self.logger.info(f"  3.1: Counting rows in both environments...")
                prod_count = self.get_row_count(prod_schema, table, start_date, end_date)
                dev_count = self.get_row_count(dev_schema, table, start_date, end_date)
                
                self.logger.info(f"  Date Range: {start_date} to {end_date}")
                self.logger.info(f"  PROD row count: {prod_count}")
                self.logger.info(f"  DEV row count: {dev_count}")
                
                row_count_diff_pct = utils.calculate_diff_percentage(prod_count, dev_count)
                
                self.logger.info(f"  Row count difference: {row_count_diff_pct:.2f}%")
                
                window_result = {
                    "time_window": (start_date, end_date),
                    "prod_row_count": prod_count,
                    "dev_row_count": dev_count,
                    "row_count_diff_pct": round(row_count_diff_pct, 2),
                    "metrics": []
                }
                
                # Compare metrics
                self.logger.info(f"  3.2: Comparing {len(numeric_metrics)} metrics between environments...")
                for idx, metric in enumerate(numeric_metrics):
                    try:
                        self.logger.info(f"  Comparing metric {idx+1}/{len(numeric_metrics)}: {metric}...")
                        prod_result = self.get_aggregate(prod_schema, table, metric, start_date, end_date)
                        dev_result = self.get_aggregate(dev_schema, table, metric, start_date, end_date)
                        
                        prod_value = prod_result["value"]
                        dev_value = dev_result["value"]
                        
                        metric_diff_pct = utils.calculate_diff_percentage(prod_value, dev_value)
                        
                        self.logger.info(f"  {metric} ({prod_result['description']}): PROD={prod_value}, DEV={dev_value}, Diff={metric_diff_pct:.2f}%")
                        
                        window_result["metrics"].append({
                            "name": metric,
                            "type": prod_result["type"],
                            "aggregation": prod_result["aggregation"],
                            "prod_value": prod_value,
                            "dev_value": dev_value,
                            "diff_pct": round(metric_diff_pct, 2)
                        })
                    except Exception as e:
                        self.logger.error(f"  Error comparing metric {metric}: {e}")
                
                window_results.append(window_result)
                
                # Compare dimensions (only for the first time window)
                if start_date == time_windows[0][0] and end_date == time_windows[0][1]:
                    self.logger.info(f"  3.3: Comparing {len(dimensions)} dimensions between environments...")
                    for dimension in dimensions:
                        result = self.compare_dimension(
                            prod_schema, dev_schema, table, dimension, 
                            start_date, end_date
                        )
                        dimension_comparisons.append(result)
                
            except Exception as e:
                self.logger.error(f"Error comparing time window {start_date} to {end_date}: {e}")
        
        # Random sample check for the most recent time window
        sample_check_result = None
        if time_windows:
            recent_window = time_windows[0]  # Use the first time window
            self.logger.info(f"\nPHASE 4: Performing random sample checks...")
            
            # Get configured sample size or use override
            if self.override_sample_size:
                sample_size = self.override_sample_size
                self.logger.info(f"  Taking {sample_size} random samples (override value) from production and checking if they exist in development...")
            else:
                sample_size = self.config.get("sample_size", 5)
                self.logger.info(f"  Taking {sample_size} random samples from production and checking if they exist in development...")
            
            sample_check_result = self.random_sample_check(
                prod_dataset=prod_schema,
                dev_dataset=dev_schema,
                table=table,
                start_date=recent_window[0],
                end_date=recent_window[1],
                sample_size=sample_size
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
            "dimension_comparisons": dimension_comparisons,
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
            self.logger.info("\nSaving comparison results to JSON file...")
            serializable_results = utils.json_serialize(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            self.logger.info(f"Comparison results successfully saved to {output_file}")
            
            # Calculate and log some summary statistics
            summary = analysis.analyze_comparison_results(results)
            
            self.logger.info("\nSUMMARY STATISTICS:")
            self.logger.info(f"- Total tables compared: {summary['tables_count']}")
            self.logger.info(f"- Schemas with column mismatches: {len(summary['schemas_with_column_mismatches'])}")
            self.logger.info(f"- Schemas with data type mismatches: {len(summary['schemas_with_type_mismatches'])}")
            self.logger.info(f"- Schemas with significant row count differences: {len(summary['schemas_with_row_count_diff'])}")
            self.logger.info(f"- Schemas with low sample match rates: {len(summary['schemas_with_low_match_rate'])}")
            if 'schemas_with_dimension_issues' in summary:
                self.logger.info(f"- Schemas with dimension distribution issues: {len(summary['schemas_with_dimension_issues'])}")
            
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