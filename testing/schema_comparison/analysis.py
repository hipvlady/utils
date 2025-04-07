"""
Analysis functions for schema comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union


def find_date_column(schema_df: pd.DataFrame, table_name: str) -> Optional[str]:
    """
    Find a suitable date column in the table schema with fallback to shard.
    
    Args:
        schema_df: DataFrame containing schema information
        table_name: Name of the table (for logging)
        
    Returns:
        Column name if found, None otherwise
    """
    if schema_df.empty:
        return None
        
    # Look for date-related columns
    potential_date_columns = ['date', 'report_date', 'event_date', 'hit_date', 'page_date', 'shard']
    
    # Convert column names to lowercase for case-insensitive comparison
    schema_columns_lower = [col.lower() for col in schema_df['column_name'].tolist()]
    
    # First try to find a date column
    for col in potential_date_columns:
        col_lower = col.lower()
        if col_lower in schema_columns_lower:
            # Get the actual column name with original casing
            idx = schema_columns_lower.index(col_lower)
            actual_col_name = schema_df['column_name'].iloc[idx]
            return actual_col_name
    
    return None


def compare_schemas(prod_schema_df: pd.DataFrame, dev_schema_df: pd.DataFrame) -> Tuple[Set[str], Set[str], List[Tuple[str, str, str]]]:
    """
    Compare production and development schemas for missing, extra, or mismatched columns.
    
    Args:
        prod_schema_df: DataFrame containing production schema
        dev_schema_df: DataFrame containing development schema
        
    Returns:
        Tuple of (missing_in_dev, extra_in_dev, type_mismatches)
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


def get_available_metrics(schema_df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric columns that can be used as metrics.
    
    Args:
        schema_df: DataFrame containing schema information
        
    Returns:
        List of column names that can be used as metrics
    """
    if schema_df.empty:
        return []
        
    # Filter for numeric types
    numeric_types = ['int64', 'float64', 'numeric', 'integer', 'float', 'number', 'bignumeric', 'decimal']
    numeric_columns = schema_df[schema_df['data_type'].str.lower().isin(numeric_types)]['column_name'].tolist()
    
    # Filter out ID columns and other non-metric columns
    excluded_terms = ['id', 'key', 'num', 'count', 'index', 'position', 'order']
    metrics = [col for col in numeric_columns if not any(term in col.lower() for term in excluded_terms)]
    
    # Add back specific count metrics that are actually valuable
    count_metrics = [col for col in numeric_columns if 'count' in col.lower()]
    return metrics + count_metrics


def get_available_dimensions(schema_df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns that can be used as dimensions.
    
    Args:
        schema_df: DataFrame containing schema information
        
    Returns:
        List of column names that can be used as dimensions
    """
    if schema_df.empty:
        return []
        
    # Filter for string/categorical types
    string_types = ['string', 'varchar', 'char', 'text', 'bytes']
    categorical_columns = schema_df[schema_df['data_type'].str.lower().isin(string_types)]['column_name'].tolist()
    
    # Filter out columns that are likely not dimensions
    excluded_terms = ['url', 'path', 'description', 'comment', 'note', 'content']
    dimensions = [col for col in categorical_columns if not any(term in col.lower() for term in excluded_terms)]
    
    # Add back specific dimension-like columns even if filtered
    dimension_terms = ['channel', 'medium', 'source', 'category', 'platform', 'device', 'country', 'region']
    additional_dims = [col for col in categorical_columns if any(term in col.lower() for term in dimension_terms)]
    
    # Combine and deduplicate
    result = list(set(dimensions + additional_dims))
    
    return result


def is_numeric_column(schema_df: pd.DataFrame, column_name: str) -> bool:
    """
    Determine if a column is numeric based on schema information.
    
    Args:
        schema_df: DataFrame containing schema information
        column_name: Name of column to check
        
    Returns:
        Boolean indicating if column is numeric
    """
    if schema_df.empty:
        return False
        
    numeric_types = ['int64', 'float64', 'numeric', 'integer', 'float', 'number', 'bignumeric', 'decimal']
    
    # Get column data type, accounting for case sensitivity
    column_lower = column_name.lower()
    matching_cols = schema_df[schema_df['column_name'].str.lower() == column_lower]
    
    if matching_cols.empty:
        return False
    
    data_type = matching_cols.iloc[0]['data_type'].lower()
    return data_type in numeric_types


def determine_column_type(schema_df: pd.DataFrame, column_name: str) -> str:
    """
    Determine the type of a column (numeric, categorical, date, boolean).
    
    Args:
        schema_df: DataFrame containing schema information
        column_name: Name of column to check
        
    Returns:
        String representing column type ('numeric', 'categorical', 'date', 'boolean', 'unknown')
    """
    if schema_df.empty:
        return 'unknown'
    
    # Get column data type, accounting for case sensitivity
    column_lower = column_name.lower()
    matching_cols = schema_df[schema_df['column_name'].str.lower() == column_lower]
    
    if matching_cols.empty:
        return 'unknown'
    
    data_type = matching_cols.iloc[0]['data_type'].lower()
    
    # Numeric types
    if data_type in ['int64', 'float64', 'numeric', 'integer', 'float', 'number', 'bignumeric', 'decimal']:
        return 'numeric'
    
    # Date types
    elif data_type in ['date', 'datetime', 'timestamp', 'time']:
        return 'date'
    
    # Boolean types
    elif data_type in ['bool', 'boolean']:
        return 'boolean'
    
    # String/categorical types
    elif data_type in ['string', 'varchar', 'char', 'text', 'bytes']:
        return 'categorical'
    
    # Default
    return 'unknown'


def get_appropriate_aggregation(column_type: str) -> Tuple[str, str]:
    """
    Get appropriate aggregation function and description for a column type.
    
    Args:
        column_type: Type of column ('numeric', 'categorical', etc.)
        
    Returns:
        Tuple of (aggregation_function, description)
    """
    if column_type == 'numeric':
        return 'SUM', 'sum'
    elif column_type == 'categorical':
        return 'COUNT(DISTINCT)', 'distinct count'
    elif column_type == 'boolean':
        return 'SUM', 'count of true values'
    elif column_type == 'date':
        return 'COUNT(DISTINCT)', 'distinct count'
    else:
        return 'COUNT(DISTINCT)', 'distinct count'


def build_row_conditions(row: pd.Series, columns: List[str], 
                        date_column: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Build WHERE conditions for a row based on its values.
    
    Args:
        row: Pandas Series containing row values
        columns: List of column names
        date_column: Name of date column (if any)
        
    Returns:
        Tuple of (conditions, key_columns)
    """
    conditions = []
    key_columns = []
    
    # Add date/shard condition if it exists
    if date_column and date_column in row and not pd.isna(row[date_column]):
        conditions.append(f"`{date_column}` = '{row[date_column]}'")
        key_columns.append(date_column)
    
    # Add other good identifier columns
    potential_key_cols = ['page_path', 'url', 'event_name', 'content_group', 'device_category', 'country']
    for col in potential_key_cols:
        if col in columns and col in row and not pd.isna(row.get(col)):
            if isinstance(row[col], str):
                # Option 1: Split the f-string to avoid complex escaping
                conditions.append(f"`{col}` = '" + row[col].replace("'", "''") + "'")
                # Alternative options:
                # conditions.append("`{}` = '{}'".format(col, row[col].replace("'", "''")))
                # conditions.append("`" + col + "` = '" + row[col].replace("'", "''") + "'")
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
    
    return conditions, key_columns


def compare_distributions(prod_dist: pd.DataFrame, dev_dist: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare categorical distributions between production and development.
    
    Args:
        prod_dist: DataFrame with production distribution (columns: value, count, percentage)
        dev_dist: DataFrame with development distribution (columns: value, count, percentage)
        
    Returns:
        Dictionary with comparison results
    """
    if prod_dist.empty or dev_dist.empty:
        return {
            "overlap_rate": 0,
            "common_values": [],
            "missing_in_dev": [],
            "missing_in_prod": []
        }
    
    # Get sets of values
    prod_values = set(prod_dist['value'])
    dev_values = set(dev_dist['value'])
    
    # Find common and missing values
    common_values = prod_values.intersection(dev_values)
    missing_in_dev = prod_values - dev_values
    missing_in_prod = dev_values - prod_values
    
    # Calculate overlap rate
    overlap_rate = len(common_values) / len(prod_values) if prod_values else 0
    
    # Compare percentages for common values
    percentage_diffs = []
    for value in common_values:
        prod_pct = prod_dist[prod_dist['value'] == value]['percentage'].iloc[0] if not prod_dist[prod_dist['value'] == value].empty else 0
        dev_pct = dev_dist[dev_dist['value'] == value]['percentage'].iloc[0] if not dev_dist[dev_dist['value'] == value].empty else 0
        
        percentage_diffs.append({
            "value": value,
            "prod_pct": prod_pct,
            "dev_pct": dev_pct,
            "diff": abs(prod_pct - dev_pct)
        })
    
    # Sort by largest differences
    percentage_diffs = sorted(percentage_diffs, key=lambda x: x['diff'], reverse=True)
    
    return {
        "overlap_rate": overlap_rate,
        "common_values": list(common_values),
        "missing_in_dev": list(missing_in_dev),
        "missing_in_prod": list(missing_in_prod),
        "percentage_diffs": percentage_diffs[:5]  # Top 5 differences
    }


def analyze_comparison_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from comparison results.
    
    Args:
        results: Dictionary containing comparison results
        
    Returns:
        Dictionary containing summary statistics
    """
    tables_count = len(results.get("results", []))
    
    # Count schemas with issues
    schemas_with_column_mismatches = set()
    schemas_with_type_mismatches = set()
    schemas_with_low_match_rate = set()
    schemas_with_row_count_diff = set()
    schemas_with_dimension_issues = set()
    
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
        for window in table_result.get("time_windows", []):
            if window.get("row_count_diff_pct", 0) > 1.0:  # More than 1% difference
                schemas_with_row_count_diff.add(schema)
                break
        
        # Check for sample match issues
        sample_check = table_result.get("sample_check", {})
        match_rate = sample_check.get("match_rate", 1.0)
        if match_rate < 0.9:  # Less than 90% match
            schemas_with_low_match_rate.add(schema)
            
        # Check for dimension distribution issues
        for dimension in table_result.get("dimension_comparisons", []):
            if dimension.get("overlap_rate", 1.0) < 0.9:  # Less than 90% overlap
                schemas_with_dimension_issues.add(schema)
                break
    
    return {
        "tables_count": tables_count,
        "schemas_with_column_mismatches": list(schemas_with_column_mismatches),
        "schemas_with_type_mismatches": list(schemas_with_type_mismatches),
        "schemas_with_row_count_diff": list(schemas_with_row_count_diff),
        "schemas_with_low_match_rate": list(schemas_with_low_match_rate),
        "schemas_with_dimension_issues": list(schemas_with_dimension_issues)
    }