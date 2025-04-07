"""
Utility functions for building SQL queries for schema comparison.
"""

from typing import List, Optional, Dict, Any, Tuple


def build_schema_query(project_id: str, dataset: str, table: str) -> str:
    """
    Build a query to get schema information for a table.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        
    Returns:
        SQL query string
    """
    return f"""
    SELECT column_name, data_type
    FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
    """


def build_row_count_query(project_id: str, dataset: str, table: str, 
                         date_column: Optional[str] = None,
                         start_date: str = None, end_date: str = None) -> str:
    """
    Build a query to count rows in a table, optionally filtered by date range.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        date_column: Column to use for date filtering
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        
    Returns:
        SQL query string
    """
    if date_column and start_date and end_date:
        # Special handling for shard column (typically YYYYMMDD format)
        if date_column.lower() == 'shard':
            # Convert YYYY-MM-DD to YYYYMMDD format for shard
            start_shard = start_date.replace('-', '')
            end_shard = end_date.replace('-', '')
            
            return f"""
            SELECT COUNT(*) AS row_count
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_shard}' AND '{end_shard}'
            """
        else:
            return f"""
            SELECT COUNT(*) AS row_count
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
            """
    else:
        # If no date column found, count all rows
        return f"""
        SELECT COUNT(*) AS row_count
        FROM `{project_id}.{dataset}.{table}`
        """


def build_aggregate_query(project_id: str, dataset: str, table: str,
                         column: str,
                         date_column: Optional[str] = None,
                         start_date: str = None, end_date: str = None,
                         agg_function: str = "SUM") -> str:
    """
    Build a query to aggregate a column in a table, optionally filtered by date range.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        column: Column to aggregate
        date_column: Column to use for date filtering
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        agg_function: Aggregation function to use (SUM, COUNT, COUNT(DISTINCT), etc.)
        
    Returns:
        SQL query string
    """
    # Handle special case for COUNT(DISTINCT)
    if agg_function.upper() == "COUNT(DISTINCT)":
        agg_expression = f"COUNT(DISTINCT `{column}`)"
    else:
        agg_expression = f"{agg_function}(`{column}`)"
    
    if date_column and start_date and end_date:
        # Special handling for shard column (typically YYYYMMDD format)
        if date_column.lower() == 'shard':
            # Convert YYYY-MM-DD to YYYYMMDD format for shard
            start_shard = start_date.replace('-', '')
            end_shard = end_date.replace('-', '')
            
            return f"""
            SELECT {agg_expression} AS total_value
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_shard}' AND '{end_shard}'
            """
        else:
            return f"""
            SELECT {agg_expression} AS total_value
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
            """
    else:
        # If no date column found, aggregate all rows
        return f"""
        SELECT {agg_expression} AS total_value
        FROM `{project_id}.{dataset}.{table}`
        """


def build_distribution_query(project_id: str, dataset: str, table: str,
                           column: str,
                           date_column: Optional[str] = None,
                           start_date: str = None, end_date: str = None,
                           top_n: int = 10) -> str:
    """
    Build a query to get distribution of values for categorical columns.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        column: Column to analyze distribution
        date_column: Column to use for date filtering
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        top_n: Number of top values to return
        
    Returns:
        SQL query string
    """
    where_clause = ""
    if date_column and start_date and end_date:
        # Special handling for shard column (typically YYYYMMDD format)
        if date_column.lower() == 'shard':
            # Convert YYYY-MM-DD to YYYYMMDD format for shard
            start_shard = start_date.replace('-', '')
            end_shard = end_date.replace('-', '')
            where_clause = f"WHERE {date_column} BETWEEN '{start_shard}' AND '{end_shard}'"
        else:
            where_clause = f"WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'"
    
    return f"""
    SELECT 
        `{column}` AS value,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
    FROM `{project_id}.{dataset}.{table}`
    {where_clause}
    GROUP BY `{column}`
    ORDER BY count DESC
    LIMIT {top_n}
    """


def build_cardinality_query(project_id: str, dataset: str, table: str,
                          column: str,
                          date_column: Optional[str] = None,
                          start_date: str = None, end_date: str = None) -> str:
    """
    Build a query to get cardinality (number of distinct values) for a column.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        column: Column to check cardinality
        date_column: Column to use for date filtering
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        
    Returns:
        SQL query string
    """
    if date_column and start_date and end_date:
        # Special handling for shard column (typically YYYYMMDD format)
        if date_column.lower() == 'shard':
            # Convert YYYY-MM-DD to YYYYMMDD format for shard
            start_shard = start_date.replace('-', '')
            end_shard = end_date.replace('-', '')
            
            return f"""
            SELECT 
                COUNT(DISTINCT `{column}`) AS distinct_count,
                COUNT(*) AS total_count,
                SAFE_DIVIDE(COUNT(DISTINCT `{column}`), COUNT(*)) AS cardinality_ratio
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_shard}' AND '{end_shard}'
            """
        else:
            return f"""
            SELECT 
                COUNT(DISTINCT `{column}`) AS distinct_count,
                COUNT(*) AS total_count,
                SAFE_DIVIDE(COUNT(DISTINCT `{column}`), COUNT(*)) AS cardinality_ratio
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
            """
    else:
        # If no date column found, calculate over all rows
        return f"""
        SELECT 
            COUNT(DISTINCT `{column}`) AS distinct_count,
            COUNT(*) AS total_count,
            SAFE_DIVIDE(COUNT(DISTINCT `{column}`), COUNT(*)) AS cardinality_ratio
        FROM `{project_id}.{dataset}.{table}`
        """


def build_sample_query(project_id: str, dataset: str, table: str,
                     columns: List[str],
                     date_column: Optional[str] = None,
                     start_date: str = None, end_date: str = None,
                     limit: int = 1000) -> str:
    """
    Build a query to get sample rows from a table, optionally filtered by date range.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        columns: List of columns to select
        date_column: Column to use for date filtering
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        limit: Maximum number of rows to return
        
    Returns:
        SQL query string
    """
    columns_str = ", ".join(f"`{col}`" for col in columns)
    
    if date_column and start_date and end_date:
        # Special handling for shard column
        if date_column.lower() == 'shard':
            # Convert YYYY-MM-DD to YYYYMMDD format for shard
            start_shard = start_date.replace('-', '')
            end_shard = end_date.replace('-', '')
            
            return f"""
            SELECT {columns_str}
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_shard}' AND '{end_shard}'
            LIMIT {limit}
            """
        else:
            return f"""
            SELECT {columns_str}
            FROM `{project_id}.{dataset}.{table}`
            WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'
            LIMIT {limit}
            """
    else:
        # If no date column found, sample without date filter
        return f"""
        SELECT {columns_str}
        FROM `{project_id}.{dataset}.{table}`
        LIMIT {limit}
        """


def build_match_query(project_id: str, dataset: str, table: str,
                    conditions: List[str]) -> str:
    """
    Build a query to count matching rows based on conditions.
    
    Args:
        project_id: GCP project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        conditions: List of WHERE conditions
        
    Returns:
        SQL query string
    """
    where_clause = " AND ".join(conditions)
    
    return f"""
    SELECT COUNT(*) AS match_count
    FROM `{project_id}.{dataset}.{table}`
    WHERE {where_clause}
    """