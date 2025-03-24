"""
Utility functions and classes for schema comparison.
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set

class Tee:
    """
    Class to duplicate output to multiple destinations.
    
    This is used to send log output to both console and file.
    """
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush()  # Ensure output is written immediately
    
    def flush(self):
        for file in self.files:
            file.flush()


def json_serialize(obj: Any) -> Any:
    """
    Convert any NumPy or pandas types to native Python types for JSON serialization.
    
    Args:
        obj: Object to serialize (can be dict, list, numpy type, etc.)
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(i) for i in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()  # Convert numpy types to native Python types
    elif pd.isna(obj):
        return None
    else:
        return obj


def calculate_diff_percentage(value1: float, value2: float) -> float:
    """
    Calculate percentage difference between two values.
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Percentage difference
    """
    if value1 == 0 and value2 == 0:
        return 0
    elif value1 == 0:
        return 100
    else:
        return abs(value1 - value2) / abs(value1) * 100


def get_formatted_date_range(time_window_type: str, days_back: int = 30, 
                           specific_date: str = None) -> List[Tuple[str, str]]:
    """
    Get time windows for testing based on the selected strategy.
    
    Args:
        time_window_type: Type of time window (fixed, floating, or specific_day)
        days_back: Number of days to look back for floating time window
        specific_date: Specific date to test (for time_window_type='specific_day')
        
    Returns:
        List of (start_date, end_date) tuples
    """
    if time_window_type == "floating":
        # Generate time windows based on current date - days_back
        today = datetime.now().date()
        start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        return [(start_date, end_date)]
    
    elif time_window_type == "specific_day":
        # Use a specific day for testing
        if not specific_date:
            # Default to yesterday if no specific date provided
            yesterday = (datetime.now().date() - timedelta(days=1)).strftime('%Y-%m-%d')
            return [(yesterday, yesterday)]
        else:
            # Validate date format
            try:
                datetime.strptime(specific_date, '%Y-%m-%d')
                return [(specific_date, specific_date)]
            except ValueError:
                # Invalid date format, default to yesterday
                yesterday = (datetime.now().date() - timedelta(days=1)).strftime('%Y-%m-%d')
                return [(yesterday, yesterday)]
    
    # Default: return empty list (will be filled from config in the tool)
    return []