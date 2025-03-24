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
import argparse
import logging
from datetime import datetime

# Import the SchemaComparisonTool from our module
from schema_comparison import SchemaComparisonTool
from schema_comparison.utils import create_timestamped_output_directory


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
                        choices=['fixed', 'floating', 'specific_day'],
                        help='Time window type (fixed, floating, or specific_day)')
    parser.add_argument('--days-back', type=int, default=30,
                        help='Number of days to look back for floating time window')
    parser.add_argument('--specific-date', type=str, default=None,
                        help='Specific date to test in format YYYY-MM-DD (used with --time-window=specific_day)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to use for random sample checks (overrides config file)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory to store results')
    return parser.parse_args()


def main():
    """Main entry point."""
    start_time = datetime.now()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create timestamped output directory
    output_dir = create_timestamped_output_directory(args.results_dir)
    
    # Update output and log file paths to use the output directory
    output_file = os.path.join(output_dir, args.output)
    log_file = os.path.join(output_dir, args.log)
    
    # Create SchemaComparisonTool instance
    comparison_tool = SchemaComparisonTool(
        config_path=args.config,
        log_file=log_file,
        log_level=args.log_level,
        time_window_type=args.time_window,
        days_back=args.days_back,
        specific_date=args.specific_date,
        override_sample_size=args.samples
    )
    
    try:
        comparison_tool.logger.info("======================================================")
        comparison_tool.logger.info("PROD VS DEV SCHEMA COMPARISON TOOL")
        comparison_tool.logger.info("======================================================")
        comparison_tool.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_tool.logger.info(f"Results will be stored in: {output_dir}")
        
        # Log time window settings
        if args.time_window == "fixed":
            comparison_tool.logger.info(f"Using fixed time windows from configuration")
        elif args.time_window == "floating":
            comparison_tool.logger.info(f"Using floating time window: last {args.days_back} days")
        elif args.time_window == "specific_day":
            comparison_tool.logger.info(f"Testing specific date: {args.specific_date or 'yesterday'}")
        
        # Log sample size settings
        if args.samples:
            comparison_tool.logger.info(f"Using custom sample size: {args.samples}")
        else:
            sample_size = comparison_tool.config.get("sample_size", 5)
            comparison_tool.logger.info(f"Using default sample size: {sample_size}")
        
        comparison_tool.logger.info(f"Output file: {output_file}")
        comparison_tool.logger.info(f"Log file: {log_file}")
        comparison_tool.logger.info("======================================================\n")
        
        # Save the configuration to the output directory for reference
        config_output_path = os.path.join(output_dir, "config_used.json")
        with open(config_output_path, 'w') as f:
            json.dump(comparison_tool.config, f, indent=2)
        comparison_tool.logger.info(f"Saved configuration to {config_output_path}")
        
        # Run comparison
        results = comparison_tool.run_comparison()
        
        # Save results
        comparison_tool.save_results(results, output_file)
        
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
        comparison_tool.logger.info(f"Results stored in: {output_dir}")
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