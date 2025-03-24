#!/bin/bash

# Example script to run schema comparison tool with different options

# Run with default config (fixed time windows)
echo "Running with default config (fixed time windows)..."
python automated_testing.py --output fixed_results.json

# Run with floating time window (last 30 days)
echo "Running with floating time window (last 30 days)..."
python automated_testing.py --time-window floating --days-back 30 --output floating_results.json

# Run with floating time window (last 7 days) and debug logging
echo "Running with shorter floating window and debug logging..."
python automated_testing.py --time-window floating --days-back 7 --log-level DEBUG --log debug_run.log --output recent_results.json

# Run comparison for specific date
echo "Running for a specific date..."
python automated_testing.py --time-window specific_day --specific-date 2024-03-15 --output specific_date_results.json

# Run with a custom sample size
echo "Running with increased sample size..."
python automated_testing.py --samples 20 --output increased_samples_results.json

# Run comparison for specific schemas (using custom config)
echo "Running with custom config..."
python automated_testing.py --config config.json --output custom_results.json

# Run a quick test for yesterday with small sample size
echo "Running a quick test for yesterday only..."
python automated_testing.py --time-window specific_day --samples 3 --output yesterday_quick_results.json