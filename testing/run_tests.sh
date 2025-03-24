#!/bin/bash

# Example script to run schema comparison tool with different options

# Run with default config (fixed time windows)
echo "Running with default config (fixed time windows)..."
python schema_comparison_tool.py --output fixed_results.json

# Run with floating time window (last 30 days)
echo "Running with floating time window (last 30 days)..."
python schema_comparison_tool.py --time-window floating --days-back 7 --output floating_results.json

# Run with floating time window (last 7 days) and debug logging
echo "Running with shorter floating window and debug logging..."
python schema_comparison_tool.py --time-window floating --days-back 7 --log-level DEBUG --log debug_run.log --output recent_results.json

# Run comparison for specific schemas (using custom config)
echo "Running with custom config..."
python schema_comparison_tool.py --config custom_config.json --output custom_results.json