#!/bin/bash

# Set the environment variable for BigQuery service account authentication
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Verify the variable is set
echo "Service account path is set to: $GOOGLE_APPLICATION_CREDENTIALS"

# Run the schema comparison tool
python schema_comparison_tool.py "$@"