# Schema and Table Comparison Tool

This tool compares data between production and development schemas in BigQuery to validate consistency after schema refactoring. It performs schema validation, row count checks, metric value comparisons, and sample record testing.

## Purpose

The primary purpose of this tool is to compare and test production schemas versus development schemas to ensure that after refactoring, the data in development tables matches the production tables.

## Features

- **Schema Validation**: Compares column names and data types between prod and dev schemas
- **Row Count Checks**: Verifies row counts are consistent between environments
- **Metric Comparison**: Compares sum of key metrics between prod and dev tables
- **Sample Record Testing**: Validates that individual records exist in both environments
- **Flexible Time Window Options**:
  - Fixed date ranges specified in configuration
  - Floating window based on current date (e.g., last 30 days)
  - Specific day testing for targeted verification
- **Comprehensive Logging**: Detailed logging to file and console
- **Config-Driven**: Easily configure schemas and tables to test
- **CLI Support**: Command-line interface for easy integration
- **Organized Results**: All test runs saved in timestamped directories

## Project Structure

```
schema-comparison-tool/
├── automated_testing.py      # Main script
├── config.json               # Configuration file
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── schema_comparison/        # Core package
│   ├── __init__.py           # Package initialization
│   ├── analysis.py           # Analysis functions
│   ├── core.py               # Main comparison functionality
│   ├── query_builder.py      # SQL query generators
│   └── utils.py              # Utility functions
└── results/                  # Test results directory
    └── results_YYYYMMDD_HHMMSS/  # Timestamped results for each run
        ├── schema_comparison_results.json  # JSON results
        ├── schema_comparison.log           # Detailed log
        └── config_used.json                # Configuration snapshot
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/schema-comparison-tool.git
   cd schema-comparison-tool
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up BigQuery authentication:
   - You can use service account authentication by specifying the path in the config
   - Set the environment variable with your service account key path:
     ```
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
     ```
   - Make sure the service account has access to both production and development schemas

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "project_id": "your-gcp-project-id",
  "service_account_path": "${GOOGLE_APPLICATION_CREDENTIALS}",
  "schema_comparisons": [
    {
      "prod_schema": "production_schema_name",
      "dev_schema": "development_schema_name",
      "tables": [
        "table1",
        "table2",
        "table3"
      ]
    },
    {
      "prod_schema": "another_prod_schema",
      "dev_schema": "another_dev_schema",
      "tables": [
        "other_table1",
        "other_table2"
      ]
    }
  ],
  "fixed_time_windows": [
    ["2024-02-01", "2024-02-28"],
    ["2024-03-01", "2024-03-15"]
  ],
  "metrics_to_compare": [
    "sessions",
    "pageviews",
    "unique_pageviews",
    "clicks",
    "users"
  ],
  "sample_size": 10
}
```

## Usage

### Basic Usage

```bash
python automated_testing.py
```

### Specify Config File

```bash
python automated_testing.py --config custom_config.json
```

### Use Floating Time Window (Last 30 Days)

```bash
python automated_testing.py --time-window floating --days-back 30
```

### Test a Specific Date

```bash
python automated_testing.py --time-window specific_day --specific-date 2024-03-21
```

### Use More Samples for Verification

```bash
python automated_testing.py --samples 20
```

### Specify Custom Results Directory

```bash
python automated_testing.py --results-dir custom_results
```

### Combine Options for Targeted Testing

```bash
python automated_testing.py --time-window specific_day --specific-date 2024-03-21 --samples 20 --results-dir my_test_results
```

### Set Log Level

```bash
python automated_testing.py --log-level DEBUG
```

## Time Window Options

The tool supports three approaches for time window selection:

1. **Fixed Time Windows**: Specific date ranges defined in the configuration file
   ```json
   "fixed_time_windows": [
     ["2024-02-01", "2024-02-28"],
     ["2024-03-01", "2024-03-15"]
   ]
   ```

2. **Floating Time Window**: A dynamic window based on the current date
   ```bash
   python automated_testing.py --time-window floating --days-back 30
   ```
   This will use a window from today minus 30 days to today.

3. **Specific Day**: Test a single specific date
   ```bash
   python automated_testing.py --time-window specific_day --specific-date 2024-03-21
   ```
   This will test only data from March 21, 2024.

## Results Management

Test results are organized in timestamped directories to prevent overwriting previous runs:

```
results/
└── results_20240324_154522/
    ├── schema_comparison_results.json  # Detailed JSON results
    ├── schema_comparison.log           # Complete log file
    └── config_used.json                # Configuration snapshot
```

This makes it easy to:
- Track test runs over time
- Compare results across different runs
- Preserve historical test data
- Maintain an audit trail of schema comparisons

## Full Command Line Options

```
usage: automated_testing.py [-h] [--config CONFIG] [--output OUTPUT] [--log LOG]
                        [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [--time-window {fixed,floating,specific_day}] [--days-back DAYS_BACK]
                        [--specific-date SPECIFIC_DATE] [--samples SAMPLES]
                        [--results-dir RESULTS_DIR]

Schema and Table Comparison Tool

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file
  --output OUTPUT       Output file for comparison results
  --log LOG             Log file path
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
  --time-window {fixed,floating,specific_day}
                        Time window type (fixed, floating, or specific_day)
  --days-back DAYS_BACK
                        Number of days to look back for floating time window
  --specific-date SPECIFIC_DATE
                        Specific date to test in format YYYY-MM-DD
  --samples SAMPLES     Number of samples to use for random sample checks
  --results-dir RESULTS_DIR
                        Base directory to store results
```

## Output

The tool generates three types of output:

1. **Log File**: Detailed log of all comparisons performed
2. **Results JSON**: Structured results of all comparisons that can be parsed for monitoring
3. **Config Snapshot**: Copy of the configuration used for the test run

Example results JSON:

```json
{
  "comparison_time": "2024-03-24T15:45:22.123456",
  "time_window_type": "fixed",
  "time_windows": [["2024-02-01", "2024-02-28"], ["2024-03-01", "2024-03-15"]],
  "results": [
    {
      "prod_table": "dbt_prod_ga4_reporting.ga4__sessions",
      "dev_table": "dev_admind_joe_ga4_reporting.ga4__sessions",
      "schema_comparison": {
        "missing_in_dev": [],
        "extra_in_dev": ["new_column"],
        "type_mismatches": [["column1", "int64", "float64"]]
      },
      "time_windows": [
        {
          "time_window": ["2024-02-01", "2024-02-28"],
          "prod_row_count": 1000000,
          "dev_row_count": 999950,
          "row_count_diff_pct": 0.005,
          "metrics": [
            {
              "name": "sessions",
              "prod_value": 500000,
              "dev_value": 499975,
              "diff_pct": 0.005
            }
          ]
        }
      ],
      "sample_check": {
        "sample_size": 10,
        "matches": 10,
        "match_rate": 1.0,
        "details": [
          {
            "matched": true,
            "match_count": 1,
            "key_columns": ["date", "page_path", "device_category"],
            "where_clause": "date = '2024-02-15' AND page_path = '/home' AND device_category = 'desktop'"
          }
        ]
      }
    }
  ]
}
```

## Common Use Cases

1. **After Schema Refactoring**: Verify that data in the development schema matches production after making structural changes
   
2. **Daily Data Validation**: Run with floating time window to validate recent data consistency
   
3. **QA Testing**: Conduct thorough testing before promoting development schema to production

4. **Targeted Date Testing**: Test a specific date when issues are reported in production

5. **Historical Audit**: Compare multiple date ranges to verify consistency over time

## Requirements

- Python 3.7+
- Google Cloud BigQuery SDK
- pandas
- numpy

## License

[MIT License](LICENSE)