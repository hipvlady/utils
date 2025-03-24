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
- **Comprehensive Logging**: Detailed logging to file and console
- **Config-Driven**: Easily configure schemas and tables to test
- **CLI Support**: Command-line interface for easy integration

## Project Structure

```
schema-comparison-tool/
├── schema_comparison_tool.py  # Main script
├── config.json                # Configuration file
├── requirements.txt           # Dependencies
└── README.md                  # This file
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
   - Set up application default credentials or use a service account
   - Make sure you have access to both production and development schemas

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "project_id": "your-gcp-project-id",
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
python schema_comparison_tool.py
```

### Specify Config File

```bash
python schema_comparison_tool.py --config custom_config.json
```

### Use Floating Time Window (Last 30 Days)

```bash
python schema_comparison_tool.py --time-window floating --days-back 30
```

### Set Custom Output and Log Files

```bash
python schema_comparison_tool.py --output results.json --log comparison_run.log
```

### Set Log Level

```bash
python schema_comparison_tool.py --log-level DEBUG
```

## Time Window Options

The tool supports two approaches for time window selection:

1. **Fixed Time Windows**: Specific date ranges defined in the configuration file
   ```json
   "fixed_time_windows": [
     ["2024-02-01", "2024-02-28"],
     ["2024-03-01", "2024-03-15"]
   ]
   ```

2. **Floating Time Window**: A dynamic window based on the current date
   ```bash
   python schema_comparison_tool.py --time-window floating --days-back 30
   ```
   This will use a window from today minus 30 days to today.

## Full Command Line Options

```
usage: schema_comparison_tool.py [-h] [--config CONFIG] [--output OUTPUT] [--log LOG]
                        [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [--time-window {fixed,floating}] [--days-back DAYS_BACK]

Schema and Table Comparison Tool

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file
  --output OUTPUT       Output file for comparison results
  --log LOG             Log file path
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
  --time-window {fixed,floating}
                        Time window type (fixed or floating)
  --days-back DAYS_BACK
                        Number of days to look back for floating time window
```

## Output

The tool generates two types of output:

1. **Log File**: Detailed log of all comparisons performed
2. **Results JSON**: Structured results of all comparisons that can be parsed for monitoring

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

## Requirements

- Python 3.7+
- Google Cloud BigQuery SDK
- pandas
- numpy

## License

[MIT License](LICENSE)