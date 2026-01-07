# SynMax Data Agent

A sophisticated chat-based data analysis agent powered by LangGraph and LLMs, capable of answering natural language questions about datasets through pattern recognition, anomaly detection, and causal analysis.

## Features

- **Natural Language Queries**: Ask questions in plain English about your data
- **Advanced Analytics**:
  - Pattern recognition (clustering, correlations, trends)
  - Anomaly detection (outliers, rule violations)
  - Causal hypothesis generation with evidence-backed reasoning
  - Data quality issue detection
- **Intelligent Agent Architecture**: Built with LangGraph for robust multi-step reasoning
- **Multiple Interfaces**:
  - CLI for direct interaction
  - REST API via FastAPI
  - Simple web UI for visual interaction
- **LLM Provider Support**: Powered by OpenAI GPT-4

## Architecture

The agent uses a multi-stage reasoning pipeline:

1. **Query Understanding**: Parse natural language into analytical intent
2. **Analysis Planning**: Determine appropriate statistical/analytical methods
3. **Execution**: Run calculations, detect patterns, identify anomalies
4. **Synthesis**: Generate evidence-backed answers with methodology transparency
5. **Validation**: Check for confounders and data quality issues

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd synmax-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-key
```

## Dataset Setup

The agent requires the dataset to be available locally. **Do not commit the dataset to the repository.**

### Option 1: Manual Download (Recommended)

1. Download the dataset from [this link](https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view?usp=drivesdk)
2. Place it in the `./data/` directory (this folder is gitignored)
3. The agent will automatically detect and load the dataset

### Option 2: Automatic Download

Run the setup script:
```bash
python scripts/download_dataset.py
```

This will download the dataset to `./data/` automatically.

### Option 3: Custom Path

You can specify a custom path when running the agent:
```bash
python -m agent.cli --dataset-path /path/to/your/dataset.csv
```

## Usage

### CLI Mode

Basic usage:
```bash
python -m agent.cli
```

With custom dataset path:
```bash
python -m agent.cli --dataset-path /custom/path/data.csv
```

Interactive session:
```
> What is the total count of records in 2024?
> Show me the top 5 patterns in customer behavior
> Are there any anomalies in the sales data?
> What might explain the correlation between X and Y?
```

### API Mode

Start the FastAPI server:
```bash
uvicorn agent.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

Example API request:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 5 anomalies in the dataset?"}'
```

### Web UI

1. Start the API server (see above)
2. Open `index.html` in your browser
3. Start asking questions through the visual interface

Or serve via Python:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080`

## Example Queries & Outputs

### Simple Retrieval
**Query**: "How many records are there from 2024?"

**Output**:
```
Answer: There are 1,247 records from 2024.

Method: Direct filtering and count aggregation
Columns Used: date_column
Filters Applied: year == 2024
```

### Pattern Recognition
**Query**: "What patterns exist in customer purchasing behavior?"

**Output**:
```
Answer: Identified 3 distinct customer segments using K-means clustering:

1. High-Value Infrequent (23% of customers)
   - Average order value: $450
   - Purchase frequency: 2.1x per year
   - Peak activity: Q4

2. Regular Moderate (58% of customers)
   - Average order value: $120
   - Purchase frequency: 8.3x per year
   - Consistent throughout year

3. Frequent Low-Value (19% of customers)
   - Average order value: $35
   - Purchase frequency: 24.7x per year
   - Peak activity: Weekends

Method: K-means clustering (k=3, silhouette score: 0.67)
Features: purchase_frequency, avg_order_value, seasonality_index
Statistical Significance: p < 0.001
```

### Anomaly Detection
**Query**: "Find anomalies in the sales data"

**Output**:
```
Answer: Detected 12 significant anomalies using Isolation Forest:

Top 3 Anomalies:
1. Record #4521 (2024-03-15): Sales spike 847% above normal
   - Potential Cause: Marketing campaign + viral social media

2. Record #8934 (2024-07-22): Zero sales during peak hour
   - Potential Data Quality Issue: System downtime not recorded

3. Record #2103 (2024-01-08): Negative inventory count
   - Data Quality Issue: Database constraint violation

Method: Isolation Forest (contamination=0.01)
Anomaly Score Threshold: > 0.6
Features Analyzed: sales_volume, transaction_count, inventory_level
```

### Causal Analysis
**Query**: "Why is there a correlation between X and Y?"

**Output**:
```
Answer: Observed strong positive correlation (r=0.78, p<0.001) between marketing_spend and revenue.

Plausible Causal Hypotheses:

1. Direct Effect (Most Likely):
   - Increased marketing → more customer awareness → higher sales
   - Evidence: 2-week lag between spend and revenue peaks
   - Supports: Granger causality test significant (p=0.003)

2. Confounding Variable (Moderate Likelihood):
   - Seasonality drives both marketing budget and natural demand
   - Evidence: Both peak in Q4
   - Caveat: Partial correlation controlling for month still significant (r=0.64)

3. Reverse Causation (Lower Likelihood):
   - Higher revenue → larger marketing budgets
   - Evidence: Some budget decisions follow revenue reports
   - Limitation: Time-lagged analysis shows spend precedes revenue

Robustness Checks Performed:
- Partial correlation controlling for seasonality
- Granger causality test
- Time-lagged correlation analysis
- Outlier sensitivity analysis

Limitations:
- Observational data; cannot prove causation
- Other unmeasured confounders may exist
- Marketing channel effectiveness not differentiated
```

## Assumptions & Limitations

### Assumptions
- Dataset is in CSV format (or can be automatically inferred)
- Dates are parseable (will attempt multiple common formats)
- Missing values are either NULL, empty strings, or "NaN"
- Numerical columns use standard decimal notation

### Limitations
- **LLM Costs**: Complex queries may require multiple LLM calls
- **Dataset Size**: Extremely large datasets (>1GB) may require sampling for some analyses
- **Statistical Validity**: Causal claims are hypotheses, not definitive conclusions
- **Context Window**: Very wide datasets (>100 columns) may require column selection
- **Real-time Data**: Agent analyzes static snapshots, not streaming data

### Design Decisions
- **Conservative Causation**: Agent explicitly labels causal claims as hypotheses with caveats
- **Methodology Transparency**: All methods and parameters are included in responses
- **Data Quality First**: Proactively flags potential data quality issues
- **Reproducibility**: Analysis steps are deterministic where possible

## Technical Stack

- **Agent Framework**: LangGraph for stateful multi-step reasoning
- **LLM Provider**: OpenAI GPT-4
- **API Framework**: FastAPI for REST endpoints
- **Data Processing**: pandas, numpy, scikit-learn
- **Statistical Analysis**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn (for analysis, not UI)

## Project Structure

```
synmax-agent/
├── agent/
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── api.py              # FastAPI server
│   ├── graph.py            # LangGraph agent definition
│   ├── nodes/              # Agent processing nodes
│   ├── tools/              # Data analysis tools
│   └── utils/              # Helper functions
├── data/                   # Dataset directory (gitignored)
├── scripts/
│   └── download_dataset.py # Dataset download utility
├── tests/                  # Unit and integration tests
├── index.html              # Simple web UI
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Type checking
mypy agent/

# Linting
ruff check agent/

# Formatting
black agent/
```

## Performance Considerations

- **Query Optimization**: Agent caches dataset schema and statistics
- **Parallel Processing**: Multiple analyses run concurrently where possible
- **Intelligent Sampling**: Large datasets are sampled for exploratory analysis
- **Result Caching**: Repeated queries use cached results when appropriate

## Troubleshooting

### Common Issues

**"Dataset not found"**
- Ensure dataset is in `./data/` directory
- Check file permissions
- Verify file is not corrupted

**"API key not configured"**
- Set environment variable: `OPENAI_API_KEY`
- Check `.env` file is in project root
- Verify key is valid and has sufficient credits

**"Out of memory errors"**
- Try with a smaller dataset sample
- Increase system memory allocation
- Use `--sample-size` flag to limit rows processed

## Contributing

This is a take-home assignment project. Contributions are not currently being accepted.

## License

This project is created as part of a job application for SynMax.

## Contact

[Your Name]
[Your Email]
[Your GitHub Profile]
