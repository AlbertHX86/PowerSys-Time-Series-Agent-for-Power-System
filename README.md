# PowerSys Time Series Agent for Power System

An intelligent AI-powered agent for exploratory data analysis (EDA) and time series forecasting, specifically designed for renewable energy and power system data. This system uses LangGraph workflows with LLM-powered nodes to automatically analyze data, generate insights, and build forecasting models.

## 🖥️ User Interface

![PowerSys Agent Interface](https://raw.githubusercontent.com/AlbertHX86/PowerSys-Time-Series-Agent-for-Power-System/main/docs/interface.png)

The web interface provides an intuitive 4-step workflow:

1. **Upload Data** - Drag & drop CSV, Excel, or TXT files
2. **Configure** - Set data description, time resolution, and system capacity
3. **Custom Model** - Optional: Describe your custom ML model in natural language
4. **Results** - View comprehensive analysis reports and forecasts

## Features

- **Automated Data Analysis**: Intelligent preprocessing, cleaning, and statistical analysis
- **Smart Visualization**: Automatic generation of relevant plots and charts
- **LLM-Powered Insights**: Uses GPT models for data validation and quality assessment
- **Time Series Forecasting**: Multiple model comparison (Random Forest, Decision Tree, Custom Models)
- **Custom Model Generation**: Natural language to code - describe your model and the agent generates it
- **Iterative Refinement**: Self-reflection and quality improvement loops
- **Physical Constraints Validation**: Guardrails for renewable energy systems (non-negative power, capacity limits)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlbertHX86/PowerSys-Time-Series-Agent-for-Power-System.git
cd PowerSys-Time-Series-Agent-for-Power-System
```

### Step 2: Install Dependencies

```bash
cd eda_agent
pip install -r requirements.txt
```

Required packages include:
- `langchain` and `langchain-openai` - LLM framework
- `langgraph` - Graph-based workflow orchestration
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning models
- `matplotlib`, `seaborn` - Visualization
- `flask` - API server (optional)

### Step 3: Set Up API Keys

You need to configure your API keys before running the agent. There are two methods:

#### Method 1: Edit Configuration File (Recommended)

Edit `eda_agent/config/settings.py`:

```python
# Add your API keys here
LANGSMITH_API_KEY = "your-langsmith-api-key"  # Optional, for tracing
OPENAI_API_KEY = "your-openai-api-key"        # Required
```

#### Method 2: Environment Variables

Set environment variables in your terminal:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional
```

### Step 4: Prepare Your Data

Place your CSV data files in the `data/` directory. The agent expects time series data with:
- A timestamp column (automatically detected)
- A target column for forecasting (e.g., power output)
- Optional: Additional feature columns

## Usage

### Option 1: Python Script (Recommended)

```python
from main import run_eda_agent

# Run the agent on your data
result = run_eda_agent(
    filepath="data/your_data.csv",
    user_request="Analyze this solar power generation data and forecast the next day"
)

# Access results
print(result['report'])
print(f"Forecast R²: {result['forecast_results']['rf']['r2']}")
```

### Option 2: API Server

Start the Flask API server:

```bash
cd eda_agent
python api_server.py
```

The API will be available at `http://localhost:5000`

**Example API Request:**

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "data/solar_data.csv",
    "user_request": "Analyze and forecast solar power generation"
  }'
```

### Option 3: Web Interface

1. Start the API server: `python api_server.py`
2. Open `eda_agent/web/index.html` in your browser
3. Upload your CSV file and enter your analysis request

### Option 4: Jupyter Notebook

Open and run `Huang Xiao Lab 7/EDA agent.ipynb` for an interactive walkthrough.

## System Architecture

### Workflow Nodes

The agent uses a graph-based workflow with the following key nodes:

#### 1. **Data Loading Node** (`load_data`)
- Reads CSV files and performs initial validation
- Detects timestamp columns automatically
- Identifies potential target columns for forecasting
- **Output**: Raw data, column information, errors (if any)

#### 2. **Data Preprocessing Node** (`preprocess_data`)
- Handles missing values (forward fill, interpolation)
- Converts timestamp columns to datetime format
- Sorts data chronologically
- Removes duplicate timestamps
- **Output**: Cleaned data, missing data statistics

#### 3. **Data Validation Node** (`llm_validate_data_sufficiency`)
- Uses LLM to assess data quality and sufficiency
- Checks minimum timespan requirements (7+ days for forecasting)
- Validates row count and data completeness
- **Output**: Validation result, warnings, stop signals for insufficient data

#### 4. **Data Analysis Node** (`analyze_data`)
- Computes summary statistics (mean, std, min, max, quartiles)
- Identifies data type and resolution (e.g., 10-minute intervals)
- Extracts nominal capacity for power systems
- Time series suitability assessment
- **Output**: Statistical summaries, data profile, time series analysis

#### 5. **Reflection on Data Quality Node** (`reflect_on_data_quality`)
- LLM-powered quality assessment
- Evaluates completeness, consistency, and usability
- Generates quality score (0-1)
- Decides if data revision is needed
- **Output**: Quality score, critique, revision flag

#### 6. **Feature Engineering Node** (`feature_engineering_decision`)
- Decides whether advanced features are needed
- Creates lag features (1, 2, 24, 168, 336 periods)
- Generates rolling statistics (3, 6, 12, 24, 168 windows)
- Adds Fourier components for daily/weekly patterns
- Hour-of-day and day-of-week encoding
- **Output**: Engineered dataset with enhanced features

#### 7. **Visualization Node** (`visualize_data`)
- Generates time series plots with train/test split
- Creates correlation heatmaps
- Exports images as base64 for LLM analysis
- **Output**: Plot images, visualization metadata

#### 8. **Vision Analysis Node** (`llm_vision_analysis`)
- Uses GPT-4 Vision to analyze generated plots
- Identifies patterns, trends, anomalies
- Provides insights from visual data
- **Output**: Visual insights, pattern descriptions

#### 9. **Forecasting Node** (`forecast_time_series`)
- Trains multiple models:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Custom LLM-generated models (optional)
- Evaluates models using MAE, RMSE, R²
- Generates prediction vs actual plots
- **Output**: Model metrics, predictions, forecast images

#### 10. **Custom Model Generation Node** (`generate_custom_model`)
- Converts natural language description to Python code
- Generates scikit-learn compatible model code
- Validates and executes generated code
- **Output**: Custom model code, performance metrics

#### 11. **Reflection on Forecast Quality Node** (`reflect_on_forecast_quality`)
- LLM evaluates forecast performance
- Analyzes residual patterns
- Suggests code improvements for custom models
- Calculates overall quality score
- **Output**: Quality assessment, improvement suggestions

#### 12. **Guardrail Validation Node** (`validate_guardrails`)
- Enforces physical constraints:
  - Power output ≥ 0 (renewable energy)
  - Predictions ≤ nominal capacity
  - Metrics (MAE, RMSE, R²) validity
- Critical violations halt processing
- **Output**: Violation warnings, stop signals

#### 13. **Report Generation Node** (`generate_final_report`)
- Compiles comprehensive analysis report
- Uses LLM to generate natural language summaries
- Includes all visualizations and metrics
- Handles error cases gracefully
- **Output**: Final report with insights and recommendations

### Workflow Flow

```
Start → Load Data → Preprocess → Validate → Analyze
  ↓
Reflect on Data Quality → Feature Engineering Decision → Visualize
  ↓
Vision Analysis → Forecast → Custom Model (optional) → Reflect on Forecast
  ↓
Guardrail Validation → Generate Report → End
```

The workflow includes **conditional edges** that can:
- Skip steps if data is insufficient
- Retry preprocessing if quality is poor
- Regenerate forecasts if performance is low
- Stop immediately on critical violations

## Configuration

Edit `eda_agent/config/settings.py` to customize:

### Model Selection
```python
REFLECTION_MODEL = "gpt-4o-mini"     # For quality assessment
VISION_MODEL = "gpt-4o"              # For image analysis (vision-capable)
REPORTING_MODEL = "gpt-4o-mini"      # For report generation
CODE_GENERATION_MODEL = "gpt-4o"     # For custom model code
```

### Workflow Parameters
```python
MAX_ITERATIONS = 50                  # Maximum revision loops
MAX_DATA_REVISIONS = 1               # Data preprocessing retries
MAX_FORECAST_REVISIONS = 2           # Forecast regeneration attempts
MISSING_DATA_THRESHOLD = 10.0        # Max missing % allowed
MIN_TIMESPAN_DAYS = 7.0              # Minimum data history
```

### Feature Engineering
```python
LAG_PERIODS = [1, 2, 24, 168, 336]           # Lag features
ROLLING_WINDOWS = [3, 6, 12, 24, 168]        # Rolling stats
FOURIER_DAILY_K = 3                          # Daily seasonality
FOURIER_WEEKLY_K = 2                         # Weekly seasonality
```

### Forecasting Models
```python
RF_N_ESTIMATORS = 100                # Random Forest trees
RF_MAX_DEPTH = 15                    # Tree depth
TEST_PERIOD_DAYS = 1                 # Test set size
```

## Example Use Cases

### 1. Solar Power Forecasting
```python
result = run_eda_agent(
    filepath="data/solar_generation.csv",
    user_request="Analyze solar power data and forecast next 24 hours"
)
```

### 2. Wind Farm Analysis
```python
result = run_eda_agent(
    filepath="data/wind_turbine_data.csv",
    user_request="Evaluate wind turbine performance and predict output"
)
```

### 3. Custom Model Request
```python
result = run_eda_agent(
    filepath="data/power_data.csv",
    user_request="Build a gradient boosting model with hyperparameter tuning for power forecasting"
)
```

## Output Structure

The agent returns a dictionary containing:

```python
{
    'report': {
        'status': 'success',
        'data_profile': {...},           # Statistical summary
        'ts_analysis': {...},            # Time series characteristics
        'forecast_summary': "...",       # LLM-generated insights
        'visualizations': {...},         # Plot metadata
        'viz_images': {...},             # Base64 encoded images
        'forecast_results': {...},       # Model metrics
        'forecast_images': {...}         # Forecast plots
    },
    'forecast_results': {
        'rf': {                          # Random Forest results
            'r2': 0.89,
            'mae': 1.5,
            'rmse': 4.2,
            'predictions': [...]
        },
        'dt': {...},                     # Decision Tree results
        'custom': {...}                  # Custom model results (if any)
    }
}
```

## File Structure

```
PowerSys-Time-Series-Agent-for-Power-System/
├── eda_agent/                       # Main agent package
│   ├── main.py                      # Entry point
│   ├── api_server.py                # Flask API server
│   ├── requirements.txt             # Python dependencies
│   ├── config/
│   │   └── settings.py              # Configuration file
│   ├── src/
│   │   ├── state.py                 # State management
│   │   ├── nodes/                   # Workflow nodes
│   │   │   ├── reflection.py        # Quality assessment nodes
│   │   │   └── vision.py            # Vision analysis nodes
│   │   ├── tools/                   # Utility tools
│   │   │   ├── data_loader.py       # Data loading
│   │   │   ├── preprocessor.py      # Data cleaning
│   │   │   ├── data_analyzer.py     # Statistical analysis
│   │   │   ├── feature_engineer.py  # Feature creation
│   │   │   ├── visualizer.py        # Plotting
│   │   │   ├── forecaster.py        # Model training
│   │   │   ├── custom_model_generator.py  # LLM code gen
│   │   │   ├── validator.py         # Data validation
│   │   │   └── reporter.py          # Report generation
│   │   └── workflows/
│   │       └── reflective_workflow.py  # Graph definition
│   └── web/                         # Web interface
│       ├── index.html
│       ├── script.js
│       └── styles.css
├── data/                            # Input data directory
├── outputs/                         # Generated reports and images
├── checkpoints/                     # Workflow state saves
└── README.md                        # This file
```

## Troubleshooting

### API Key Errors
- Ensure `OPENAI_API_KEY` is set correctly
- Check that your API key has sufficient credits
- Verify the key has access to required models (gpt-4o, gpt-4o-mini)

### Insufficient Data Error
- The agent requires at least 7 days of time series data
- Ensure your data has a valid timestamp column
- Check for large gaps in your time series

### Memory Errors
- For large datasets, consider sampling or aggregating data
- Reduce `MAX_DISPLAY_POINTS` in settings.py
- Use a machine with more RAM

### Model Performance Issues
- Try enabling feature engineering for better results
- Adjust `MAX_FORECAST_REVISIONS` for more optimization attempts
- Provide specific model requirements in your user request

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this agent in your research, please cite:

```bibtex
@software{powersys_timeseries_agent,
  author = {Huang, Xiao Albert},
  title = {PowerSys Time Series Agent for Power System},
  year = {2025},
  url = {https://github.com/AlbertHX86/PowerSys-Time-Series-Agent-for-Power-System}
}
```

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

- Built with LangChain and LangGraph
- Powered by OpenAI GPT models
- Designed for renewable energy and power system applications
