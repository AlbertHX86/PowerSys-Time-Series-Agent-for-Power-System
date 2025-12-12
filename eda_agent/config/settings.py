"""
Configuration settings for EDA Agent
"""
import os

# ============================================================================
# API KEYS (Set your own keys here or via environment variables)
# ============================================================================

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "cee322-eda")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# PATHS
# ============================================================================

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "..", "checkpoints")

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# LLM models - you can switch to gpt-4o-mini/gpt-4o if available
REFLECTION_MODEL = "gpt-4o-mini"  # Lightweight model for quality assessment
VISION_MODEL = "gpt-4o"           # Vision-capable model for image analysis
REPORTING_MODEL = "gpt-4o-mini"   # Model for report generation

# Model parameters
REFLECTION_TEMPERATURE = 0.3
VISION_TEMPERATURE = 0.3
REPORTING_TEMPERATURE = 0.0

# Code generation (for custom models)
CODE_GENERATION_MODEL = "gpt-4o"
CODE_GENERATION_TEMPERATURE = 0.3

# ============================================================================
# WORKFLOW PARAMETERS
# ============================================================================

# Maximum iterations for revision loops
MAX_ITERATIONS = 50
MAX_DATA_REVISIONS = 1
MAX_ANALYSIS_REVISIONS = 1
MAX_FORECAST_REVISIONS = 2

# Missing data threshold (%)
MISSING_DATA_THRESHOLD = 10.0

# Time series requirements
MIN_TIMESPAN_DAYS = 7.0
MIN_ROWS_FOR_ANALYSIS = 50

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure DPI
FIGURE_DPI = 100

# Display points for time series plots
MAX_DISPLAY_POINTS = 200

# Color schemes
COLORS = {
    'primary': '#8ECFC9',
    'training': '#95B8D1',
    'actual': '#2E86AB',
    'rf': '#00B894',
    'dt': '#FF6B6B',
    'split': 'red'
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Lag features
LAG_PERIODS = [1, 2, 24, 168, 336]

# Rolling window sizes
ROLLING_WINDOWS = [3, 6, 12, 24, 168]

# Fourier components
FOURIER_DAILY_K = 3
FOURIER_WEEKLY_K = 2

# ============================================================================
# FORECASTING
# ============================================================================

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_RANDOM_STATE = 42

# Decision Tree parameters
DT_MAX_DEPTH = 15
DT_RANDOM_STATE = 42

# Train/test split
TEST_PERIOD_DAYS = 1  # Use last N days for testing


def setup_environment():
    """Set up environment variables for API keys"""
    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    if LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    if LANGSMITH_PROJECT:
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    elif "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not set. Please set it in config/settings.py or as environment variable.")


# Auto-setup on import
setup_environment()
