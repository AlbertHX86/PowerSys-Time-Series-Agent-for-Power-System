"""
Custom Model Generator - LLM-powered model code generation from natural language
"""
import re
import traceback
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def get_code_generation_llm():
    """Get LLM for code generation (lazy loading)"""
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


def generate_custom_model_code(user_description: str, data_context: dict) -> Dict[str, Any]:
    """
    Generate custom model code from natural language description
    
    Args:
        user_description: User's natural language model description
        data_context: Context about the data (type, resolution, etc.)
    
    Returns:
        dict with 'code', 'model_name', 'explanation', 'success'
    """
    
    llm = get_code_generation_llm()
    
    prompt = f"""You are an expert machine learning engineer specializing in time series forecasting for renewable energy.

**User's Model Request:**
"{user_description}"

**Data Context:**
- Data Type: {data_context.get('description', 'energy forecasting')}
- Resolution: {data_context.get('resolution', '10 minutes')}
- Goal: {data_context.get('goal', 'forecasting')}
- Capacity: {data_context.get('capacity', 'unknown')}

**Task:**
Generate Python code for a custom forecasting model based on the user's description.

**CRITICAL REQUIREMENTS:**
1. ⚠️ **DO NOT CREATE OR LOAD DATA** - The data is already prepared!
   - X_train, X_test, y_train, y_test are ALREADY PROVIDED
   - DO NOT use pd.read_csv() or create example data with np.random
   - DO NOT create dummy DataFrames
   - Just focus on creating the model object

2. The code will be executed like this:
   ```python
   exec(code, globals_dict)
   model = globals_dict['model']  # Must have a variable named 'model'
   model.fit(X_train, y_train)    # X_train and y_train are ALREADY available
   predictions = model.predict(X_test)
   ```

3. You MUST create a variable named 'model' (NOT 'custom_model'):
   ✓ CORRECT: model = LinearRegression()
   ✗ WRONG: custom_model = LinearRegression()

4. Keep it SIMPLE - use sklearn models directly unless custom class is truly needed:
   ✓ GOOD: from sklearn.linear_model import LinearRegression; model = LinearRegression()
   ✗ BAD: Creating complex wrapper classes unnecessarily

5. The model must have fit() and predict() methods (sklearn API)

**Recommended Approach for Simple Models:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create model with preprocessing
# Note: X_train, y_train are already available - no need to create data
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
```

**For Ensemble Models:**
```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Data is already prepared - just create the model
model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', XGBRegressor(n_estimators=100))
    ],
    final_estimator=Ridge()
)
```

**Important Guidelines:**
- ⚠️ **NEVER create or load data** - X_train, X_test, y_train, y_test are ALREADY prepared
- ⚠️ **NEVER use pd.DataFrame with random data** - no fake data creation
- ⚠️ **NEVER use pd.read_csv()** - data is already loaded
- If user wants "ensemble", combine multiple models
- If user wants "neural network" or "LSTM", use keras/tensorflow
- If user wants "gradient boosting" variants, use XGBoost/LightGBM/CatBoost
- If user mentions "feature selection", include SelectKBest or similar
- If user wants "stacking", use StackingRegressor
- Keep it simple and efficient
- Add comments explaining key decisions
- Focus ONLY on creating the 'model' object

**Output Format:**
Return ONLY valid Python code wrapped in ```python ``` markers. No explanations outside the code.
The code must be executable and self-contained.
DO NOT include any data loading, splitting, or example data creation - ONLY the model definition.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        # Extract code from markdown
        code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # Try without python marker
            code_match = re.search(r'```\n(.*?)\n```', generated_code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = generated_code
        
        # Extract model name from code
        model_name_match = re.search(r'# Custom Model: (.+)', code)
        model_name = model_name_match.group(1).strip() if model_name_match else "CustomModel"
        
        # Extract description
        desc_match = re.search(r'# Description: (.+)', code)
        description = desc_match.group(1).strip() if desc_match else user_description
        
        return {
            'success': True,
            'code': code,
            'model_name': model_name,
            'description': description,
            'raw_response': generated_code
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def execute_custom_model_code(code: str, X_train, y_train, X_test) -> Dict[str, Any]:
    """
    Safely execute generated model code and return trained model + predictions
    
    Args:
        code: Generated Python code
        X_train, y_train: Training data
        X_test: Test data
    
    Returns:
        dict with 'model', 'predictions', 'success', 'error'
    """
    import pandas as pd  # Import pandas at function level to use immediately
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import re
    
    # Clean up the generated code to remove sample data initialization
    # This prevents the code from overwriting the actual X_train, y_train, X_test
    code = _clean_generated_code(code)
    
    # Ensure X_train and X_test have DatetimeIndex if they don't already
    # This is needed for time-of-day feature extraction
    if not isinstance(X_train.index, pd.DatetimeIndex):
        try:
            # Try to convert index to DatetimeIndex
            X_train = X_train.copy()
            X_train.index = pd.to_datetime(X_train.index)
        except:
            # If conversion fails, create a default datetime index
            X_train = X_train.copy()
            X_train.index = pd.date_range(start='2020-01-01', periods=len(X_train), freq='H')
    
    if not isinstance(X_test.index, pd.DatetimeIndex):
        try:
            # Try to convert index to DatetimeIndex
            X_test = X_test.copy()
            X_test.index = pd.to_datetime(X_test.index)
        except:
            # If conversion fails, create a default datetime index
            X_test = X_test.copy()
            X_test.index = pd.date_range(start=X_train.index[-1], periods=len(X_test), freq='H')
    
    # Create safe execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'np': np,
        'pd': pd,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'Pipeline': Pipeline,
        'StandardScaler': StandardScaler
    }
    
    # Import commonly used libraries
    try:
        exec_globals['sklearn'] = __import__('sklearn')
        exec_globals['BaseEstimator'] = __import__('sklearn.base').base.BaseEstimator
        exec_globals['RegressorMixin'] = __import__('sklearn.base').base.RegressorMixin
        exec_globals['RandomForestRegressor'] = __import__('sklearn.ensemble').ensemble.RandomForestRegressor
        exec_globals['GradientBoostingRegressor'] = __import__('sklearn.ensemble').ensemble.GradientBoostingRegressor
        exec_globals['StackingRegressor'] = __import__('sklearn.ensemble').ensemble.StackingRegressor
        exec_globals['LinearRegression'] = __import__('sklearn.linear_model').linear_model.LinearRegression
        exec_globals['Ridge'] = __import__('sklearn.linear_model').linear_model.Ridge
        exec_globals['Lasso'] = __import__('sklearn.linear_model').linear_model.Lasso
    except ImportError:
        pass
    
    # Try to import optional libraries
    try:
        exec_globals['xgb'] = __import__('xgboost')
        exec_globals['XGBRegressor'] = __import__('xgboost').XGBRegressor
    except ImportError:
        pass
    
    try:
        exec_globals['lgb'] = __import__('lightgbm')
        exec_globals['LGBMRegressor'] = __import__('lightgbm').LGBMRegressor
    except ImportError:
        pass
    
    try:
        exec_globals['catboost'] = __import__('catboost')
        exec_globals['CatBoostRegressor'] = __import__('catboost').CatBoostRegressor
    except ImportError:
        pass
    
    try:
        # Execute the generated code
        exec(code, exec_globals)
        
        # Get the model instance (check both 'model' and 'custom_model' for backward compatibility)
        if 'model' not in exec_globals and 'custom_model' not in exec_globals:
            raise ValueError("Generated code must create a 'model' variable (e.g., model = LinearRegression())")
        
        model = exec_globals.get('model', exec_globals.get('custom_model'))
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        return {
            'success': True,
            'model': model,
            'predictions': predictions
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def _clean_generated_code(code: str) -> str:
    """
    Clean up generated code to remove problematic patterns:
    - Remove data loading/creation sections that would override X_train, y_train, X_test
    - Remove invalid early_stopping_rounds parameters
    - Remove train_test_split that would override provided data
    """
    lines = code.split('\n')
    cleaned_lines = []
    skip_until_empty_line = False
    
    for i, line in enumerate(lines):
        # Skip lines that create example/dummy data
        if any(x in line for x in [
            'np.random.seed', 'np.random.rand', 'np.random.randn',
            'X = np.random.', 'y = np.random.',
            'train_test_split', 'X_train, X_test',
            '# Example data', '# Assuming X and y',
            '# For demonstration'
        ]):
            skip_until_empty_line = True
            continue
        
        # Stop skipping at empty line
        if skip_until_empty_line and line.strip() == '':
            skip_until_empty_line = False
            continue
        
        # Skip if we're in the middle of skipping
        if skip_until_empty_line:
            continue
        
        # Remove invalid early_stopping_rounds from fit() call
        if 'early_stopping_rounds' in line and '.fit(' in line:
            # This parameter should not be in fit() for sklearn-like models
            line = line.replace(', early_stopping_rounds=10', '')
            line = line.replace(', early_stopping_rounds=5', '')
            line = line.replace('early_stopping_rounds=10, ', '')
            line = line.replace('early_stopping_rounds=5, ', '')
        
        # Remove eval_set from fit() if not using XGBoost with callbacks
        if 'eval_set=' in line and '.fit(' in line:
            # Check if this is a plain sklearn model (not XGBoost with callbacks)
            if 'callbacks=' not in line and 'eval_metric=' not in line:
                line = re.sub(r',?\s*eval_set=\[\([^)]+\)\]', '', line)
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def validate_custom_model(model, X_sample) -> tuple[bool, str]:
    """Validate that model has required methods and can make predictions"""
    try:
        # Check for required methods
        if not hasattr(model, 'fit'):
            return False, "Model missing 'fit' method"
        if not hasattr(model, 'predict'):
            return False, "Model missing 'predict' method"
        
        # Try a prediction
        pred = model.predict(X_sample[:5])
        
        # Check output shape
        if len(pred) != 5:
            return False, f"Prediction shape mismatch: expected 5, got {len(pred)}"
        
        return True, "Model validated successfully"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def detect_and_install_missing_packages(error_message: str) -> tuple[bool, str]:
    """Detect missing packages from error message and attempt installation
    
    Args:
        error_message: Error message from code execution
        
    Returns:
        tuple: (installed_successfully, message)
    """
    import subprocess
    import sys
    
    # Common package name mappings
    package_mappings = {
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'tensorflow': 'tensorflow',
        'keras': 'tensorflow',  # keras is now part of tensorflow
        'torch': 'torch',
        'prophet': 'prophet',
        'statsmodels': 'statsmodels'
    }
    
    # Detect missing module from error message
    missing_module = None
    
    if "No module named" in error_message:
        # Extract module name from "No module named 'xxx'"
        import re
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        if match:
            missing_module = match.group(1).split('.')[0]  # Get top-level package
    
    if not missing_module:
        return False, "No missing package detected"
    
    # Check if it's a known package
    if missing_module not in package_mappings:
        return False, f"Unknown package '{missing_module}' - cannot auto-install"
    
    package_to_install = package_mappings[missing_module]
    
    try:
        print(f"\n📦 Detected missing package: {missing_module}")
        print(f"   Attempting to install {package_to_install}...")
        
        # Install package using pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_to_install],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Successfully installed {package_to_install}")
            return True, f"Installed {package_to_install}"
        else:
            print(f"   ❌ Installation failed: {result.stderr[:200]}")
            return False, f"Failed to install {package_to_install}: {result.stderr[:200]}"
    
    except subprocess.TimeoutExpired:
        print(f"   ❌ Installation timeout")
        return False, f"Installation of {package_to_install} timed out"
    except Exception as e:
        print(f"   ❌ Installation error: {e}")
        return False, f"Installation error: {str(e)}"


def reflect_and_regenerate(user_description: str, data_context: dict, 
                           previous_code: str, error_message: str, attempt: int) -> dict:
    """
    Reflect on error and regenerate improved code
    
    Args:
        user_description: Original user request
        data_context: Data context information
        previous_code: Code that failed
        error_message: Error message from failure
        attempt: Current attempt number
    
    Returns:
        dict with keys: success, code, error, reflection
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config.settings import CODE_GENERATION_MODEL, CODE_GENERATION_TEMPERATURE
    
    # Try to auto-install missing packages
    installation_note = ""
    if "No module named" in error_message:
        installed, install_msg = detect_and_install_missing_packages(error_message)
        if installed:
            installation_note = f"\n\nNOTE: Package was just installed: {install_msg}. You can now use it in the corrected code."
        else:
            installation_note = f"\n\nNOTE: Could not install missing package ({install_msg}). Please use only available packages: sklearn, numpy, pandas, xgboost (if installed)."
    
    reflection_prompt = f"""You are a Python ML engineer debugging code generation errors.

ORIGINAL REQUEST: {user_description}

DATA CONTEXT:
- Description: {data_context.get('description', 'time series data')}
- Resolution: {data_context.get('resolution', 'unknown')}
- Goal: {data_context.get('goal', 'forecasting')}

PREVIOUS CODE (ATTEMPT {attempt}):
```python
{previous_code}
```

ERROR ENCOUNTERED:
{error_message}{installation_note}

TASK: Analyze the error and generate CORRECTED code.

CRITICAL REQUIREMENTS:
1. The code will be executed in this environment:
   ```python
   exec(code, globals_dict)
   model = globals_dict['model']  # Must have a variable named 'model'
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

2. You MUST create a variable named 'model' (not 'custom_model'):
   ✓ CORRECT: model = LinearRegression()
   ✗ WRONG: custom_model = LinearRegression()

3. Keep it SIMPLE - use sklearn models directly:
   ✓ GOOD: from sklearn.linear_model import LinearRegression; model = LinearRegression()
   ✗ BAD: Creating custom wrapper classes (unless absolutely necessary)

4. Common sklearn models for regression:
   - LinearRegression (simple baseline)
   - Ridge, Lasso (regularized linear)
   - RandomForestRegressor (ensemble)
   - GradientBoostingRegressor (boosting)
   - XGBRegressor (from xgboost)
   - LGBMRegressor (from lightgbm)

5. The model must have fit() and predict() methods (sklearn API)

GENERATE CORRECTED CODE (respond with ONLY Python code in markdown):
"""
    
    try:
        llm = ChatOpenAI(model=CODE_GENERATION_MODEL, temperature=CODE_GENERATION_TEMPERATURE)
        response = llm.invoke([HumanMessage(content=reflection_prompt)])
        code = response.content
        
        # Extract code from markdown if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        return {
            'success': True,
            'code': code,
            'reflection': f"Attempt {attempt}: Analyzed error '{error_message[:100]}...' and regenerated code"
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Reflection failed: {str(e)}"
        }

