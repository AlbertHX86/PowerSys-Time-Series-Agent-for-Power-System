"""
Feature engineering tool
"""
import pandas as pd
import numpy as np
from ..state import EDAState, memory_update


def engineer_features(state: EDAState) -> dict:
    """
    Create time-based and lag features for time series forecasting
    Only applies if data is sparse or has low correlation with target
    """
    step_name = "feature_eng"
    data = state.get("preprocessed_data")
    data_profile = state.get("data_profile", {})

    if data is None or state.get("stop_processing"):
        out = {"engineered_data": data, "feature_engineering_applied": False}
        out.update(memory_update(step_name, "Feature engineering skipped: no preprocessed data"))
        return out

    time_col = data_profile.get('timeseries', {}).get('time_column')
    if time_col is None:
        out = {"engineered_data": data, "feature_engineering_applied": False}
        out.update(memory_update(step_name, "Feature engineering skipped: no time column"))
        return out

    df = data.copy()

    # Ensure parsed time column exists
    if '_parsed_time' not in df.columns:
        df['_parsed_time'] = pd.to_datetime(df[time_col], utc=True, errors='coerce')

    time_idx = df['_parsed_time']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Identify target column
    target_col = None
    for col in numeric_cols:
        if 'power' in col.lower() or 'generation' in col.lower() or 'mw' in col.lower():
            target_col = col
            break
    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]
    
    if target_col is None:
        out = {"engineered_data": df, "feature_engineering_applied": False}
        out.update(memory_update(step_name, "Feature engineering skipped: no numeric target"))
        return out

    feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('_')]
    needs_engineering = False
    engineering_reason = ""

    # Decide if feature engineering is needed
    if len(feature_cols) <= 2:
        needs_engineering = True
        engineering_reason = f"Sparse input data ({len(feature_cols)+1} numeric features). Performing comprehensive feature engineering."
    else:
        correlations = df[feature_cols + [target_col]].corr()[target_col]
        relevant_features = (correlations.abs() > 0.3).sum() - 1
        if relevant_features < max(2, len(feature_cols) // 3):
            needs_engineering = True
            engineering_reason = f"Low correlation with target ({relevant_features}/{len(feature_cols)} features |corr|>0.3). Performing feature engineering."
        else:
            needs_engineering = False
            engineering_reason = f"Sufficient relevant features ({relevant_features}/{len(feature_cols)} with |corr|>0.3). Skipping feature engineering."

    if not needs_engineering:
        out = {
            "engineered_data": df,
            "feature_engineering_applied": False,
            "feature_engineering_reason": engineering_reason
        }
        out.update(memory_update(step_name, f"FE skipped: {engineering_reason}"))
        return out

    # Create time-based features
    df['hour'] = time_idx.dt.hour
    df['day'] = time_idx.dt.day
    df['weekday'] = time_idx.dt.dayofweek
    df['month'] = time_idx.dt.month
    df['year'] = time_idx.dt.year
    df['season'] = df['month'].apply(lambda x: (x-1)//3 + 1)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['year_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['year_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    lags = [1, 2, 24, 168, 336]
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Rolling window features
    windows = [3, 6, 12, 24, 168]
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()

    # Difference features
    df[f'{target_col}_roc'] = df[target_col].pct_change()
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_24'] = df[target_col].diff(24)
    df[f'{target_col}_cumsum_normalized'] = (
        (df[target_col].cumsum() - df[target_col].cumsum().min()) / 
        (df[target_col].cumsum().max() - df[target_col].cumsum().min() + 1e-8)
    )

    # Fourier features
    for k in range(1, 4):
        df[f'{target_col}_fourier_daily_sin_{k}'] = np.sin(2 * np.pi * k * df['hour'] / 24)
        df[f'{target_col}_fourier_daily_cos_{k}'] = np.cos(2 * np.pi * k * df['hour'] / 24)
    
    for k in range(1, 3):
        df[f'{target_col}_fourier_weekly_sin_{k}'] = np.sin(2 * np.pi * k * df['weekday'] / 7)
        df[f'{target_col}_fourier_weekly_cos_{k}'] = np.cos(2 * np.pi * k * df['weekday'] / 7)

    # Drop NaN rows created by lag/rolling features
    df = df.dropna()

    out = {
        "engineered_data": df,
        "feature_engineering_applied": True,
        "feature_engineering_reason": engineering_reason
    }
    
    msg = f"Feature engineering applied: {engineering_reason}; final cols={df.shape[1]}"
    out.update(memory_update(step_name, msg))
    return out
