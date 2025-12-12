"""
Data analysis tool
"""
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from ..state import EDAState, memory_update


def _safe_date_parse(v):
    """Safe date parsing with fallback"""
    try:
        return dateparser.parse(str(v))
    except Exception:
        return pd.NaT


def analyze_data(state: EDAState) -> dict:
    """Comprehensive data analysis including statistics, profiling, and time series detection"""
    step_name = "analyze"
    data = state.get("data")
    
    if data is None:
        return memory_update(
            step_name, 
            "No data available for analysis",
            errors=state.get("errors", []) + ["Analyze: no data"],
            needs_revision=True
        )

    df = data.copy()
    
    # Detect time-like column
    time_col = None
    candidate_cols = []
    
    # Check column names for time-related keywords
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ("time", "date", "datetime", "ts")):
            candidate_cols.append(col)
    
    # Check for parseable date strings
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in candidate_cols:
            sample = df[col].dropna().astype(str).head(10).tolist()
            parsed = 0
            for v in sample:
                try:
                    _ = dateparser.parse(v)
                    parsed += 1
                except Exception:
                    break
            if parsed >= min(3, len(sample)) and len(sample) > 0:
                candidate_cols.append(col)
    
    if len(candidate_cols) > 0:
        candidate_cols.sort(
            key=lambda c: (0 if any(k in c.lower() for k in ("time", "date", "datetime", "ts")) else 1)
        )
        time_col = candidate_cols[0]

    # Parse time column if detected
    parsed_time_col = None
    if time_col:
        try:
            parsed = pd.to_datetime(df[time_col], utc=True, errors='coerce')
        except Exception:
            parsed = pd.to_datetime(df[time_col].astype(str), utc=True, errors='coerce')
        
        if parsed.isna().sum() > 0.5 * len(parsed):
            parsed = df[time_col].apply(lambda v: _safe_date_parse(v))
            parsed = pd.to_datetime(parsed, utc=True, errors='coerce')
        
        parsed_time_col = parsed
        df['_parsed_time'] = parsed_time_col
        df = df.sort_values('_parsed_time', ascending=True).reset_index(drop=True)

    # Compute summary statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary_stats = {}
    
    for col in numeric_cols:
        summary_stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "count": int(df[col].count())
        }

    # Build data profile
    data_profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_details": []
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percent": float(round(df[col].isnull().sum() / len(df) * 100, 2)),
            "non_null_count": int(df[col].count())
        }
        
        if col in numeric_cols:
            col_info.update({
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75))
            })
        else:
            col_info.update({
                "unique_count": int(df[col].nunique()),
                "top_value": str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None
            })
        
        data_profile["column_details"].append(col_info)

    # Missing data summary
    data_profile["missing_data_summary"] = {
        "columns_with_missing": sum(1 for col_info in data_profile["column_details"] if col_info["missing_count"] > 0),
        "total_missing_cells": int(df.isnull().sum().sum()),
        "total_cells": int(len(df) * len(df.columns)),
        "missing_percent": float(round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2))
    }

    # Time series analysis
    timespan_days = None
    enough_history = True
    
    if parsed_time_col is not None:
        times = pd.to_datetime(df['_parsed_time'], utc=True)
        if times.dropna().shape[0] >= 2:
            timespan = times.max() - times.min()
            timespan_days = float(timespan / pd.Timedelta(days=1))
            enough_history = timespan_days >= 7.0
        else:
            timespan_days = 0.0
            enough_history = False
    else:
        enough_history = False

    data_profile['timeseries'] = {
        'time_column': time_col,
        'timespan_days': timespan_days,
        'enough_history': enough_history
    }

    # Identify target column
    target_col = None
    for col in numeric_cols:
        if 'power' in col.lower() or 'generation' in col.lower() or 'mw' in col.lower():
            target_col = col
            break
    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[0]
    
    # Add target column to data_profile
    data_profile['target_column'] = target_col

    # Build output
    missing_pct = data_profile.get('missing_data_summary', {}).get('missing_percent', 0)
    num_vars = len(numeric_cols)
    
    out = {
        "summary_stats": summary_stats,
        "data_profile": data_profile,
        "data": df,
        "missing_percentage": missing_pct,
        "num_variables": num_vars,
        "target_column": target_col,
        "stop_processing": False,
        "errors": state.get("errors", [])
    }

    timespan_days_val = data_profile.get('timeseries', {}).get('timespan_days', 0)
    if timespan_days_val is not None and timespan_days_val < 7.0:
        out["errors"] = out["errors"] + [
            f"Insufficient time series data: only {timespan_days_val:.1f} days (need >= 7 days for forecasting)"
        ]
        out["stop_processing"] = True

    msg_summary = (
        f"Analysis complete: {len(df)} rows, {num_vars} numeric vars, "
        f"missing {missing_pct:.2f}%, target={target_col}, timespan_days={timespan_days_val}"
        + (" (insufficient history)" if out.get("stop_processing") else "")
    )
    
    out.update(
        memory_update(
            step_name, 
            msg_summary,
            errors=out.get("errors", []),
            needs_revision=out.get("stop_processing", False)
        )
    )
    
    return out
