"""
Data validation tool
"""
from ..state import EDAState, memory_update


def validate_data(state: EDAState) -> dict:
    """Comprehensive data validation"""
    step_name = "validate"
    errors = state.get("errors", [])
    data = state.get("data")

    if data is None:
        out = {
            "errors": errors + ["CRITICAL: Data failed to load"],
            "stop_processing": True
        }
        out.update(
            memory_update(
                step_name,
                "Validation failed: no data",
                errors=out["errors"],
                needs_revision=True
            )
        )
        return out

    if len(data) == 0:
        out = {
            "errors": errors + ["CRITICAL: Dataset is empty"],
            "stop_processing": True
        }
        out.update(
            memory_update(
                step_name,
                "Validation failed: empty dataset",
                errors=out["errors"],
                needs_revision=True
            )
        )
        return out

    if len(data) < 50:
        errors.append(
            f"WARNING: Dataset has only {len(data)} rows. "
            f"Minimum 50 rows recommended for robust analysis"
        )

    # Check missing data
    missing_counts = data.isnull().sum()
    missing_pct = (missing_counts / len(data) * 100).round(2)
    missing_info = {
        col: {"count": int(missing_counts[col]), "percent": float(missing_pct[col])}
        for col in data.columns if missing_counts[col] > 0
    }

    total_missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
    if total_missing_pct == 100:
        out = {
            "errors": errors + ["CRITICAL: All data is missing"],
            "stop_processing": True
        }
        out.update(
            memory_update(
                step_name,
                "Validation failed: all data missing",
                errors=out["errors"],
                needs_revision=True
            )
        )
        return out

    # Check for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        out = {
            "errors": errors + ["CRITICAL: No numeric columns found for analysis"],
            "stop_processing": True
        }
        out.update(
            memory_update(
                step_name,
                "Validation failed: no numeric columns",
                errors=out["errors"],
                needs_revision=True
            )
        )
        return out

    # Check for duplicate column names
    if len(data.columns) != len(set(data.columns)):
        errors.append("WARNING: Duplicate column names detected")

    out = {
        "missing_info": missing_info,
        "errors": errors,
        "stop_processing": False
    }
    
    msg = (
        f"Validation passed: rows={len(data)}, numeric_cols={len(numeric_cols)}, "
        f"missing_cols={len(missing_info)}"
    )
    out.update(memory_update(step_name, msg, errors=errors))
    return out


def llm_validate_data_sufficiency(state: EDAState) -> dict:
    """LLM-based validation for data sufficiency (critical gate)"""
    step_name = "llm_validate"
    
    data_profile = state.get('data_profile', {})
    timeseries_info = data_profile.get('timeseries', {})
    timespan_days = timeseries_info.get('timespan_days', 0)
    time_column = timeseries_info.get('time_column')
    total_rows = data_profile.get('total_rows', 0)

    # CRITICAL: Direct check for data sufficiency
    # If less than 7 days, immediately set stop_processing and skip all further analysis
    if timespan_days < 7:
        error_msg = (
            f"Insufficient time series data: only {timespan_days:.1f} days "
            f"(need >= 7 days for forecasting)"
        )
        errors = state.get('errors', []) + [error_msg]
        out = {
            "stop_processing": True,
            "errors": errors,
            "llm_validation": "insufficient",
            "guardrail_passed": False,
            "guardrail_warning": error_msg
        }
        out.update(
            memory_update(
                step_name,
                f"Data insufficient ({timespan_days:.1f} days < 7 days required)",
                errors=errors,
                needs_revision=True
            )
        )
        return out

    # If sufficient data, optionally use LLM for validation commentary
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        validation_context = f"""
Dataset Information:
- Total rows: {total_rows}
- Time column: {time_column}
- Timespan: {timespan_days:.2f} days
- Minimum required for forecasting: 7 days

Determine if this dataset has sufficient history for time series forecasting.

Respond concisely with a description of the situation and a brief suggestion

GUARDRAIL:
- do not makeup data, avoid hallucination
"""
        llm = ChatOpenAI(model=REPORTING_MODEL, temperature=REPORTING_TEMPERATURE)
        response = llm.invoke([HumanMessage(content=validation_context)])
        validation_result = response.content.strip()
    except Exception:
        validation_result = "sufficient"

    out = {
        "stop_processing": False,
        "llm_validation": validation_result
    }
    out.update(
        memory_update(
            step_name,
            f"LLM validate: sufficient ({timespan_days:.1f} days)"
        )
    )
    return out
