"""
Data preprocessing tool
"""
from ..state import EDAState, memory_update


def preprocess_data(state: EDAState) -> dict:
    """Preprocess data: handle missing values via interpolation"""
    step_name = "preprocess"
    
    # Early exit if stop_processing flag is set
    if state.get("stop_processing", False):
        out = {
            "preprocessed_data": None,
            "stop_processing": True,
            "errors": state.get("errors", [])
        }
        out.update(
            memory_update(
                step_name,
                "Preprocessing skipped: stop_processing flag is set",
                errors=out["errors"]
            )
        )
        return out
    
    data = state.get("data")
    data_profile = state.get("data_profile", {})
    
    if data is None:
        return memory_update(
            step_name,
            "No data for preprocessing",
            errors=state.get("errors", []) + ["Preprocess: no data"],
            needs_revision=True
        )

    missing_pct = data_profile.get('missing_data_summary', {}).get('missing_percent', 0)
    num_vars = len(data.select_dtypes(include=['number']).columns)

    # Stop if too much missing data
    if missing_pct > 10:
        out = {
            "preprocessed_data": None,
            "stop_processing": True,
            "missing_percentage": missing_pct,
            "num_variables": num_vars,
            "errors": state.get("errors", []) + [
                f"Data contains {missing_pct}% missing values (>10%). Processing stopped."
            ]
        }
        out.update(
            memory_update(
                step_name,
                f"Preprocessing halted: missing {missing_pct:.2f}% exceeds threshold",
                errors=out["errors"],
                needs_revision=True
            )
        )
        return out

    # Apply forward-fill and backward-fill for missing values
    preprocessed = data.copy()
    if missing_pct > 0:
        numeric_cols = preprocessed.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if preprocessed[col].isnull().sum() > 0:
                preprocessed[col] = preprocessed[col].ffill()
                preprocessed[col] = preprocessed[col].bfill()
    
    # Remove constant columns (zero variance)
    numeric_cols = preprocessed.select_dtypes(include=['number']).columns
    constant_cols = []
    for col in numeric_cols:
        if preprocessed[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        preprocessed = preprocessed.drop(columns=constant_cols)

    out = {
        "preprocessed_data": preprocessed,
        "stop_processing": False,
        "missing_percentage": missing_pct,
        "num_variables": num_vars,
        "constant_columns_removed": constant_cols
    }
    
    msg = (
        f"Preprocessing complete: missing {missing_pct:.2f}% handled"
        if missing_pct > 0 else "Preprocessing complete: no missing values"
    )
    
    out.update(memory_update(step_name, msg))
    return out
