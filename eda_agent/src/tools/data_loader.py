"""
Data loading tool
"""
import os
import pandas as pd
from datetime import datetime
from langchain_core.messages import AIMessage
from ..state import EDAState, memory_update


def load_data(state: EDAState) -> dict:
    """Load data from file with support for CSV, Excel, and TXT formats"""
    step_name = "load_data"
    try:
        filepath = state["filepath"]
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".csv":
            data = pd.read_csv(filepath)
        elif ext in [".xls", ".xlsx"]:
            data = pd.read_excel(filepath)
        elif ext == ".txt":
            data = pd.read_csv(filepath, delimiter="\t")
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        msg = AIMessage(
            content=f"Loaded {len(data)} rows, {len(data.columns)} columns from {os.path.basename(filepath)}"
        )
        
        return {
            "data": data,
            "messages": [msg],
            "steps_completed": [step_name],
            "current_step": step_name,
            "last_updated": datetime.now().isoformat(),
            "errors": state.get("errors", [])
        }
        
    except Exception as e:
        msg = AIMessage(content=f"Failed to load data: {str(e)}")
        return {
            "data": None,
            "messages": [msg],
            "steps_completed": [step_name],
            "current_step": step_name,
            "last_updated": datetime.now().isoformat(),
            "errors": state.get("errors", []) + [f"Load error: {str(e)}"],
            "stop_processing": True
        }
