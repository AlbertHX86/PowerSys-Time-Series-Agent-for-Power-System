"""
State definitions and reducers for EDA Agent
"""
from typing import TypedDict, Optional, Annotated
import pandas as pd
from datetime import datetime
from langgraph.graph import add_messages
from langchain_core.messages import AIMessage


def append_unique(current: list, update: list) -> list:
    """Append new items without duplicates"""
    if current is None:
        current = []
    if update is None:
        return current
    return current + [item for item in update if item not in current]


def merge_dicts(current: dict, update: dict) -> dict:
    """Deep merge dictionaries"""
    if current is None:
        current = {}
    if update is None:
        return current
    result = current.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def memory_update(step_name: str, content: str, *, errors=None, warnings=None, 
                  reflections=None, needs_revision: bool = False):
    """Return a dict containing standardized memory fields for a node step."""
    return {
        "messages": [AIMessage(content=content)],
        "steps_completed": [step_name],
        "current_step": step_name,
        "last_updated": datetime.now().isoformat(),
        "errors": errors or [],
        "warnings": warnings or [],
        "reflections": reflections or [],
        "needs_revision": needs_revision
    }


class EDAState(TypedDict):
    """Enhanced state with memory management and reflection support"""
    # Input
    filepath: str
    
    # Data
    data: any
    preprocessed_data: any
    engineered_data: any
    
    # Analysis results
    summary_stats: str
    missing_info: str
    data_profile: str
    visualizations: str
    viz_images: dict
    
    # Forecast results
    forecast_results: dict
    forecast_images: dict
    
    # Time series analysis
    ts_suitable: bool
    ts_method: str
    ts_analysis: str
    model_state: dict
    
    # Metadata
    errors: Annotated[list, append_unique]
    stop_processing: bool
    report: str
    target_column: str
    missing_percentage: float
    num_variables: int
    feature_engineering_applied: bool
    feature_engineering_reason: str
    
    # Memory fields
    messages: Annotated[list, add_messages]
    session_id: str
    session_start_time: str
    last_updated: str
    steps_completed: Annotated[list, append_unique]
    current_step: str
    iteration_count: int
    max_iterations: int
    warnings: Annotated[list, append_unique]
    
    # Reflection fields
    reflections: Annotated[list, append_unique]
    critique_history: Annotated[list, append_unique]
    needs_revision: bool
    revision_count: int
    quality_score: float
    data_quality_score: float
    analysis_quality_score: float
    forecast_quality_score: float
    checkpoint_tags: Annotated[list, append_unique]
    guardrail_passed: bool
    guardrail_warning: str
    
    # Vision analysis fields
    exported_images: dict
    export_directory: str
    vision_analysis: dict
    vision_summary: str
    vision_quality_passed: bool
    vision_quality_issues: Annotated[list, append_unique]
    
    # Domain context
    data_context: dict
    
    # Custom model info
    custom_model_info: dict  # Stores custom model request, generated code, and metrics
