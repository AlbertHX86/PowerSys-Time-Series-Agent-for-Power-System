"""
Reflective workflow configuration and builder
"""
from langgraph.graph import StateGraph
from ..state import EDAState
from ..tools import (
    load_data,
    validate_data,
    analyze_data,
    llm_validate_data_sufficiency,
    preprocess_data,
    engineer_features,
    visualize_data,
    analyze_ts_suitability,
    train_forecast_models,
    visualize_forecast,
    generate_report,
)
from ..nodes import (
    mark_checkpoint,
    reflect_on_data_quality,
    reflect_on_analysis_quality,
    reflect_on_forecast_quality,
    validate_guardrails,
    export_images_to_workspace,
    analyze_visualizations_with_vision,
    vision_quality_check
)


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def check_after_load(state):
    """Route after data loading"""
    if state.get("stop_processing", False) or state.get("data") is None:
        return "final_report"
    return "validate"


def check_after_validate(state):
    """Route after validation"""
    if state.get("stop_processing", False):
        return "final_report"
    return "analyze"


def check_after_llm_validate(state):
    """Critical gate: if data is insufficient, skip ALL processing"""
    if state.get("stop_processing", False):
        return "final_report"
    return "preprocess"


def check_after_data_reflection(state):
    """Check if data quality needs revision. Hard limit at 1 revision."""
    # EARLY EXIT: Skip if stop_processing is set
    if state.get("stop_processing", False):
        return "final_report"
    
    needs_revision = state.get("needs_revision", False)
    revision_count = state.get("revision_count", 0)
    max_data_revisions = 1
    
    if revision_count >= max_data_revisions:
        return "feature_eng"  # Hard exit after max revisions
    
    if needs_revision:
        return "preprocess"  # Retry
    
    return "feature_eng"  # Good quality, move on


def check_after_analysis_reflection(state):
    """Check if analysis quality needs revision. Hard limit at 1 revision."""
    # EARLY EXIT: Skip if stop_processing is set
    if state.get("stop_processing", False):
        return "final_report"
    
    needs_revision = state.get("needs_revision", False)
    revision_count = state.get("revision_count", 0)
    max_analysis_revisions = 1
    
    if revision_count >= max_analysis_revisions:
        return "visualize"  # Hard exit after max revisions
    
    if needs_revision:
        return "feature_eng"  # Retry
    
    return "visualize"  # Good quality, move on


def check_after_forecast_reflection(state):
    """
    Check if forecast quality needs revision. Hard limit at 2 revisions.
    This is the critical function that prevents infinite loops.
    """
    # EARLY EXIT: Skip if stop_processing is set
    if state.get("stop_processing", False):
        return "final_report"
    
    needs_revision = state.get("needs_revision", False)
    revision_count = state.get("revision_count", 0)
    quality_score = state.get("forecast_quality_score", 0.5)
    max_revisions = 2
    
    print(f"\n[check_after_forecast_reflection] Decision point:")
    print(f"  revision_count={revision_count} (max={max_revisions})")
    print(f"  quality_score={quality_score:.3f}")
    print(f"  needs_revision={needs_revision}")
    
    # CRITICAL: Hard stop after max revisions to prevent infinite loops
    if revision_count >= max_revisions:
        print(f"  → Max revisions reached. Exiting to final_report.")
        return "final_report"
    
    # Only retry if quality is still poor
    if needs_revision and quality_score < 0.4:
        print(f"  → Quality too low. Looping back to ts_train for revision {revision_count + 1}/{max_revisions}")
        return "ts_train"
    
    print(f"  → Quality acceptable or no revision needed. Exiting to final_report.")
    return "final_report"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_reflective_workflow(checkpointer=None):
    """
    Build the reflective workflow with all nodes and edges
    
    Args:
        checkpointer: Optional checkpointer for persistent memory
        
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(EDAState)
    
    # Add processing nodes
    workflow.add_node("load", load_data)
    workflow.add_node("validate", validate_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("llm_validate", llm_validate_data_sufficiency)
    workflow.add_node("preprocess", preprocess_data)
    workflow.add_node("feature_eng", engineer_features)
    workflow.add_node("visualize", visualize_data)
    workflow.add_node("ts_suitable", analyze_ts_suitability)
    workflow.add_node("ts_train", train_forecast_models)
    workflow.add_node("ts_visualize", visualize_forecast)
    workflow.add_node("final_report", generate_report)
    
    # Add checkpoint marker nodes
    workflow.add_node("cp_preprocess", lambda s: mark_checkpoint(s, "preprocess"))
    workflow.add_node("cp_feature", lambda s: mark_checkpoint(s, "feature_eng"))
    workflow.add_node("cp_forecast", lambda s: mark_checkpoint(s, "forecast"))
    
    # Add reflection nodes
    workflow.add_node("reflect_data", reflect_on_data_quality)
    workflow.add_node("reflect_analysis", reflect_on_analysis_quality)
    workflow.add_node("reflect_forecast", reflect_on_forecast_quality)
    
    # Add guardrail validation node
    workflow.add_node("guardrail", validate_guardrails)
    
    # Add vision analysis nodes
    workflow.add_node("export_images", export_images_to_workspace)
    workflow.add_node("vision_analysis", analyze_visualizations_with_vision)
    workflow.add_node("vision_check", vision_quality_check)
    
    # ========================================================================
    # BUILD EDGES: Phase 1 (load/validate/analyze/llm_gate)
    # ========================================================================
    workflow.add_conditional_edges(
        "load",
        check_after_load,
        {"validate": "validate", "final_report": "final_report"}
    )
    workflow.add_conditional_edges(
        "validate",
        check_after_validate,
        {"analyze": "analyze", "final_report": "final_report"}
    )
    workflow.add_edge("analyze", "llm_validate")
    workflow.add_conditional_edges(
        "llm_validate",
        check_after_llm_validate,
        {"preprocess": "preprocess", "final_report": "final_report"}
    )
    
    # ========================================================================
    # Phase 2: Preprocessing + checkpoint + reflection loop
    # ========================================================================
    workflow.add_edge("preprocess", "cp_preprocess")
    workflow.add_edge("cp_preprocess", "reflect_data")
    workflow.add_conditional_edges(
        "reflect_data",
        check_after_data_reflection,
        {
            "preprocess": "preprocess",
            "feature_eng": "feature_eng",
            "final_report": "final_report"
        }
    )
    
    # ========================================================================
    # Phase 3: Feature engineering + checkpoint + reflection loop
    # ========================================================================
    workflow.add_edge("feature_eng", "cp_feature")
    workflow.add_edge("cp_feature", "reflect_analysis")
    workflow.add_conditional_edges(
        "reflect_analysis",
        check_after_analysis_reflection,
        {
            "feature_eng": "feature_eng",
            "visualize": "visualize",
            "final_report": "final_report"
        }
    )
    
    # ========================================================================
    # Phase 4: Visualization + training + checkpoint + forecast reflection + guardrail + vision
    # ========================================================================
    workflow.add_edge("visualize", "ts_suitable")
    workflow.add_edge("ts_suitable", "ts_train")
    workflow.add_edge("ts_train", "ts_visualize")
    workflow.add_edge("ts_visualize", "cp_forecast")
    workflow.add_edge("cp_forecast", "reflect_forecast")
    workflow.add_edge("reflect_forecast", "guardrail")
    
    # Add vision analysis pipeline after guardrail
    workflow.add_edge("guardrail", "export_images")
    workflow.add_edge("export_images", "vision_analysis")
    workflow.add_edge("vision_analysis", "vision_check")
    
    workflow.add_conditional_edges(
        "vision_check",
        check_after_forecast_reflection,
        {
            "ts_train": "ts_train",
            "final_report": "final_report"
        }
    )
    
    # Entry/finish
    workflow.set_entry_point("load")
    workflow.set_finish_point("final_report")
    
    # Compile with checkpointer if provided
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
