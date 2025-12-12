"""
Reflection and critique nodes for iterative improvement
"""
from ..state import EDAState, memory_update


def mark_checkpoint(state: EDAState, label: str) -> dict:
    """Explicitly mark a logical checkpoint with a label for revision boundaries"""
    tags = state.get("checkpoint_tags", [])
    tags.append({
        "tag": label,
        "revision_count": state.get("revision_count", 0)
    })
    
    out = {
        "checkpoint_tags": tags
    }
    out.update(
        memory_update(
            f"checkpoint_{label}",
            f"Checkpoint saved at {label} (rev={state.get('revision_count',0)})"
        )
    )
    return out


def _reflection_memory(step: str, score: float, needs_revision: bool, 
                       critique: str, state: EDAState, label: str):
    """Helper to create standardized reflection memory update"""
    entry = f"{label} Reflection (Score: {score:.2f}) needs_revision={needs_revision}: {critique}"[:6000]
    
    result = memory_update(
        step,
        f"{label} reflection: score={score:.2f} needs_revision={needs_revision}",
        reflections=[entry],
        needs_revision=needs_revision,
        errors=state.get("errors", [])
    )
    
    result["critique_history"] = [{
        "step": step,
        "score": score,
        "critique": critique
    }]
    result["revision_count"] = state.get("revision_count", 0) + (1 if needs_revision else 0)
    
    return result


def reflect_on_data_quality(state: EDAState) -> dict:
    """Reflect on data quality and decide if revision needed"""
    step = "reflect_data"
    
    # BUG FIX: Avoid "or" with DataFrame - check explicitly for None/emptiness
    data = state.get("preprocessed_data")
    if data is None or (hasattr(data, 'empty') and data.empty):
        data = state.get("data")
    if data is None or (hasattr(data, 'empty') and data.empty):
        result = _reflection_memory(
            step, 0.0, True, "No data available", state, "Data Quality"
        )
        result["data_quality_score"] = 0.0
        return result
    
    missing_pct = state.get("missing_percentage", 0.0)
    num_rows = len(data)
    num_cols = state.get("num_variables", 0)
    
    prompt = f"""Evaluate dataset quality.
Rows: {num_rows}
Vars: {num_cols}
Missing%: {missing_pct:.2f}

Provide:
QUALITY_SCORE: <0-1>
NEEDS_REVISION: <yes/no>
REASONING: <brief>
RECOMMENDATIONS: <brief>
"""
    
    try:
        response = get_reflection_llm().invoke(prompt)
        critique = response.content
        quality_score = 0.5
        needs_revision = False
        
        for line in critique.split('\n'):
            if 'QUALITY_SCORE:' in line:
                try:
                    quality_score = float(line.split(':')[1].strip())
                except:
                    pass
            if 'NEEDS_REVISION:' in line:
                needs_revision = 'yes' in line.lower()
        
        result = _reflection_memory(
            step, quality_score, needs_revision, critique, state, "Data Quality"
        )
        result["data_quality_score"] = quality_score
        return result
        
    except Exception as e:
        result = _reflection_memory(
            step, 0.5, False, f"LLM error: {e}", state, "Data Quality"
        )
        result["data_quality_score"] = 0.5
        return result


def reflect_on_analysis_quality(state: EDAState) -> dict:
    """Reflect on analysis quality (feature engineering decisions)"""
    step = "reflect_analysis"
    
    fe_applied = state.get("feature_engineering_applied", False)
    fe_reason = state.get("feature_engineering_reason", "")
    num_vars = state.get("num_variables", 0)
    
    prompt = f"""Evaluate analysis quality.
Vars: {num_vars}
FeatureEngineeringApplied: {fe_applied}
Reason: {fe_reason}

Provide:
QUALITY_SCORE: <0-1>
NEEDS_REVISION: <yes/no>
REASONING: <brief>
RECOMMENDATIONS: <brief>
"""
    
    try:
        response = get_reflection_llm().invoke(prompt)
        critique = response.content
        quality_score = 0.5
        needs_revision = False
        
        for line in critique.split('\n'):
            if 'QUALITY_SCORE:' in line:
                try:
                    quality_score = float(line.split(':')[1].strip())
                except:
                    pass
            if 'NEEDS_REVISION:' in line:
                needs_revision = 'yes' in line.lower()
        
        result = _reflection_memory(
            step, quality_score, needs_revision, critique, state, "Analysis Quality"
        )
        result["analysis_quality_score"] = quality_score
        return result
        
    except Exception as e:
        result = _reflection_memory(
            step, 0.5, False, f"LLM error: {e}", state, "Analysis Quality"
        )
        result["analysis_quality_score"] = 0.5
        return result


def reflect_on_forecast_quality(state: EDAState) -> dict:
    """Reflect on forecast model quality and provide code improvement suggestions for custom models"""
    step = "reflect_forecast"
    
    forecast_results = state.get("forecast_results")
    if not forecast_results:
        result = _reflection_memory(
            step, 0.0, True, "No forecast results", state, "Forecast Quality"
        )
        result["forecast_quality_score"] = 0.0
        result["quality_score"] = 0.0
        return result
    
    # Extract metrics from all models (new structure: forecast_results['rf'], ['dt'], etc.)
    all_r2_scores = []
    model_summary = []
    has_custom = False
    custom_code = ""
    custom_request = ""
    custom_metrics = {}
    
    for model_key, model_data in forecast_results.items():
        if isinstance(model_data, dict) and 'r2' in model_data:
            r2 = model_data.get('r2', -999)
            mae = model_data.get('mae', 0)
            rmse = model_data.get('rmse', 0)
            name = model_data.get('name', model_key.upper())
            all_r2_scores.append(r2)
            model_summary.append(f"{name}: R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
            
            if model_key == 'custom':
                has_custom = True
                custom_metrics = model_data
    
    # Get custom model info for code analysis
    custom_model_info = state.get('custom_model_info', {})
    if custom_model_info and custom_model_info.get('success'):
        has_custom = True
        custom_code = custom_model_info.get('generated_code', '')
        custom_request = custom_model_info.get('request', '')
    
    if not all_r2_scores:
        result = _reflection_memory(
            step, 0.0, True, "No valid metrics found", state, "Forecast Quality"
        )
        result["forecast_quality_score"] = 0.0
        result["quality_score"] = 0.0
        return result
    
    best_r2 = max(all_r2_scores)
    avg_r2 = sum(all_r2_scores) / len(all_r2_scores)
    
    # Build custom model analysis section if applicable
    custom_analysis_section = ""
    if has_custom and custom_code:
        custom_analysis_section = f"""

CUSTOM MODEL ANALYSIS:
User Request: {custom_request}

Generated Code:
```python
{custom_code}
```

Custom Model Performance:
- MAE: {custom_metrics.get('mae', 'N/A')}
- RMSE: {custom_metrics.get('rmse', 'N/A')}
- R²: {custom_metrics.get('r2', 'N/A')}

Based on Figure 4 (Custom Model Detail) showing:
1. Actual vs Predicted comparison
2. Residual patterns over time
3. Residual distribution (should be centered at 0)
4. Performance metrics summary

Analyze the generated code and visual patterns to provide:
CODE_IMPROVEMENT_SUGGESTIONS: <Specific recommendations to improve the custom model code, such as:
- Hyperparameter tuning suggestions (e.g., max_depth, n_estimators, learning_rate)
- Feature engineering improvements (e.g., lag features, rolling statistics)
- Model architecture changes (e.g., ensemble methods, different algorithms)
- Regularization techniques (e.g., L1/L2 penalties, dropout for neural networks)
- Data preprocessing enhancements (e.g., scaling, normalization, outlier handling)
- Cross-validation strategy improvements
- If residuals show patterns: suggest capturing those patterns
- If residuals are non-normal: suggest transformations or robust methods>
"""
    
    prompt = f"""Evaluate forecast quality for renewable energy forecasting.

MODELS TRAINED: {len(all_r2_scores)}
{chr(10).join(model_summary)}

BEST R²: {best_r2:.3f}
AVERAGE R²: {avg_r2:.3f}{custom_analysis_section}

Provide:
QUALITY_SCORE: <0-1> (0.8+ for R²>0.85, 0.6-0.8 for R²>0.7, 0.4-0.6 for R²>0.5, <0.4 for R²<0.5)
NEEDS_REVISION: <yes/no>
REASONING: <brief explanation of quality>
RECOMMENDATIONS: <suggestions for improvement>
{"CODE_IMPROVEMENT_SUGGESTIONS: <specific code improvements for custom model>" if has_custom else ""}
"""
    
    try:
        response = get_reflection_llm().invoke(prompt)
        critique = response.content
        quality_score = 0.5
        needs_revision = False
        
        for line in critique.split('\n'):
            if 'QUALITY_SCORE:' in line:
                try:
                    quality_score = float(line.split(':')[1].strip())
                except:
                    pass
            if 'NEEDS_REVISION:' in line:
                needs_revision = 'yes' in line.lower()
        
        # Auto-adjust based on best R²
        if best_r2 >= 0.85:
            quality_score = max(quality_score, 0.8)
        elif best_r2 < 0.5:
            needs_revision = True
            quality_score = min(quality_score, 0.4)
        
        result = _reflection_memory(
            step, quality_score, needs_revision, critique, state, "Forecast Quality"
        )
        result["forecast_quality_score"] = quality_score
        
        # Calculate overall quality as average of all quality scores
        data_q = state.get("data_quality_score", 0.5)
        analysis_q = state.get("analysis_quality_score", 0.5)
        overall = (data_q + analysis_q + quality_score) / 3.0
        result["quality_score"] = overall
        
        return result
        
    except Exception as e:
        result = _reflection_memory(
            step, 0.5, False, f"LLM error: {e}", state, "Forecast Quality"
        )
        result["forecast_quality_score"] = 0.5
        
        # Calculate overall quality
        data_q = state.get("data_quality_score", 0.5)
        analysis_q = state.get("analysis_quality_score", 0.5)
        overall = (data_q + analysis_q + 0.5) / 3.0
        result["quality_score"] = overall
        
        return result


def validate_guardrails(state: EDAState) -> dict:
    """
    Validate physical constraints:
    1. Predictions must not exceed nominal capacity
    2. MAE, RMSE, R2 must be > 0
    3. All power output data must be >= 0 (renewable energy constraint)
    """
    step = "guardrail_validation"
    violations = []
    warnings = []
    critical_data_violation = False  # Track if it's a data-level issue
    
    # Extract capacity from data profile
    data_profile = state.get("data_profile", {})
    capacity = data_profile.get("nominal_capacity")
    
    # Check data non-negativity (CRITICAL CHECK - data quality issue)
    data = state.get("engineered_data")
    if data is None:
        data = state.get("preprocessed_data")
    if data is None:
        data = state.get("data")
    
    if data is not None:
        target_col = state.get("target_column")
        if target_col and target_col in data.columns:
            min_value = data[target_col].min()
            if min_value < 0:
                critical_data_violation = True  # Mark as critical
                violations.append(
                    f"Negative power output detected: min={min_value:.3f} "
                    f"(renewable energy systems must have power output >= 0)"
                )
    
    # Check forecast results
    forecast_results = state.get("forecast_results")
    if forecast_results:
        methods = forecast_results.get("methods", {})
        
        for method_name, method_data in methods.items():
            metrics = method_data.get("metrics", {})
            predictions = method_data.get("predictions")
            
            # Validate metrics > 0
            mae = metrics.get("MAE", 0)
            rmse = metrics.get("RMSE", 0)
            r2 = metrics.get("R2", -999)
            
            if mae < 0:
                violations.append(f"{method_name}: MAE={mae:.3f} is negative (invalid)")
            if rmse < 0:
                violations.append(f"{method_name}: RMSE={rmse:.3f} is negative (invalid)")
            if r2 < 0:
                warnings.append(
                    f"{method_name}: R2={r2:.3f} is negative (poor model fit)"
                )
            
            # Validate predictions <= capacity
            if predictions is not None and capacity is not None:
                import numpy as np
                max_pred = np.max(predictions)
                if max_pred > capacity:
                    violations.append(
                        f"{method_name}: max prediction={max_pred:.3f} exceeds "
                        f"capacity={capacity} MW"
                    )
            
            # Validate predictions >= 0
            if predictions is not None:
                import numpy as np
                min_pred = np.min(predictions)
                if min_pred < 0:
                    violations.append(
                        f"{method_name}: min prediction={min_pred:.3f} is negative "
                        f"(renewable energy must be >= 0)"
                    )
    
    # Determine if revision needed
    needs_revision = len(violations) > 0
    
    # Build guardrail report
    if violations:
        violation_text = "; ".join(violations)
        content = f"Guardrail violations detected: {violation_text}"
    else:
        content = "All physical constraints satisfied"
        violation_text = ""
    
    result = memory_update(
        step,
        content,
        warnings=warnings,
        errors=violations if needs_revision else []
    )
    
    result["needs_revision"] = needs_revision
    result["guardrail_passed"] = not needs_revision
    result["guardrail_warning"] = violation_text if needs_revision else ""
    
    # CRITICAL: If negative power in raw data, halt all processing immediately
    if critical_data_violation:
        result["stop_processing"] = True
        print(f"\n⚠️ CRITICAL GUARDRAIL VIOLATION: {violation_text}")
        print(f"⚠️ Halting analysis - data quality issue must be resolved")
    
    if needs_revision:
        result["revision_count"] = state.get("revision_count", 0) + 1
    
    return result
