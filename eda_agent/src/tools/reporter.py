"""
Report generation tool
"""
from config.settings import REPORTING_MODEL, REPORTING_TEMPERATURE
from ..state import EDAState, memory_update


def generate_report(state: EDAState) -> dict:
    """Generate final analysis report using LLM"""
    step_name = "final_report"
    
    # Extract data context for domain-aware report generation
    data_context = state.get('data_context', {})
    business_objective = data_context.get('business_objective', '')
    business_context_str = f"\n\nBUSINESS OBJECTIVE: {business_objective}" if business_objective else ""
    data_context_str = (
        f"This is {data_context.get('description', 'time series data')} "
        f"with {data_context.get('resolution', 'unknown')} resolution. "
        f"Objective: {data_context.get('goal', 'forecasting')}. "
        f"System capacity: {data_context.get('capacity', 'unknown')}."
        f"{business_context_str}"
    )

    errors = state.get('errors', [])
    ts_analysis = state.get('ts_analysis') or {}
    forecast_results = state.get('forecast_results') or {}
    forecast_images = state.get('forecast_images') or {}
    data_profile = state.get('data_profile') or {}
    summary_stats = state.get('summary_stats') or {}
    engineered_data = state.get('engineered_data')
    visualizations = state.get('visualizations') or {}
    viz_images = state.get('viz_images') or {}

    timeseries_info = data_profile.get('timeseries', {})
    timespan_days = timeseries_info.get('timespan_days', 0)
    total_rows = data_profile.get('total_rows', 0)
    feature_count = engineered_data.shape[1] if engineered_data is not None else 0
    recommended_method = ts_analysis.get('recommended_method', 'statistical')

    feature_engineering_applied = state.get('feature_engineering_applied', False)
    feature_engineering_reason = state.get('feature_engineering_reason', '')
    original_feature_count = data_profile.get('total_columns', 0)

    missing_pct = state.get('missing_percentage', 0)
    missing_info = (
        f"Missing values: {missing_pct:.2f}% of dataset. "
        f"Interpolation method: forward-fill then backward-fill for numeric columns."
    )

    # CRITICAL: If stop_processing is True, generate ONLY insufficient data report
    # NO visualizations, NO forecast results, NO images should be included
    if state.get('stop_processing', False) and len(errors) > 0:
        # Check if it's a guardrail violation (negative power) vs insufficient data
        guardrail_warning = state.get('guardrail_warning', '')
        
        if 'Negative power output' in guardrail_warning:
            # Data quality issue - negative power
            summary_text = f"""⚠️ **Data Quality Alert**

{guardrail_warning}

**Issue:** The dataset contains negative power output values, which violates physical constraints for renewable energy systems. Solar and wind power generation must always be non-negative (≥ 0).

**Possible Causes:**
1. **Sensor calibration error:** Power sensors may need recalibration
2. **Data processing error:** Sign flip or unit conversion mistake during data export
3. **Incorrect column selection:** The target column may not represent actual power output

**Recommended Actions:**
1. Review and clean the raw data to remove or correct negative values
2. Verify sensor calibration and data export procedures
3. Confirm that the selected column represents actual power output (not delta/difference values)
4. Re-upload the corrected dataset

**Note:** Analysis cannot proceed with physically impossible data values. Please fix the data and try again."""
            status = 'data_quality_issue'
        else:
            # Insufficient data timespan
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage
                
                insufficient_prompt = f"""You are a data analyst. The dataset contains only {timespan_days:.2f} days of data ({total_rows} rows), which is insufficient for reliable time series forecasting (minimum 7 days required).

{missing_info}

Provide a brief 1-2 paragraph assessment of:
1. Why this dataset is unsuitable for forecasting
2. What additional data collection would help
3. Alternative analyses that could be performed on this limited data

Data summary: {total_rows} rows, {timespan_days:.2f} days timespan"""
                
                llm = ChatOpenAI(model=REPORTING_MODEL, temperature=REPORTING_TEMPERATURE)
                response = llm.invoke([HumanMessage(content=insufficient_prompt)])
                summary_text = response.content
            except Exception:
                summary_text = (
                    f"**Insufficient Data**\n\n"
                    f"Dataset contains only {total_rows} rows over {timespan_days:.2f} days. "
                    f"Minimum 7 days required for forecasting. Please upload more data and try again."
                )
            status = 'insufficient_data'
        
        # Return ONLY the error report with NO images
        report_obj = {
            'status': status,
            'errors': errors,
            'forecast_summary': summary_text,
            'data_profile': data_profile,
            'ts_analysis': ts_analysis,
            'visualizations': {},  # Empty - no visualizations
            'viz_images': {},  # Empty - no images
            'forecast_results': {},  # Empty forecast results
            'forecast_images': {},
            'forecast_performance_image': None
        }
        
        out = {
            'report': report_obj,
            'forecast_results': {}  # PRESERVE forecast_results at top level (empty due to guardrail)
        }
        out.update(
            memory_update(
                step_name,
                f"Final report generated: {status}",
                errors=errors
            )
        )
        return out

    # Normal success path - include all visualizations and results
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Extract all model metrics
        model_metrics = {}
        has_custom_model = False
        custom_model_code = ""
        custom_model_request = ""
        
        if forecast_results:
            # Check for both benchmark and custom models
            for model_key, result_data in forecast_results.items():
                model_metrics[model_key] = result_data
                if model_key == 'custom':
                    has_custom_model = True
        
        # Get custom model info from state
        custom_model_info = state.get('custom_model_info', {})
        if custom_model_info and custom_model_info.get('success'):
            has_custom_model = True
            custom_model_code = custom_model_info.get('generated_code', '')
            custom_model_request = custom_model_info.get('request', '')
        
        # Extract vision analysis for detailed tuning recommendations
        vision_analysis = state.get('vision_analysis', {})
        vision_tuning_insights = ""
        print(f"DEBUG reporter: vision_analysis keys = {list(vision_analysis.keys()) if vision_analysis else 'None'}")
        print(f"DEBUG reporter: has_custom_model = {has_custom_model}")
        if vision_analysis and has_custom_model:
            # Find the custom model detail analysis
            for viz_key, analysis_data in vision_analysis.items():
                print(f"DEBUG reporter: checking viz_key = {viz_key}")
                if 'custom_model_detail' in viz_key or 'custom_model' in viz_key:
                    analysis_text = analysis_data.get('analysis', '') if isinstance(analysis_data, dict) else str(analysis_data)
                    print(f"DEBUG reporter: found analysis, length = {len(analysis_text)}")
                    if analysis_text:
                        vision_tuning_insights = f"\n\n**VISION-BASED ANALYSIS (Figure 4 Diagnostics):**\n{analysis_text}"
                    break
        print(f"DEBUG reporter: vision_tuning_insights length = {len(vision_tuning_insights)}")
        
        fe_insights = ""
        if feature_engineering_applied:
            fe_insights = (
                f"\n- Feature Engineering: Applied | Reason: {feature_engineering_reason} | "
                f"Original features: {original_feature_count}, Final features: {feature_count}"
            )
        else:
            fe_insights = f"\n- Feature Engineering: Skipped | Reason: {feature_engineering_reason}"
        
        # Include Figure 4 if custom model exists
        figure_list = [
            "- Figure 1 (Power vs Time)",
            "- Figure 2 (Correlation Matrix)",
            "- Figure 3 (Forecast Performance - Multi-Model Comparison)"
        ]
        if has_custom_model and state.get('custom_model_detail_image'):
            figure_list.append("- Figure 4 (Custom Model Detail - Performance Analysis)")
        
        figure_descriptions = (
            "VISUALIZATIONS GENERATED:\n" + 
            "\n".join(figure_list)
        )
        
        # Build forecasting results section
        forecast_summary = []
        forecast_summary.append(f"- Recommended approach: {recommended_method}")
        
        # Add benchmark models
        if 'rf' in model_metrics:
            m = model_metrics['rf']
            forecast_summary.append(f"- Random Forest: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
        if 'dt' in model_metrics:
            m = model_metrics['dt']
            forecast_summary.append(f"- Decision Tree: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
        if 'xgb' in model_metrics:
            m = model_metrics['xgb']
            forecast_summary.append(f"- XGBoost: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
        if 'prophet' in model_metrics:
            m = model_metrics['prophet']
            forecast_summary.append(f"- Prophet: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
        
        # Add custom model if present
        custom_model_analysis = ""
        if 'custom' in model_metrics:
            m = model_metrics['custom']
            forecast_summary.append(f"- Custom Model (LLM-generated): MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
            if custom_model_request:
                forecast_summary.append(f"\nCUSTOM MODEL REQUEST:\n'{custom_model_request}'")
            if custom_model_code:
                forecast_summary.append(f"\nCUSTOM MODEL CODE:\n```python\n{custom_model_code}\n```")
            
            # Add detailed code analysis section with benchmark comparison
            # Get benchmark performance for comparison
            benchmark_summary = []
            best_benchmark_r2 = 0
            best_benchmark_name = ""
            
            for model_key, model_data in model_metrics.items():
                if model_key != 'custom' and isinstance(model_data, dict) and 'r2' in model_data:
                    r2 = model_data.get('r2', 0)
                    mae = model_data.get('mae', 0)
                    name = model_data.get('name', model_key.upper())
                    benchmark_summary.append(f"  {name}: R²={r2:.3f}, MAE={mae:.2f}")
                    if r2 > best_benchmark_r2:
                        best_benchmark_r2 = r2
                        best_benchmark_name = name
            
            benchmark_str = "\n".join(benchmark_summary)
            performance_gap = m['r2'] - best_benchmark_r2
            
            custom_model_analysis = f"""

**CUSTOM MODEL HYPERPARAMETER TUNING ANALYSIS:**

Performance Comparison:
- Custom Model: R²={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}
- Best Benchmark ({best_benchmark_name}): R²={best_benchmark_r2:.4f}
- Performance Gap: {performance_gap:+.4f} {'(Better than benchmarks ✓)' if performance_gap > 0 else '(Worse than benchmarks - needs tuning)' if performance_gap < -0.01 else '(Comparable)'}

Benchmark Models for Reference:
{benchmark_str}

Generated Code:
```python
{custom_model_code}
```

Dataset Context:
- Time series: {data_context.get('resolution', 'unknown')} resolution
- Size: {total_rows} rows × {feature_count} features ({timespan_days:.1f} days)
- Feature engineering: {feature_engineering_applied}

Figure 4 Diagnostic Panels:
1. Actual vs Predicted - check prediction accuracy and systematic bias
2. Residual Time Series - identify temporal patterns (hourly, daily cycles)
3. Residual Distribution - assess normality and outliers
4. Metrics Summary - MAE, RMSE, R², residual statistics

As a hyperparameter tuner, analyze:
- Compare custom model performance vs {best_benchmark_name} (best benchmark)
- Extract current hyperparameters from code (n_estimators, max_depth, learning_rate, etc.)
- Based on performance gap, suggest specific parameter adjustments with expected impact
- If underperforming, recommend architectural changes (ensemble, stacking, regularization)
- Identify missing temporal features from residual patterns
"""
        
        forecast_results_str = "\n".join(forecast_summary)
        
        # Build the main report prompt
        suitable_prompt = f"""You are a data analyst reporting on time series analysis results. Summarize this dataset and forecasting analysis in 3-4 paragraphs:

**IMPORTANT DOMAIN CONTEXT:**
{data_context_str}

**DATA SUMMARY:**
- Total rows: {total_rows}, Timespan: {timespan_days:.1f} days
- Features: {feature_count} (after preprocessing)
- {missing_info}{fe_insights}

**FORECASTING RESULTS:**
{forecast_results_str}

{figure_descriptions}{custom_model_analysis}{vision_tuning_insights}

REQUIREMENTS:
1. First paragraph: describe dataset characteristics, data quality (missing values, preprocessing applied), feature engineering decision and reason
2. Second paragraph: compare ALL models' performance (MAE, RMSE, R², rank them and explain which is best and why). {'NOTE: A custom model was generated using LLM-based code generation from user description.' if has_custom_model else ''} Briefly describe visualizations produced
3. Third paragraph: recommend next steps (validation approach, model selection, limitations to address)
"""
        
        # Add tuning recommendations section if custom model exists
        if has_custom_model and vision_tuning_insights:
            suitable_prompt += """4. Fourth paragraph: Include the MODEL TUNING RECOMMENDATIONS section from the Vision-Based Analysis above (COPY IT DIRECTLY from the analysis).
   - Do NOT re-generate or rephrase the recommendations
   - Include ALL detailed tuning suggestions with specific parameter values and code examples
   - Include the prioritized implementation plan with complete code snippets
   
   IMPORTANT: The Vision Analysis already contains specific, data-driven recommendations. Use them as-is in your response.
"""
        
        # Add business recommendations section if business objective exists
        if business_objective:
            suitable_prompt += f"""4. CRITICAL - BUSINESS RECOMMENDATIONS SECTION:
   You MUST include a dedicated "**Business Recommendations:**" section at the end of your response.
   
   Based on the forecast results, error patterns, peak/low periods, and confidence levels observed:
   - Address this specific business question: "{business_objective}"
   - Provide 3-5 concrete, actionable recommendations
   - Use specific numbers from the forecasts (e.g., "peak generation expected at 12:30pm with 85% confidence")
   - Consider forecast error ranges when making recommendations
   - Format each recommendation clearly with rationale
   
   Example format:
   **Business Recommendations:**
   1. **Energy Market Participation Strategy**: Based on forecast showing peak generation of 42-45 MW between 11am-2pm with ±3 kW error margin, recommend participating in day-ahead market during these hours with 90% of forecasted capacity to account for uncertainty.
   2. **Storage System Scheduling**: Deploy battery charging during low-generation morning hours (6-8am, <10 MW) and discharge during evening peak demand (6-8pm) when solar drops but grid prices remain high.
   3. **Ancillary Services**: Forecast shows stable generation during 10am-3pm window (±2 kW variation), suitable for frequency regulation services with high reliability.
   
   Be SPECIFIC to the data, ACTIONABLE for operations, and DIRECTLY answer the business objective.
"""
        
        suitable_prompt += """
Keep it technical and concise (3-4 sentences per paragraph). DO NOT fabricate information."""
        
        llm = ChatOpenAI(model=REPORTING_MODEL, temperature=REPORTING_TEMPERATURE)
        response = llm.invoke([HumanMessage(content=suitable_prompt)])
        summary_text = response.content
    except Exception as e:
        summary_text = (
            f"Analysis complete. Dataset: {total_rows} rows, {timespan_days:.1f} days. "
            f"Models evaluated."
        )

    report_obj = {
        'status': 'success',
        'forecast_summary': summary_text,
        'forecast_results': forecast_results,
        'forecast_images': forecast_images,
        'forecast_performance_image': state.get('forecast_performance_image'),
        'custom_model_detail_image': state.get('custom_model_detail_image'),
        'ts_analysis': ts_analysis,
        'ts_method': state.get('ts_method'),
        'ts_suitable': state.get('ts_suitable'),
        'data_profile': data_profile,
        'summary_stats': summary_stats,
        'engineered_data': engineered_data,
        'visualizations': visualizations,
        'viz_images': viz_images,
        'vision_analysis': vision_analysis  # Include detailed vision analysis for reference
    }
    
    out = {
        'report': report_obj,
        'forecast_results': forecast_results,  # PRESERVE forecast_results at top level
        'custom_model_detail_image': state.get('custom_model_detail_image'),  # PRESERVE Figure 4 at top level
        'vision_analysis': vision_analysis  # PRESERVE vision analysis at top level
    }
    out.update(memory_update(step_name, "Final report generated: success"))
    return out
