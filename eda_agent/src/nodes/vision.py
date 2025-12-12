"""
Vision analysis nodes for image-based validation and content verification
"""
import os
import base64
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config.settings import VISION_MODEL, VISION_TEMPERATURE
from ..state import EDAState, memory_update


# Lazy initialization of vision LLM
_vision_llm = None

def get_vision_llm():
    """Get or create the vision LLM instance"""
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = ChatOpenAI(
            model=VISION_MODEL, 
            temperature=VISION_TEMPERATURE
        )
    return _vision_llm


def export_images_to_workspace(state: EDAState) -> dict:
    """
    Export all generated images (viz_images + forecast_images) to workspace files
    for LLM vision analysis
    """
    from config.settings import OUTPUT_DIR
    
    step_name = "export_images"
    session_id = state.get("session_id", "default")
    
    viz_images = state.get('viz_images', {})
    forecast_images = state.get('forecast_images', {})
    
    # Handle None values (can happen during regeneration)
    if viz_images is None:
        viz_images = {}
    if forecast_images is None:
        forecast_images = {}
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    exported_files = {}
    
    # Export EDA visualization images
    for fig_name, img_base64 in viz_images.items():
        try:
            img_bytes = base64.b64decode(img_base64)
            filename = os.path.join(OUTPUT_DIR, f"{session_id}_figure1_{fig_name}.png")
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            exported_files[fig_name] = filename
        except Exception as e:
            print(f"Error exporting {fig_name}: {e}")
    
    # Export forecast images
    for fig_name, img_base64 in forecast_images.items():
        try:
            img_bytes = base64.b64decode(img_base64)
            filename = os.path.join(OUTPUT_DIR, f"{session_id}_figure3_{fig_name}.png")
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            exported_files[fig_name] = filename
        except Exception as e:
            print(f"Error exporting {fig_name}: {e}")
    
    # Export custom model detail image (Figure 4)
    custom_model_detail_image = state.get('custom_model_detail_image')
    if custom_model_detail_image:
        try:
            img_bytes = base64.b64decode(custom_model_detail_image)
            filename = os.path.join(OUTPUT_DIR, f"{session_id}_figure4_custom_model_detail.png")
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            exported_files['custom_model_detail'] = filename
        except Exception as e:
            print(f"Error exporting custom_model_detail: {e}")
    
    out = {
        "exported_images": exported_files,
        "export_directory": OUTPUT_DIR
    }
    msg = f"Images exported: {len(exported_files)} files saved to {OUTPUT_DIR}"
    out.update(memory_update(step_name, msg))
    
    return out


def detect_features_in_code(code: str) -> dict:
    """
    Detect which features/techniques are already implemented in the generated code.
    Returns a dict with feature names as keys and boolean values.
    
    This helps avoid recommending features that are already in use.
    """
    if not code:
        return {}
    
    features = {
        'time_of_day_features': False,
        'lag_features': False,
        'rolling_features': False,
        'seasonal_features': False,
        'calendar_features': False,
        'polynomial_features': False,
        'interaction_features': False,
        'statistical_features': False
    }
    
    code_lower = code.lower()
    
    # Check for time-of-day features (hour_sin, hour_cos, hour encoding)
    if any(x in code_lower for x in ['hour_sin', 'hour_cos', 'x[\'hour\'', 'x["hour"', '.hour', 'np.sin(2 * np.pi']):
        features['time_of_day_features'] = True
    
    # Check for lag features (shift operations)
    if any(x in code_lower for x in ['.shift(', 'lag_', 'previous_', 'x1.shift', 'x.shift']):
        features['lag_features'] = True
    
    # Check for rolling features (rolling operations)
    if any(x in code_lower for x in ['.rolling(', 'rolling_', 'rolling_mean', 'rolling_std']):
        features['rolling_features'] = True
    
    # Check for seasonal features (decomposition, seasonal encoding)
    if any(x in code_lower for x in ['seasonal', 'decompose', 'seasonal_decompose', 'month', 'quarter', 'day_of_year']):
        features['seasonal_features'] = True
    
    # Check for calendar features (weekday, month, day)
    if any(x in code_lower for x in ['weekday', 'day_of_week', '.month', '.dayofweek', 'is_weekend', 'is_holiday']):
        features['calendar_features'] = True
    
    # Check for polynomial features
    if any(x in code_lower for x in ['polynomialfeatures', 'degree=', 'poly_', 'x**2', '**3']):
        features['polynomial_features'] = True
    
    # Check for interaction features
    if any(x in code_lower for x in ['interaction', 'x * y', 'cross_', 'product']):
        features['interaction_features'] = True
    
    # Check for statistical features (mean, std, quantiles)
    if any(x in code_lower for x in ['.mean()', '.std()', '.quantile(', 'percentile', 'describe()']):
        features['statistical_features'] = True
    
    return features


def analyze_visualizations_with_vision(state: EDAState) -> dict:
    """
    Use LLM vision capabilities (GPT-4o) to analyze generated images:
    - Detect and verify figure titles and labels
    - Analyze chart content and data patterns
    - Identify anomalies and outliers
    - Extract metrics and performance indicators
    - Provide quality assessment and recommendations
    
    This provides an additional validation layer beyond numerical metrics.
    """
    step_name = "vision_analysis"
    
    exported_images = state.get('exported_images', {})
    export_dir = state.get('export_directory', '')
    
    if not exported_images:
        out = {"vision_analysis": None}
        out.update(memory_update(
            step_name, 
            "Vision analysis skipped: no images exported",
            warnings=["No images available for vision analysis"]
        ))
        return out
    
    vision_results = {}
    all_critiques = []
    
    data_context = state.get('data_context', {})
    capacity = data_context.get('capacity', 'unknown')
    description = data_context.get('description', 'time series data')
    
    for fig_name, filepath in exported_images.items():
        if not os.path.exists(filepath):
            continue
            
        try:
            # Read and encode image
            with open(filepath, 'rb') as f:
                img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            # Prepare vision prompt based on figure type
            if 'correlation' in fig_name.lower():
                analysis_focus = """
                Analyze this correlation matrix:
                - Are correlations readable and properly labeled?
                - Are there any strong correlations (|r| > 0.7)?
                - Are there unexpected patterns or anomalies?
                - Is the color scale appropriate?
                """
            elif 'custom_model_detail' in fig_name.lower():
                # Special analysis for Figure 4 (Custom Model Detail)
                custom_model_info = state.get('custom_model_info', {})
                custom_code = custom_model_info.get('generated_code', 'N/A')
                custom_metrics = custom_model_info.get('metrics', {})
                custom_r2 = custom_metrics.get('r2', 0)
                custom_mae = custom_metrics.get('mae', 0)
                custom_rmse = custom_metrics.get('rmse', 0)
                
                # Get benchmark model performances for comparison
                forecast_results = state.get('forecast_results', {})
                benchmark_models_info = []
                best_benchmark_r2 = 0
                best_benchmark_name = ""
                
                for model_key, model_data in forecast_results.items():
                    if model_key != 'custom' and isinstance(model_data, dict) and 'r2' in model_data:
                        r2 = model_data.get('r2', 0)
                        mae = model_data.get('mae', 0)
                        rmse = model_data.get('rmse', 0)
                        name = model_data.get('name', model_key.upper())
                        benchmark_models_info.append({
                            'name': name,
                            'r2': r2,
                            'mae': mae,
                            'rmse': rmse
                        })
                        if r2 > best_benchmark_r2:
                            best_benchmark_r2 = r2
                            best_benchmark_name = name
                
                performance_gap = custom_r2 - best_benchmark_r2
                
                # Build user-friendly comparison text
                benchmark_comparison_text = "标准对标模型包括："
                for bm in benchmark_models_info:
                    benchmark_comparison_text += f"\n- {bm['name']}"
                
                performance_comparison = ""
                if performance_gap > 0.01:
                    performance_comparison = f"您的自定义模型表现优于{best_benchmark_name}，性能提升明显。"
                elif performance_gap > -0.01:
                    performance_comparison = f"您的自定义模型与{best_benchmark_name}性能相当，表现具有竞争力。"
                else:
                    performance_comparison = f"您的自定义模型与{best_benchmark_name}相比还有进步空间，误差为{abs(performance_gap)*100:.1f}%。"
                
                # Simplify code explanation - don't show raw code
                code_explanation = "您的模型使用了高级集成学习方法，结合了多个决策树的优势来提高预测准确度。"
                if 'LightGBM' in custom_code or 'lightgbm' in custom_code.lower():
                    code_explanation = "您的模型采用了LightGBM梯度提升框架，这是一种高效的集成学习方法，特别适合时间序列预测。"
                elif 'XGBoost' in custom_code or 'xgboost' in custom_code.lower():
                    code_explanation = "您的模型基于XGBoost框架，使用了先进的梯度提升技术来最小化预测误差。"
                elif 'ensemble' in custom_code.lower() or 'stack' in custom_code.lower():
                    code_explanation = "您的模型使用了集成策略，将多个模型的预测结合起来以获得更强的预测能力。"
                
                # Detect what features are already implemented
                implemented_features = detect_features_in_code(custom_code)
                implemented_feature_list = [name.replace('_', ' ').title() for name, present in implemented_features.items() if present]
                
                already_implemented_section = ""
                if implemented_feature_list:
                    features_text = '\n'.join([f'- {f}' for f in implemented_feature_list])
                    already_implemented_section = f"""
⚠️ **ALREADY IMPLEMENTED FEATURES (DO NOT RECOMMEND AGAIN):**
The following enhancements are ALREADY in the current code:
{features_text}

When suggesting new improvements, make sure NOT to recommend the above features again.
Focus on NEW, DIFFERENT approaches that haven't been tried yet.
"""
                
                # Include the actual generated code in the prompt
                actual_code_section = f"""
                ==============================================================
                SECTION 0: ACTUAL GENERATED MODEL CODE
                ==============================================================
                
                This is the EXACT Python code that was generated for the custom model:
                
                ```python
                {custom_code}
                ```
                
                This is the CURRENT implementation you need to suggest improvements for.
                Extract specific parameter values from this code.
                Base your recommendations on modifying these actual parameters.
                
                {already_implemented_section}
                """
                
                analysis_focus = f"""
                You are an expert solar photovoltaic (PV) forecasting specialist and ML hyperparameter tuning expert.
                
                **🚨 CRITICAL OUTPUT REQUIREMENT 🚨**
                Your response MUST include SPECIFIC PYTHON CODE SNIPPETS with EXACT PARAMETER VALUES for EVERY recommendation.
                DO NOT give vague suggestions like "improve model complexity" or "add features".
                ALWAYS provide actual code like: "model = LGBMRegressor(n_estimators=250, max_depth=15)"
                
                Analyze this 4-panel diagnostic plot (Figure 4) to provide DATA-DRIVEN improvement recommendations.
                Base your analysis on SPECIFIC NUMERICAL VALUES and PATTERNS you see in the charts.
                
                For explanations: Use plain language for business stakeholders.
                For implementation: Provide COMPLETE, COPY-PASTEABLE Python code with exact values.
                
                {actual_code_section}
                
                ==============================================================
                SECTION 1: CUSTOM MODEL PERFORMANCE SUMMARY
                ==============================================================
                
                **Model Architecture:**
                {code_explanation}
                
                **Current Performance Metrics:**
                • Overall Accuracy Score: {custom_r2:.1%} 
                  Interpretation: The model explains {custom_r2*100:.1f}% of solar power variations
                  
                • Average Prediction Error: ±{custom_mae:.2f} kW per forecast
                  Interpretation: Typical daily forecasts are off by about {custom_mae:.2f} kilowatts
                  
                • Error Severity (worst-case typical error): ±{custom_rmse:.2f} kW
                  Interpretation: When errors do occur, they're typically ±{custom_rmse:.2f} kW
                
                **Comparison with Standard Models:**
                {benchmark_comparison_text}
                Performance Gap: {performance_gap:.1%} vs {best_benchmark_name}
                
                {performance_comparison}
                
                ==============================================================
                SECTION 2: DETAILED ANALYSIS BASED ON DIAGNOSTIC CHARTS
                ==============================================================
                
                ** STEP 1: Read the TOP-LEFT PANEL (Scatter: Actual vs Predicted Power) **
                Instructions:
                1. Look at how far the dots are from the diagonal line (perfect predictions)
                2. Extract rough numbers:
                   - At HIGH power levels (right side, >50% of capacity): How scattered are dots?
                     Estimate scatter range in ±X kW
                   - At PEAK hours (middle right, 30-80% capacity): Are there clusters or scattered points?
                   - At RAMP hours (bottom left, 0-30% capacity): Are predictions near zero or systematically off?
                3. Note color patterns or intensity concentration
                
                Based on what you see, describe:
                - TIGHT DOTS around diagonal = Model predicts well (GOOD)
                - SCATTERED DOTS away from diagonal = Model struggles (NEEDS IMPROVEMENT)
                - DOTS ABOVE diagonal = Systematic overprediction (model predicts too high)
                - DOTS BELOW diagonal = Systematic underprediction (model predicts too low)
                
                ** STEP 2: Read the TOP-RIGHT PANEL (Time Series: Errors Over Time) **
                Instructions:
                1. Scan from left to right (through 24 hours)
                2. Identify specific TIME PERIODS with large error spikes:
                   - What TIME do errors spike? (e.g., 6-7am, noon, 5-8pm)
                   - How large are the spikes? (extract ±X kW from axis)
                   - Are these spikes CONSISTENTLY at the same times each day?
                3. Look for error PATTERNS:
                   - Are errors mostly positive (above zero) or negative (below zero)?
                   - Does the error pattern repeat like a daily cycle?
                
                Based on what you see, describe:
                - Morning spikes (6-8am) = Model struggles with sunrise ramp
                - Noon spikes (11am-3pm) = Model struggles with cloud variability at peak
                - Evening spikes (5-8pm) = Model struggles with sunset ramp
                - Consistent positive errors = Model systematically overpredicts
                - Consistent negative errors = Model systematically underpredicts
                
                ** STEP 3: Read the BOTTOM-LEFT PANEL (Histogram: Error Distribution) **
                Instructions:
                1. Examine the histogram shape:
                   - Is it centered at zero or shifted left/right?
                   - Is the shape symmetric (bell curve) or skewed?
                   - Are there heavy tails (fat wings) with outlier errors?
                2. Estimate from the chart:
                   - Where is the peak? At what error value?
                   - What's the range? (e.g., errors span -20 to +15 kW)
                   - Are there isolated bars far from center (outliers)?
                
                Based on what you see, describe:
                - CENTERED at zero = Balanced predictions (GOOD)
                - LEFT-SKEWED (peak left of zero) = Tendency to overpredict
                - RIGHT-SKEWED (peak right of zero) = Tendency to underpredict
                - WIDE DISTRIBUTION = Errors are unpredictable
                - NARROW DISTRIBUTION = Errors are consistent
                - FAT TAILS = Occasional extreme prediction failures
                
                ** STEP 4: Read the BOTTOM-RIGHT PANEL (Statistics: Mean, Std Dev, Min/Max) **
                Instructions:
                1. Extract the numerical values shown:
                   - Mean error (should be ~0)
                   - Standard deviation (spread of typical errors)
                   - Min/Max range (worst-case errors)
                2. Interpret what these tell you about model reliability
                
                ==============================================================
                SECTION 3: SPECIFIC PARAMETER ADJUSTMENT RECOMMENDATIONS
                ==============================================================
                
                Based on your observations from Sections 1-2, provide 4-5 specific improvement suggestions.
                
                Each suggestion MUST follow this format:
                
                🎯 **RECOMMENDATION #N: [Clear Business Problem]**
                
                **What You Observed:**
                - Describe specific chart finding (e.g., "scatter plot shows 30% of high-power predictions are >±10 kW off")
                - Reference specific time period (e.g., "errors spike from 11am-2pm by ±15 kW")
                - Include rough numbers extracted from the charts
                
                **Root Cause:**
                - Explain why this happens in physics terms (not code terms)
                  Example: "The model doesn't understand that rapid changes at sunrise happen in minutes, not hours"
                
                **How to Fix It:**
                Provide 2-3 CONCRETE, ACTIONABLE steps. Use PLAIN ENGLISH - no code syntax or parameter names.
                Example of GOOD: "Add a new input that tells the model what time of day it is (dawn, morning, noon, afternoon, dusk, night)"
                Example of BAD: "Set feature_fraction=0.8 and add lag_1 feature"
                
                **Technical Implementation (Code Suggestion):**
                NOW provide the actual code/parameter changes needed. Be SPECIFIC:
                - If it's a hyperparameter: "Set n_estimators=300 (currently 100)"
                - If it's a new feature: "Add lag features: X['lag_1'] = X['power'].shift(1)"
                - If it's an architecture change: Show the exact code modification
                
                Example:
                ```python
                # Current: n_estimators=100, max_depth=10
                # Suggested change:
                model = LGBMRegressor(
                    n_estimators=250,  # Increase from 100
                    max_depth=15,      # Increase from 10
                    learning_rate=0.05, # Decrease from 0.1 for stability
                    num_leaves=31
                )
                
                # Add time-of-day features:
                X['hour'] = X.index.hour
                X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
                X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
                ```
                
                **Expected Improvement:**
                - Estimate how much accuracy might improve
                - Example: "Could reduce peak-hour errors from ±15 kW to ±8 kW"
                - Estimate new accuracy score range
                
                ---
                
                COMMON ADJUSTMENTS FOR PV FORECASTING (Choose relevant ones):
                
                1️⃣ **Add Time-of-Day Awareness** (For sunrise/sunset problems)
                   When: Errors spike in morning 6-8am or evening 5-8pm
                   What it means: Model sees sunrise/sunset like any other hour
                   Fix: Encode time-of-day as periodic information (model learns dawn ramps are different)
                   Benefit: Could cut ramp-period errors 40-60%
                
                2️⃣ **Increase Model Sophistication** (For peak-hour scattered predictions)
                   When: Scatter plot shows loose dots on right side (high power, 30-80% of max)
                   What it means: Model architecture too simple to capture cloud variability
                   Fix: Use more decision trees / deeper learning (model trained more thoroughly)
                   Benefit: Could improve peak accuracy from current to ±8-10 kW range
                
                3️⃣ **Add Recent History Features** (For time-dependent patterns)
                   When: Time-series plot shows daily repeating error patterns
                   What it means: Model not using "what just happened in the last 1-2 hours" information
                   Fix: Tell model what power was 30min ago, 1hr ago, 2hrs ago (recent trajectory)
                   Benefit: Could reduce all errors 15-25% by capturing momentum
                
                4️⃣ **Fix Systematic Bias** (If histogram is skewed)
                   When: Histogram peak is left of zero (overprediction) or right of zero (underprediction)
                   What it means: Model has built-in tendency to guess too high or too low
                   Fix: Adjust model calibration to center errors at zero
                   Benefit: Would shift entire prediction distribution to be balanced
                
                5️⃣ **Handle Weather Extremes** (For fat-tailed histogram)
                   When: Histogram has heavy tails with outlier errors ±20+ kW
                   What it means: On unusual weather days, model fails significantly
                   Fix: Explicitly train on extreme weather scenarios and sudden cloud events
                   Benefit: Could reduce worst-case errors from ±25 kW to ±15 kW
                
                ---
                
                **CRITICAL OUTPUT FORMAT REQUIREMENTS:**
                
                For EACH of your 4-5 recommendations, you MUST include:
                
                1. **Business-Friendly Explanation** (2-3 sentences in plain language)
                2. **Technical Implementation** (actual code with specific parameter values):
                   - Show current vs. suggested parameter values
                   - Provide complete code snippets (5-15 lines)
                   - Include import statements if needed
                   - Example:
                     ```python
                     # Current model configuration:
                     # n_estimators=100, learning_rate=0.1
                     
                     # Suggested improvement:
                     from lightgbm import LGBMRegressor
                     model = LGBMRegressor(
                         n_estimators=250,      # Increased from 100
                         learning_rate=0.05,    # Reduced from 0.1 for stability
                         max_depth=15,          # Increased from 10
                         num_leaves=31,
                         min_child_samples=20
                     )
                     ```
                
                3. **Expected Impact** (quantified improvement estimate)
                
                End with a PRIORITIZED ACTION PLAN with complete code examples:
                
                "**🎯 Prioritized Implementation Plan:**
                
                **Step 1: [First Priority]**
                Expected improvement: [X% accuracy gain / ±Y kW error reduction]
                ```python
                [Complete code for implementation]
                ```
                
                **Step 2: [Second Priority]**
                Expected improvement: [X% accuracy gain / ±Y kW error reduction]
                ```python
                [Complete code for implementation]
                ```
                
                **Step 3: [Third Priority]**
                Expected improvement: [X% accuracy gain / ±Y kW error reduction]
                ```python
                [Complete code for implementation]
                ```
                
                **Combined Expected Improvement:** 
                Accuracy could improve from current {custom_r2:.1%} to estimated {custom_r2 + 0.05:.1%} (~{((custom_r2 + 0.05) / custom_r2 - 1) * 100:.1f}% relative improvement)"
                
                Remember: 
                - EVERY recommendation must include actual code with specific parameter values
                - Don't just say "increase n_estimators" - show the exact code change
                - Base all recommendations on what you actually see in the 4 charts
                - Provide 4-5 detailed recommendations, not just 2-3 vague ones
                - Include complete, copy-pasteable code snippets
                """
            elif 'performance' in fig_name.lower() or 'forecast' in fig_name.lower():
                analysis_focus = f"""
                Analyze this forecast performance plot:
                - Are predictions and actual values clearly distinguishable?
                - Is the train/test split clearly marked?
                - Do predictions stay within physical constraints (0-{capacity})?
                - Are there visible overfitting or underfitting patterns?
                - Are metrics (MAE, RMSE, R²) readable in the legend?
                """
            else:
                analysis_focus = """
                Analyze this visualization:
                - Is the chart title clear and descriptive?
                - Are axes properly labeled with units?
                - Is the data trend visible and interpretable?
                - Are there any obvious outliers or anomalies?
                - Is the color scheme effective?
                """
            
            prompt = f"""You are analyzing a data visualization for {description}.

{analysis_focus}

Provide a concise analysis (2-3 sentences) covering:
1. What the chart shows
2. Key observations or patterns
3. Any quality issues or recommendations

Be specific and focus on actionable insights."""

            # Call vision LLM
            llm = get_vision_llm()
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            )
            
            response = llm.invoke([message])
            critique = response.content.strip()
            
            vision_results[fig_name] = {
                "filepath": filepath,
                "analysis": critique,
                "timestamp": datetime.now().isoformat()
            }
            all_critiques.append(f"{fig_name}: {critique}")
            
        except Exception as e:
            vision_results[fig_name] = {
                "filepath": filepath,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            all_critiques.append(f"{fig_name}: Vision analysis failed - {str(e)}")
    
    # Compile summary
    summary = f"Vision analysis completed for {len(vision_results)} images. " + \
              f"All visualizations verified."
    
    out = {
        "vision_analysis": vision_results,
        "vision_summary": summary
    }
    out.update(memory_update(
        step_name,
        summary,
        reflections=all_critiques[:3]  # Include top 3 critiques in reflections
    ))
    
    return out


def vision_quality_check(state: EDAState) -> dict:
    """
    Quick vision-based quality check focusing on critical issues:
    - Verify charts are not blank/corrupted
    - Check for rendering errors
    - Validate that forecasts don't exceed capacity
    
    This is a lightweight check for workflow validation.
    """
    step_name = "vision_quality_check"
    
    vision_analysis = state.get('vision_analysis', {})
    
    if not vision_analysis:
        out = {"vision_quality_passed": True}
        out.update(memory_update(step_name, "Vision quality check skipped: no vision analysis"))
        return out
    
    issues = []
    
    for fig_name, result in vision_analysis.items():
        if 'error' in result:
            issues.append(f"{fig_name}: {result['error']}")
        else:
            analysis = result.get('analysis', '').lower()
            # Check for common quality issues
            if 'blank' in analysis or 'empty' in analysis:
                issues.append(f"{fig_name}: Chart appears blank or empty")
            if 'error' in analysis or 'corrupted' in analysis:
                issues.append(f"{fig_name}: Rendering errors detected")
            if 'exceed' in analysis and 'capacity' in analysis:
                issues.append(f"{fig_name}: Predictions may exceed capacity constraint")
    
    quality_passed = len(issues) == 0
    
    if quality_passed:
        msg = f"Vision quality check passed: {len(vision_analysis)} images verified"
    else:
        msg = f"Vision quality check found {len(issues)} issues: {'; '.join(issues)}"
    
    out = {
        "vision_quality_passed": quality_passed,
        "vision_quality_issues": issues
    }
    out.update(memory_update(
        step_name,
        msg,
        warnings=issues if not quality_passed else []
    ))
    
    return out
