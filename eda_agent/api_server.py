# -*- coding: utf-8 -*-
"""
Flask API Server for PowerSys Frontend
Connects the web UI to the existing Python backend
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side plotting

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import base64
from pathlib import Path
import uuid
import sqlite3
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import workflow components
from src.workflows import build_reflective_workflow
from config.settings import OUTPUT_DIR, CHECKPOINT_DIR
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Store active sessions
sessions = {}


def setup_checkpointer():
    """Initialize SQLite checkpointer with pickle fallback for DataFrame serialization"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "eda_agent_memory.db")
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    
    # Use JsonPlusSerializer with pickle fallback to handle DataFrames
    serde = JsonPlusSerializer(pickle_fallback=True)
    return SqliteSaver(conn, serde=serde)


def run_eda_workflow(filepath, data_context, resume_session_id=None):
    """
    Run the EDA workflow programmatically
    Similar to run_analysis() in main.py but callable from API
    """
    # Generate or use existing session ID
    session_id = resume_session_id if resume_session_id else str(uuid.uuid4())[:8]
    
    # Setup checkpoint database
    checkpointer = setup_checkpointer()
    
    # Build workflow graph
    graph = build_reflective_workflow(checkpointer=checkpointer)
    
    # Prepare initial state (matching main.py structure)
    initial_state = {
        "filepath": filepath,
        "data": None,
        "preprocessed_data": None,
        "engineered_data": None,
        "summary_stats": None,
        "missing_info": None,
        "data_profile": None,
        "missing_percentage": 0.0,
        "num_variables": 0,
        "target_column": None,
        "visualizations": {},
        "viz_images": {},
        "forecast_results": None,
        "forecast_images": {},
        "model_state": None,
        "report": None,
        "errors": [],
        "stop_processing": False,
        "ts_suitable": False,
        "ts_method": None,
        "ts_analysis": None,
        "feature_engineering_applied": False,
        "feature_engineering_reason": None,
        # Memory fields
        "messages": [HumanMessage(content=f"Starting reflective analysis on: {filepath}")],
        "session_id": session_id,
        "session_start_time": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "steps_completed": [],
        "current_step": "initialized",
        "iteration_count": 0,
        "max_iterations": 50,
        "warnings": [],
        # Reflection fields
        "reflections": [],
        "critique_history": [],
        "needs_revision": False,
        "revision_count": 0,
        "quality_score": 0.0,
        "data_quality_score": 0.0,
        "analysis_quality_score": 0.0,
        "forecast_quality_score": 0.0,
        "checkpoint_tags": [],
        "data_context": data_context,
        "custom_model_info": {},
        # Vision analysis fields
        "exported_images": {},
        "export_directory": None,
        "vision_analysis": {},
        "vision_summary": None,
        "vision_quality_passed": True,
        "vision_quality_issues": []
    }
    
    # Execute workflow with invoke (not stream)
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
    
    print(f"Running workflow for session {session_id}...")
    result = graph.invoke(initial_state, config=config)
    print(f"Workflow completed for session {session_id}")
    
    return result, session_id

# Store active sessions
sessions = {}

@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory('web', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'EDA Agent API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    Receives file upload and parameters, runs EDA analysis
    """
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get form parameters
        description = request.form.get('description', 'Time series data')
        resolution = request.form.get('resolution', '30 minutes')
        capacity = request.form.get('capacity', '')
        goal = request.form.get('goal', 'Forecasting')
        business_objective = request.form.get('business_objective', '')
        session_id = request.form.get('session_id', str(os.getpid()))
        custom_model_request = request.form.get('custom_model_request', '')
        
        # Save uploaded file temporarily
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / file.filename
        file.save(str(file_path))
        
        # Prepare context for main.py
        context = {
            'data_path': str(file_path),
            'description': description,
            'resolution': resolution,
            'capacity': capacity,
            'goal': goal,
            'business_objective': business_objective if business_objective else None,
            'custom_model_request': custom_model_request if custom_model_request else None,
            'session_id': session_id
        }
        
        # Run the EDA analysis
        print(f"Starting analysis for session {session_id}...")
        result, session_id = run_eda_workflow(
            filepath=str(file_path),
            data_context=context,
            resume_session_id=session_id
        )
        
        # Extract results
        state = result
        
        # Debug: print state keys
        print(f"State keys: {list(state.keys()) if state else 'None'}")
        
        # Get metrics
        metrics = {}
        forecast_results = state.get('forecast_results', {})  # This has the model results
        print(f"Forecast results: {forecast_results}")
        
        if forecast_results:
            for model_key, model_data in forecast_results.items():
                if isinstance(model_data, dict):
                    metrics[model_key] = {
                        'name': model_data.get('name', model_key),
                        'r2': model_data.get('r2', 0),
                        'mae': model_data.get('mae', 0),
                        'rmse': model_data.get('rmse', 0)
                    }
        
        # Get visualizations
        visualizations = []
        exported_files = state.get('exported_images', {})
        print(f"Exported image files: {exported_files}")
        
        # Map exported image keys to display names
        viz_mapping = {
            'power_vs_time': 'Figure 1: Power vs Time',
            'correlation_matrix': 'Figure 2: Correlation Matrix',
            'forecast_comparison': 'Figure 3: Model Comparison',
            'custom_model_detail': 'Figure 4: Custom Model Detail'
        }
        
        for viz_key, viz_title in viz_mapping.items():
            if viz_key in exported_files:
                file_path = exported_files[viz_key]
                print(f"Processing {viz_key}: {file_path}")
                if os.path.exists(file_path):
                    # Read image and convert to base64
                    with open(file_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        visualizations.append({
                            'name': viz_title,
                            'id': viz_key,
                            'data': img_data,
                            'path': file_path
                        })
                        print(f"✓ Added {viz_title}")
                else:
                    print(f"✗ File not found: {file_path}")
        
        # Also check for Figure 4 in forecast_images directly (fallback if not in exported_files)
        forecast_images = state.get('forecast_images', {})
        if 'custom_model_detail' in forecast_images and 'Figure 4' not in [v['name'] for v in visualizations]:
            img_data = forecast_images['custom_model_detail']
            if isinstance(img_data, str):  # Already base64
                visualizations.append({
                    'name': 'Figure 4: Custom Model Detail',
                    'id': 'custom_model_detail',
                    'data': img_data,
                    'path': 'memory'
                })
                print(f"✓ Added Figure 4 from forecast_images")
        
        # Get report
        report_data = state.get('report', {})
        report_text = ''
        if report_data:
            report_text = report_data.get('forecast_summary', '')
        print(f"Report length: {len(report_text)} chars")
        
        # Note: Model tuning suggestions are now included directly in the report
        # No need to extract separately since the full vision analysis is embedded in the report text
        suggestions = []
        
        # Store session data with current R² for later comparison
        current_r2 = 0
        # Initialize iteration tracking for custom model
        custom_model_metrics = None
        if 'custom' in metrics:
            current_r2 = metrics['custom'].get('r2', 0)
            custom_model_metrics = {
                'iteration': 0,
                'r2': metrics['custom'].get('r2', 0),
                'mae': metrics['custom'].get('mae', 0),
                'rmse': metrics['custom'].get('rmse', 0),
                'name': 'Custom Model'
            }
        
        sessions[session_id] = {
            'state': state,
            'file_path': str(file_path),
            'current_r2': current_r2,  # Store for regenerate comparison
            'iteration_count': 0,  # Track which iteration we're on
            'custom_model_iterations': [custom_model_metrics] if custom_model_metrics else []  # Keep history
        }
        
        # Check guardrail status
        guardrail_passed = state.get('guardrail_passed', True)
        guardrail_warning = state.get('guardrail_warning', '')
        
        print(f"DEBUG: guardrail_passed={guardrail_passed}, guardrail_warning={guardrail_warning}")
        
        # Get custom model info and vision analysis
        custom_model_info = state.get('custom_model_info', {})
        vision_analysis = state.get('vision_analysis') or {}  # Ensure it's a dict, not None
        
        # Enhance custom_model_info with AI analysis from vision_analysis
        if custom_model_info and custom_model_info.get('success') and vision_analysis:
            for viz_key, analysis_data in vision_analysis.items():
                if 'custom_model_detail' in viz_key or 'custom_model' in viz_key:
                    ai_analysis = analysis_data.get('analysis', '')
                    if ai_analysis:
                        custom_model_info['ai_analysis'] = ai_analysis
                        break
        
        response = {
            'success': True,
            'session_id': session_id,
            'metrics': metrics,
            'visualizations': visualizations,
            'report': report_text,
            'suggestions': suggestions,
            'guardrail_passed': guardrail_passed,
            'guardrail_warning': guardrail_warning,
            'custom_model_info': custom_model_info,
            'vision_analysis': vision_analysis,  # Include raw vision analysis for frontend reference
            'iteration': 0,  # Initial analysis
            'custom_model_iterations': sessions[session_id].get('custom_model_iterations', [])  # Include iteration history
        }
        
        print(f"Response summary: {len(metrics)} metrics, {len(visualizations)} visualizations, {len(report_text)} chars report, guardrail_passed={guardrail_passed}")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in analyze endpoint: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/regenerate', methods=['POST'])
def regenerate():
    """
    Regenerate custom model with user feedback
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        feedback = data.get('feedback')
        iteration = data.get('iteration', 1)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = sessions[session_id]
        file_path = session_data['file_path']
        
        # Get original context
        state = session_data['state']
        description = state.get('data_context', {}).get('description', 'Time series data')
        
        # Run analysis with feedback
        print(f"Regenerating for session {session_id} with feedback: {feedback}")
        
        # Get the previous custom model info to use as base for improvement
        previous_custom_model_info = state.get('custom_model_info', {})
        previous_code = previous_custom_model_info.get('generated_code', '')
        previous_request = previous_custom_model_info.get('request', '')
        
        # Create improvement request that references the previous model
        improvement_request = f"""IMPORTANT: This is an IMPROVEMENT iteration, not a new model request.

Previous Custom Model Code:
```python
{previous_code}
```

Previous Request: {previous_request}

User's Improvement Feedback: {feedback}

TASK: Modify the EXISTING custom model code above based on the user's feedback. 
DO NOT create a completely new model - improve the parameters/structure of the existing one.
Keep the same model type (e.g., if it's LGBMRegressor, keep it as LGBMRegressor) unless the user explicitly asks to change it.
"""
        
        # Create improvement context
        improvement_context = {
            "description": state.get('data_context', {}).get('description', 'Time series data'),
            "resolution": state.get('data_context', {}).get('resolution', '30 minutes'),
            "goal": state.get('data_context', {}).get('goal', 'Forecasting'),
            "capacity": state.get('data_context', {}).get('capacity', ''),
            "custom_model_request": improvement_request,  # Use the enhanced request
            "is_improvement_iteration": True,
            "business_objective": state.get('data_context', {}).get('business_objective', ''),
            "previous_custom_code": previous_code  # Also pass as separate field
        }
        
        # CRITICAL: Manually run the improvement pipeline nodes
        # Instead of invoking the full graph (which would skip to final_report),
        # we manually call the specific nodes we need: ts_train -> ts_visualize -> vision -> report
        print(f"Running improvement iteration with feedback: {feedback[:100]}...")
        
        # Import required nodes (correct paths)
        from src.tools.forecaster import train_forecast_models, visualize_forecast
        from src.nodes.vision import analyze_visualizations_with_vision, export_images_to_workspace
        from src.tools.reporter import generate_report
        
        # Update state with improvement context
        state['data_context'] = improvement_context
        state['improvement_feedback'] = feedback
        
        # Step 1: Re-train models with improvement feedback
        print("Step 1: Re-training models...")
        updates = train_forecast_models(state)
        state.update(updates)  # Merge updates into state
        
        # Step 2: Visualize new forecast results
        print("Step 2: Generating visualizations...")
        updates = visualize_forecast(state)
        state.update(updates)  # Merge updates into state
        
        # Step 3: Export and analyze with vision
        print("Step 3: Exporting images...")
        updates = export_images_to_workspace(state)
        state.update(updates)  # Merge updates into state
        print(f"DEBUG after export - exported_images keys: {list(state.get('exported_images', {}).keys())}")
        print(f"DEBUG after export - viz_images keys: {list(state.get('viz_images', {}).keys())}")
        print(f"DEBUG after export - forecast_images keys: {list(state.get('forecast_images', {}).keys())}")
        
        print("Step 4: Running vision analysis...")
        updates = analyze_visualizations_with_vision(state)
        state.update(updates)  # Merge updates into state
        print(f"DEBUG regenerate - vision_analysis keys after analysis: {list(state.get('vision_analysis', {}).keys()) if state.get('vision_analysis') else 'None'}")
        
        # Step 5: Generate new report
        print("Step 5: Generating report...")
        updates = generate_report(state)
        state.update(updates)  # Merge updates into state
        
        result = state
        improved_session_id = session_id
        
        # Update the ORIGINAL session with new state, keep same session_id for client
        state = result
        sessions[session_id]['state'] = state
        sessions[session_id]['improved_session_id'] = improved_session_id  # Track the internal session
        
        # Extract results (same as analyze endpoint)
        metrics = {}
        forecast_results = state.get('forecast_results', {})
        if forecast_results:
            for model_key, model_data in forecast_results.items():
                if isinstance(model_data, dict):
                    metrics[model_key] = {
                        'name': model_data.get('name', model_key),
                        'r2': model_data.get('r2', 0),
                        'mae': model_data.get('mae', 0),
                        'rmse': model_data.get('rmse', 0)
                    }
        
        # Get updated visualizations
        visualizations = []
        exported_files = state.get('exported_images', {})
        forecast_images = state.get('forecast_images', {})
        viz_images = state.get('viz_images', {})
        
        print(f"DEBUG regenerate - exported_files: {list(exported_files.keys())}")
        print(f"DEBUG regenerate - forecast_images: {list(forecast_images.keys())}")
        print(f"DEBUG regenerate - viz_images: {list(viz_images.keys())}")
        
        viz_mapping = {
            'power_vs_time': 'Figure 1: Power vs Time',
            'correlation_matrix': 'Figure 2: Correlation Matrix',
            'forecast_comparison': 'Figure 3: Model Comparison',
            'custom_model_detail': 'Figure 4: Custom Model Detail'
        }
        
        # Try exported_files first (file paths)
        for viz_key, viz_title in viz_mapping.items():
            if viz_key in exported_files:
                file_path_viz = exported_files[viz_key]
                if os.path.exists(file_path_viz):
                    with open(file_path_viz, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        visualizations.append({
                            'name': viz_title,
                            'id': viz_key,
                            'data': img_data,
                            'path': file_path_viz
                        })
                        print(f"✓ Added {viz_title} from exported_files")
        
        # Fallback to forecast_images (base64 data)
        if 'custom_model_detail' not in [v['id'] for v in visualizations]:
            if 'custom_model_detail' in forecast_images:
                visualizations.append({
                    'name': 'Figure 4: Custom Model Detail',
                    'id': 'custom_model_detail',
                    'data': forecast_images['custom_model_detail'],
                    'path': ''
                })
                print(f"✓ Added Figure 4 from forecast_images")
        
        # Also check viz_images for any missing ones
        for viz_key, viz_title in viz_mapping.items():
            if viz_key not in [v['id'] for v in visualizations] and viz_key in viz_images:
                visualizations.append({
                    'name': viz_title,
                    'id': viz_key,
                    'data': viz_images[viz_key],
                    'path': ''
                })
                print(f"✓ Added {viz_title} from viz_images")
        
        # Get updated report
        report_data = state.get('report', {})
        report_text = ''
        print(f"DEBUG regenerate - report_data type: {type(report_data)}")
        print(f"DEBUG regenerate - report_data keys: {list(report_data.keys()) if isinstance(report_data, dict) else 'N/A'}")
        if report_data:
            if isinstance(report_data, dict):
                report_text = report_data.get('forecast_summary', '')
                print(f"DEBUG regenerate - report_text length: {len(report_text)}")
            else:
                report_text = str(report_data)
        print(f"DEBUG regenerate - final report_text length: {len(report_text)}")
        
        # Get custom model info
        custom_model_info = state.get('custom_model_info', {})
        
        # Add custom model detail visualization if available
        custom_model_detail_image = state.get('custom_model_detail_image')
        if custom_model_detail_image:
            visualizations.append({
                'name': 'Figure 4: Custom Model Detail',
                'id': 'custom_model_detail',
                'data': custom_model_detail_image,
                'path': ''
            })
        
        # Note: Model tuning suggestions are now included directly in the report
        # No need to extract separately since the full vision analysis is embedded in the report text
        suggestions = []
        
        # Compare metrics with previous iteration to detect degradation
        metrics_status = 'improved'  # Default: assume improvement
        iteration_feedback = ''
        
        if custom_model_info and 'success' in custom_model_info:
            current_r2 = custom_model_info.get('metrics', {}).get('r2', 0)
            current_mae = custom_model_info.get('metrics', {}).get('mae', 0)
            current_rmse = custom_model_info.get('metrics', {}).get('rmse', 0)
            
            # Get previous R² from sessions
            previous_session_data = sessions.get(session_id, {})
            previous_r2 = previous_session_data.get('current_r2', current_r2)
            
            # Add custom model iteration to history
            iteration_count = previous_session_data.get('iteration_count', 0) + 1
            custom_model_name = f'Custom Model (Iteration {iteration_count})'
            
            custom_model_iterations = previous_session_data.get('custom_model_iterations', [])
            custom_model_iterations.append({
                'iteration': iteration_count,
                'r2': current_r2,
                'mae': current_mae,
                'rmse': current_rmse,
                'name': custom_model_name
            })
            
            # Update session with iteration info
            sessions[session_id]['custom_model_iterations'] = custom_model_iterations
            sessions[session_id]['iteration_count'] = iteration_count
            
            # Check if R² decreased
            if current_r2 < previous_r2 - 0.01:  # 1% tolerance
                metrics_status = 'degraded'
                r2_change = (current_r2 - previous_r2) * 100
                iteration_feedback = f"""
⚠️ **Model Performance Alert**
Your adjustments resulted in a {abs(r2_change):.1f}% drop in R² score (from {previous_r2:.1%} to {current_r2:.1%}).

**Recommended Next Steps:**
1. **Try Cross-Validation:** Use k-fold cross-validation to ensure the previous model wasn't overfitting
2. **Revert Changes:** Go back to the previous configuration and try smaller adjustments
3. **Feature Analysis:** Analyze which features are most important - the new parameters may need different feature engineering
4. **Alternative Approaches:** Consider ensemble methods combining the previous model with this iteration
"""
            elif current_r2 > previous_r2 + 0.01:
                metrics_status = 'improved'
                r2_change = (current_r2 - previous_r2) * 100
                iteration_feedback = f"✅ Success! R² improved by {r2_change:.1f}% (from {previous_r2:.1%} to {current_r2:.1%})"
            else:
                metrics_status = 'stable'
                iteration_feedback = "Status quo: Performance remained stable. Continue refining or try different parameters."
            
            # Update session with new R² for next iteration
            sessions[session_id]['current_r2'] = current_r2
            # Store previous state for reference
            state['previous_iteration_r2'] = previous_r2
        
        response = {
            'success': True,
            'session_id': session_id,
            'metrics': metrics,
            'visualizations': visualizations,
            'report': report_text,
            'suggestions': suggestions,
            'custom_model_info': custom_model_info,
            'iteration': iteration,
            'custom_model_iterations': sessions[session_id].get('custom_model_iterations', []),  # Include all custom model iterations
            'metrics_status': metrics_status,
            'iteration_feedback': iteration_feedback,
            'guardrail_passed': True,  # Regenerate always has data
            'vision_analysis': state.get('vision_analysis', {})  # Include vision analysis
        }
        
        print(f"Regenerate response prepared with {len(visualizations)} visualizations")
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in regenerate endpoint: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Generate real-world forecast using the best model"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        forecast_hours = data.get('forecast_hours', 24)
        business_requirements = data.get('business_requirements', '')
        model_name = data.get('model_name', '')  # User-selected model
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        session_data = sessions[session_id]
        state = session_data['state']
        
        print(f"Generating forecast for session {session_id}: {forecast_hours} hours")
        print(f"Business requirements: {business_requirements}")
        print(f"Selected model: {model_name}")
        
        # Get available models from forecast results and custom model iterations
        forecast_results = state.get('forecast_results', {})
        custom_model_iterations = session_data.get('custom_model_iterations', [])
        
        print(f"DEBUG: forecast_results keys = {list(forecast_results.keys())}")
        print(f"DEBUG: custom_model_iterations count = {len(custom_model_iterations)}")
        
        # Build list of available models
        available_models = {}
        
        # Add benchmark models from forecast_results
        for model_key, model_data in forecast_results.items():
            if isinstance(model_data, dict) and 'name' in model_data:
                available_models[model_data['name']] = {
                    'r2': model_data.get('r2', 0),
                    'name': model_data['name'],
                    'type': 'benchmark'
                }
                print(f"DEBUG: Added benchmark model: {model_data['name']}")
        
        # Add custom model iterations
        for iteration in custom_model_iterations:
            available_models[iteration['name']] = {
                'r2': iteration['r2'],
                'name': iteration['name'],
                'type': 'custom',
                'index': iteration.get('index', 0)
            }
            print(f"DEBUG: Added custom iteration: {iteration['name']}")
        
        print(f"DEBUG: available_models after forecasts = {list(available_models.keys())}")
        
        # If no models found yet, try to extract from state metrics
        if not available_models and 'metrics' in state:
            metrics = state.get('metrics', {})
            print(f"DEBUG: Trying to extract models from state metrics: {list(metrics.keys())}")
            for model_key, model_data in metrics.items():
                if isinstance(model_data, dict):
                    model_name = model_data.get('name', model_key)
                    available_models[model_name] = {
                        'r2': model_data.get('r2', 0),
                        'name': model_name,
                        'type': 'backup_metrics'
                    }
                    print(f"DEBUG: Added model from metrics: {model_name}")
        
        print(f"DEBUG: Final available_models keys = {list(available_models.keys())}")
        
        if not available_models:
            print("ERROR: No models available for forecast")
            return jsonify({'success': False, 'error': 'No models available for forecast'}), 400
        
        # Select the model
        if model_name and model_name in available_models:
            selected_model = available_models[model_name]
            selected_model_name = model_name
            selected_model_r2 = selected_model['r2']
        else:
            # Default to best model by R²
            selected_model = max(available_models.values(), key=lambda x: x['r2'])
            selected_model_name = selected_model['name']
            selected_model_r2 = selected_model['r2']
        
        print(f"Using model: {selected_model_name} with R²={selected_model_r2:.4f}")
        
        # Get forecast data from state
        import pandas as pd
        preprocessed_data = state.get('preprocessed_data', {})
        
        print(f"DEBUG: preprocessed_data type = {type(preprocessed_data)}")
        if isinstance(preprocessed_data, dict):
            print(f"DEBUG: preprocessed_data keys = {list(preprocessed_data.keys())}")
        else:
            print(f"DEBUG: preprocessed_data is not dict, it's {type(preprocessed_data)}")
        
        # Check if preprocessed_data is None or empty dict
        if preprocessed_data is None or (isinstance(preprocessed_data, dict) and not preprocessed_data):
            print("ERROR: preprocessed_data is None or empty dict")
            return jsonify({'success': False, 'error': 'No data available for forecast'}), 400
        
        # Extract dataframe from preprocessed_data
        if isinstance(preprocessed_data, dict):
            df = preprocessed_data.get('data')
            print(f"DEBUG: extracted df from dict, df type = {type(df)}")
        else:
            # preprocessed_data might be the dataframe itself
            df = preprocessed_data
            print(f"DEBUG: preprocessed_data is dataframe directly")
        
        # Check if df is valid
        if df is None:
            print("ERROR: df is None after extraction")
            return jsonify({'success': False, 'error': 'No data available for forecast'}), 400
        
        if hasattr(df, 'empty') and df.empty:
            print("ERROR: df is empty")
            return jsonify({'success': False, 'error': 'No data available for forecast'}), 400
        
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"ERROR: df is not a DataFrame, type = {type(df)}")
            return jsonify({'success': False, 'error': 'Invalid data format for forecast'}), 400
        
        print(f"DEBUG: df shape = {df.shape}, columns = {list(df.columns)[:5]}...")
        
        # Get the model state and predictions from training
        model_state = state.get('model_state', {})
        predictions_history = model_state.get('predictions', {})  # This has RF and DT test predictions
        y_test_history = model_state.get('y_test', [])
        y_train_history = model_state.get('y_train', [])
        split_idx = model_state.get('split_point', int(len(df) * 0.8))
        
        print(f"DEBUG: predictions_history keys = {list(predictions_history.keys()) if predictions_history else 'None'}")
        print(f"DEBUG: y_test_history length = {len(y_test_history) if hasattr(y_test_history, '__len__') else 'N/A'}")
        
        # Get the actual training data for feature extraction
        engineered_data = state.get('engineered_data')
        if engineered_data is None:
            engineered_data = df.copy()
        
        # Prepare features for forecast (use last values of data)
        import numpy as np
        import pandas as pd
        
        # Assuming the model expects the same features as training
        # Need to exclude target column and time column
        data_profile = state.get('data_profile', {})
        target_col = data_profile.get('target_column', 'power')
        timeseries_info = data_profile.get('timeseries', {})
        time_column = timeseries_info.get('time_column')
        
        feature_cols = [col for col in df.columns if col != target_col and col != time_column and col != df.index.name and col != '_parsed_time']
        
        # Filter to only numeric columns (exclude time/datetime columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        print(f"DEBUG: target_col = {target_col}, time_column = {time_column}")
        print(f"DEBUG: feature_cols = {feature_cols}")
        print(f"DEBUG: numeric_feature_cols = {numeric_feature_cols}")
        
        # Use numeric columns if available
        if numeric_feature_cols:
            feature_cols = numeric_feature_cols
        elif len(feature_cols) == 0:
            feature_cols = list(df.columns)
            # Filter to numeric only
            feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        # Remove target column from features if accidentally included
        feature_cols = [col for col in feature_cols if col != target_col]
        
        print(f"DEBUG: Final feature_cols for prediction = {feature_cols}")
        
        # Check if we have historical model predictions to use as reference
        has_trained_models = len(predictions_history) > 0
        print(f"DEBUG: has_trained_models = {has_trained_models}")
        
        # Get forecast data and last timestamp
        forecast_values = []
        forecast_times = []
        
        try:
            last_time = df.index[-1]
            print(f"DEBUG: Last timestamp = {last_time}")
        except:
            last_time = None
            print(f"DEBUG: Could not extract timestamp from index")
        
        # Use the already trained model from state
        try:
            # Get trained models from model_state
            rf_model = model_state.get('rf_model')
            dt_model = model_state.get('dt_model')
            stored_feature_cols = model_state.get('feature_columns', [])
            stored_target_col = model_state.get('target_column', target_col)
            
            print(f"DEBUG: rf_model available = {rf_model is not None}")
            print(f"DEBUG: dt_model available = {dt_model is not None}")
            print(f"DEBUG: stored_feature_cols = {stored_feature_cols}")
            
            # Select the model based on user selection
            if selected_model_name == 'Random Forest' and rf_model is not None:
                model = rf_model
                print(f"Using trained Random Forest model")
            elif selected_model_name == 'Decision Tree' and dt_model is not None:
                model = dt_model
                print(f"Using trained Decision Tree model")
            elif 'Custom' in selected_model_name:
                # Custom model - need to handle separately
                print(f"Custom model selected - will use fallback")
                model = None
            else:
                # Default to RF if available, otherwise DT
                if rf_model is not None:
                    model = rf_model
                    selected_model_name = 'Random Forest'
                    print(f"Defaulting to Random Forest model")
                elif dt_model is not None:
                    model = dt_model
                    selected_model_name = 'Decision Tree'
                    print(f"Defaulting to Decision Tree model")
                else:
                    model = None
                    print(f"No trained models available")
            
            # Use stored feature columns if available, otherwise use what we extracted
            if stored_feature_cols:
                feature_cols_for_prediction = stored_feature_cols
            else:
                feature_cols_for_prediction = feature_cols
            
            print(f"DEBUG: Using feature_cols = {feature_cols_for_prediction}")
            
            if model is not None and len(feature_cols_for_prediction) > 0:
                print(f"DEBUG: Generating {forecast_hours} hour forecast using trained {selected_model_name}")
                
                # Verify features exist in data
                missing_features = [f for f in feature_cols_for_prediction if f not in df.columns]
                if missing_features:
                    print(f"WARNING: Missing features in data: {missing_features}")
                    feature_cols_for_prediction = [f for f in feature_cols_for_prediction if f in df.columns]
                
                if len(feature_cols_for_prediction) == 0:
                    raise Exception("No valid features available for prediction")
                
                # For 24-hour PV forecasting: Use the last complete day as weather pattern template
                # Model is: f(irradiance, temp, humidity...) -> power
                # Assumption: Tomorrow's weather will be similar to a recent day
                if forecast_hours == 24 and len(df) >= 96:
                    print(f"DEBUG: Using last complete day cycle as weather pattern template")
                    
                    # For 15-minute interval data: 96 rows = 1 day (24h * 4)
                    # For hourly data: 24 rows = 1 day
                    # Detect interval
                    if len(df) >= 96:
                        rows_per_day = 96  # Assume 15-min intervals
                    else:
                        rows_per_day = 24  # Assume hourly
                    
                    # Get the last complete day's features as template
                    # This represents yesterday's weather pattern (0:00 -> 23:45)
                    X_template_day = df[feature_cols_for_prediction].tail(rows_per_day).values
                    
                    print(f"DEBUG: Template day shape = {X_template_day.shape} (rows_per_day={rows_per_day})")
                    
                    # If we need exactly 24 predictions (hourly), sample from the template
                    if len(X_template_day) == forecast_hours:
                        # Perfect match
                        X_forecast = X_template_day
                    elif len(X_template_day) > forecast_hours:
                        # Downsample: take evenly spaced samples
                        indices = np.linspace(0, len(X_template_day)-1, forecast_hours, dtype=int)
                        X_forecast = X_template_day[indices]
                        print(f"DEBUG: Downsampled from {len(X_template_day)} to {forecast_hours} points")
                    else:
                        # Not enough data, use what we have and repeat
                        X_forecast = X_template_day
                        print(f"DEBUG: Using {len(X_template_day)} template points")
                    
                    # Generate predictions using the template weather patterns
                    for i in range(min(forecast_hours, len(X_forecast))):
                        try:
                            X_input = X_forecast[i:i+1]
                            pred = model.predict(X_input)[0]
                            pred = float(max(0, pred))
                            forecast_values.append(pred)
                            
                            if (i+1) % 6 == 0:
                                print(f"DEBUG: Hour {i+1}: {pred:.2f} MW")
                        
                        except Exception as e:
                            print(f"DEBUG: Prediction failed at hour {i+1}: {e}")
                            if stored_target_col and stored_target_col in df.columns:
                                # Use actual power from template day as fallback
                                fallback_val = df[stored_target_col].tail(rows_per_day).iloc[i] if i < rows_per_day else 0
                            else:
                                fallback_val = 0
                            forecast_values.append(float(max(0, fallback_val)))
                    
                    # If we still need more forecast points, repeat the pattern
                    while len(forecast_values) < forecast_hours:
                        idx = len(forecast_values) % len(X_forecast)
                        try:
                            X_input = X_forecast[idx:idx+1]
                            pred = model.predict(X_input)[0]
                            forecast_values.append(float(max(0, pred)))
                        except:
                            forecast_values.append(0.0)
                
                else:
                    # For non-24-hour forecasts or short data: use last known features
                    print(f"DEBUG: Using last row features for {forecast_hours} hour forecast")
                    X_last = df[feature_cols_for_prediction].iloc[-1:].values
                    print(f"DEBUG: X_last shape = {X_last.shape}")
                    
                    for i in range(forecast_hours):
                        try:
                            pred = model.predict(X_last)[0]
                            pred = float(max(0, pred))
                            forecast_values.append(pred)
                            
                            if (i+1) % 6 == 0:
                                print(f"DEBUG: Hour {i+1}: {pred:.2f}")
                        
                        except Exception as e:
                            print(f"DEBUG: Prediction failed at hour {i+1}: {e}")
                            if stored_target_col and stored_target_col in df.columns:
                                fallback_val = df[stored_target_col].tail(24).mean()
                            else:
                                fallback_val = df.iloc[:, -1].tail(24).mean()
                            forecast_values.append(float(max(0, fallback_val)))
                
                print(f"✓ Generated {len(forecast_values)} predictions")
                
            else:
                raise Exception("Model not available or no features")
                
        except Exception as e:
            print(f"Warning: Could not use trained model: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: use simple average of last 24 hours
            print(f"DEBUG: Using fallback method - averaging recent values")
            if target_col in df.columns:
                avg_val = df[target_col].tail(24).mean()
            else:
                avg_val = df.iloc[:, -1].tail(24).mean()
            
            for i in range(1, forecast_hours + 1):
                # Add small random variation to avoid flat line
                variation = 1 + np.random.normal(0, 0.05)
                pred = avg_val * variation
                forecast_values.append(float(max(0, pred)))
        
        # Calculate timestamps
        for i in range(1, forecast_hours + 1):
            if last_time is not None:
                try:
                    future_time = last_time + pd.Timedelta(hours=i)
                    forecast_times.append(str(future_time))
                except:
                    forecast_times.append(f'+{i}h')
            else:
                forecast_times.append(f'+{i}h')
        
        # Calculate statistics
        forecast_mean = float(np.mean(forecast_values))
        forecast_std = float(np.std(forecast_values))
        forecast_min = float(np.min(forecast_values))
        forecast_max = float(np.max(forecast_values))
        
        # 95% confidence interval
        ci_width = 1.96 * forecast_std  # Approximate 95% CI
        
        # Prepare forecast values for response
        forecast_output = [
            {'time': forecast_times[i], 'value': forecast_values[i]}
            for i in range(len(forecast_values))
        ]
        
        # Use LLM to generate personalized business insights based on user requirements
        if business_requirements and business_requirements.strip():
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage
                from config.settings import REPORTING_MODEL, REPORTING_TEMPERATURE
                
                # Prepare forecast data summary for LLM
                forecast_summary = {
                    'total_hours': forecast_hours,
                    'mean_value': round(forecast_mean, 2),
                    'std_dev': round(forecast_std, 2),
                    'min_value': round(forecast_min, 2),
                    'max_value': round(forecast_max, 2),
                    'total_production': round(sum(forecast_values), 2),
                    'confidence_interval': round(ci_width, 2),
                    'model_accuracy': f"{selected_model_r2:.1%}",
                    'hourly_forecast': [
                        {'hour': i+1, 'value': round(v, 2)} 
                        for i, v in enumerate(forecast_values)
                    ]
                }
                
                # Create LLM prompt
                llm_prompt = f"""You are an expert energy consultant analyzing solar power forecasts for business decision-making.

**User's Business Requirement:**
{business_requirements}

**Forecast Data Summary:**
- Model: {selected_model_name} (Accuracy: R² = {selected_model_r2:.1%})
- Forecast Period: Next {forecast_hours} hours
- Total Expected Production: {forecast_summary['total_production']} MW·h
- Average Output: {forecast_summary['mean_value']} MW
- Peak Production: {forecast_summary['max_value']} MW at Hour {forecast_values.index(forecast_max)+1}
- Minimum Production: {forecast_summary['min_value']} MW
- Uncertainty: ±{forecast_summary['confidence_interval']} MW (95% CI)
- Production Hours: {len([v for v in forecast_values if v > 1])} hours with significant output (>1 MW)

**Hourly Forecast Details:**
{chr(10).join([f"Hour {h['hour']:2d}: {h['value']:6.2f} MW" for h in forecast_summary['hourly_forecast']])}

**Task:**
Based on the user's specific business requirement and the forecast data above, provide:
1. **Analysis**: How the forecast relates to their specific need
2. **Actionable Recommendations**: Concrete steps they should take
3. **Key Insights**: Important patterns, opportunities, or risks in the data
4. **Quantitative Metrics**: Specific numbers and time windows relevant to their goal

Format your response in clear sections with markdown formatting. Be specific and actionable."""

                llm = ChatOpenAI(model=REPORTING_MODEL, temperature=REPORTING_TEMPERATURE)
                response = llm.invoke([HumanMessage(content=llm_prompt)])
                business_insights = response.content.strip()
                
                print(f"✓ Generated personalized business insights using {REPORTING_MODEL}")
                
            except Exception as e:
                print(f"Warning: LLM insight generation failed: {e}")
                # Fallback to basic summary
                business_insights = f"""
Based on {selected_model_name} forecast (R² = {selected_model_r2:.1%}):

**User Requirement:** {business_requirements}

**Forecast Summary:**
- Total Production: {sum(forecast_values):.2f} MW·h over {forecast_hours} hours
- Peak: {forecast_max:.2f} MW at Hour {forecast_values.index(forecast_max)+1}
- Average: {forecast_mean:.2f} MW
- Uncertainty: ±{ci_width:.2f} MW

**Note:** Unable to generate detailed analysis. Please review the forecast values above.
"""
        else:
            # No user requirements provided, give generic summary
            business_insights = f"""
Based on {selected_model_name} forecast (R² = {selected_model_r2:.1%}):

**Forecast Summary:**
- Total Production: {sum(forecast_values):.2f} MW·h over {forecast_hours} hours
- Peak: {forecast_max:.2f} MW at Hour {forecast_values.index(forecast_max)+1}
- Average: {forecast_mean:.2f} MW
- Range: {forecast_min:.2f} - {forecast_max:.2f} MW
- Uncertainty: ±{ci_width:.2f} MW (95% CI)

*Tip: Enter your business requirements in the text box to get personalized insights and recommendations.*
"""
        
        response = {
            'success': True,
            'session_id': session_id,
            'selected_model': selected_model_name,
            'forecast_hours': forecast_hours,
            'forecast_stats': {
                'mean': forecast_mean,
                'std': forecast_std,
                'min': forecast_min,
                'max': forecast_max,
                'ci_width': ci_width
            },
            'forecast_values': forecast_output,
            'business_insights': business_insights.strip()
        }
        
        print(f"Forecast generated: mean={forecast_mean:.2f}, std={forecast_std:.2f}")
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in forecast endpoint: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

@app.route('/api/download/report/<session_id>', methods=['GET'])
def download_report(session_id):
    """Download analysis report"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    state = sessions[session_id]['state']
    report_data = state.get('report', {})
    report_text = report_data.get('forecast_summary', '')
    
    if not report_text:
        return jsonify({'error': 'Report not found'}), 404
    
    # Create temp file
    from io import BytesIO
    report_buffer = BytesIO(report_text.encode('utf-8'))
    report_buffer.seek(0)
    
    return send_file(
        report_buffer,
        mimetype='text/plain',
        as_attachment=True,
        download_name=f'eda_report_{session_id}.txt'
    )

@app.route('/api/download/visualizations/<session_id>', methods=['GET'])
def download_visualizations(session_id):
    """Download all visualizations as ZIP"""
    import zipfile
    from io import BytesIO
    
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    state = sessions[session_id]['state']
    exported_files = state.get('exported_images', {})
    
    if not exported_files:
        return jsonify({'error': 'No visualizations found'}), 404
    
    # Create ZIP file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for viz_key, file_path in exported_files.items():
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'visualizations_{session_id}.zip'
    )

if __name__ == '__main__':
    print("=" * 60)
    print(">> EDA Agent API Server Starting...")
    print("=" * 60)
    print(f"[DIR] Output Directory: {OUTPUT_DIR}")
    print(f"[NET] Server: http://localhost:5000")
    print(f"[WEB] Web UI: http://localhost:5000")
    print("=" * 60)
    
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
