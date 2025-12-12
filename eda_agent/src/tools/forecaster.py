"""
Time series forecasting tool with multiple models:
- Random Forest (benchmark)
- Decision Tree (benchmark)
- XGBoost (benchmark)
- Prophet (benchmark)
- Custom Models (LLM-generated from natural language)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ..state import EDAState, memory_update
from .custom_model_generator import (
    generate_custom_model_code,
    execute_custom_model_code,
    validate_custom_model,
    reflect_and_regenerate
)

# Import optional dependencies with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. XGBoost model will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Prophet model will be skipped.")


def analyze_ts_suitability(state: EDAState) -> dict:
    """Analyze if data is suitable for time series forecasting"""
    step_name = "ts_suitable"
    data_profile = state.get('data_profile', {})

    timeseries_info = data_profile.get('timeseries', {})
    time_column = timeseries_info.get('time_column')
    timespan_days = timeseries_info.get('timespan_days', 0)
    enough_history = timeseries_info.get('enough_history', False)

    ts_analysis = {
        'has_time_column': time_column is not None,
        'timespan_days': timespan_days,
        'enough_history': enough_history,
        'suitable_for_neural_network': False,
        'suitable_for_statistical': False,
        'recommended_method': None,
        'prerequisites_met': True,
        'method_rationale': ''
    }

    if not time_column:
        ts_analysis['prerequisites_met'] = False
        ts_analysis['method_rationale'] = 'No time column detected. Cannot perform time series forecasting. Consider regression models instead.'
        ts_analysis['recommended_method'] = 'regression'
        out = {
            'ts_suitable': False,
            'ts_method': 'regression',
            'ts_analysis': ts_analysis
        }
        out.update(
            memory_update(
                step_name,
                'TS suitability: no time column -> regression',
                warnings=['No time column']
            )
        )
        return out

    if timespan_days >= 30 and enough_history:
        ts_analysis['suitable_for_neural_network'] = True
        ts_analysis['suitable_for_statistical'] = True
        ts_analysis['recommended_method'] = 'neural_network'
        ts_analysis['method_rationale'] = f'Sufficient historical data ({timespan_days:.1f} days). Neural networks recommended.'
    elif timespan_days >= 7:
        ts_analysis['suitable_for_neural_network'] = False
        ts_analysis['suitable_for_statistical'] = True
        ts_analysis['recommended_method'] = 'statistical'
        ts_analysis['method_rationale'] = f'Limited data ({timespan_days:.1f} days). Statistical methods recommended.'
    else:
        ts_analysis['prerequisites_met'] = False
        ts_analysis['recommended_method'] = 'baseline'
        ts_analysis['method_rationale'] = f'Insufficient historical data ({timespan_days:.1f} days). Only baseline methods feasible.'

    out = {
        'ts_suitable': ts_analysis['suitable_for_neural_network'] or ts_analysis['suitable_for_statistical'],
        'ts_method': ts_analysis['recommended_method'],
        'ts_analysis': ts_analysis
    }
    
    msg = f"TS suitability: method={ts_analysis['recommended_method']} span={timespan_days:.1f}d suitable={out['ts_suitable']}"
    out.update(
        memory_update(
            step_name,
            msg,
            warnings=[] if out['ts_suitable'] else ["Limited/insufficient timespan"]
        )
    )
    return out


def train_forecast_models(state: EDAState) -> dict:
    """
    Train multiple forecast models:
    - Random Forest (benchmark)
    - Decision Tree (benchmark)  
    - XGBoost (benchmark, if available)
    - Prophet (benchmark, if available)
    - Custom Models (LLM-generated from user's natural language description)
    
    User can specify custom models via 'custom_model_request' in data_context:
    Example: data_context['custom_model_request'] = "I want a stacking ensemble of XGBoost and LightGBM"
    """
    step_name = "train_forecast"
    
    engineered_data = state.get('engineered_data')
    if engineered_data is None or engineered_data.empty:
        out = {'model_state': None, 'forecast_results': {}}
        out.update(
            memory_update(
                step_name,
                "Forecast skipped: no engineered data",
                warnings=["No data to train on"]
            )
        )
        return out

    data_profile = state.get('data_profile', {})
    target_col = data_profile.get('target_column')
    if not target_col or target_col not in engineered_data.columns:
        out = {'model_state': None, 'forecast_results': {}}
        out.update(
            memory_update(
                step_name,
                "Forecast skipped: no valid target column",
                warnings=[f"Target column '{target_col}' not found"]
            )
        )
        return out

    # Check for time column
    timeseries_info = data_profile.get('timeseries', {})
    time_column = timeseries_info.get('time_column')
    has_timestamps = time_column is not None and '_parsed_time' in engineered_data.columns

    # Prepare features and target
    drop_cols = [target_col, '_parsed_time'] if '_parsed_time' in engineered_data.columns else [target_col]
    if time_column and time_column in engineered_data.columns and time_column != '_parsed_time':
        drop_cols.append(time_column)
    
    X = engineered_data.drop(columns=drop_cols, errors='ignore')
    y = engineered_data[target_col]
    
    # Check for highly correlated features (multicollinearity)
    if len(X.columns) > 1:
        corr_matrix = X.corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # Note: High correlation detected but not removing features automatically
        # This preserves all information for the models to use

    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) < 10 or len(X_test) < 2:
        out = {'model_state': None, 'forecast_results': {}}
        out.update(
            memory_update(
                step_name,
                f"Forecast skipped: insufficient data (train={len(X_train)}, test={len(X_test)})",
                warnings=["Too few samples for reliable forecasting"]
            )
        )
        return out

    # Dictionary to store all predictions and metrics
    predictions = {}
    metrics = {}
    
    # Check for custom model request
    data_context = state.get('data_context', {})
    custom_request = data_context.get('custom_model_request')
    
    if custom_request:
        print(f"\n🤖 Generating custom model from request: '{custom_request}'\n")
        
        max_attempts = 7
        attempt = 0
        custom_model_generated = False
        last_code = ""  # Initialize for reflection attempts
        last_error = ""  # Initialize for reflection attempts
        
        while attempt < max_attempts and not custom_model_generated:
            attempt += 1
            
            try:
                # First attempt: generate code
                if attempt == 1:
                    print(f"[Attempt {attempt}/{max_attempts}] Generating code...")
                    gen_result = generate_custom_model_code(custom_request, data_context)
                else:
                    # Subsequent attempts: reflect and regenerate
                    print(f"\n[Attempt {attempt}/{max_attempts}] Reflecting on error and regenerating...")
                    gen_result = reflect_and_regenerate(
                        custom_request, 
                        data_context, 
                        last_code, 
                        last_error,
                        attempt - 1
                    )
                
                if not gen_result['success']:
                    print(f"✗ Code generation failed: {gen_result['error']}")
                    break
                
                last_code = gen_result['code']
                print(f"✓ Code generation successful")
                if attempt > 1:
                    print(f"  Reflection: {gen_result.get('reflection', 'Code regenerated')}")
                print(f"Generated code:\n{'-'*60}\n{last_code}\n{'-'*60}\n")
                
                # Execute generated code
                print(f"[Attempt {attempt}/{max_attempts}] Executing code...")
                exec_result = execute_custom_model_code(
                    last_code,
                    X_train,
                    y_train,
                    X_test
                )
                
                if not exec_result['success']:
                    last_error = exec_result['error']
                    print(f"✗ Execution failed: {last_error}")
                    if attempt < max_attempts:
                        print(f"  Will retry with reflection...")
                    continue
                
                custom_model = exec_result['model']
                custom_pred = exec_result['predictions']
                print(f"✓ Code executed successfully")
                
                # Validate model
                print(f"[Attempt {attempt}/{max_attempts}] Validating model...")
                is_valid, validation_msg = validate_custom_model(custom_model, X_train[:5])
                
                if not is_valid:
                    last_error = validation_msg
                    print(f"✗ Validation failed: {validation_msg}")
                    if attempt < max_attempts:
                        print(f"  Will retry with reflection...")
                    continue
                
                # Success!
                print(f"✓ Custom model validated successfully")
                
                # Add to predictions
                predictions['custom'] = custom_pred
                metrics['custom'] = {
                    'name': 'Custom Model',
                    'mae': mean_absolute_error(y_test, custom_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, custom_pred)),
                    'r2': r2_score(y_test, custom_pred),
                    'generated_code': last_code,
                    'attempts': attempt
                }
                print(f"✓ Custom model metrics - MAE: {metrics['custom']['mae']:.4f}, "
                      f"RMSE: {metrics['custom']['rmse']:.4f}, R²: {metrics['custom']['r2']:.4f}")
                print(f"✓ Generation successful after {attempt} attempt(s)\n")
                
                custom_model_generated = True
                
            except Exception as e:
                last_error = str(e)
                print(f"✗ Unexpected error: {e}")
                if attempt < max_attempts:
                    print(f"  Will retry with reflection...")
                import traceback
                traceback.print_exc()
        
        if not custom_model_generated:
            print(f"\n✗ Failed to generate custom model after {max_attempts} attempts")
            print(f"  Last error: {last_error}")
            
            # Check if there's a checkpoint to fallback to
            previous_checkpoint = data_context.get('previous_checkpoint')
            if previous_checkpoint and data_context.get('is_improvement_iteration'):
                print(f"  📦 Restoring from checkpoint (previous successful model)...")
                
                # Restore checkpoint model
                predictions['custom'] = None  # Will be filled from checkpoint later
                metrics['custom'] = {
                    'name': 'Custom Model (Checkpoint)',
                    'mae': previous_checkpoint['mae'],
                    'rmse': previous_checkpoint['rmse'],
                    'r2': previous_checkpoint['r2']
                }
                
                custom_model_info = {
                    'success': True,
                    'generated_code': previous_checkpoint['code'],
                    'request': f"{previous_checkpoint['request']} (restored from checkpoint)",
                    'metrics': metrics['custom'],
                    'model': previous_checkpoint.get('model_object'),
                    'is_checkpoint_restore': True
                }
                
                print(f"  ✓ Checkpoint restored: R²={previous_checkpoint['r2']:.3f}")
                print(f"  Note: Using previous successful model instead of failed improvement\n")
            else:
                print(f"  Continuing with benchmark models only...\n")
                custom_model_info = None
    
    # 1. Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    predictions['rf'] = rf_pred
    metrics['rf'] = {
        'name': 'Random Forest',
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred)
    }

    # 2. Train Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    predictions['dt'] = dt_pred
    metrics['dt'] = {
        'name': 'Decision Tree',
        'mae': mean_absolute_error(y_test, dt_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'r2': r2_score(y_test, dt_pred)
    }

    # XGBoost and Prophet removed - only using RF and DT benchmarks

    # Create forecast results summary
    forecast_results = {}
    for model_key, model_metrics in metrics.items():
        forecast_results[model_key] = {
            'name': model_metrics['name'],
            'mae': float(model_metrics['mae']),
            'rmse': float(model_metrics['rmse']),
            'r2': float(model_metrics['r2'])
        }

    # Store model state (including the trained model objects)
    model_state = {
        'predictions': predictions,
        'y_test': y_test.values,
        'y_train': y_train.values,
        'split_point': split_idx,
        'test_size': len(X_test),
        'metrics': metrics,
        'rf_model': rf_model,  # Store Random Forest model
        'dt_model': dt_model,  # Store Decision Tree model
        'feature_columns': list(X.columns),  # Store feature column names
        'target_column': target_col  # Store target column name
    }

    # Store custom model info if present
    custom_model_info = {}
    if 'custom' in metrics:
        custom_model_info = {
            'request': custom_request,
            'generated_code': metrics['custom'].get('generated_code', ''),
            'metrics': {
                'mae': float(metrics['custom']['mae']),
                'rmse': float(metrics['custom']['rmse']),
                'r2': float(metrics['custom']['r2'])
            },
            'success': True
        }
    
    # Create summary message
    model_count = len(predictions)
    r2_summary = ', '.join([f"{m['name']} R²={m['r2']:.3f}" for m in metrics.values()])
    msg = f"Training complete ({model_count} models): {r2_summary}"
    
    if custom_model_info:
        msg += f" | Custom model generated successfully"

    out = {
        'model_state': model_state,
        'forecast_results': forecast_results,
        'custom_model_info': custom_model_info
    }
    out.update(memory_update(step_name, msg))
    return out


def visualize_forecast(state: EDAState) -> dict:
    """Visualize all trained forecast models"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    import numpy as np
    
    step_name = "viz_forecast"
    
    model_state = state.get('model_state')
    forecast_results = state.get('forecast_results', {})
    target_col = state.get('target_column')
    custom_model_info = state.get('custom_model_info', {})

    if not model_state:
        out = {
            'forecast_images': {},
            'forecast_results': forecast_results,
            'target_column': target_col,
            'ts_analysis': state.get('ts_analysis'),
            'ts_method': state.get('ts_method'),
            'ts_suitable': state.get('ts_suitable'),
            'data_profile': state.get('data_profile'),
            'summary_stats': state.get('summary_stats'),
            'engineered_data': state.get('engineered_data'),
            'visualizations': state.get('visualizations'),
            'viz_images': state.get('viz_images')
        }
        out.update(
            memory_update(
                step_name,
                "Forecast visualization skipped: no model state",
                warnings=["No model state"]
            )
        )
        return out

    try:
        predictions = model_state['predictions']
        y_test = model_state['y_test']
        y_train = model_state['y_train']
        split_point = model_state['split_point']
        test_size = model_state['test_size']
        metrics = model_state['metrics']
        
        # Limit forecast to 1 day (assuming 10-min resolution = 144 points per day)
        forecast_horizon = min(144, len(y_test))  # 1 day or less if test set is smaller
        y_test_display = y_test[:forecast_horizon]

        fig, ax = plt.subplots(figsize=(16, 6))
        
        engineered_data = state.get('engineered_data')
        data_profile = state.get('data_profile', {})
        timeseries_info = data_profile.get('timeseries', {})
        time_column = timeseries_info.get('time_column')

        all_timestamps = None
        if engineered_data is not None and '_parsed_time' in engineered_data.columns and time_column:
            all_timestamps = engineered_data['_parsed_time'].values

        # Display last 1 day of training data + 1 day forecast
        train_display_points = 144  # Last 1 day of training
        train_display_start = max(0, len(y_train) - train_display_points)
        train_display_end = len(y_train)
        train_y_display = y_train[train_display_start:train_display_end]
        
        if all_timestamps is not None:
            train_timestamps_display = all_timestamps[train_display_start:train_display_end]
            test_timestamps_display = all_timestamps[split_point:split_point + forecast_horizon]
            
            if len(train_y_display) > 0:
                ax.plot(
                    train_timestamps_display,
                    train_y_display,
                    'o-',
                    linewidth=2,
                    markersize=4,
                    label='Training Data',
                    color='#95B8D1',
                    alpha=0.6,
                    zorder=2
                )
            
            ax.plot(
                test_timestamps_display,
                y_test_display,
                'o-',
                linewidth=3,
                markersize=5,
                label='Actual (Test)',
                color='#2E4057',
                zorder=5
            )
            
            # Plot all model predictions with different colors (1-day forecast)
            model_styles = {
                'rf': {'color': '#4CAF50', 'marker': 's', 'label': 'RF Pred'},
                'dt': {'color': '#F44336', 'marker': '^', 'label': 'DT Pred'},
                'xgb': {'color': '#9C27B0', 'marker': 'd', 'label': 'XGB Pred'},
                'prophet': {'color': '#FF9800', 'marker': 'v', 'label': 'Prophet Pred'},
                'custom': {'color': '#00BCD4', 'marker': '*', 'label': 'Custom Pred'}
            }
            
            for model_key, pred in predictions.items():
                # Use defined style or fallback for unknown models
                if model_key in model_styles:
                    style = model_styles[model_key]
                else:
                    # Fallback style for any additional custom models
                    style = {'color': '#607D8B', 'marker': 'x', 'label': f'{model_key.upper()} Pred'}
                
                r2 = metrics[model_key]['r2']
                pred_display = pred[:forecast_horizon]
                ax.plot(
                    test_timestamps_display,
                    pred_display,
                    marker=style['marker'],
                    linestyle='--',
                    linewidth=2,
                    markersize=4,
                    label=f"{style['label']} (R²={r2:.3f})",
                    color=style['color'],
                    alpha=0.7,
                    zorder=4
                )
            
            ax.set_xlabel('Time', fontsize=12)
        else:
            # No timestamps, use indices
            train_indices_display = np.arange(train_display_start, train_display_end)
            test_indices_display = np.arange(split_point, split_point + forecast_horizon)
            
            if len(train_y_display) > 0:
                ax.plot(
                    train_indices_display,
                    train_y_display,
                    'o-',
                    linewidth=2,
                    markersize=4,
                    label='Training Data',
                    color='#95B8D1',
                    alpha=0.6,
                    zorder=2
                )
            
            ax.plot(
                test_indices_display,
                y_test_display,
                'o-',
                linewidth=3,
                markersize=5,
                label='Actual (Test)',
                color='#2E4057',
                zorder=5
            )
            
            model_styles = {
                'rf': {'color': '#4CAF50', 'marker': 's', 'label': 'RF Pred'},
                'dt': {'color': '#F44336', 'marker': '^', 'label': 'DT Pred'},
                'xgb': {'color': '#9C27B0', 'marker': 'd', 'label': 'XGB Pred'},
                'prophet': {'color': '#FF9800', 'marker': 'v', 'label': 'Prophet Pred'},
                'custom': {'color': '#00BCD4', 'marker': '*', 'label': 'Custom Pred'}
            }
            
            for model_key, pred in predictions.items():
                # Use defined style or fallback for unknown models
                if model_key in model_styles:
                    style = model_styles[model_key]
                else:
                    # Fallback style for any additional custom models
                    style = {'color': '#607D8B', 'marker': 'x', 'label': f'{model_key.upper()} Pred'}
                
                r2 = metrics[model_key]['r2']
                pred_display = pred[:forecast_horizon]
                ax.plot(
                    test_indices_display,
                    pred_display,
                    marker=style['marker'],
                    linestyle='--',
                    linewidth=2,
                    markersize=4,
                    label=f"{style['label']} (R²={r2:.3f})",
                    color=style['color'],
                    alpha=0.7,
                    zorder=4
                )
            
            ax.set_xlabel('Index', fontsize=12)

        ax.set_ylabel(target_col if target_col else 'Target', fontsize=12)
        ax.set_title('Figure 3: Multi-Model Forecast Comparison (1-Day Ahead)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        forecast_images = {'forecast_comparison': img_base64}

    except Exception as e:
        forecast_images = {}
        print(f"Warning: Forecast visualization failed: {e}")

    # Create Figure 4: Custom Model Detailed Analysis (if custom model exists)
    custom_model_detail_image = None
    if 'custom' in predictions and custom_model_info.get('success'):
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Determine if we have timestamps (reuse logic from Figure 3)
            engineered_data = state.get('engineered_data')
            data_profile = state.get('data_profile', {})
            timeseries_info = data_profile.get('timeseries', {})
            time_column = timeseries_info.get('time_column')
            
            all_timestamps = None
            if engineered_data is not None and '_parsed_time' in engineered_data.columns and time_column:
                all_timestamps = engineered_data['_parsed_time'].values
            
            has_timestamps = all_timestamps is not None
            
            # Get display data
            if has_timestamps:
                test_timestamps_display = all_timestamps[split_point:split_point + forecast_horizon]
            else:
                test_indices_display = np.arange(split_point, split_point + forecast_horizon)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Figure 4: Custom Model (LLM-Generated) - Detailed Performance Analysis', 
                        fontsize=14, fontweight='bold')
            
            custom_pred = predictions['custom'][:forecast_horizon]
            custom_metrics = metrics['custom']
            
            # Subplot 1: Actual vs Predicted
            ax1 = axes[0, 0]
            if has_timestamps:
                ax1.plot(test_timestamps_display, y_test_display, 'o-', linewidth=2, 
                        markersize=5, label='Actual', color='#2E4057', zorder=5)
                ax1.plot(test_timestamps_display, custom_pred, 's--', linewidth=2, 
                        markersize=4, label='Custom Model Prediction', color='#00BCD4', alpha=0.7)
                ax1.set_xlabel('Time', fontsize=10)
            else:
                ax1.plot(test_indices_display, y_test_display, 'o-', linewidth=2, 
                        markersize=5, label='Actual', color='#2E4057', zorder=5)
                ax1.plot(test_indices_display, custom_pred, 's--', linewidth=2, 
                        markersize=4, label='Custom Model Prediction', color='#00BCD4', alpha=0.7)
                ax1.set_xlabel('Index', fontsize=10)
            ax1.set_ylabel(target_col if target_col else 'Target', fontsize=10)
            ax1.set_title('Actual vs Predicted', fontsize=11, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Residuals over time
            ax2 = axes[0, 1]
            residuals = y_test_display - custom_pred
            if has_timestamps:
                ax2.plot(test_timestamps_display, residuals, 'o-', linewidth=2, 
                        markersize=4, color='#E91E63', alpha=0.7)
                ax2.set_xlabel('Time', fontsize=10)
            else:
                ax2.plot(test_indices_display, residuals, 'o-', linewidth=2, 
                        markersize=4, color='#E91E63', alpha=0.7)
                ax2.set_xlabel('Index', fontsize=10)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
            ax2.set_title('Residual Analysis', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: Residual Distribution
            ax3 = axes[1, 0]
            ax3.hist(residuals, bins=20, color='#9C27B0', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax3.set_xlabel('Residual Value', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            ax3.set_title('Residual Distribution', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Subplot 4: Performance Metrics Summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            metrics_text = (
                f"Performance Metrics:\n\n"
                f"MAE:  {custom_metrics['mae']:.3f}\n"
                f"RMSE: {custom_metrics['rmse']:.3f}\n"
                f"R²:   {custom_metrics['r2']:.4f}\n\n"
                f"User Request:\n{custom_model_info.get('request', 'N/A')[:100]}\n\n"
                f"Residual Stats:\n"
                f"Mean: {np.mean(residuals):.3f}\n"
                f"Std:  {np.std(residuals):.3f}\n"
                f"Min:  {np.min(residuals):.3f}\n"
                f"Max:  {np.max(residuals):.3f}"
            )
            ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            custom_model_detail_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            forecast_images['custom_model_detail'] = custom_model_detail_image
            
        except Exception as e:
            print(f"Warning: Custom model detail visualization failed: {e}")

    out = {
        'forecast_images': forecast_images,
        'forecast_results': forecast_results,
        'custom_model_detail_image': custom_model_detail_image,
        'target_column': target_col,
        'ts_analysis': state.get('ts_analysis'),
        'ts_method': state.get('ts_method'),
        'ts_suitable': state.get('ts_suitable'),
        'data_profile': state.get('data_profile'),
        'summary_stats': state.get('summary_stats'),
        'engineered_data': state.get('engineered_data'),
        'visualizations': state.get('visualizations'),
        'viz_images': state.get('viz_images')
    }
    
    msg = f"Forecast visualization complete: {len(predictions)} models plotted, {len(forecast_images)} images"
    out.update(memory_update(step_name, msg))
    return out
