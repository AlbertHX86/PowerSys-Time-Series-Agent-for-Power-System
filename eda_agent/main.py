"""
EDA Agent - Main Entry Point

Usage:
    python main.py --filepath data/your_data.csv
    python main.py --filepath data/your_data.csv --resume SESSION_ID
"""
import argparse
import uuid
import sqlite3
from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.workflows import build_reflective_workflow
from config.settings import CHECKPOINT_DIR, OUTPUT_DIR
import os


# ========================================================================
DEFAULT_FILEPATH = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
DEFAULT_DESCRIPTION = "solar power output data"
DEFAULT_RESOLUTION = "10 minutes"
DEFAULT_CAPACITY = "50 MW"
DEFAULT_GOAL = "forecasting"
# ========================================================================


def prompt_custom_model():
    """Prompt user for custom model request (interactive)"""
    print("\n" + "="*70)
    print("🤖 CUSTOM MODEL GENERATOR")
    print("="*70)
    print("\nWould you like to create a custom forecasting model?")
    print("The system can generate model code from your natural language description.")
    print("\nExamples:")
    print("  - 'Create a simple linear regression model'")
    print("  - 'Build a stacking ensemble of XGBoost and Random Forest'")
    print("  - 'I want a Gradient Boosting model with 200 estimators'")
    print("  - 'Create a LightGBM regressor optimized for time series'")
    print("\nPress ENTER to skip, or type your model description:")
    print("-" * 70)
    
    user_input = input(">>> ").strip()
    
    if user_input:
        print(f"\n✓ Custom model request recorded: '{user_input}'")
        return user_input
    else:
        print("\n✓ Skipping custom model (using benchmark models only)")
        return None


def prompt_human_feedback(result, session_id):
    """Prompt user for feedback after viewing the report"""
    print("\n" + "="*70)
    print("👁️  REPORT REVIEW & FEEDBACK")
    print("="*70)
    print("\nThe analysis report has been generated and saved.")
    print(f"Report location: outputs/report_{session_id}.txt")
    
    # Show custom model info if available
    custom_model_info = result.get('custom_model_info', {})
    if custom_model_info and custom_model_info.get('success'):
        print("\n🌟 Custom Model Summary:")
        print(f"   Request: '{custom_model_info.get('request', 'N/A')}'")
        metrics = custom_model_info.get('metrics', {})
        print(f"   Performance: MAE={metrics.get('mae', 0):.2f}, "
              f"RMSE={metrics.get('rmse', 0):.2f}, R²={metrics.get('r2', 0):.3f}")
    
    # Show forecast quality
    forecast_results = result.get('forecast_results', {})
    if forecast_results:
        print("\n📊 All Model Performance:")
        for model_key, model_data in forecast_results.items():
            if isinstance(model_data, dict) and 'r2' in model_data:
                name = model_data.get('name', model_key.upper())
                print(f"   {name}: R²={model_data['r2']:.3f}, "
                      f"MAE={model_data['mae']:.2f}, RMSE={model_data['rmse']:.2f}")
    
    print("\n" + "-" * 70)
    print("\nWould you like to provide feedback for further improvements?")
    print("\nYou can suggest:")
    print("  - Hyperparameter adjustments (e.g., 'increase n_estimators to 200')")
    print("  - Different model architecture (e.g., 'try ensemble of RF and XGBoost')")
    print("  - Feature engineering ideas (e.g., 'add more lag features')")
    print("  - Code improvements (e.g., 'add cross-validation')")
    print("\nPress ENTER to finish, or type your feedback:")
    print("-" * 70)
    
    user_feedback = input(">>> ").strip()
    
    if user_feedback:
        return {
            "human_feedback": user_feedback,
            "needs_improvement": True
        }
    else:
        return {
            "human_feedback": None,
            "needs_improvement": False
        }


def setup_checkpointer():
    """Initialize SQLite checkpointer with pickle fallback for DataFrame serialization"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "eda_agent_memory.db")
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    
    # Use JsonPlusSerializer with pickle fallback to handle DataFrames
    serde = JsonPlusSerializer(pickle_fallback=True)
    return SqliteSaver(conn, serde=serde)


def run_analysis(filepath: str, resume_session_id: str = None, data_context: dict = None):
    """
    Run EDA analysis with reflective workflow
    
    Args:
        filepath: Path to data file
        resume_session_id: Optional session ID to resume
        data_context: Domain context (description, resolution, goal, capacity)
        
    Returns:
        tuple: (result, session_id)
    """
    # Setup checkpointer with pickle fallback for DataFrame serialization
    checkpointer = setup_checkpointer()
    
    # Build workflow with checkpointer
    agent = build_reflective_workflow(checkpointer=checkpointer)
    
    # Generate or use provided session ID
    session_id = resume_session_id if resume_session_id else str(uuid.uuid4())
    thread = {"configurable": {"thread_id": session_id}}
    
    # Default data context
    if data_context is None:
        data_context = {
            "description": "power generation data",
            "resolution": "variable",
            "goal": "forecasting",
            "capacity": "unknown"
        }
    
    # Initialize state
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
    
    # Run agent with checkpointing
    print(f"\n{'='*70}")
    print(f"Starting EDA Analysis")
    print(f"Session ID: {session_id}")
    print(f"File: {filepath}")
    print(f"{'='*70}\n")
    
    result = agent.invoke(initial_state, config={"recursion_limit": 50, **thread})
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"Session ID: {session_id}")
    print(f"Revision count: {result.get('revision_count', 0)}")
    
    # Quality scores
    data_q = result.get('data_quality_score', 0)
    analysis_q = result.get('analysis_quality_score', 0)
    forecast_q = result.get('forecast_quality_score', 0)
    overall_q = result.get('quality_score', 0)
    
    print(f"Quality scores: Data={data_q:.2f} Analysis={analysis_q:.2f} "
          f"Forecast={forecast_q:.2f} Overall={overall_q:.2f}")
    print(f"Checkpoint tags: {result.get('checkpoint_tags', [])}")
    
    # Custom model info
    custom_model_info = result.get('custom_model_info', {})
    if custom_model_info and custom_model_info.get('success'):
        print(f"\n🌟 Custom Model Generated:")
        print(f"   Request: '{custom_model_info.get('request', 'N/A')}'")
        metrics = custom_model_info.get('metrics', {})
        print(f"   Performance: MAE={metrics.get('mae', 0):.2f}, "
              f"RMSE={metrics.get('rmse', 0):.2f}, R²={metrics.get('r2', 0):.3f}")
    
    # Vision analysis summary
    vision_summary = result.get('vision_summary')
    vision_quality_passed = result.get('vision_quality_passed', True)
    if vision_summary:
        print(f"Vision Analysis: {'PASSED' if vision_quality_passed else 'ISSUES FOUND'}")
        print(f"Vision Summary: {vision_summary}")
    
    print(f"{'='*70}\n")
    
    return result, session_id


def save_report(result, session_id):
    """Save analysis report to file"""
    report = result.get("report", {})
    summary = report.get("forecast_summary", "")
    
    if summary:
        output_file = os.path.join(OUTPUT_DIR, f"report_{session_id}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write(f"EDA ANALYSIS REPORT\n")
            f.write(f"Session ID: {session_id}\n")
            f.write("="*70 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "="*70 + "\n")
            
            # Add quality scores if available
            if 'data_quality_score' in result:
                f.write("\nQUALITY SCORES:\n")
                f.write(f"Data Quality: {result.get('data_quality_score', 0):.2f}\n")
                f.write(f"Analysis Quality: {result.get('analysis_quality_score', 0):.2f}\n")
                f.write(f"Forecast Quality: {result.get('forecast_quality_score', 0):.2f}\n")
                f.write(f"Overall Quality: {result.get('quality_score', 0):.2f}\n")
            
            # Add custom model info if available
            custom_model_info = result.get('custom_model_info', {})
            if custom_model_info and custom_model_info.get('success'):
                f.write("\n" + "="*70 + "\n")
                f.write("🌟 CUSTOM MODEL DETAILS:\n")
                f.write("="*70 + "\n\n")
                f.write(f"User Request: '{custom_model_info.get('request', 'N/A')}'\n\n")
                
                metrics = custom_model_info.get('metrics', {})
                f.write("Performance Metrics:\n")
                f.write(f"  - MAE (Mean Absolute Error): {metrics.get('mae', 0):.4f}\n")
                f.write(f"  - RMSE (Root Mean Squared Error): {metrics.get('rmse', 0):.4f}\n")
                f.write(f"  - R² (R-Squared): {metrics.get('r2', 0):.4f}\n\n")
                
                generated_code = custom_model_info.get('generated_code', '')
                if generated_code:
                    f.write("Generated Code:\n")
                    f.write("-" * 70 + "\n")
                    f.write(generated_code + "\n")
                    f.write("-" * 70 + "\n")
            
            # Add vision analysis if available
            vision_analysis = result.get('vision_analysis', {})
            if vision_analysis:
                f.write("\n" + "="*70 + "\n")
                f.write("VISION ANALYSIS RESULTS:\n")
                f.write("="*70 + "\n\n")
                
                # Separate custom_model_detail from other figures
                custom_model_analysis = None
                other_analyses = {}
                
                for fig_name, analysis_data in vision_analysis.items():
                    if 'custom_model_detail' in fig_name:
                        custom_model_analysis = analysis_data
                    else:
                        other_analyses[fig_name] = analysis_data
                
                # Write other analyses first
                for fig_name, analysis_data in other_analyses.items():
                    f.write(f"[{fig_name}]:\n")
                    if 'error' in analysis_data:
                        f.write(f"   Error: {analysis_data['error']}\n")
                    else:
                        f.write(f"   {analysis_data.get('analysis', 'No analysis')}\n")
                    f.write(f"   File: {analysis_data.get('filepath', 'N/A')}\n\n")
                
                # Write custom model analysis separately with emphasis
                if custom_model_analysis:
                    f.write("\n" + "-"*70 + "\n")
                    f.write("🔍 FIGURE 4 - CUSTOM MODEL IMPROVEMENT SUGGESTIONS:\n")
                    f.write("-"*70 + "\n\n")
                    if 'error' in custom_model_analysis:
                        f.write(f"   Error: {custom_model_analysis['error']}\n")
                    else:
                        improvement_suggestions = custom_model_analysis.get('analysis', 'No analysis')
                        f.write(improvement_suggestions + "\n\n")
                        f.write("💡 Use these suggestions when providing feedback for model improvement!\n")
                    f.write(f"   File: {custom_model_analysis.get('filepath', 'N/A')}\n\n")
                
                vision_quality_passed = result.get('vision_quality_passed', True)
                f.write(f"\nVision Quality Check: {'PASSED' if vision_quality_passed else 'ISSUES FOUND'}\n")
                
                vision_issues = result.get('vision_quality_issues', [])
                if vision_issues:
                    f.write("\nVision Quality Issues:\n")
                    for issue in vision_issues:
                        f.write(f"  - {issue}\n")
                f.write(f"Overall Quality: {result.get('quality_score', 0):.2f}\n")
        
        print(f"Report saved to: {output_file}")
        return output_file
    
    return None


def export_visualizations(result, session_id):
    """Export all visualizations to files"""
    import base64
    
    report = result.get("report", {})
    viz_images = report.get("viz_images", {})
    forecast_images = report.get("forecast_images", {})
    
    exported = []
    
    # Export EDA visualizations
    for fig_name, img_base64 in viz_images.items():
        try:
            img_bytes = base64.b64decode(img_base64)
            filename = os.path.join(OUTPUT_DIR, f"{session_id}_figure1_{fig_name}.png")
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            exported.append(filename)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error exporting {fig_name}: {e}")
    
    # Export forecast visualizations
    for fig_name, img_base64 in forecast_images.items():
        try:
            img_bytes = base64.b64decode(img_base64)
            filename = os.path.join(OUTPUT_DIR, f"{session_id}_figure3_{fig_name}.png")
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            exported.append(filename)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error exporting {fig_name}: {e}")
    
    return exported


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="EDA Agent - Automated Exploratory Data Analysis")
    parser.add_argument(
        "--filepath",
        type=str,
        default=DEFAULT_FILEPATH,
        help=f"Path to the data file (CSV, Excel, or TXT). Default: {os.path.basename(DEFAULT_FILEPATH)}"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous session ID"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=DEFAULT_DESCRIPTION,
        help=f"Data description for domain context. Default: {DEFAULT_DESCRIPTION}"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=DEFAULT_RESOLUTION,
        help=f"Time resolution (e.g., '15 minutes', '1 hour'). Default: {DEFAULT_RESOLUTION}"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=DEFAULT_GOAL,
        help=f"Analysis goal (e.g., 'Day-ahead forecasting'). Default: {DEFAULT_GOAL}"
    )
    parser.add_argument(
        "--capacity",
        type=str,
        default=DEFAULT_CAPACITY,
        help=f"System capacity (e.g., '100 MW'). Default: {DEFAULT_CAPACITY}"
    )
    parser.add_argument(
        "--custom-model",
        type=str,
        default=None,
        help="Custom model description (skip interactive prompt)"
    )
    parser.add_argument(
        "--no-custom-prompt",
        action="store_true",
        help="Skip interactive custom model prompt"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting visualizations and report"
    )
    
    args = parser.parse_args()
    
    # Prompt for custom model (interactive) unless disabled or provided
    custom_model_request = None
    if args.custom_model:
        # Use command-line argument
        custom_model_request = args.custom_model
        print(f"\n✓ Custom model from CLI: '{custom_model_request}'")
    elif not args.no_custom_prompt:
        # Interactive prompt
        custom_model_request = prompt_custom_model()
    
    # Build data context
    data_context = {
        "description": args.description,
        "resolution": args.resolution,
        "goal": args.goal,
        "capacity": args.capacity
    }
    
    # Add custom model request if provided
    if custom_model_request:
        data_context["custom_model_request"] = custom_model_request
    
    # Run analysis
    result, session_id = run_analysis(
        filepath=args.filepath,
        resume_session_id=args.resume,
        data_context=data_context
    )
    
    # Export results
    if not args.no_export:
        print("\nExporting results...")
        save_report(result, session_id)
        export_visualizations(result, session_id)
    
    # Interactive feedback loop
    max_feedback_iterations = 2
    feedback_iteration = 0
    
    while feedback_iteration < max_feedback_iterations:
        # Prompt for human feedback
        feedback_info = prompt_human_feedback(result, session_id)
        
        if not feedback_info.get('needs_improvement'):
            print("\n✓ Analysis accepted. No further improvements requested.")
            break
        
        feedback_iteration += 1
        print(f"\n🔄 Starting improvement iteration {feedback_iteration}/{max_feedback_iterations}...")
        
        # Save checkpoint: store previous successful custom model info
        previous_checkpoint = None
        custom_model_info = result.get('custom_model_info', {})
        if custom_model_info and custom_model_info.get('success'):
            previous_checkpoint = {
                'mae': custom_model_info.get('metrics', {}).get('mae', 0),
                'rmse': custom_model_info.get('metrics', {}).get('rmse', 0),
                'r2': custom_model_info.get('metrics', {}).get('r2', 0),
                'code': custom_model_info.get('generated_code', ''),
                'request': custom_model_info.get('request', ''),
                'model_object': custom_model_info.get('model')  # Store the actual model
            }
            print(f"💾 Checkpoint saved: Previous model R²={previous_checkpoint['r2']:.3f}")
        
        # Update data_context with human feedback
        human_feedback = feedback_info.get('human_feedback', '')
        improvement_request = (
            f"Based on previous model performance (R²={previous_checkpoint.get('r2', 0) if previous_checkpoint else 0:.3f}) "
            f"and human feedback: '{human_feedback}', "
            f"improve the custom model or generate a new one."
        )
        
        # Create fresh data_context for improvement iteration
        improvement_context = {
            "description": args.description,
            "resolution": args.resolution,
            "goal": args.goal,
            "capacity": args.capacity,
            "custom_model_request": improvement_request,
            "is_improvement_iteration": True,
            "previous_checkpoint": previous_checkpoint  # Pass checkpoint for fallback
        }
        
        # Re-run analysis with feedback
        improved_result, improved_session_id = run_analysis(
            filepath=args.filepath,
            resume_session_id=None,  # New session for improvement
            data_context=improvement_context
        )
        
        # Check if improvement succeeded or need to rollback
        improved_custom_info = improved_result.get('custom_model_info', {})
        if improved_custom_info and improved_custom_info.get('success'):
            # Improvement succeeded
            print(f"\n✅ Improvement successful!")
            new_r2 = improved_custom_info.get('metrics', {}).get('r2', 0)
            old_r2 = previous_checkpoint.get('r2', 0) if previous_checkpoint else 0
            if new_r2 > old_r2:
                print(f"   Performance improved: R² {old_r2:.3f} → {new_r2:.3f} (+{(new_r2-old_r2):.3f})")
            else:
                print(f"   Performance: R² {old_r2:.3f} → {new_r2:.3f} ({(new_r2-old_r2):+.3f})")
            result = improved_result
            session_id = improved_session_id
        elif previous_checkpoint:
            # Improvement failed, rollback to checkpoint
            print(f"\n⚠️ Improvement failed. Rolling back to previous checkpoint...")
            print(f"   Using previous model: R²={previous_checkpoint['r2']:.3f}")
            
            # Restore checkpoint info into result
            result['custom_model_info'] = {
                'success': True,
                'generated_code': previous_checkpoint['code'],
                'request': f"{previous_checkpoint['request']} (checkpoint restored)",
                'metrics': {
                    'mae': previous_checkpoint['mae'],
                    'rmse': previous_checkpoint['rmse'],
                    'r2': previous_checkpoint['r2']
                },
                'model': previous_checkpoint.get('model_object')
            }
            
            # Keep the improved_session_id but restore checkpoint data
            session_id = improved_session_id
            print(f"   💾 Checkpoint restored successfully")
        else:
            # No checkpoint to rollback to, use improved result anyway
            print(f"\n⚠️ No previous checkpoint available. Using benchmark models.")
            result = improved_result
            session_id = improved_session_id
        
        # Save updated results
        if not args.no_export:
            print("\nExporting updated results...")
            save_report(result, session_id)
            export_visualizations(result, session_id)
    
    if feedback_iteration >= max_feedback_iterations:
        print(f"\n⚠️ Maximum feedback iterations ({max_feedback_iterations}) reached.")
    
    print("\nAnalysis complete!")
    print(f"Session ID: {session_id}")
    
    return result, session_id


if __name__ == "__main__":
    main()
