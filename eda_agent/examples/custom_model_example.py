"""
Custom Model Generation Example

This example demonstrates how to use the LLM-powered custom model generator
to create models from natural language descriptions.
"""
import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_analysis, save_report, export_visualizations


def example_1_simple_linear_regression():
    """Example 1: Request a simple linear regression model"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Linear Regression")
    print("="*70 + "\n")
    
    filepath = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
    
    data_context = {
        "description": "solar power output data",
        "resolution": "10 minutes",
        "goal": "forecasting",
        "capacity": "50 MW",
        # Request a custom model via natural language
        "custom_model_request": "Create a simple linear regression model for baseline comparison"
    }
    
    result, session_id = run_analysis(filepath=filepath, data_context=data_context)
    
    save_report(result, session_id)
    export_visualizations(result, session_id)
    
    print(f"\n✓ Example 1 complete! Session ID: {session_id}")
    return result, session_id


def example_2_stacking_ensemble():
    """Example 2: Request a stacking ensemble"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Stacking Ensemble")
    print("="*70 + "\n")
    
    filepath = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
    
    data_context = {
        "description": "solar power output data",
        "resolution": "10 minutes",
        "goal": "forecasting",
        "capacity": "50 MW",
        # Request a more complex custom model
        "custom_model_request": """
        Build a stacking ensemble that combines Random Forest and XGBoost as base models,
        with a Ridge regression meta-learner. Use 5-fold cross-validation for the base models.
        """
    }
    
    result, session_id = run_analysis(filepath=filepath, data_context=data_context)
    
    save_report(result, session_id)
    export_visualizations(result, session_id)
    
    print(f"\n✓ Example 2 complete! Session ID: {session_id}")
    return result, session_id


def example_3_gradient_boosting():
    """Example 3: Request a gradient boosting model with specific hyperparameters"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Gradient Boosting")
    print("="*70 + "\n")
    
    filepath = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
    
    data_context = {
        "description": "solar power output data",
        "resolution": "10 minutes",
        "goal": "forecasting",
        "capacity": "50 MW",
        # Request custom hyperparameters
        "custom_model_request": """
        Create a Gradient Boosting Regressor with these settings:
        - 200 estimators
        - max depth of 5
        - learning rate of 0.05
        - subsample ratio of 0.8
        Optimize for power generation forecasting accuracy
        """
    }
    
    result, session_id = run_analysis(filepath=filepath, data_context=data_context)
    
    save_report(result, session_id)
    export_visualizations(result, session_id)
    
    print(f"\n✓ Example 3 complete! Session ID: {session_id}")
    return result, session_id


def example_4_lightgbm():
    """Example 4: Request a LightGBM model"""
    print("\n" + "="*70)
    print("EXAMPLE 4: LightGBM Model")
    print("="*70 + "\n")
    
    filepath = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
    
    data_context = {
        "description": "solar power output data",
        "resolution": "10 minutes",
        "goal": "forecasting",
        "capacity": "50 MW",
        # Request a different library
        "custom_model_request": """
        I want a LightGBM regressor optimized for time series forecasting.
        Use these hyperparameters:
        - 150 boosting iterations
        - 31 leaves per tree
        - learning rate of 0.1
        - feature fraction of 0.9
        """
    }
    
    result, session_id = run_analysis(filepath=filepath, data_context=data_context)
    
    save_report(result, session_id)
    export_visualizations(result, session_id)
    
    print(f"\n✓ Example 4 complete! Session ID: {session_id}")
    return result, session_id


def example_5_catboost():
    """Example 5: Request a CatBoost model"""
    print("\n" + "="*70)
    print("EXAMPLE 5: CatBoost Model")
    print("="*70 + "\n")
    
    filepath = r"C:\Users\AlbertHX\OneDrive - Stanford\桌面\lab7\data\Solar station site 1 (Nominal capacity-50MW)(1).xlsx"
    
    data_context = {
        "description": "solar power output data",
        "resolution": "10 minutes",
        "goal": "forecasting",
        "capacity": "50 MW",
        # Request CatBoost
        "custom_model_request": """
        Build a CatBoost regression model for power forecasting.
        Configure it with 100 iterations, depth of 6, and learning rate of 0.03.
        Enable verbose output to track training progress.
        """
    }
    
    result, session_id = run_analysis(filepath=filepath, data_context=data_context)
    
    save_report(result, session_id)
    export_visualizations(result, session_id)
    
    print(f"\n✓ Example 5 complete! Session ID: {session_id}")
    return result, session_id


def run_all_examples():
    """Run all custom model examples"""
    print("\n" + "="*70)
    print("RUNNING ALL CUSTOM MODEL EXAMPLES")
    print("="*70 + "\n")
    
    results = []
    
    # Note: These examples may take a while to run
    # Comment out the ones you don't want to test
    
    results.append(example_1_simple_linear_regression())
    # results.append(example_2_stacking_ensemble())
    # results.append(example_3_gradient_boosting())
    # results.append(example_4_lightgbm())
    # results.append(example_5_catboost())
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Run example 1 by default
    # To run all examples, call run_all_examples()
    
    example_1_simple_linear_regression()
    
    # Or run a specific example:
    # example_2_stacking_ensemble()
    # example_3_gradient_boosting()
    # example_4_lightgbm()
    # example_5_catboost()
    
    # Or run all:
    # run_all_examples()
