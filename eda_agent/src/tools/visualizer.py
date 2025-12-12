"""
Data visualization tool
"""
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from ..state import EDAState, memory_update


def visualize_data(state: EDAState) -> dict:
    """Generate EDA visualizations: time series plot and correlation matrix"""
    step_name = "visualize"
    
    # Early exit if stop_processing flag is set
    if state.get("stop_processing", False):
        out = {
            "plot_paths": [],
            "visualizations": {},
            "viz_images": {},
            "stop_processing": True,
            "errors": state.get("errors", [])
        }
        out.update(
            memory_update(
                step_name,
                "Visualization skipped: stop_processing flag is set",
                errors=out["errors"]
            )
        )
        return out
    
    data = state.get("data")
    if data is None:
        base = {
            "plot_paths": [],
            "visualizations": {},
            "viz_images": {},
            "missing_percentage": state.get("missing_percentage"),
            "num_variables": state.get("num_variables"),
            "stop_processing": state.get("stop_processing", False),
            "preprocessed_data": state.get("preprocessed_data"),
            "engineered_data": state.get("engineered_data")
        }
        base.update(
            memory_update(
                step_name,
                "No data available for visualization",
                warnings=["Visualization skipped (no data)"]
            )
        )
        return base

    data_profile = state.get('data_profile', {})
    timeseries_info = data_profile.get('timeseries', {})
    time_column = timeseries_info.get('time_column')

    numeric_cols = [
        col for col in data.select_dtypes(include=['number']).columns 
        if not col.startswith('_')
    ]

    viz_descriptions = {}
    viz_images = {}

    # Time series plot
    if time_column and '_parsed_time' in data.columns:
        power_col = None
        for col in numeric_cols:
            if 'power' in col.lower() or 'generation' in col.lower() or 'mw' in col.lower():
                power_col = col
                break
        if power_col is None and len(numeric_cols) > 0:
            power_col = numeric_cols[0]
        
        if power_col:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                data['_parsed_time'], 
                data[power_col],
                marker='o',
                linewidth=2,
                markersize=4,
                color='#8ECFC9'
            )
            ax.set_xlabel(time_column, fontsize=12)
            ax.set_ylabel(power_col, fontsize=12)
            fig_name = 'Power Vs Time'
            ax.set_title(f'Figure 1: {fig_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(fig)
            
            viz_descriptions['power_vs_time'] = {
                'title': f'{power_col} over Time',
                'description': f'Time series visualization of {power_col}'
            }
            viz_images['power_vs_time'] = img_base64

    # Correlation matrix
    if len(numeric_cols) > 1:
        corr_data = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_data,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'},
            vmin=-1,
            vmax=1
        )
        ax.set_title('Figure 2: Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        
        viz_descriptions['correlation_matrix'] = {
            'title': 'Correlation Matrix',
            'description': 'Pairwise correlations between numeric features'
        }
        viz_images['correlation_matrix'] = img_base64

    out = {
        "plot_paths": [],
        "visualizations": viz_descriptions,
        "viz_images": viz_images,
        "missing_percentage": state.get("missing_percentage"),
        "num_variables": state.get("num_variables")
    }
    
    msg = f"Visualization complete: {len(viz_images)} plot(s) generated"
    out.update(memory_update(step_name, msg))
    return out
