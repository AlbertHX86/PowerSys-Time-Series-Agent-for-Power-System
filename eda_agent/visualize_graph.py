"""
Generate visualization of the LangGraph workflow
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.workflows import build_reflective_workflow
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def visualize_workflow():
    """Generate and save workflow graph visualization"""
    print("Building workflow graph...")
    
    # Create a temporary checkpointer for graph building
    conn = sqlite3.connect(":memory:")
    checkpointer = SqliteSaver(conn)
    
    # Build the workflow
    graph = build_reflective_workflow(checkpointer=checkpointer)
    
    # Generate visualization
    print("Generating visualization...")
    try:
        # Get the Mermaid diagram
        mermaid_code = graph.get_graph().draw_mermaid()
        
        # Save to file
        output_file = "workflow_graph.mmd"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        print(f"✓ Mermaid diagram saved to: {output_file}")
        print("\nYou can visualize this at: https://mermaid.live/")
        print("\nMermaid code:")
        print("="*70)
        print(mermaid_code)
        print("="*70)
        
    except Exception as e:
        print(f"Mermaid generation failed: {e}")
    
    # Try PNG generation if dependencies available
    try:
        from IPython.display import Image
        png_data = graph.get_graph().draw_mermaid_png()
        
        output_png = "workflow_graph.png"
        with open(output_png, 'wb') as f:
            f.write(png_data)
        print(f"\n✓ PNG diagram saved to: {output_png}")
        
    except ImportError:
        print("\nNote: Install 'pygraphviz' or use Mermaid online for visual diagram")
    except Exception as e:
        print(f"\nPNG generation not available: {e}")
    
    # Print graph structure info
    print("\n" + "="*70)
    print("WORKFLOW STRUCTURE")
    print("="*70)
    
    graph_info = graph.get_graph()
    print(f"\nNodes: {len(graph_info.nodes)}")
    for node in graph_info.nodes:
        print(f"  - {node}")
    
    print(f"\nEdges: {len(graph_info.edges)}")
    for edge in graph_info.edges[:20]:  # Show first 20 edges
        print(f"  - {edge}")
    if len(graph_info.edges) > 20:
        print(f"  ... and {len(graph_info.edges) - 20} more edges")

if __name__ == "__main__":
    visualize_workflow()
