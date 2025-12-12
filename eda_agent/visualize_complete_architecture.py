"""
Generate complete PowerSys architecture diagram including user interactions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_complete_architecture():
    """Generate complete system architecture with user interactions"""
    
    architecture_mmd = """---
config:
  flowchart:
    curve: basis
---
graph TB
    subgraph "User Interface (Frontend)"
        UI[Web Browser]
        Upload[Upload Data + Context]
        BusinessObj[Business Objective Input]
        Analyze[Run Analysis Button]
        Feedback[Regenerate with Feedback]
        ForecastUI[Forecast Section]
        ForecastReq[Business Requirements]
        ModelSelect[Model Selection]
    end
    
    subgraph "API Layer"
        Health[/api/health]
        AnalyzeAPI[/api/analyze]
        RegenerateAPI[/api/regenerate]
        ForecastAPI[/api/forecast]
    end
    
    subgraph "Main Workflow (LangGraph)"
        Start([__start__])
        Load[load]
        Validate[validate]
        Analyze2[analyze]
        LLMValidate[llm_validate]
        Preprocess[preprocess]
        FeatureEng[feature_eng]
        Visualize[visualize]
        TsSuitable[ts_suitable]
        TsTrain[ts_train]
        TsVisualize[ts_visualize]
        FinalReport[final_report]
        End([__end__])
        
        subgraph "Reflection Loop"
            CPPreprocess[cp_preprocess]
            CPFeature[cp_feature]
            CPForecast[cp_forecast]
            ReflectData[reflect_data]
            ReflectAnalysis[reflect_analysis]
            ReflectForecast[reflect_forecast]
        end
        
        subgraph "Quality Control"
            Guardrail[guardrail]
            ExportImages[export_images]
            VisionAnalysis[vision_analysis]
            VisionCheck[vision_check]
        end
    end
    
    subgraph "Business Context Processing"
        BusinessContext[Business Objective]
        DataContext[Data Context]
        UserGoals[User Goals]
    end
    
    subgraph "Forecast Generation (Separate Flow)"
        ForecastInput[User Requirements]
        ModelState[Trained Models]
        ForecastLogic[Template-based Prediction]
        LLMInsights[LLM Analysis]
        ForecastOutput[Personalized Report]
    end
    
    subgraph "Regeneration Flow"
        UserFeedback[User Feedback]
        ResumeSession[Resume from Checkpoint]
        ImproveModel[Improve Custom Model]
        RerunWorkflow[Re-execute from ts_train]
    end
    
    %% User interactions
    UI --> Upload
    UI --> BusinessObj
    UI --> Analyze
    UI --> Feedback
    UI --> ForecastUI
    
    Upload --> AnalyzeAPI
    BusinessObj --> AnalyzeAPI
    Analyze --> AnalyzeAPI
    
    Feedback --> RegenerateAPI
    ForecastUI --> ForecastAPI
    ForecastReq --> ForecastAPI
    ModelSelect --> ForecastAPI
    
    %% API to workflow
    AnalyzeAPI --> BusinessContext
    AnalyzeAPI --> Start
    
    BusinessContext --> Load
    DataContext --> Load
    UserGoals --> Load
    
    %% Main workflow flow
    Start --> Load
    Load --> Validate
    Validate --> Analyze2
    Analyze2 --> LLMValidate
    LLMValidate --> Preprocess
    Preprocess --> CPPreprocess
    CPPreprocess --> ReflectData
    ReflectData --> FeatureEng
    FeatureEng --> CPFeature
    CPFeature --> ReflectAnalysis
    ReflectAnalysis --> Visualize
    Visualize --> TsSuitable
    TsSuitable --> TsTrain
    TsTrain --> TsVisualize
    TsVisualize --> CPForecast
    CPForecast --> ReflectForecast
    ReflectForecast --> Guardrail
    Guardrail --> ExportImages
    ExportImages --> VisionAnalysis
    VisionAnalysis --> VisionCheck
    VisionCheck --> FinalReport
    FinalReport --> End
    
    %% Regeneration flow
    RegenerateAPI --> UserFeedback
    UserFeedback --> ResumeSession
    ResumeSession --> ImproveModel
    ImproveModel --> RerunWorkflow
    RerunWorkflow --> TsTrain
    
    %% Forecast flow
    ForecastAPI --> ForecastInput
    ForecastInput --> ModelState
    End -.->|Saves Models| ModelState
    ModelState --> ForecastLogic
    ForecastLogic --> LLMInsights
    LLMInsights --> ForecastOutput
    ForecastOutput --> ForecastUI
    
    %% Results back to UI
    End -.->|Results| AnalyzeAPI
    AnalyzeAPI -.->|Response| UI
    ForecastOutput -.->|Response| UI
    
    style UI fill:#e1f5ff
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style BusinessContext fill:#FFE4B5
    style ForecastInput fill:#FFE4B5
    style UserFeedback fill:#FFE4B5
    style LLMInsights fill:#DDA0DD
    style ReflectData fill:#FFD700
    style ReflectAnalysis fill:#FFD700
    style ReflectForecast fill:#FFD700
"""
    
    # Save Mermaid diagram
    output_file = "complete_architecture.mmd"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(architecture_mmd)
    
    print("✓ Complete architecture diagram saved to: complete_architecture.mmd")
    print("\nVisualize at: https://mermaid.live/")
    print("\n" + "="*70)
    print("SYSTEM ARCHITECTURE OVERVIEW")
    print("="*70)
    print("\n1. USER INTERACTIONS:")
    print("   - Upload Data + Business Objective")
    print("   - Run Analysis (triggers main workflow)")
    print("   - Regenerate with Feedback (resumes from checkpoint)")
    print("   - Generate Forecast + Business Requirements (separate flow)")
    
    print("\n2. MAIN WORKFLOW (23 nodes):")
    print("   - Data loading & validation")
    print("   - Preprocessing & feature engineering")
    print("   - Visualization & time series analysis")
    print("   - Model training & evaluation")
    print("   - Reflection loops (3 checkpoints)")
    print("   - Quality control (guardrail + vision)")
    print("   - Final report generation")
    
    print("\n3. REGENERATION FLOW:")
    print("   - User provides feedback")
    print("   - Resume from last checkpoint")
    print("   - Improve custom model with LLM")
    print("   - Re-execute from ts_train node")
    
    print("\n4. FORECAST FLOW (Separate):")
    print("   - Uses trained models from workflow")
    print("   - User selects model + provides business requirements")
    print("   - Template-based prediction (last day)")
    print("   - LLM generates personalized insights")
    print("   - Returns actionable business recommendations")
    
    print("\n5. BUSINESS CONTEXT INTEGRATION:")
    print("   - Business Objective → affects report generation")
    print("   - Data Context → guides analysis decisions")
    print("   - User Goals → influences model selection")
    print("   - Forecast Requirements → LLM-powered insights")
    
    print("\n" + "="*70)
    print("\nMermaid Code:")
    print("="*70)
    print(architecture_mmd)

if __name__ == "__main__":
    generate_complete_architecture()
