// Global State
let state = {
    file: null,
    analysisRunning: false,
    currentStep: 1,
    sessionId: null,
    iterationCount: 0,
    maxIterations: 2
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    browseBtn: document.getElementById('browse-btn'),
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    removeFile: document.getElementById('remove-file'),
    
    enableCustomModel: document.getElementById('enable-custom-model'),
    customModelSection: document.getElementById('custom-model-section'),
    modelDescription: document.getElementById('model-description'),
    
    startAnalysisBtn: document.getElementById('start-analysis'),
    statusCard: document.getElementById('status-card'),
    progressBar: document.getElementById('progress-bar'),
    progressFill: document.getElementById('progress-fill'),
    logConsole: document.getElementById('log-console'),
    logContent: document.getElementById('log-content'),
    clearLogs: document.getElementById('clear-logs'),
    
    resultsSection: document.getElementById('results-section'),
    feedbackSection: document.getElementById('feedback-section'),
    forecastSection: document.getElementById('forecast-section'),
    downloadSection: document.getElementById('download-section'),
    
    bestModel: document.getElementById('best-model'),
    r2Score: document.getElementById('r2-score'),
    maeValue: document.getElementById('mae-value'),
    rmseValue: document.getElementById('rmse-value'),
    
    comparisonTable: document.getElementById('comparison-table'),
    vizGrid: document.getElementById('viz-grid'),
    reportContent: document.getElementById('report-content'),
    tuningSuggestions: document.getElementById('tuning-suggestions'),
    
    feedbackText: document.getElementById('feedback-text'),
    submitFeedback: document.getElementById('submit-feedback'),
    skipFeedback: document.getElementById('skip-feedback'),
    
    // Forecast elements
    forecastHours: document.getElementById('forecast-hours'),
    forecastUnit: document.getElementById('forecast-unit'),
    forecastRequirements: document.getElementById('forecast-requirements'),
    generateForecast: document.getElementById('generate-forecast'),
    forecastResults: document.getElementById('forecast-results'),
    forecastMean: document.getElementById('forecast-mean'),
    forecastCI: document.getElementById('forecast-ci'),
    forecastTable: document.getElementById('forecast-table'),
    forecastInsights: document.getElementById('forecast-insights'),
    
    downloadReport: document.getElementById('download-report'),
    downloadViz: document.getElementById('download-viz')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkServerHealth();
    addLog('System ready', 'info');
});

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch('http://localhost:5000/api/health');
        if (response.ok) {
            const data = await response.json();
            console.log('Server health check:', data);
            addLog('✓ Connected to backend server', 'success');
        } else {
            addLog('⚠ Backend server connection issue', 'warning');
        }
    } catch (error) {
        addLog('⚠ Cannot connect to backend server. Make sure it is running on port 5000.', 'warning');
        console.error('Server health check failed:', error);
    }
}

// Event Listeners
function initializeEventListeners() {
    // File Upload
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.removeFile.addEventListener('click', removeFile);
    
    // Drag & Drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // Custom Model Toggle
    elements.enableCustomModel.addEventListener('change', (e) => {
        const isChecked = e.target.checked;
        elements.customModelSection.style.display = isChecked ? 'block' : 'none';
        if (isChecked) {
            elements.modelDescription.disabled = false;
            elements.modelDescription.focus();
        } else {
            elements.modelDescription.disabled = true;
        }
    });
    
    // Example Chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const example = chip.dataset.example;
            const examples = {
                'lightgbm': 'Create a LightGBM model with n_estimators=250, learning_rate=0.05, max_depth=8, and add lag features for past 24 hours',
                'ensemble': 'Build a stacking ensemble with RandomForest, XGBoost, and LightGBM as base models, use LinearRegression as meta-learner',
                'xgboost': 'Implement XGBoost with early stopping, learning_rate=0.03, n_estimators=300, max_depth=6, colsample_bytree=0.8'
            };
            elements.modelDescription.value = examples[example];
        });
    });
    
    // Analysis Start
    elements.startAnalysisBtn.addEventListener('click', startAnalysis);
    
    // Logs
    elements.clearLogs.addEventListener('click', clearLogs);
    
    // Feedback
    elements.submitFeedback.addEventListener('click', submitFeedback);
    elements.skipFeedback.addEventListener('click', acceptResults);
    
    // Forecast
    elements.generateForecast.addEventListener('click', generateForecast);
    
    // Downloads
    elements.downloadReport.addEventListener('click', downloadReport);
    elements.downloadViz.addEventListener('click', downloadVisualizations);
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       'text/plain'];
    const validExtensions = ['.csv', '.xlsx', '.xls', '.txt'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        alert('Please upload a valid file (CSV, Excel, or TXT)');
        return;
    }
    
    state.file = file;
    
    // Update UI
    elements.uploadArea.style.display = 'none';
    elements.fileInfo.style.display = 'block';
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    
    updateStep(1, 'completed');
    updateStep(2, 'active');
    
    addLog(`File "${file.name}" uploaded successfully`, 'success');
}

function removeFile() {
    state.file = null;
    elements.uploadArea.style.display = 'block';
    elements.fileInfo.style.display = 'none';
    elements.fileInput.value = '';
    
    updateStep(1, 'active');
    updateStep(2, '');
    
    addLog('File removed', 'info');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Analysis
async function startAnalysis() {
    if (!state.file) {
        alert('Please upload a file first');
        return;
    }
    
    if (state.analysisRunning) return;
    
    state.analysisRunning = true;
    state.iterationCount = 0;
    state.sessionId = Date.now().toString();
    
    // Update UI
    elements.startAnalysisBtn.disabled = true;
    elements.startAnalysisBtn.textContent = 'Running Analysis...';
    updateStep(2, 'completed');
    updateStep(3, 'active');
    
    // Show progress
    elements.progressBar.style.display = 'block';
    elements.logConsole.style.display = 'block';
    
    updateStatus('running', 'Analysis in progress...');
    addLog('Starting EDA analysis...', 'info');
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', state.file);
    formData.append('description', document.getElementById('description').value);
    formData.append('resolution', document.getElementById('resolution').value);
    formData.append('capacity', document.getElementById('capacity').value);
    formData.append('goal', document.getElementById('goal').value);
    formData.append('session_id', state.sessionId);
    
    const businessObjective = document.getElementById('business-objective').value.trim();
    if (businessObjective) {
        formData.append('business_objective', businessObjective);
    }
    
    if (elements.enableCustomModel.checked) {
        formData.append('custom_model_request', elements.modelDescription.value);
    }
    
    try {
        // Call actual backend API with timeout
        updateProgress(10);
        addLog('Uploading data to server...', 'info');
        
        // Add timeout to fetch
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
        
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        updateProgress(30);
        addLog('Data uploaded successfully', 'success');
        addLog('Starting comprehensive analysis pipeline...', 'info');
        
        // Simulate progress during long operation with detailed messages
        let progressValue = 30;
        const progressInterval = setInterval(() => {
            if (progressValue < 90) {
                progressValue += 1.5;
                updateProgress(progressValue);
                
                // Add detailed periodic status messages
                if (progressValue >= 32 && progressValue < 34) addLog('📊 Loading and validating dataset...', 'info');
                if (progressValue >= 38 && progressValue < 40) addLog('🔍 Analyzing data quality and structure...', 'info');
                if (progressValue >= 44 && progressValue < 46) addLog('🧹 Running data preprocessing...', 'info');
                if (progressValue >= 50 && progressValue < 52) addLog('⚙️ Engineering time series features...', 'info');
                if (progressValue >= 56 && progressValue < 58) addLog('📈 Creating exploratory visualizations...', 'info');
                if (progressValue >= 62 && progressValue < 64) addLog('🎯 Training Decision Tree benchmark...', 'info');
                if (progressValue >= 68 && progressValue < 70) addLog('🌲 Training Random Forest benchmark...', 'info');
                if (progressValue >= 74 && progressValue < 76) addLog('🤖 Generating custom AI model from your request...', 'info');
                if (progressValue >= 80 && progressValue < 82) addLog('📊 Creating forecast comparison plots...', 'info');
                if (progressValue >= 85 && progressValue < 87) addLog('👁️ Running vision analysis on diagnostics...', 'info');
                if (progressValue >= 89 && progressValue < 91) addLog('📝 Generating comprehensive report...', 'info');
            }
        }, 1500); // Update every 1.5 seconds
        
        const results = await response.json();
        
        clearInterval(progressInterval);
        
        // DEBUG: Log the full response to check guardrail status
        console.log('API Response:', results);
        
        if (!results.success) {
            throw new Error(results.error || 'Analysis failed');
        }
        
        updateProgress(100);
        addLog('Analysis completed successfully!', 'success');
        
        // Store session ID from response
        if (results.session_id) {
            state.sessionId = results.session_id;
        }
        
        displayResults(results);
        
        updateStep(3, 'completed');
        updateStep(4, 'active');
        updateStatus('success', 'Analysis completed successfully!');
        addLog('Analysis completed', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        let errorMessage = 'Analysis failed. Please try again.';
        if (error.name === 'AbortError') {
            errorMessage = 'Analysis timed out after 5 minutes. The dataset may be too large or complex.';
        } else if (error.message) {
            errorMessage = error.message;
        }
        
        updateStatus('error', errorMessage);
        addLog(`Error: ${errorMessage}`, 'error');
    } finally {
        state.analysisRunning = false;
        elements.startAnalysisBtn.disabled = false;
        elements.startAnalysisBtn.textContent = 'Start Analysis';
    }
}

async function submitFeedback() {
    if (state.iterationCount >= state.maxIterations) {
        alert(`Maximum ${state.maxIterations} iterations reached`);
        return;
    }
    
    const feedback = elements.feedbackText.value.trim();
    if (!feedback) {
        alert('Please enter your improvement request');
        return;
    }
    
    state.iterationCount++;
    
    // Update status display to show regeneration in progress
    updateStatus('Regenerating model with improvements...', 'running');
    addLog(`Iteration ${state.iterationCount}: Regenerating with feedback...`, 'info');
    
    elements.submitFeedback.disabled = true;
    updateProgress(0);
    
    let progressInterval = null;
    
    try {
        // Call backend regenerate API
        updateProgress(10);
        addLog('📤 Sending regenerate request to server...', 'info');
        
        // Simulate progress during regeneration with detailed messages
        let progressValue = 10;
        progressInterval = setInterval(() => {
            if (progressValue < 90) {
                progressValue += 2.5;
                updateProgress(progressValue);
                
                // Add detailed periodic status messages
                if (progressValue >= 15 && progressValue < 18) addLog('🔄 Applying your improvement suggestions...', 'info');
                if (progressValue >= 25 && progressValue < 28) addLog('🤖 Re-training model with updated parameters...', 'info');
                if (progressValue >= 35 && progressValue < 38) addLog('📊 Evaluating new model performance...', 'info');
                if (progressValue >= 45 && progressValue < 48) addLog('📈 Generating updated forecast visualizations...', 'info');
                if (progressValue >= 55 && progressValue < 58) addLog('🖼️ Exporting diagnostic plots...', 'info');
                if (progressValue >= 65 && progressValue < 68) addLog('👁️ Running vision analysis on Figure 4...', 'info');
                if (progressValue >= 75 && progressValue < 78) addLog('💡 Extracting improvement insights...', 'info');
                if (progressValue >= 85 && progressValue < 88) addLog('📝 Updating analysis report...', 'info');
            }
        }, 1200); // Update every 1.2 seconds
        
        const response = await fetch('http://localhost:5000/api/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                feedback: feedback,
                iteration: state.iterationCount
            })
        });
        
        clearInterval(progressInterval);
        
        updateProgress(60);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error response:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const results = await response.json();
        console.log('Regenerate results received:', results);
        
        if (!results.success) {
            throw new Error(results.error || 'Regeneration failed');
        }
        
        updateProgress(100);
        addLog('✓ Model regenerated successfully', 'success');
        addLog(`Updating display with iteration ${state.iterationCount} results...`, 'info');
        
        // Update status to completed
        updateStatus('Regeneration completed successfully', 'completed');
        
        console.log('About to call displayResults with:', {
            metricsCount: Object.keys(results.metrics || {}).length,
            visualizationsCount: (results.visualizations || []).length,
            reportLength: (results.report || '').length,
            hasCustomModel: !!results.custom_model_info
        });
        
        // Update display with new results
        try {
            displayResults(results);
            addLog('✓ Display updated with new model', 'success');
            
            // Show iteration feedback if available
            if (results.iteration_feedback) {
                addLog(results.iteration_feedback, results.metrics_status === 'degraded' ? 'warning' : 'info');
            }
        } catch (displayError) {
            console.error('Error in displayResults:', displayError);
            addLog(`⚠️ Display update error: ${displayError.message}`, 'warning');
            throw displayError;
        }
        
        // Clear feedback text for next iteration
        elements.feedbackText.value = '';
        
        addLog(`Ready for iteration ${state.iterationCount + 1}`, 'info');
        
    } catch (error) {
        console.error('Regeneration error:', error);
        addLog(`❌ Regeneration failed: ${error.message}`, 'error');
        updateStatus('Regeneration failed', 'error');
        alert(`Regeneration failed: ${error.message}`);
    } finally {
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        elements.submitFeedback.disabled = false;
        updateProgress(0);
    }
}

function acceptResults() {
    elements.feedbackSection.style.display = 'none';
    elements.forecastSection.style.display = 'block';  // Show forecast section
    elements.downloadSection.style.display = 'none';   // Hide download for now
    addLog('Ready to generate forecasts or download results.', 'success');
}

// Forecast Function
async function generateForecast() {
    const forecastHours = parseInt(elements.forecastHours.value) || 24;
    const forecastUnit = elements.forecastUnit.value;
    const requirements = elements.forecastRequirements.value;
    const modelName = document.getElementById('forecast-model').value;  // Get selected model
    
    // Convert to hours if in days
    let totalHours = forecastHours;
    if (forecastUnit === 'days') {
        totalHours = forecastHours * 24;
    }
    
    if (!state.sessionId) {
        addLog('❌ No active session. Please run analysis first.', 'error');
        return;
    }
    
    if (!modelName) {
        addLog('❌ Please select a model for forecast.', 'error');
        return;
    }
    
    if (totalHours <= 0 || totalHours > 168) {
        addLog('❌ Forecast horizon must be between 1 and 168 hours (7 days).', 'error');
        return;
    }
    
    elements.generateForecast.disabled = true;
    addLog('📤 Generating forecast...', 'info');
    
    try {
        const response = await fetch('http://localhost:5000/api/forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                forecast_hours: totalHours,
                business_requirements: requirements,
                model_name: modelName  // Send the selected model name
            })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const results = await response.json();
        console.log('Forecast results:', results);
        
        if (!results.success) {
            throw new Error(results.error || 'Forecast generation failed');
        }
        
        // Display forecast results
        displayForecastResults(results);
        addLog('✓ Forecast generated successfully', 'success');
        
    } catch (error) {
        console.error('Forecast error:', error);
        addLog(`❌ Forecast failed: ${error.message}`, 'error');
    } finally {
        elements.generateForecast.disabled = false;
    }
}

function displayForecastResults(results) {
    console.log('Displaying forecast results:', results);
    
    // Show results container
    elements.forecastResults.style.display = 'block';
    
    // Display metrics
    if (results.forecast_stats) {
        const mean = results.forecast_stats.mean || 0;
        const ci_width = results.forecast_stats.ci_width || 0;
        
        elements.forecastMean.textContent = mean.toFixed(2);
        elements.forecastCI.textContent = `±${ci_width.toFixed(2)}`;
    }
    
    // Display forecast table
    if (results.forecast_values && results.forecast_values.length > 0) {
        let tableHTML = `
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <thead style="background: #f0f0f0; font-weight: 600;">
                    <tr>
                        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Time</th>
                        <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">Forecast</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        results.forecast_values.forEach((item, index) => {
            const bgColor = index % 2 === 0 ? '#fafafa' : '#fff';
            tableHTML += `
                <tr style="background: ${bgColor};">
                    <td style="padding: 6px 8px; border: 1px solid #ddd;">${item.time || `+${index}h`}</td>
                    <td style="padding: 6px 8px; border: 1px solid #ddd; text-align: right; font-weight: 500;">${parseFloat(item.value).toFixed(2)}</td>
                </tr>
            `;
        });
        
        tableHTML += '</tbody></table>';
        elements.forecastTable.innerHTML = tableHTML;
    }
    
    // Display insights
    if (results.business_insights) {
        let insights = results.business_insights;
        // Format markdown-like text to HTML
        insights = insights
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        
        elements.forecastInsights.innerHTML = `<div style="line-height: 1.6;">${insights}</div>`;
    } else {
        elements.forecastInsights.innerHTML = '<p style="color: #999;">No additional insights available.</p>';
    }
    
    // Show download section after forecast is generated
    setTimeout(() => {
        elements.downloadSection.style.display = 'block';
        addLog('✓ Forecast complete. You can now download results.', 'info');
    }, 500);
}

// Display Functions
function displayResults(results) {
    console.log('=== displayResults called ===');
    console.log('Results object:', results);
    
    // Show results section
    elements.resultsSection.style.display = 'block';
    console.log('✓ Results section displayed');
    
    // Check for guardrail violations - if guardrail failed, skip feedback section
    const guardrailPassed = results.guardrail_passed !== false;
    const guardrailWarning = results.guardrail_warning || '';
    
    // DEBUG: Log guardrail status
    console.log('Guardrail Status - passed:', guardrailPassed, 'warning:', guardrailWarning);
    
    if (!guardrailPassed && guardrailWarning) {
        // Guardrail triggered - show N/A for metrics
        elements.bestModel.textContent = 'N/A';
        elements.r2Score.textContent = 'N/A';
        elements.maeValue.textContent = 'N/A';
        elements.rmseValue.textContent = 'N/A';
        
        // Clear comparison table and visualizations
        elements.comparisonTable.innerHTML = '<p>Analysis not available due to data constraints.</p>';
        elements.vizGrid.innerHTML = '<p>Visualizations not available due to data constraints.</p>';
        
        // Show only report with guardrail warning
        elements.reportContent.innerHTML = `
            <div style="background: #fff3cd; border-left: 4px solid #ff9800; padding: 16px; margin-bottom: 20px; border-radius: 4px;">
                <h3 style="margin: 0 0 8px 0; color: #ff9800;">⚠️ Data Quality Alert</h3>
                <p style="margin: 0 0 8px 0; color: #333;">${guardrailWarning}</p>
                <p style="margin: 0; color: #666; font-size: 14px;">Complete analysis requires sufficient data. Please upload more data and try again.</p>
            </div>
            <h3>Analysis Report</h3>
            <p>${results.report || 'No analysis report available due to data constraints.'}</p>
        `;
        
        elements.feedbackSection.style.display = 'none';
        elements.downloadSection.style.display = 'block';
        addLog(`⚠️ Guardrail Warning: ${guardrailWarning}`, 'warning');
        addLog('Analysis stopped due to data quality constraints. Only report available for download.', 'info');
    } else {
        // No guardrail issue - show full analysis
        console.log('Processing full analysis results...');
        
        // Update metrics
        const metricsArray = Object.values(results.metrics || {});
        console.log('Metrics array:', metricsArray);
        
        if (metricsArray.length > 0) {
            console.log('Updating metric displays...');
            const bestMetric = metricsArray.reduce((a, b) => 
                (a.r2 || 0) > (b.r2 || 0) ? a : b
            );
            
            elements.bestModel.textContent = bestMetric.name || 'N/A';
            elements.r2Score.textContent = (bestMetric.r2 || 0).toFixed(3);
            elements.maeValue.textContent = (bestMetric.mae || 0).toFixed(2);
            elements.rmseValue.textContent = (bestMetric.rmse || 0).toFixed(2);
            console.log('✓ Metrics updated');
            
            // Comparison table
            console.log('Building comparison table...');
            let tableHTML = '<div class="comparison-row header">';
            tableHTML += '<div>Model</div><div>R²</div><div>MAE</div><div>RMSE</div>';
            tableHTML += '</div>';
            
            // First add custom model iterations if available
            if (results.custom_model_iterations && results.custom_model_iterations.length > 0) {
                console.log('Adding custom model iterations:', results.custom_model_iterations);
                results.custom_model_iterations.forEach(customModel => {
                    const isHighlight = customModel.r2 === bestMetric.r2;
                    tableHTML += `<div class="comparison-row ${isHighlight ? 'highlight' : ''}">`;
                    tableHTML += `<div>${customModel.name || 'Custom Model'}</div>`;
                    tableHTML += `<div>${(customModel.r2 || 0).toFixed(3)}</div>`;
                    tableHTML += `<div>${(customModel.mae || 0).toFixed(2)}</div>`;
                    tableHTML += `<div>${(customModel.rmse || 0).toFixed(2)}</div>`;
                    tableHTML += '</div>';
                });
            }
            
            // Then add other benchmark models (excluding 'custom' since it's now in iterations)
            metricsArray.forEach(model => {
                // Skip custom model if it's already been added via custom_model_iterations
                if (results.custom_model_iterations && results.custom_model_iterations.length > 0 && model.name === 'Custom Model') {
                    return;
                }
                
                const isHighlight = model.name === bestMetric.name;
                tableHTML += `<div class="comparison-row ${isHighlight ? 'highlight' : ''}">`;
                tableHTML += `<div>${model.name || 'Unknown'}</div>`;
                tableHTML += `<div>${(model.r2 || 0).toFixed(3)}</div>`;
                tableHTML += `<div>${(model.mae || 0).toFixed(2)}</div>`;
                tableHTML += `<div>${(model.rmse || 0).toFixed(2)}</div>`;
                tableHTML += '</div>';
            });
            
            elements.comparisonTable.innerHTML = tableHTML;
            console.log('✓ Comparison table updated');
        } else {
            console.log('No metrics to display');
        }
        
        // Visualizations
        console.log('Processing visualizations...', results.visualizations?.length || 0, 'items');
        let vizHTML = '';
        if (results.visualizations && results.visualizations.length > 0) {
            results.visualizations.forEach((viz, index) => {
                console.log(`Processing viz ${index + 1}:`, viz.name);
                const imgSrc = viz.data ? `data:image/png;base64,${viz.data}` : '';
                vizHTML += `
                    <div class="viz-card">
                        <div class="viz-card-title">${viz.name || 'Visualization'}</div>
                        ${imgSrc ? `<img src="${imgSrc}" alt="${viz.name || 'Visualization'}">` : '<p>Image not available</p>'}
                    </div>
                `;
            });
            console.log('✓ Visualizations HTML generated');
        } else {
            vizHTML = '<p>No visualizations available</p>';
        }
        elements.vizGrid.innerHTML = vizHTML;
        
        // Report - extract and separate business recommendations
        let reportText = results.report || 'No report available';
        let businessRecommendations = '';
        
        // Extract business recommendations section if present
        // Try multiple patterns to find business recommendations
        const patterns = [
            /\*\*Business Recommendations:\*\*[\s\S]*$/i,  // From "**Business Recommendations:**" to end
            /Business Recommendations:[\s\S]*$/i           // Without markdown formatting
        ];
        
        let match = null;
        for (const pattern of patterns) {
            match = reportText.match(pattern);
            if (match) {
                console.log('Found business recommendations with pattern:', pattern);
                break;
            }
        }
        
        if (match) {
            businessRecommendations = match[0];
            // Remove from main report
            reportText = reportText.replace(match[0], '').trim();
            
            // Display business recommendations in dedicated section
            const businessSection = document.getElementById('business-recommendations-section');
            const businessContent = document.getElementById('business-recommendations-content');
            if (businessSection && businessContent) {
                // Format business recommendations with better styling
                let formattedBizRec = businessRecommendations
                    .replace(/\*\*Business Recommendations:\*\*/i, '<h5 style="margin-top: 0; color: #047857; font-size: 16px;">Business Recommendations</h5>')
                    .replace(/Business Recommendations:/i, '<h5 style="margin-top: 0; color: #047857; font-size: 16px;">Business Recommendations</h5>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #047857;">$1</strong>')
                    .replace(/(\d+\.\s+)/g, '<div style="margin: 12px 0;">$1')
                    .replace(/\n\n/g, '</div>')
                    .replace(/\n/g, '<br>');
                
                businessContent.innerHTML = `
                    <div style="font-size: 14px; line-height: 1.8; color: #1f2937;">
                        ${formattedBizRec}
                    </div>
                `;
                businessSection.style.display = 'block';
                console.log('✓ Business recommendations section displayed');
            }
        } else {
            console.log('No business recommendations found in report');
        }
        
        // Extract Model Tuning Recommendations section from report
        let tuningSuggestionsHTML = '<h4 class="subsection-title">🎯 Model Tuning Recommendations</h4>';
        
        let foundAnalysis = false;
        
        // First try: Extract from vision_analysis object if available
        if (results.vision_analysis && typeof results.vision_analysis === 'object') {
            for (const [key, value] of Object.entries(results.vision_analysis)) {
                if (key.includes('custom_model') || key.includes('detail')) {
                    if (value && value.analysis) {
                        tuningSuggestionsHTML += `<div style="white-space: pre-wrap; font-size: 0.95em; line-height: 1.8; margin-top: 10px; max-height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">${value.analysis}</div>`;
                        foundAnalysis = true;
                        break;
                    }
                }
            }
        }
        
        // Second try: Look for the VISION-BASED ANALYSIS section in the report text
        if (!foundAnalysis) {
            const visionMatch = reportText.match(/\*\*VISION-BASED ANALYSIS[\s\S]*?(?=\n##|\n\*\*Business Recommendations|\n\*\*[A-Z]|$)/i);
            if (visionMatch) {
                tuningSuggestionsHTML += `<div style="white-space: pre-wrap; font-size: 0.95em; line-height: 1.8; margin-top: 10px; max-height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">${visionMatch[0]}</div>`;
                foundAnalysis = true;
            }
        }
        
        // Fallback: show message if nothing found
        if (!foundAnalysis) {
            tuningSuggestionsHTML += '<p style="color: #999; font-style: italic; margin-top: 10px;">💡 Model tuning recommendations are being analyzed. Check the detailed report above for insights.</p>';
        }
        
        elements.tuningSuggestions.innerHTML = tuningSuggestionsHTML;
        
        // Display main report without business recommendations
        elements.reportContent.innerHTML = `<p style="white-space: pre-wrap;">${reportText}</p>`;
        
        // Custom Model Code & Figure Analysis
        const customModelAnalysis = document.getElementById('custom-model-analysis');
        if (results.custom_model_info && Object.keys(results.custom_model_info).length > 0) {
            // Extract custom model data
            const customModel = results.custom_model_info;
            
            // Display code
            const codeElement = document.querySelector('#custom-model-code code');
            if (codeElement && customModel.generated_code) {
                codeElement.textContent = customModel.generated_code;
            }
            
            // Display Figure 4 (custom model figure)
            const figureElement = document.getElementById('custom-model-figure');
            if (figureElement && results.visualizations) {
                const figure4 = results.visualizations.find(v => 
                    v.name && (v.name.includes('Figure 4') || v.id === 'custom_model_detail' || v.name.includes('Custom Model Detail'))
                );
                if (figure4 && figure4.data) {
                    figureElement.innerHTML = `<img src="data:image/png;base64,${figure4.data}" alt="Custom Model Performance" style="width: 100%; max-width: 100%; border-radius: 4px;">`;
                    console.log('✓ Figure 4 displayed successfully');
                } else {
                    // Don't show "not available" message, just leave it empty
                    console.log('Figure 4 not found in visualizations:', results.visualizations.map(v => v.name));
                    figureElement.innerHTML = '';
                }
            }
            
            // Display insights from vision analysis
            const insightsElement = document.getElementById('custom-model-insights');
            if (insightsElement && customModel.ai_analysis) {
                // Format the analysis text for better readability
                let formattedAnalysis = customModel.ai_analysis;
                // Convert markdown-like formatting to HTML
                formattedAnalysis = formattedAnalysis
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/__(.*?)__/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/-\s+/g, '• ')
                    .replace(/(\d+\.\s+)/g, '$1');
                
                insightsElement.innerHTML = `
                    <h5 style="margin-top: 0; color: #0066cc;">📈 Performance Insights & Recommendations</h5>
                    <div style="line-height: 1.8; color: #333; font-size: 14px; white-space: pre-wrap; word-wrap: break-word;">
                        ${formattedAnalysis.replace(/\n/g, '<br>')}
                    </div>
                    <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #cce0ff;">
                        <strong style="color: #0066cc;">💡 Next Steps:</strong>
                        <ul style="margin: 8px 0; padding-left: 20px; font-size: 13px;">
                            <li>Review the generated code above</li>
                            <li>Study the Figure 4 diagnostic plots</li>
                            <li>Use the recommendations to provide specific improvements below</li>
                            <li>Example feedback: "Increase n_estimators to 300, add lag_24 feature, set learning_rate to 0.05"</li>
                        </ul>
                    </div>
                `;
            } else if (insightsElement) {
                insightsElement.innerHTML = '<p>Analysis not available for this model.</p>';
            }
            
            // Show the custom model analysis section
            if (customModelAnalysis) {
                customModelAnalysis.style.display = 'block';
            }
        } else {
            // Hide custom model analysis if no custom model data
            if (customModelAnalysis) {
                customModelAnalysis.style.display = 'none';
            }
        }
        
        // Show feedback section for iteration (if custom model exists, allow refinement)
        if (results.custom_model_info && results.custom_model_info.generated_code) {
            elements.feedbackSection.style.display = 'block';
        } else {
            elements.feedbackSection.style.display = 'none';
        }
        
        // Show forecast section for real-world prediction (after iteration testing is done or user decides to skip)
        // Store best model info in state for forecast
        state.bestModelInfo = {
            bestModel: elements.bestModel.textContent,
            r2: parseFloat(elements.r2Score.textContent),
            sessionId: state.sessionId
        };
        
        // Populate forecast model dropdown with available models
        const forecastModelSelect = document.getElementById('forecast-model');
        if (forecastModelSelect) {
            forecastModelSelect.innerHTML = '<option value="">-- Choose a model --</option>';
            
            // Add custom model iterations if available
            if (results.custom_model_iterations && results.custom_model_iterations.length > 0) {
                results.custom_model_iterations.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.name} (R² = ${(model.r2 || 0).toFixed(3)})`;
                    forecastModelSelect.appendChild(option);
                });
            }
            
            // Add benchmark models
            metricsArray.forEach(model => {
                // Skip if it's a custom model that was already added
                if (results.custom_model_iterations && results.custom_model_iterations.length > 0 && model.name === 'Custom Model') {
                    return;
                }
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name} (R² = ${(model.r2 || 0).toFixed(3)})`;
                forecastModelSelect.appendChild(option);
            });
            
            // Set default to best model
            forecastModelSelect.value = elements.bestModel.textContent;
            console.log('✓ Forecast model dropdown populated');
        }
        
        elements.forecastSection.style.display = 'block';
    }
}

// UI Updates
function updateStep(stepNum, status) {
    const stepElement = document.getElementById(`step-${['upload', 'configure', 'model', 'results'][stepNum - 1]}`);
    if (stepElement) {
        stepElement.className = 'step';
        if (status) stepElement.classList.add(status);
    }
}

function updateStatus(type, message) {
    const icons = {
        idle: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
        running: '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
        success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>'
    };
    
    const statusHTML = `
        <div class="status-${type}">
            <svg class="status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                ${icons[type]}
            </svg>
            <p>${message}</p>
        </div>
    `;
    
    elements.statusCard.innerHTML = statusHTML;
}

function updateProgress(percent) {
    elements.progressFill.style.width = percent + '%';
}

function addLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    logEntry.innerHTML = `
        <span class="log-time">${timestamp}</span>
        <span class="log-message">${message}</span>
    `;
    elements.logContent.appendChild(logEntry);
    elements.logContent.scrollTop = elements.logContent.scrollHeight;
}

function clearLogs() {
    elements.logContent.innerHTML = '<div class="log-entry log-info"><span class="log-time">00:00:00</span><span class="log-message">Logs cleared</span></div>';
}

// Download Functions
function downloadReport() {
    if (!state.sessionId) {
        alert('No active session');
        return;
    }
    window.location.href = `http://localhost:5000/api/download/report/${state.sessionId}`;
    addLog('Report download started', 'success');
}

function downloadVisualizations() {
    if (!state.sessionId) {
        alert('No active session');
        return;
    }
    window.location.href = `http://localhost:5000/api/download/visualizations/${state.sessionId}`;
    addLog('Visualizations download started', 'success');
}

// Utility
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
