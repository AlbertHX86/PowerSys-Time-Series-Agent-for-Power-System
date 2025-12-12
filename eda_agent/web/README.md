# EDA Agent - Frontend UI/UX Documentation

## Overview
Modern web-based interface for AI-powered time series analysis and forecasting. Built with vanilla HTML/CSS/JavaScript for maximum compatibility and easy integration with any backend.

## Features

### 1. **Data Upload**
- Drag & drop file upload
- Supports CSV, Excel (.xlsx, .xls), and TXT formats
- File validation and preview
- Visual feedback during upload

### 2. **Configuration Panel**
- Data description input
- Time resolution selector (10 min, 15 min, 30 min, 1 hour, 1 day)
- System capacity specification
- Analysis goal definition

### 3. **Custom Model Generator** (AI-Powered)
- Toggle to enable/disable custom model generation
- Natural language model description input
- Quick example templates:
  - LightGBM with tuning
  - Stacking ensemble
  - XGBoost optimized
- Integrates with GPT-4o for code generation

### 4. **Interactive Analysis Workflow**
- Visual step indicator (Upload → Configure → Model → Results)
- Real-time progress bar
- Console log output with timestamps
- Color-coded log messages (info, success, warning, error)

### 5. **Results Dashboard**
- **Performance Metrics Cards**: Best Model, R², MAE, RMSE
- **Model Comparison Table**: Side-by-side comparison of all models
- **Visualizations Grid**: Interactive image gallery
  - Figure 1: Data Overview
  - Figure 3: Model Comparison
  - Figure 4: Custom Model Detail
- **Analysis Report**: Detailed text summary

### 6. **Hyperparameter Tuning Feedback**
- AI-generated improvement suggestions with:
  - Parameter name and current/suggested values
  - Expected performance impact
  - Code snippets for implementation
- Interactive feedback textarea for custom requests
- Iteration counter (max 2 iterations)
- Accept/Regenerate actions

### 7. **Download Section**
- Analysis report download (.txt)
- All visualizations download (.zip)
- One-click download buttons

## File Structure

```
web/
├── index.html          # Main HTML structure
├── styles.css          # Complete styling with modern design system
├── script.js           # Interactive functionality and API integration
└── README.md           # This documentation
```

## Design System

### Colors
- **Primary**: Blue gradient (#3b82f6 → #2563eb)
- **Secondary**: Purple (#8b5cf6)
- **Success**: Green (#10b981)
- **Warning**: Orange (#f59e0b)
- **Error**: Red (#ef4444)
- **Neutrals**: Gray scale (50-900)

### Typography
- Font Family: Inter (Google Fonts)
- Weights: 300, 400, 500, 600, 700

### Layout
- Max Width: 1400px
- 2-column grid layout (left: input, right: output)
- Responsive breakpoints: 1200px, 768px

### Components
- Cards with shadow elevation
- Smooth transitions (200ms)
- Border radius: 8px, 12px, 16px
- Consistent spacing and padding

## Integration Guide

### Backend API Endpoints (Mock - Replace with Actual)

```javascript
// 1. Upload and Start Analysis
POST /api/analyze
FormData: {
    file: File,
    description: string,
    resolution: string,
    capacity: string,
    goal: string,
    session_id: string,
    custom_model_request?: string
}
Response: {
    metrics: {
        [model_name]: { name, r2, mae, rmse }
    },
    visualizations: [{ name, url, id }],
    report: string,
    suggestions: [{ title, description, impact, code }]
}

// 2. Submit Feedback for Regeneration
POST /api/regenerate
JSON: {
    session_id: string,
    feedback: string,
    iteration: number
}
Response: Same as above

// 3. Download Files
GET /api/download/report/{session_id}
GET /api/download/visualizations/{session_id}
```

### JavaScript Integration Points

**File: script.js**

```javascript
// Line 175-220: startAnalysis()
// Replace mock API call with actual fetch:
const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData
});
const results = await response.json();
displayResults(results);

// Line 265-290: submitFeedback()
// Replace mock regeneration with actual API:
const response = await fetch('/api/regenerate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        session_id: state.sessionId,
        feedback: feedback,
        iteration: state.iterationCount
    })
});
const results = await response.json();
displayResults(results);

// Line 380-395: downloadReport() & downloadVisualizations()
// Replace with actual download endpoints:
window.location.href = `/api/download/report/${state.sessionId}`;
window.location.href = `/api/download/visualizations/${state.sessionId}`;
```

## How to Use

### 1. **Local Development**
```bash
# Open index.html in browser (no server needed for static demo)
open web/index.html

# Or use a simple HTTP server
cd web
python -m http.server 8000
# Visit http://localhost:8000
```

### 2. **Integration Steps**
1. Update API endpoints in `script.js` (lines marked with "// Replace with actual API")
2. Ensure backend returns JSON in expected format
3. Configure CORS if backend is on different domain
4. Test file upload size limits (adjust if needed)
5. Customize branding (logo, colors in `styles.css`)

### 3. **Customization**

**Change Colors:**
```css
/* In styles.css, modify :root variables */
:root {
    --primary: #your-color;
    --secondary: #your-color;
}
```

**Modify Workflow Steps:**
```javascript
// In script.js, update step labels
const steps = ['upload', 'configure', 'model', 'results'];
```

**Add More Metrics:**
```javascript
// In displayResults(), add to metrics grid
elements.newMetric.textContent = results.metrics.new_value;
```

## Features Highlight

### ✅ Implemented
- Fully responsive design (desktop, tablet, mobile)
- Drag & drop file upload with validation
- Real-time progress tracking
- Interactive console logs
- Custom model generator toggle
- AI-powered tuning suggestions display
- Iteration counter with max limit
- Model comparison table with highlighting
- Visualization gallery
- Download functionality
- Modern gradient backgrounds
- Smooth animations and transitions

### 🔄 Ready for Backend Integration
- File upload API
- Analysis execution API
- Feedback/regeneration API
- Download endpoints
- Real visualization images (currently placeholders)
- Actual performance metrics

## Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance
- No external dependencies (except Google Fonts)
- Optimized CSS with variables
- Efficient DOM manipulation
- Lazy-loaded images (if implemented)

## Accessibility
- Semantic HTML5
- ARIA labels where needed
- Keyboard navigation support
- Color contrast ratio > 4.5:1
- Focus indicators

## Future Enhancements (Optional)
- Dark mode toggle
- Session history panel
- Real-time WebSocket updates during analysis
- Interactive chart editing (Plotly.js integration)
- Export to PDF report
- Multi-language support
- Advanced parameter configuration UI
- Model comparison side-by-side view

## Notes
- Current version uses mock data for demonstration
- Replace all `mockResults` with actual API responses
- Visualization placeholders should be replaced with actual images from backend
- Session management currently client-side only (add backend session handling)

## Questions?
Refer to inline comments in each file for detailed implementation notes.
