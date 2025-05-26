# SciVisualizer - Scientific Data Visualization and Analysis Tool

## Overview
SciVisualizer is a comprehensive Python-based application for scientific data visualization, analysis, and machine learning. Built with PySide6 and Qt, it provides an intuitive graphical interface for various scientific computing tasks.

## Key Features

### Data Analysis
- **D-S Evidence Theory**: Combine evidence from multiple sources
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Factor Analysis**: Identify latent variables in datasets
- Interactive data visualization with matplotlib

### Machine Learning
- Support Vector Machines (SVM)
- Random Forest algorithms
- Extreme Learning Machines (ELM)
- Radial Basis Function Networks (RBF)

### Specialized Modules
- Crack identification and analysis
- Deep learning integration
- Data collection and processing tools

## Installation
1. Ensure Python 3.8+ is installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

### Getting Started
1. Run the application:
```bash
python main.py
```

2. Main interface components:
- Navigation sidebar for module selection
- Data visualization area
- Analysis controls
- Results display

### Data Analysis Workflow
1. Load your dataset (CSV format recommended)
2. Select analysis method:
   - D-S Evidence Theory
   - PCA
   - Factor Analysis
3. View results in tables and interactive plots
4. Export results as needed

### Machine Learning Workflow
1. Prepare your dataset
2. Select ML algorithm
3. Configure parameters
4. Train and evaluate model
5. Visualize results

## Algorithm Implementations

### Data Analysis Algorithms
- `algorithm/dataanalysisalgo.py`: Contains core implementations for:
  - D-S evidence calculations
  - PCA transformations
  - Factor analysis computations

### Machine Learning Algorithms
- SVM: `page/SVM.py`
- Random Forest: `page/randomforest.py`
- ELM: `page/ELM.py`
- RBF: `page/RBF.py`

## Configuration
Customize the application through:
- `resource/style.qss`: UI stylesheet
- `config/`: Configuration files
- Application settings menu

## Troubleshooting
- Ensure all dependencies are installed
- Check file permissions for data access
- Verify dataset formats match expected inputs

## License
MIT License - See LICENSE file for details