# Dry Bean Classification Analysis

## Project Overview
This project focuses on classifying seven different types of dry beans using computer vision features. The dataset contains 13,611 instances of dry beans with 16 morphological features extracted through image processing. The goal is to build machine learning models that can accurately classify bean types, which has significant implications for market quality control and pricing.

## Dataset Information
**Source:** [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

**Classes:** BARBUNYA, SIRA, HOROZ, DERMASON, CALI, BOMBAY, SEKER

**Size:** 13,611 instances with 16 features + target variable

## Features
The dataset contains 17 attributes:
- **ID**: Unique identifier for each instance
- **12 Dimensional Features**: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness
- **4 Shape Factors**: ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
- **Class**: Target variable (7 bean types)

## Data Preprocessing & Analysis

### Data Quality Checks
- ✅ **Null Values**: Comprehensive check for missing data
- ✅ **Duplicates**: Identification and handling of duplicate entries
- ✅ **Multicollinearity**: Analysis of correlation between features and handle multicollinearity using PCA
- ✅ **Skewness**: Assessment of feature distribution symmetry and handle it through normalization

### Data Visualization
- Numerical feature distributions
- Categorical target variable analysis
- Correlation matrix and heatmap
- Multicollinearity detection

### Feature Engineering
- **Outlier Handling**: Z-score and IQR method for outlier detection and treatment
- **Skewness Correction**: Yeo-Johnson transformation applied to normalize feature distributions
- **Scaling**: Standardization of numerical features
- **Encoding**: Categorical variable encoding for the target class using one hot encoding 

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Applied to obtain explained variance and reduce feature dimensionality while preserving information

## Machine Learning Models

### Algorithms Implemented
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier (RFC)**
3. **K-Nearest Neighbors (KNN)**

### Hyperparameter Optimization
- **BayesSearchCV**: Bayesian optimization for hyperparameter tuning to find optimal model parameters

### Performance Evaluation
Comprehensive model evaluation using:
- Accuracy scores
- Precision and Recall metrics
- F1-score
- Confusion matrices
- Cross-validation results

## Key Techniques & Benefits

### Yeo-Johnson Transformation
- Handles both positive and negative data
- Effectively reduces skewness in feature distributions
- Improves model performance by normalizing data

### Z-Score Outlier Detection
- Identifies statistical outliers
- Robust method for outlier treatment
- Prevents model performance degradation

### PCA Benefits
- Reduces computational complexity
- Mitigates multicollinearity issues
- Preserves maximum variance with fewer components

## Installation & Requirements

```bash
# Required libraries
pip install pandas numpy matplotlib seaborn scikit-learn scipy