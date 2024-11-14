# Sobri-Naive-PSO
Intradialytic Hypotension Prediction using PSO-Optimized Naive Bayes

# Intradialytic Hypotension Prediction using PSO-Optimized Naive Bayes

## Overview
This project implements an advanced machine learning approach for predicting Intradialytic Hypotension (IDH) in hemodialysis patients. The system combines Naive Bayes classification with Particle Swarm Optimization (PSO) for feature selection optimization, aiming to improve prediction accuracy and identify the most relevant clinical indicators.

## Author
**Muhammad Sobri**
- GitHub: [@sobri3195](https://github.com/sobri3195)
- Created: November 2024

## Features
- Optimized feature selection using Particle Swarm Optimization
- Gaussian Naive Bayes classifier implementation
- Automated data preprocessing and standardization
- Cross-validation for robust performance evaluation
- Comprehensive performance metrics reporting
- Easy-to-use interface for medical data analysis

## Technical Details
The implementation includes:
1. `SobriParticleSwarmOptimization`: Custom PSO implementation for feature selection
2. `SobriOptimizedNaiveBayes`: Enhanced Naive Bayes classifier with PSO optimization
3. `SobriDataProcessor`: Automated data preprocessing and standardization

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Random

## Installation
```bash
git clone https://github.com/sobri3195/idh-prediction-pso-naive-bayes.git
cd idh-prediction-pso-naive-bayes
pip install -r requirements.txt
