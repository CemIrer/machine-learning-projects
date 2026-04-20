# Bornova Housing Dataset - Decision Tree Analysis

Group 4 submission for Week 2 Machine Learning assignment.

## Overview

This project analyzes housing prices in Bornova district (Izmir) using Decision Tree classification. We created a dataset of 100 houses and built a model to predict price categories (Low/Medium/High).

## Files

### Data
- `bornova_housing_dataset.xlsx` - Housing dataset (100 houses from 6 neighborhoods)

### Code
- `dataset_generator.py` - Generates the synthetic housing dataset
- `decision_tree_analysis.py` - Trains and evaluates Decision Tree model

### Outputs
- `decision_tree_visualization.png` - Visual representation of the trained tree
- `feature_importance.png` - Chart showing which features matter most
- `confusion_matrix.png` - Classification performance matrix

### Documentation
- `REPORT.md` - Comprehensive analysis report
- `README.md` - This file

## Quick Start

### 1. Generate Dataset (Optional - already done)

```bash
cd w2
python dataset_generator.py
```

This creates `bornova_housing_dataset.xlsx` with 100 houses.

### 2. Run Decision Tree Analysis

```bash
python decision_tree_analysis.py
```

This will:
- Load the dataset
- Create price categories (Low/Medium/High)
- Train Decision Trees with different depths
- Evaluate performance
- Generate visualizations
- Show results

## Dataset Details

**Features:**
- Price (Turkish Lira)
- Neighborhood (6 neighborhoods in Bornova)
- Age (years)
- NetSquareMeters

**Standard Criteria:**
- All houses: 3+1 rooms
- All houses: Natural gas heating
- All houses: Completed construction

**Neighborhoods:**
- Erzene
- Kazimdirik
- Evka-3
- Yesilova
- Altas
- Kizilcikli

## Results Summary

**Model Performance:**
- Best accuracy: 60%
- Classification: 3 classes (Low/Medium/High price)
- Most important features:
  1. NetSquareMeters (43%)
  2. Age (40%)
  3. Neighborhood (17%)

**Key Findings:**
- Larger apartments are more expensive
- Newer buildings command premium prices
- Neighborhood affects price but less than size and age

## Requirements

Python libraries (already installed from Week 1):
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl

## Assignment Checklist

- [x] Dataset created (100+ houses)
- [x] Features: Price, Neighborhood, Age, NetSquareMeters
- [x] At least 4 different neighborhoods (we have 6)
- [x] Standard criteria applied (3+1, natural gas, completed)
- [x] Decision Tree implemented
- [x] Multi-class classification demonstrated
- [x] Analysis and report completed
- [ ] Group member information (add to REPORT.md if needed)

## Understanding the Code

### dataset_generator.py

Creates realistic synthetic data:
- Uses actual Bornova neighborhood names
- Realistic price ranges based on neighborhood
- Age affects price (newer = more expensive)
- Size affects price (bigger = more expensive)
- Random variation added for realism

### decision_tree_analysis.py

Performs complete analysis:
1. Loads data
2. Creates price categories
3. Encodes categorical features
4. Splits train/test
5. Trains multiple trees
6. Evaluates performance
7. Shows feature importance
8. Creates visualizations

## Next Steps

To improve the model:
1. Collect more real data (200+ houses)
2. Add more features (floor, elevator, parking)
3. Use tree pruning (max_depth=5 showed promise)
4. Try Random Forest for better accuracy
5. Apply to other districts for comparison

