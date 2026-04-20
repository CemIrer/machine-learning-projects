# K-NN Gender Classification Assignment

This folder contains all materials for the K-NN machine learning assignment.

## Files

### Data Files
- `bas-boy-kilo.ods` - Feature data (head circumference, height, weight)
- `cinsiyet.ods` - Label data (gender: 0=Female, 1=Male)

### Code Files
- `knn_model.py` - Main K-NN implementation
- `rule_analysis.py` - Error analysis and rule extraction

### Documentation
- `REPORT.md` - Complete assignment report
- `README.md` - This file

### Output (generated when you run the code)
- `knn_results.png` - Performance visualization

## How to Run

1. Make sure you're in the w1 directory:
```bash
cd w1
```

2. Run the main K-NN model:
```bash
python knn_model.py
```

This will:
- Load the data
- Train K-NN models with K=1 to K=15
- Find the best K value
- Show detailed results
- Generate visualization

3. Run the rule analysis:
```bash
python rule_analysis.py
```

This will:
- Analyze misclassified samples
- Test various classification rules
- Show which rules work best

## Results Summary

- **Best K value**: 1
- **Test accuracy**: 72.50%
- **Errors**: 11 out of 40 test samples
- **Main error pattern**: Females misclassified as males (9/11 errors)
- **Best simple rule**: HeadCircumference > 45 cm (57.50% accuracy)

## Requirements

The code uses these Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- odfpy

All should already be installed.



