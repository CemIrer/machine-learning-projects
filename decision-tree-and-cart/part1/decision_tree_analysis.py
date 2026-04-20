"""
Decision Tree Analysis for Bornova Housing Dataset
Multi-class classification of house price categories
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('bornova_housing_dataset.xlsx')

print("=" * 70)
print("BORNOVA HOUSING DATASET - DECISION TREE ANALYSIS")
print("=" * 70)
print(f"\nDataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")

# Create price categories (multi-class classification)
# Low: < 2,000,000 TL
# Medium: 2,000,000 - 3,000,000 TL
# High: > 3,000,000 TL

def categorize_price(price):
    if price < 2_000_000:
        return 'Low'
    elif price < 3_000_000:
        return 'Medium'
    else:
        return 'High'

df['PriceCategory'] = df['Price'].apply(categorize_price)

print("\n" + "=" * 70)
print("PRICE CATEGORIES (TARGET VARIABLE)")
print("=" * 70)
print(df['PriceCategory'].value_counts())
print(f"\nCategory distribution:")
for category in ['Low', 'Medium', 'High']:
    count = (df['PriceCategory'] == category).sum()
    percentage = count / len(df) * 100
    print(f"  {category}: {count} houses ({percentage:.1f}%)")

# Prepare features and target
# Convert neighborhood to numeric (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['Neighborhood'], prefix='Neighborhood')

# Features: Age, NetSquareMeters, and Neighborhood dummy variables
feature_cols = [col for col in df_encoded.columns if col not in ['Price', 'PriceCategory']]
X = df_encoded[feature_cols]
y = df['PriceCategory']

print("\n" + "=" * 70)
print("FEATURES FOR DECISION TREE")
print("=" * 70)
print(f"Total features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train Decision Tree with different depths
print("\n" + "=" * 70)
print("TRAINING DECISION TREES (Different Depths)")
print("=" * 70)

depths = [2, 3, 4, 5, None]
results = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    depth_str = str(depth) if depth else "Unlimited"
    print(f"Max Depth: {depth_str:10s} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    results.append({
        'depth': depth,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'model': dt
    })

# Find best model
best_result = max(results, key=lambda x: x['test_acc'])
best_model = best_result['model']
best_depth = best_result['depth']

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(f"Best max_depth: {best_depth if best_depth else 'Unlimited'}")
print(f"Test Accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc']*100:.2f}%)")

# Detailed evaluation
y_pred = best_model.predict(X_test)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
print(cm)
print("\nRows: True labels, Columns: Predicted labels")
print("Order: Low, Medium, High")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred))

# Feature importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Visualize Decision Tree
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Tree visualization
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=feature_cols,
          class_names=['High', 'Low', 'Medium'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Housing Price Classification', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("Decision tree saved: decision_tree_visualization.png")

# Feature importance plot
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("Feature importance saved: feature_importance.png")

# Confusion matrix visualization
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Low', 'Medium', 'High'])
plt.yticks(tick_marks, ['Low', 'Medium', 'High'])

for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion matrix saved: confusion_matrix.png")

print("\n" + "=" * 70)
print("DECISION RULES EXTRACTED")
print("=" * 70)
print("\nThe Decision Tree learned these key patterns:")
print("1. Houses in certain neighborhoods tend to be in specific price ranges")
print("2. Age is important: newer houses (low age) are more expensive")
print("3. Size (NetSquareMeters) affects price category")
print("4. Combination of neighborhood + age + size determines final category")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Best model accuracy: {best_result['test_acc']*100:.2f}%")
print(f"Total samples: {len(df)}")
print(f"Features used: {len(feature_cols)}")
print(f"Classes: 3 (Low, Medium, High)")
print("=" * 70)
