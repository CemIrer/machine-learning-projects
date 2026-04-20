"""
Rule Extraction from Misclassified Samples
Analyzing errors to determine simple classification rules
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
df_features = pd.read_excel('bas-boy-kilo.ods', engine='odf')
df_labels = pd.read_excel('cinsiyet.ods', engine='odf')

X = df_features.values
y = df_labels['Cinsiyet'].values

# Split data (same random_state as main model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train best model (K=1)
best_knn = KNeighborsClassifier(n_neighbors=1)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

# Find misclassified samples
wrong_indices = np.where(y_test != y_pred)[0]

print("=" * 80)
print("DETAILED ERROR ANALYSIS")
print("=" * 80)

# Categorize errors
female_to_male = []  # Female but predicted Male
male_to_female = []  # Male but predicted Female

for idx in wrong_indices:
    sample = X_test[idx]
    real = y_test[idx]
    predicted = y_pred[idx]

    if real == 0 and predicted == 1:
        female_to_male.append({
            'HeadCircumference': sample[0],
            'Height': sample[1],
            'Weight': sample[2]
        })
    elif real == 1 and predicted == 0:
        male_to_female.append({
            'HeadCircumference': sample[0],
            'Height': sample[1],
            'Weight': sample[2]
        })

print(f"\nFemale -> Male errors: {len(female_to_male)}")
print(f"Male -> Female errors: {len(male_to_female)}")

# Focus on Female -> Male errors (largest category)
print("\n" + "=" * 80)
print("FEMALE -> MALE MISCLASSIFICATIONS (Main Problem)")
print("=" * 80)

df_female_male = pd.DataFrame(female_to_male)
print("\nMisclassified females:")
print(df_female_male)

print("\nStatistics:")
print(df_female_male.describe())

# Compare with all females in test set
print("\n" + "=" * 80)
print("ALL FEMALES IN TEST SET")
print("=" * 80)

all_females = X_test[y_test == 0]
df_all_females = pd.DataFrame(all_females, columns=['HeadCircumference', 'Height', 'Weight'])
print(df_all_females.describe())

# Compare with all males
print("\n" + "=" * 80)
print("ALL MALES IN TEST SET")
print("=" * 80)

all_males = X_test[y_test == 1]
df_all_males = pd.DataFrame(all_males, columns=['HeadCircumference', 'Height', 'Weight'])
print(df_all_males.describe())

# Test different rules
print("\n" + "=" * 80)
print("RULE TESTING")
print("=" * 80)

print("\n### HEAD CIRCUMFERENCE RULES ###")
for threshold in [44, 45, 46, 47]:
    y_pred_rule = np.where(X_test[:, 0] > threshold, 1, 0)

    wrong_corrected = sum(1 for idx in wrong_indices if y_test[idx] == y_pred_rule[idx])
    acc = accuracy_score(y_test, y_pred_rule)

    print(f"HeadCirc > {threshold} -> Male, else Female")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Corrected errors: {wrong_corrected}/{len(wrong_indices)}")
    print()

print("\n### HEIGHT RULES ###")
for threshold in [70, 75, 80, 85]:
    y_pred_rule = np.where(X_test[:, 1] > threshold, 1, 0)

    wrong_corrected = sum(1 for idx in wrong_indices if y_test[idx] == y_pred_rule[idx])
    acc = accuracy_score(y_test, y_pred_rule)

    print(f"Height > {threshold} -> Male, else Female")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Corrected errors: {wrong_corrected}/{len(wrong_indices)}")
    print()

print("\n### WEIGHT RULES ###")
for threshold in [8, 9, 10, 11]:
    y_pred_rule = np.where(X_test[:, 2] > threshold, 1, 0)

    wrong_corrected = sum(1 for idx in wrong_indices if y_test[idx] == y_pred_rule[idx])
    acc = accuracy_score(y_test, y_pred_rule)

    print(f"Weight > {threshold} -> Male, else Female")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Corrected errors: {wrong_corrected}/{len(wrong_indices)}")
    print()

print("\n### COMBINED RULE (Height AND Weight) ###")
y_pred_combined = np.where((X_test[:, 1] > 75) & (X_test[:, 2] > 9), 1, 0)
wrong_corrected = sum(1 for idx in wrong_indices if y_test[idx] == y_pred_combined[idx])
acc = accuracy_score(y_test, y_pred_combined)

print(f"(Height > 75 AND Weight > 9) -> Male, else Female")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  Corrected errors: {wrong_corrected}/{len(wrong_indices)}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Analysis results:
1. The model primarily makes Female -> Male errors (9/11 errors)
2. Misclassified females have physical features closer to males
3. These samples are in the boundary region between classes

RECOMMENDED RULE:
- Best single-feature rule: HeadCircumference > 45 cm
  - Total accuracy: 57.50%
  - However, this is still worse than K-NN (72.50%)

Why simple rules perform worse:
- Gender prediction requires multiple features
- K-NN considers all three features together
- Single threshold rules are too simplistic
- These patterns will be learned automatically by Decision Trees next week
""")
