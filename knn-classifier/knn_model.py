"""
K-Nearest Neighbors Classification for Gender Prediction
Using head circumference, height, and weight features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load data
df_features = pd.read_excel('bas-boy-kilo.ods', engine='odf')
df_labels = pd.read_excel('cinsiyet.ods', engine='odf')

X = df_features.values
y = df_labels['Cinsiyet'].values

print("=" * 60)
print("DATA LOADED")
print("=" * 60)
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"\nClass distribution:")
print(f"  Class 0 (Female): {np.sum(y == 0)} samples")
print(f"  Class 1 (Male): {np.sum(y == 1)} samples")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Test K values from 1 to 15
print("\n" + "=" * 60)
print("TESTING K VALUES (1 to 15)")
print("=" * 60)

k_values = range(1, 16)
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"K={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# Find best K
best_k_index = np.argmax(test_accuracies)
best_k = k_values[best_k_index]
best_accuracy = test_accuracies[best_k_index]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"Best K value: {best_k}")
print(f"Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Train best model and analyze
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Female (0)', 'Male (1)']))

# Find wrong predictions
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

wrong_indices = np.where(y_test != y_pred)[0]
print(f"\nTotal errors: {len(wrong_indices)} / {len(y_test)}")

if len(wrong_indices) > 0:
    print("\nMisclassified samples:")
    print("-" * 80)
    print(f"{'Index':<8} {'HeadCirc':<12} {'Height':<12} {'Weight':<12} {'True':<10} {'Predicted':<10}")
    print("-" * 80)

    for idx in wrong_indices:
        sample = X_test[idx]
        real = y_test[idx]
        predicted = y_pred[idx]

        print(f"{idx:<8} {sample[0]:<12.2f} {sample[1]:<12.2f} {sample[2]:<12.2f} "
              f"{'Female' if real == 0 else 'Male':<10} {'Female' if predicted == 0 else 'Male':<10}")

    wrong_samples = X_test[wrong_indices]
    print("\n" + "=" * 60)
    print("STATISTICS OF MISCLASSIFIED SAMPLES")
    print("=" * 60)
    print(f"\nHead Circumference - Mean: {wrong_samples[:, 0].mean():.2f}, "
          f"Min: {wrong_samples[:, 0].min():.2f}, Max: {wrong_samples[:, 0].max():.2f}")
    print(f"Height - Mean: {wrong_samples[:, 1].mean():.2f}, "
          f"Min: {wrong_samples[:, 1].min():.2f}, Max: {wrong_samples[:, 1].max():.2f}")
    print(f"Weight - Mean: {wrong_samples[:, 2].mean():.2f}, "
          f"Min: {wrong_samples[:, 2].min():.2f}, Max: {wrong_samples[:, 2].max():.2f}")

# Plot results
print("\n" + "=" * 60)
print("CREATING VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy', linewidth=2)
plt.plot(k_values, test_accuracies, marker='s', label='Test Accuracy', linewidth=2)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}', linewidth=2)
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('K-NN: K Value vs Model Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('knn_results.png', dpi=300)
print("Plot saved: knn_results.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Best K value: {best_k}")
print(f"Test accuracy: {best_accuracy*100:.2f}%")
print(f"Errors: {len(wrong_indices)}")
print(f"Correct: {len(y_test) - len(wrong_indices)}")
print("=" * 60)
