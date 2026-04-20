"""
Week 2 - Part 2: Decision Tree Assignment Solution
Group 3

Q1: Apply the given decision tree to Table 1 (test data)
    - Compute 4x4 confusion matrix and accuracy
    - Compute macro and micro Precision, Recall, F1

Q2: Build a CART decision tree from Table 2 (training data)
    - Show Gini impurity calculations
    - Classify new record: Price=10.700.000, Neighbourhood=Adalet, Age=20, m2=140
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']

# ===========================================================================
# Q1: APPLY GIVEN DECISION TREE TO TABLE 1
# ===========================================================================

print("=" * 70)
print("Q1: GIVEN DECISION TREE APPLIED TO TABLE 1 (TEST DATA)")
print("=" * 70)

print("""
Given Decision Tree Structure:
   PRİCE
   ├── <6M          --> LOW PROFİL      (Düşük)
   └── >=6M
        ├── AGE <= 5YEARS --> NEİGHBOORHOOD
        │    ├── Mansuroğlu             --> VERY HIGH PROFILE (Çok Yüksek)
        │    └── The Others             --> HIGH PROFILE    (Yüksek)
        └── AGE > 5YEARS  --> SQUARE METERS
             ├── >= 120                 --> HIGH PROFILE    (Yüksek)
             └── <  120                 --> MEDIUM PROFILE  (Orta)
""")

# ---- Table 1: Test Data (20 records) ----
table1 = pd.DataFrame({
    'ID':            [1, 3, 6, 11, 15, 21, 26, 28, 32, 35,
                      39, 43, 47, 51, 56, 61, 72, 78, 64, 68],
    'Price':         [11250000, 9400000, 4675000, 5150000, 9250000,
                      10500000, 4500000, 6500000, 6750000, 8200000,
                      3975000,  5850000, 4450000, 5500000, 5150000,
                      6300000,  6750000, 5200000, 7300000, 7350000],
    'Neighbourhood': ['Mansuroğlu Mh.', 'Mansuroğlu Mh.', 'Mansuroğlu Mh.',
                      'Mansuroğlu Mh.', 'Mansuroğlu Mh.', 'Mansuroğlu Mh.',
                      'Mansuroğlu Mh.', 'Adalet Mh.', 'Adalet Mh.',
                      'Adalet Mh.', 'Bayraklı Mh.', 'Bayraklı Mh.',
                      'Bayraklı Mh.', 'Bayraklı Mh.', '75. Yıl Mh.',
                      '75. Yıl Mh.', '75. Yıl Mh.', '75. Yıl Mh.',
                      'Cengizhan Mh.', 'Turan Mh.'],
    'Age':           [0, 0, 31, 23, 23, 23, 33, 13, 20, 18,
                      18, 10, 23, 18, 23, 23, 28, 23, 8, 8],
    'SquareMeters':  [100, 85, 100, 120, 126, 135, 120, 110, 130, 128,
                      115, 132, 115, 130, 110, 135, 135, 121, 125, 95],
    'TrueClass':     ['Çok Yüksek', 'Yüksek', 'Orta', 'Orta', 'Yüksek',
                      'Yüksek', 'Orta', 'Orta', 'Yüksek', 'Yüksek',
                      'Düşük', 'Orta', 'Düşük', 'Orta', 'Düşük',
                      'Orta', 'Orta', 'Düşük', 'Yüksek', 'Orta']
})

# ---- Decision tree function (Group 3 tree) ----
def given_decision_tree(price, neighbourhood, age, sq_meters):
    """
    Group 3 Decision Tree:
      PRİCE < 6M                        --> Düşük
      PRİCE >= 6M, AGE <= 5, Mansuroğlu --> Yüksek
      PRİCE >= 6M, AGE <= 5, Others     --> Çok Yüksek
      PRİCE >= 6M, AGE > 5,  m² >= 120  --> Yüksek
      PRİCE >= 6M, AGE > 5,  m² < 120   --> Orta
    """
    if price < 6_000_000:
        return 'Düşük'
    else:
        if age <= 5:
            if 'Mansuroğlu' in neighbourhood:
                return 'Çok Yüksek'   # VERY HIGH PROFILE
            else:
                return 'Yüksek'       # HIGH PROFILE
        else:
            if sq_meters >= 120:
                return 'Yüksek'
            else:
                return 'Orta'

# Apply tree to all test records
table1['Predicted'] = table1.apply(
    lambda r: given_decision_tree(r['Price'], r['Neighbourhood'],
                                   r['Age'], r['SquareMeters']), axis=1)

# Print record-by-record predictions
print("Record-by-record Predictions:")
print("-" * 105)
print(f"{'#':<4} {'Price':>12} {'Neighbourhood':<16} {'Age':>4} {'m²':>4} "
      f"{'True':>12} {'Predicted':>12}  {'Match'}")
print("-" * 105)
for _, row in table1.iterrows():
    match = "OK" if row['TrueClass'] == row['Predicted'] else "WRONG"
    print(f"{int(row['ID']):<4} {row['Price']:>12,} {row['Neighbourhood']:<16} "
          f"{row['Age']:>4} {row['SquareMeters']:>4} "
          f"{row['TrueClass']:>12} {row['Predicted']:>12}  {match}")

# ---------------------------------------------------------------------------
# 4x4 Confusion Matrix  (rows = True class, cols = Predicted class)
# ---------------------------------------------------------------------------
cm = np.zeros((4, 4), dtype=int)
for true_c, pred_c in zip(table1['TrueClass'], table1['Predicted']):
    i = CLASSES.index(true_c)
    j = CLASSES.index(pred_c)
    cm[i][j] += 1

correct  = int(np.trace(cm))
total    = len(table1)
accuracy = correct / total

print("\n" + "=" * 70)
print("4x4 CONFUSION MATRIX  (Rows = True Class, Cols = Predicted Class)")
print("=" * 70)
print(f"\n{'':>16}" + "".join(f"  {c:>12}" for c in CLASSES))
print("-" * 68)
for i, cls in enumerate(CLASSES):
    print(f"{cls:>16}" + "".join(f"  {cm[i][j]:>12}" for j in range(4)))

print(f"\nTotal Correct : {correct} / {total}")
print(f"Accuracy      : {correct}/{total} = {accuracy:.4f}  ({accuracy*100:.2f}%)")

# ---------------------------------------------------------------------------
# Per-class Precision, Recall, F1
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

metrics = {}
for i, cls in enumerate(CLASSES):
    TP = int(cm[i, i])
    FP = int(cm[:, i].sum()) - TP
    FN = int(cm[i, :].sum()) - TP

    P  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    metrics[cls] = {'TP': TP, 'FP': FP, 'FN': FN,
                    'Precision': P, 'Recall': R, 'F1': F1}

    print(f"\n  Class: {cls}")
    print(f"    TP={TP}  |  FP={FP}  |  FN={FN}")
    if (TP + FP) > 0:
        print(f"    Precision = {TP} / ({TP}+{FP}) = {P:.4f}")
    else:
        print(f"    Precision = undefined (no positive predictions) -> 0.0000")
    print(f"    Recall    = {TP} / ({TP}+{FN}) = {R:.4f}")
    if (P + R) > 0:
        print(f"    F1        = 2 x {P:.4f} x {R:.4f} / ({P:.4f}+{R:.4f}) = {F1:.4f}")
    else:
        print(f"    F1        = 0.0000")

# ---------------------------------------------------------------------------
# Macro Averages
# ---------------------------------------------------------------------------
macro_P  = np.mean([m['Precision'] for m in metrics.values()])
macro_R  = np.mean([m['Recall']    for m in metrics.values()])
macro_F1 = np.mean([m['F1']        for m in metrics.values()])

print("\n" + "=" * 70)
print("MACRO AVERAGES  (unweighted mean over all 4 classes)")
print("=" * 70)
p_str  = " + ".join(f"{m['Precision']:.4f}" for m in metrics.values())
r_str  = " + ".join(f"{m['Recall']:.4f}"    for m in metrics.values())
f1_str = " + ".join(f"{m['F1']:.4f}"        for m in metrics.values())
print(f"  Macro Precision = ({p_str})  / 4 = {macro_P:.4f}")
print(f"  Macro Recall    = ({r_str})  / 4 = {macro_R:.4f}")
print(f"  Macro F1        = ({f1_str}) / 4 = {macro_F1:.4f}")

# ---------------------------------------------------------------------------
# Micro Averages
# ---------------------------------------------------------------------------
total_TP = sum(m['TP'] for m in metrics.values())
total_FP = sum(m['FP'] for m in metrics.values())
total_FN = sum(m['FN'] for m in metrics.values())

micro_P  = total_TP / (total_TP + total_FP)
micro_R  = total_TP / (total_TP + total_FN)
micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R)

print("\n" + "=" * 70)
print("MICRO AVERAGES  (aggregate TP/FP/FN across all classes)")
print("=" * 70)
print(f"  Sum TP={total_TP}  |  Sum FP={total_FP}  |  Sum FN={total_FN}")
print(f"  Micro Precision = {total_TP} / ({total_TP}+{total_FP}) = {micro_P:.4f}")
print(f"  Micro Recall    = {total_TP} / ({total_TP}+{total_FN}) = {micro_R:.4f}")
print(f"  Micro F1        = 2 x {micro_P:.4f} x {micro_R:.4f} "
      f"/ ({micro_P:.4f}+{micro_R:.4f}) = {micro_F1:.4f}")
print(f"\n  (Note: Micro F1 = Accuracy = {micro_F1:.4f} in multi-class)")

# ---------------------------------------------------------------------------
# Confusion Matrix Visualization
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_yticklabels(CLASSES, fontsize=10)
ax.set_xlabel('Predicted Class', fontsize=12)
ax.set_ylabel('True Class', fontsize=12)
ax.set_title(f'Q1 - 4x4 Confusion Matrix\nAccuracy = {accuracy*100:.1f}%', fontsize=13)
for i in range(4):
    for j in range(4):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
plt.tight_layout()
plt.savefig('q1_confusion_matrix.png', dpi=200, bbox_inches='tight')
print("\n[Saved] q1_confusion_matrix.png")
plt.close()

# ===========================================================================
# Q2: CART DECISION TREE FROM TABLE 2 (TRAINING DATA)
# ===========================================================================

print("\n\n" + "=" * 70)
print("Q2: CART DECISION TREE — TABLE 2 (TRAINING DATA)")
print("=" * 70)

table2 = pd.DataFrame({
    'ID':            [2, 8, 14, 23, 29, 34, 40, 45, 52, 57, 63, 74],
    'Price':         [4140000, 4750000, 4200000, 7500000, 8750000, 4875000,
                      4100000, 5500000, 3100000, 6800000, 7500000, 5850000],
    'Neighbourhood': ['Mansuroğlu Mh.', 'Mansuroğlu Mh.', 'Mansuroğlu Mh.',
                      'Mansuroğlu Mh.', 'Adalet Mh.', 'Adalet Mh.',
                      'Bayraklı Mh.', 'Bayraklı Mh.', 'Bayraklı Mh.',
                      '75. Yıl Mh.', '75. Yıl Mh.', '75. Yıl Mh.'],
    'Age':           [28, 28, 28, 33, 18, 23, 23, 10, 33, 23, 23, 23],
    'SquareMeters':  [115, 110, 100, 121, 135, 115, 120, 145, 125, 125, 130, 135],
    'Class':         ['Orta', 'Orta', 'Orta', 'Yüksek', 'Yüksek', 'Düşük',
                      'Düşük', 'Orta', 'Düşük', 'Orta', 'Orta', 'Orta']
})

print("\nTraining Data (Table 2) — 12 samples:")
print(table2[['ID','Price','Neighbourhood','Age','SquareMeters','Class']].to_string(index=False))

print("\nClass distribution at root:")
for cls in CLASSES:
    n = (table2['Class'] == cls).sum()
    pct = n / len(table2) * 100
    print(f"  {cls:<14}: {n} samples  ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Gini Impurity Calculations
# ---------------------------------------------------------------------------
def gini(labels):
    n = len(labels)
    if n == 0: return 0.0
    counts = pd.Series(labels).value_counts()
    return 1.0 - sum((c / n) ** 2 for c in counts)

def weighted_gini(left, right):
    n = len(left) + len(right)
    return (len(left)/n)*gini(left) + (len(right)/n)*gini(right)

n2 = len(table2)
d  = (table2['Class']=='Düşük').sum()
o  = (table2['Class']=='Orta').sum()
y  = (table2['Class']=='Yüksek').sum()
cy = (table2['Class']=='Çok Yüksek').sum()

print(f"\nRoot Gini = 1 - (({d}/{n2})^2 + ({o}/{n2})^2 + ({y}/{n2})^2 + ({cy}/{n2})^2)")
root_gini = gini(table2['Class'].tolist())
print(f"Root Gini = {root_gini:.4f}")

print("\n" + "=" * 70)
print("GINI IMPURITY SEARCH — ALL NUMERIC FEATURES")
print("=" * 70)

numeric_features = ['Price', 'Age', 'SquareMeters']
best_splits = {}

for feat in numeric_features:
    sorted_vals = sorted(table2[feat].unique())
    thresholds  = [(sorted_vals[k]+sorted_vals[k+1])/2
                   for k in range(len(sorted_vals)-1)]

    print(f"\nFeature: {feat}")
    print(f"  {'Threshold':>15} {'Left':>5} {'Right':>5} "
          f"{'Gini_L':>8} {'Gini_R':>8} {'Weighted':>10}")
    print(f"  {'-'*58}")

    best_wg, best_thr = float('inf'), None
    for thr in thresholds:
        left  = table2[table2[feat] <= thr]['Class'].tolist()
        right = table2[table2[feat] >  thr]['Class'].tolist()
        g_l   = gini(left)
        g_r   = gini(right)
        wg    = weighted_gini(left, right)
        tag   = "  <-- BEST" if wg < best_wg else ""
        if wg < best_wg:
            best_wg, best_thr = wg, thr
        print(f"  {thr:>15,.1f} {len(left):>5} {len(right):>5} "
              f"{g_l:>8.4f} {g_r:>8.4f} {wg:>10.4f}{tag}")

    best_splits[feat] = (best_thr, best_wg)
    reduction = root_gini - best_wg
    print(f"  --> Best: {feat} <= {best_thr:,.1f}  "
          f"Weighted Gini={best_wg:.4f}  Reduction={reduction:.4f}")

print("\n" + "-" * 55)
print("BEST SPLIT COMPARISON (root level):")
for feat, (thr, wg) in best_splits.items():
    print(f"  {feat:<15}: threshold={thr:>10,.1f}  "
          f"Weighted Gini={wg:.4f}  Reduction={root_gini-wg:.4f}")

# ---------------------------------------------------------------------------
# Build CART with sklearn
# ---------------------------------------------------------------------------
table2_enc = pd.get_dummies(table2[['Price','Neighbourhood','Age','SquareMeters']],
                             columns=['Neighbourhood'], prefix='Nbhd')
feat_cols  = list(table2_enc.columns)
X_train    = table2_enc.astype(float)
y_train    = table2['Class']

cart = DecisionTreeClassifier(criterion='gini', random_state=42)
cart.fit(X_train, y_train)

print("\n" + "=" * 70)
print("CART TREE STRUCTURE (sklearn — fully grown, no pruning)")
print("=" * 70)
print(export_text(cart, feature_names=feat_cols))

# Visualize
plt.figure(figsize=(22, 10))
plot_tree(cart,
          feature_names=feat_cols,
          class_names=list(cart.classes_),
          filled=True, rounded=True, fontsize=8, impurity=True)
plt.title('Q2 - CART Decision Tree (Gini, fully grown)\nGroup 3 - Table 2 Training Data',
          fontsize=13)
plt.tight_layout()
plt.savefig('q2_cart_tree.png', dpi=200, bbox_inches='tight')
print("[Saved] q2_cart_tree.png")
plt.close()

# ---------------------------------------------------------------------------
# Q2.2: Classify new record
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Q2.2: CLASSIFY NEW RECORD")
print("=" * 70)
print("  Price         = 10,700,000 TL")
print("  Neighbourhood = Adalet Mh.")
print("  Age           = 20 years")
print("  Square Meters = 140 m2")

new_raw = pd.DataFrame([{
    'Price': 10700000, 'Neighbourhood': 'Adalet Mh.',
    'Age': 20, 'SquareMeters': 140
}])
new_enc = pd.get_dummies(new_raw, columns=['Neighbourhood'], prefix='Nbhd')
new_enc = new_enc.reindex(columns=feat_cols, fill_value=0).astype(float)

predicted_class = cart.predict(new_enc)[0]
predicted_proba = cart.predict_proba(new_enc)[0]

print(f"\n  --> Predicted Class: {predicted_class}")
print("\n  Class probabilities:")
for cls, prob in zip(cart.classes_, predicted_proba):
    bar = '#' * int(prob * 30)
    print(f"    {cls:<14}: {prob:.4f}  {bar}")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY — GROUP 3")
print("=" * 70)
print(f"\nQ1 — Given Decision Tree on Table 1 (20 test samples)")
print(f"  Accuracy        : {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro Precision : {macro_P:.4f}")
print(f"  Macro Recall    : {macro_R:.4f}")
print(f"  Macro F1        : {macro_F1:.4f}")
print(f"  Micro Precision : {micro_P:.4f}")
print(f"  Micro Recall    : {micro_R:.4f}")
print(f"  Micro F1        : {micro_F1:.4f}")
print(f"\nQ2 — CART Tree trained on Table 2 (12 training samples)")
print(f"  Algorithm       : CART (Gini Impurity, fully grown, no pruning)")
print(f"  New record prediction: {predicted_class}")
print("=" * 70)
