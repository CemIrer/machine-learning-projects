"""
============================================================
COMP4350 - Machine Learning Assignment
============================================================
Part A: CART Decision Tree (built from scratch)
Part B: Random Forest using Random Subspace Method (built from scratch)

Allowed libraries for the algorithms: numpy, pandas only.
Allowed libraries for metrics & visualization: any.

Author: [Student Name]
Date: March 2026
============================================================
"""

import sys
import numpy as np
import pandas as pd
from collections import Counter
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Increase recursion limit since a fully grown tree can be very deep
sys.setrecursionlimit(10000)

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================
# SECTION 1 — GINI IMPURITY
# Core splitting criterion for CART algorithm.
# Gini = 1 - sum(p_i^2)
# A value of 0 means the node is perfectly pure (all same class).
# Higher value means more mixed classes.
# ============================================================

def gini_impurity(y):
    """
    Calculate Gini Impurity for an array of class labels.
    Formula: Gini = 1 - sum(p_i^2) where p_i = proportion of class i.
    """
    n = len(y)
    if n == 0:
        return 0.0
    # Count occurrences of each class and compute proportions
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - float(np.sum(probs ** 2))


def weighted_gini(y_left, y_right):
    """
    Weighted average Gini of two child nodes after a split.
    Weight is proportional to the number of samples in each side.
    """
    n_left  = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left / n_total)  * gini_impurity(y_left) + \
           (n_right / n_total) * gini_impurity(y_right)


# ============================================================
# SECTION 2 — TREE NODE CLASS
# Each node stores the split rule (or leaf label) and links
# to left/right children.
# ============================================================

class DecisionNode:
    """Represents one node in the CART decision tree."""
    def __init__(self):
        self.feature        = None   # feature name used for splitting
        self.threshold      = None   # split value (numeric) or category value (categorical)
        self.is_categorical = False  # True for categorical splits
        self.left           = None   # left child  (condition is True)
        self.right          = None   # right child (condition is False)
        self.is_leaf        = False  # True if this is a terminal node
        self.label          = None   # predicted class (only for leaf nodes)
        self.gini           = 0.0    # Gini impurity at this node
        self.n_samples      = 0      # number of training samples at this node
        self.class_counts   = {}     # class distribution at this node


# ============================================================
# SECTION 3 — FIND BEST SPLIT
# Tries every possible binary split for every feature and
# returns the split that gives the lowest weighted Gini.
#
# For numeric features: try all midpoints between consecutive
#   sorted unique values as thresholds (condition: value <= threshold).
# For categorical features: try each unique value as
#   "equals this value" vs "not equals" (one-vs-rest).
# ============================================================

def find_best_split(X, y, features):
    """
    Search all features and all their candidate thresholds/values.
    Returns: (best_feature, best_threshold, best_weighted_gini, is_categorical)
    """
    best_wg          = float('inf')
    best_feature     = None
    best_threshold   = None
    best_categorical = False

    y_arr = np.array(y)

    for feat in features:
        col = X[feat].values

        if pd.api.types.is_numeric_dtype(X[feat]):
            # ---- Numeric feature ----
            # Candidate thresholds: midpoints between consecutive sorted unique values
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue  # Cannot split if all values identical
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thr in thresholds:
                left_mask  = col <= thr
                right_mask = ~left_mask
                y_left     = y_arr[left_mask]
                y_right    = y_arr[right_mask]

                # Skip if one side is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                wg = weighted_gini(y_left, y_right)
                if wg < best_wg:
                    best_wg, best_feature, best_threshold, best_categorical = \
                        wg, feat, float(thr), False

        else:
            # ---- Categorical feature ----
            # Candidate splits: each unique value vs all others (one-vs-rest)
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            for val in unique_vals:
                left_mask  = col == val
                right_mask = ~left_mask
                y_left     = y_arr[left_mask]
                y_right    = y_arr[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                wg = weighted_gini(y_left, y_right)
                if wg < best_wg:
                    best_wg, best_feature, best_threshold, best_categorical = \
                        wg, feat, val, True

    return best_feature, best_threshold, best_wg, best_categorical


# ============================================================
# SECTION 4 — BUILD CART TREE (recursive)
# Recursively splits data until stopping conditions are met.
# No pruning is applied (fully grown tree).
#
# Stopping conditions:
#   1. All samples in node belong to the same class (Gini = 0).
#   2. No valid split found (all feature values identical).
#   3. Optional: max_depth reached.
# ============================================================

def build_cart(X, y, features=None, depth=0, max_depth=None):
    """
    Recursively build a fully grown CART decision tree.

    Parameters:
        X          : DataFrame of features
        y          : Series/array of class labels
        features   : list of feature names to consider (None = all)
        depth      : current depth (used for max_depth check)
        max_depth  : maximum allowed depth (None = no limit)

    Returns: root DecisionNode of the built subtree
    """
    node = DecisionNode()
    y_arr = np.array(y)

    # Store node statistics
    node.gini        = gini_impurity(y_arr)
    node.n_samples   = len(y_arr)
    node.class_counts = dict(zip(*np.unique(y_arr, return_counts=True)))

    # Majority class (used if this becomes a leaf)
    majority = Counter(y_arr.tolist()).most_common(1)[0][0]

    # --- Stopping Condition 1: Pure node (all same class) ---
    if len(np.unique(y_arr)) == 1:
        node.is_leaf = True
        node.label   = y_arr[0]
        return node

    # --- Stopping Condition 2: Max depth reached ---
    if max_depth is not None and depth >= max_depth:
        node.is_leaf = True
        node.label   = majority
        return node

    if features is None:
        features = list(X.columns)

    # --- Find the best split across all features and thresholds ---
    best_feat, best_thr, best_wg, best_cat = find_best_split(X, y_arr, features)

    # --- Stopping Condition 3: No valid split found ---
    if best_feat is None:
        node.is_leaf = True
        node.label   = majority
        return node

    # Set split parameters for this node
    node.feature        = best_feat
    node.threshold      = best_thr
    node.is_categorical = best_cat

    # --- Split the data ---
    col = X[best_feat].values
    if best_cat:
        mask = col == best_thr        # left: equals the value
    else:
        mask = col <= best_thr        # left: less than or equal to threshold

    X_left  = X[mask].reset_index(drop=True)
    y_left  = pd.Series(y_arr[mask])
    X_right = X[~mask].reset_index(drop=True)
    y_right = pd.Series(y_arr[~mask])

    # --- Recursively build left and right subtrees ---
    node.left  = build_cart(X_left,  y_left,  features, depth + 1, max_depth)
    node.right = build_cart(X_right, y_right, features, depth + 1, max_depth)

    return node


# ============================================================
# SECTION 5 — PREDICTION
# Traverse the tree from root to leaf for each sample.
# ============================================================

def predict_one(node, x):
    """Predict the class of a single sample by traversing the tree."""
    # If leaf node, return the stored class label
    if node.is_leaf:
        return node.label

    # Navigate left or right based on the split condition
    if node.is_categorical:
        go_left = (x[node.feature] == node.threshold)
    else:
        go_left = (x[node.feature] <= node.threshold)

    return predict_one(node.left if go_left else node.right, x)


def predict_all(tree, X):
    """Predict classes for all rows in a DataFrame."""
    return [predict_one(tree, X.iloc[i]) for i in range(len(X))]


# ============================================================
# SECTION 6 — TREE STATISTICS
# Helper functions to compute depth and leaf count.
# ============================================================

def tree_depth(node):
    """Compute the maximum depth of the tree."""
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_leaves(node):
    """Count the number of leaf nodes."""
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def count_nodes(node):
    """Count total nodes (internal + leaf)."""
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


# ============================================================
# SECTION 7 — PERFORMANCE METRICS
# Multi-class evaluation using macro averaging.
# For each class c:
#   TP_c = correctly predicted as c
#   FP_c = incorrectly predicted as c (not truly c)
#   FN_c = truly c but predicted as something else
#   TN_c = neither truly c nor predicted as c
# Macro average = simple mean over all classes.
# ============================================================

def compute_metrics(y_true, y_pred, classes):
    """
    Compute classification performance metrics for multi-class problems.
    Uses macro-averaging (equal weight for each class).

    Returns a dict with Accuracy, TP Rate, TN Rate, Precision, F-Score,
    Total TP, Total TN.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)

    # Total TP = number of correctly classified samples
    total_tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    precisions, recalls, tn_rates, f1s = [], [], [], []
    total_tn = 0

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != c and p != c)

        total_tn += tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tn_rate   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        tn_rates.append(tn_rate)
        f1s.append(f1)

    return {
        'Accuracy':             round(total_tp / n, 4),
        'TP Rate (Recall)':     round(float(np.mean(recalls)), 4),
        'TN Rate':              round(float(np.mean(tn_rates)), 4),
        'Precision':            round(float(np.mean(precisions)), 4),
        'F-Score':              round(float(np.mean(f1s)), 4),
        'Total Number of TP':   total_tp,
        'Total Number of TN':   total_tn,
    }


def print_and_write_metrics(title, metrics, file_handle=None):
    """Print metrics to console and optionally write to a file."""
    lines = [
        f"\n{title}",
        "-" * 40,
        f"Accuracy:              {metrics['Accuracy']:.4f}",
        f"TP Rate (Recall):      {metrics['TP Rate (Recall)']:.4f}",
        f"TN Rate:               {metrics['TN Rate']:.4f}",
        f"Precision:             {metrics['Precision']:.4f}",
        f"F-Score:               {metrics['F-Score']:.4f}",
        f"Total Number of TP:    {metrics['Total Number of TP']}",
        f"Total Number of TN:    {metrics['Total Number of TN']}",
    ]
    for line in lines:
        print(line)
        if file_handle:
            file_handle.write(line + "\n")


# ============================================================
# SECTION 8 — DECISION TREE VISUALIZATION
# Draws the tree using matplotlib.
# Because a fully grown tree on 501 samples can have hundreds
# of nodes, we display up to a configurable depth for clarity.
# The full tree is still built and used for predictions.
# ============================================================

# Color scheme for leaf classes
CLASS_COLORS = {
    'low':       '#ff9999',
    'medium':    '#ffcc88',
    'high':      '#88cc88',
    'very high': '#8888ff',
}
NODE_COLOR = '#aaddff'


def _draw_node(ax, node, x, y, x_span, dy, depth, max_depth,
               parent_xy=None):
    """
    Recursively draw one node and its children.
    x_span: horizontal space allocated to this subtree.
    dy: vertical distance between levels.
    """
    if depth > max_depth:
        # Draw a placeholder to show tree continues
        ax.text(x, y, '...', ha='center', va='center', fontsize=6,
                color='gray', style='italic')
        if parent_xy:
            ax.plot([parent_xy[0], x], [parent_xy[1] - 0.25, y + 0.25],
                    '-', color='#888888', lw=0.6, zorder=1)
        return

    # Build label text
    if node.is_leaf:
        color = CLASS_COLORS.get(str(node.label), '#dddddd')
        txt = f"{node.label}\nn={node.n_samples}"
    else:
        color = NODE_COLOR
        if node.is_categorical:
            txt = (f"{node.feature}\n"
                   f"== {node.threshold}\n"
                   f"n={node.n_samples}  G={node.gini:.3f}")
        else:
            thr = node.threshold
            thr_str = f"{thr/1e6:.2f}M" if thr >= 1_000_000 else f"{thr:.1f}"
            txt = (f"{node.feature}\n"
                   f"<= {thr_str}\n"
                   f"n={node.n_samples}  G={node.gini:.3f}")

    # Draw box
    box_props = dict(boxstyle='round,pad=0.25', facecolor=color,
                     edgecolor='#333333', linewidth=0.8, alpha=0.9)
    ax.text(x, y, txt, ha='center', va='center',
            fontsize=5.5, bbox=box_props, zorder=3)

    # Draw edge from parent
    if parent_xy:
        ax.plot([parent_xy[0], x], [parent_xy[1] - 0.3, y + 0.3],
                '-', color='#444444', lw=0.7, zorder=1)

    if not node.is_leaf:
        half = x_span / 2.0
        _draw_node(ax, node.left,  x - half/2, y - dy, half, dy,
                   depth + 1, max_depth, parent_xy=(x, y))
        _draw_node(ax, node.right, x + half/2, y - dy, half, dy,
                   depth + 1, max_depth, parent_xy=(x, y))


def visualize_tree(root, max_display_depth=4, filename='decision_tree.png'):
    """
    Draw the CART decision tree.
    Full tree is built but only `max_display_depth` levels are shown.
    """
    actual_depth  = tree_depth(root)
    total_leaves  = count_leaves(root)
    total_nodes   = count_nodes(root)

    fig_w = max(14, min(28, 2 ** (max_display_depth + 1)))
    fig_h = max(10, (max_display_depth + 1) * 2.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-(max_display_depth + 1) * 2.4, 1.2)
    ax.axis('off')

    title = (f"CART Decision Tree  (Fully Grown — No Pruning)\n"
             f"Total depth: {actual_depth}  |  Total nodes: {total_nodes}  "
             f"|  Leaf nodes: {total_leaves}  |  "
             f"Displaying top {max_display_depth} levels")
    ax.set_title(title, fontsize=9, pad=8)

    _draw_node(ax, root, x=0, y=0, x_span=1.8, dy=2.0,
               depth=0, max_depth=max_display_depth)

    # Legend
    legend_handles = [
        mpatches.Patch(color=NODE_COLOR,   label='Decision node'),
        mpatches.Patch(color='#ff9999',    label='Leaf: low'),
        mpatches.Patch(color='#ffcc88',    label='Leaf: medium'),
        mpatches.Patch(color='#88cc88',    label='Leaf: high'),
        mpatches.Patch(color='#8888ff',    label='Leaf: very high'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7,
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {filename}  (depth={actual_depth}, leaves={total_leaves})")


# ============================================================
# SECTION 9 — RANDOM FOREST (Random Subspace Method)
# Builds multiple CART trees, each trained on:
#   1. A bootstrap sample of the training data (sampling with replacement).
#   2. A random subset of features (Random Subspace Method).
# Final prediction = majority vote across all trees.
# Number of trees: at least 15, at most 100 (we use 25).
# ============================================================

class RandomForest:
    """
    Random Forest using Random Subspace Method.
    Each tree sees: bootstrap sample + random feature subset.
    """

    def __init__(self, n_trees=25, n_features_per_tree=None):
        """
        n_trees             : number of decision trees (15–100).
        n_features_per_tree : features per tree (None = sqrt of total features).
        """
        self.n_trees             = n_trees
        self.n_features_per_tree = n_features_per_tree
        self.trees               = []   # list of DecisionNode (root)
        self.feature_subsets     = []   # list of feature subsets used per tree

    def fit(self, X, y):
        """
        Train the random forest on (X, y).
        Steps per tree:
          1. Select random feature subset (Random Subspace).
          2. Create bootstrap sample (Bagging).
          3. Build a fully grown CART tree on those features+samples.
        """
        all_features = list(X.columns)
        n_feat = len(all_features)

        # Default: use sqrt(n_features) features per tree
        k = self.n_features_per_tree or max(2, int(np.sqrt(n_feat)))

        print(f"  Features per tree: {k} out of {n_feat}")
        print(f"  Building {self.n_trees} trees...")

        self.trees           = []
        self.feature_subsets = []

        for i in range(self.n_trees):
            # --- Step 1: Random Subspace — pick k random features ---
            feat_subset = random.sample(all_features, k)
            self.feature_subsets.append(feat_subset)

            # --- Step 2: Bootstrap sample — sample with replacement ---
            n = len(X)
            boot_idx = np.random.choice(n, n, replace=True)
            X_boot = X.iloc[boot_idx][feat_subset].reset_index(drop=True)
            y_boot = y.iloc[boot_idx].reset_index(drop=True)

            # --- Step 3: Build CART tree on the subset ---
            tree = build_cart(X_boot, y_boot, features=feat_subset)
            self.trees.append(tree)

            d = tree_depth(tree)
            l = count_leaves(tree)
            print(f"    Tree {i+1:2d}/{self.n_trees} | "
                  f"features={feat_subset} | depth={d} | leaves={l}")

    def predict(self, X):
        """
        Predict by majority vote from all trees.
        Each tree predicts using only its own feature subset.
        """
        # Collect predictions from each tree
        all_preds = []
        for tree, feats in zip(self.trees, self.feature_subsets):
            preds = predict_all(tree, X[feats])
            all_preds.append(preds)

        # Majority vote for each sample
        n = len(X)
        final_preds = []
        for i in range(n):
            votes = [all_preds[t][i] for t in range(self.n_trees)]
            winner = Counter(votes).most_common(1)[0][0]
            final_preds.append(winner)

        return final_preds


# ============================================================
# MAIN — Load data, train, evaluate, save results
# ============================================================

if __name__ == '__main__':

    # ---- Load datasets ----
    print("Loading data...")
    train_df = pd.read_excel('X_train.xlsx')
    test_df  = pd.read_excel('X_test.xlsx')

    FEATURE_COLS = ['Neighborhood',
                    'Price (TRY)',
                    'Age (Years)',
                    'Net Square Meters (m2)']
    CLASS_COL    = 'Sınıf'
    CLASSES      = ['low', 'medium', 'high', 'very high']

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[CLASS_COL].copy()
    X_test  = test_df[FEATURE_COLS].copy()
    y_test  = test_df[CLASS_COL].copy()

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Features: {FEATURE_COLS}")
    print(f"Classes: {CLASSES}")
    print(f"Train class distribution:\n{y_train.value_counts().to_string()}")

    # ============================================================
    # PART A — CART DECISION TREE
    # ============================================================
    print("\n" + "=" * 60)
    print("PART A: CART DECISION TREE (from scratch)")
    print("=" * 60)

    print("\nBuilding fully grown CART tree on training data...")
    print("(This may take a moment — trying all splits at every node)\n")

    cart_tree = build_cart(X_train, y_train)

    depth   = tree_depth(cart_tree)
    leaves  = count_leaves(cart_tree)
    nodes   = count_nodes(cart_tree)
    print(f"\nTree built successfully!")
    print(f"  Max depth : {depth}")
    print(f"  Total nodes : {nodes}")
    print(f"  Leaf nodes  : {leaves}")

    # Predictions
    print("\nGenerating predictions on training set...")
    y_train_pred = predict_all(cart_tree, X_train)

    print("Generating predictions on test set...")
    y_test_pred = predict_all(cart_tree, X_test)

    # Metrics
    train_metrics = compute_metrics(y_train, y_train_pred, CLASSES)
    test_metrics  = compute_metrics(y_test,  y_test_pred,  CLASSES)

    # Write report
    with open('report.txt', 'w', encoding='utf-8') as f:
        header = [
            "============================================================",
            "COMP4350 - Machine Learning Assignment Report",
            "============================================================",
            "",
            "PART A: CART Decision Tree",
            f"Training set size : {len(X_train)}",
            f"Test set size     : {len(X_test)}",
            f"Tree depth        : {depth}",
            f"Total nodes       : {nodes}",
            f"Leaf nodes        : {leaves}",
        ]
        for line in header:
            print(line)
            f.write(line + "\n")

        print_and_write_metrics("Train Results:", train_metrics, f)
        print_and_write_metrics("Test Results:", test_metrics, f)

    # Visualize tree
    print("\nDrawing decision tree (showing top 4 levels of the full tree)...")
    visualize_tree(cart_tree, max_display_depth=4,
                   filename='decision_tree.png')

    # ============================================================
    # PART B — RANDOM FOREST
    # ============================================================
    print("\n" + "=" * 60)
    print("PART B: RANDOM FOREST (from scratch)")
    print("=" * 60)
    print("\nBuilding Random Forest (25 trees, Random Subspace Method)...")

    rf = RandomForest(n_trees=25)
    rf.fit(X_train, y_train)

    print("\nGenerating Random Forest predictions on test set...")
    y_rf_pred = rf.predict(X_test)

    rf_metrics = compute_metrics(y_test, y_rf_pred, CLASSES)

    # Write RF results to report
    with open('report.txt', 'a', encoding='utf-8') as f:
        section = [
            "",
            "============================================================",
            "PART B: Random Forest",
            f"Number of trees          : {rf.n_trees}",
            f"Method                   : Random Subspace + Bootstrap",
            f"Features per tree        : {len(rf.feature_subsets[0])} "
            f"(sqrt of {len(FEATURE_COLS)})",
        ]
        for line in section:
            print(line)
            f.write(line + "\n")

        print_and_write_metrics("Test Results (Random Forest):", rf_metrics, f)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nPart A — CART Decision Tree:")
    print(f"  Train Accuracy : {train_metrics['Accuracy']:.4f}  "
          f"({train_metrics['Total Number of TP']}/{len(X_train)} correct)")
    print(f"  Test  Accuracy : {test_metrics['Accuracy']:.4f}  "
          f"({test_metrics['Total Number of TP']}/{len(X_test)} correct)")
    print(f"\nPart B — Random Forest (25 trees):")
    print(f"  Test  Accuracy : {rf_metrics['Accuracy']:.4f}  "
          f"({rf_metrics['Total Number of TP']}/{len(X_test)} correct)")
    print("\nOutput files:")
    print("  solution.py       — this source code")
    print("  decision_tree.png — tree visualization (Part A)")
    print("  report.txt        — performance metrics (Part A + B)")
    print("=" * 60)
