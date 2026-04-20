"""
COMP4350 - Midterm Exam Project
Part B: K-Means Clustering — built from scratch
================================================
Dataset       : midtermProject-part2-data.xlsx  (200 records, 3 numeric columns)
Algorithm     : K-Means with K-Means++ initialisation  (smarter than random init)
Normalisation : Min-Max scaling applied to all 3 variables BEFORE clustering
Input         : k value entered by the user from the keyboard
Outputs       : result.txt (cluster assignments + WCSS / BCSS / Dunn Index)
                PNG files  (scatter plots for every pair of variables)
                Interactive matplotlib window with axis-selector buttons
Allowed libs  : numpy, pandas  (for the algorithm itself)
                matplotlib      (for visualisation — explicitly allowed)
Reproducible  : Fixed random seed ensures identical cluster output every run
================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')          # use Tk-backed interactive window on Windows
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.patches as mpatches

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

DATA_PATH   = 'midtermProject-part2-data.xlsx'
RESULT_FILE = 'result.txt'

# Colour palette for up to 10 clusters (distinct, colour-blind-friendly)
PALETTE = [
    '#e6194b', '#3cb44b', '#4169e1', '#f58231',
    '#9b59b6', '#17becf', '#d62728', '#8c564b',
    '#e377c2', '#7f7f7f',
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_normalize(path):
    """
    Load the Excel file and apply min-max normalisation to all 3 columns.
    Returns:
        data_norm : (200, 3) float64 array — values in [0, 1]
        col_names : list of original column name strings
    """
    df        = pd.read_excel(path)
    col_names = df.columns.tolist()
    raw       = df.values.astype(np.float64)

    col_min   = raw.min(axis=0)
    col_max   = raw.max(axis=0)
    col_range = col_max - col_min + 1e-9    # avoid division by zero

    data_norm = (raw - col_min) / col_range
    return data_norm, col_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. K-MEANS++ INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_plus_plus_init(X, k):
    """
    K-Means++ centroid seeding strategy.

    How it works:
        1. Pick the first centroid uniformly at random.
        2. For each subsequent centroid, sample a data point with probability
           proportional to its squared distance from the nearest existing centroid.
    This biases the initial seeds to be spread out, which leads to faster
    convergence and better final clusters compared to purely random initialisation.
    """
    n = X.shape[0]

    # Step 1 — choose first centroid randomly
    first_idx  = np.random.randint(n)
    centroids  = [X[first_idx].copy()]

    for _ in range(1, k):
        # Step 2 — for each point, find its minimum squared distance to any centroid
        min_sq_dists = np.array([
            min(float(np.sum((x - c) ** 2)) for c in centroids)
            for x in X
        ])

        # Sample next centroid proportional to distance²
        probs      = min_sq_dists / min_sq_dists.sum()
        cumulative = np.cumsum(probs)
        r          = np.random.rand()
        next_idx   = int(np.searchsorted(cumulative, r))
        centroids.append(X[next_idx].copy())

    return np.array(centroids)   # shape (k, n_features)


# ══════════════════════════════════════════════════════════════════════════════
# 3. K-MEANS CORE ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def assign_to_clusters(X, centroids):
    """
    Assign every data point to the nearest centroid (Euclidean distance).
    Uses numpy broadcasting for efficiency — no sklearn involved.
        X          : (n, d)
        centroids  : (k, d)
    Returns integer array of shape (n,) with values in [0, k-1].
    """
    # Broadcast subtraction → (n, k, d), then sum squares → (n, k)
    diffs    = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)      # (n, k)
    return np.argmin(sq_dists, axis=1)         # (n,)


def recompute_centroids(X, assignments, k):
    """
    Recompute each centroid as the mean of its assigned points.
    If a cluster becomes empty (edge case), its centroid is reset to a random point.
    """
    n_features   = X.shape[1]
    new_centroids = np.empty((k, n_features))

    for j in range(k):
        members = X[assignments == j]
        if len(members) == 0:
            # Empty cluster — reinitialise to a random data point
            new_centroids[j] = X[np.random.randint(len(X))].copy()
        else:
            new_centroids[j] = members.mean(axis=0)

    return new_centroids


def run_kmeans(X, k, max_iter=500):
    """
    Execute the K-Means algorithm until convergence or max_iter iterations.
    Convergence criterion: maximum centroid displacement < 1e-8.
    Returns (final_centroids, cluster_assignments, iterations_used).
    """
    centroids = kmeans_plus_plus_init(X, k)

    for iteration in range(1, max_iter + 1):
        assignments   = assign_to_clusters(X, centroids)
        new_centroids = recompute_centroids(X, assignments, k)

        # Measure how much centroids moved this iteration
        shifts        = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))
        max_shift     = float(shifts.max())
        centroids     = new_centroids

        print(f"  Iter {iteration:>3d}  │  max centroid shift = {max_shift:.10f}")

        if max_shift < 1e-8:
            print(f"  ✔  Converged after {iteration} iteration(s).")
            break

    return centroids, assignments, iteration


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_wcss(X, assignments, centroids):
    """
    WCSS — Within-Cluster Sum of Squares.
    Measures compactness: sum of squared distances from each point to its centroid.
    Lower WCSS → tighter, more compact clusters.
    """
    wcss = 0.0
    for j in range(len(centroids)):
        members = X[assignments == j]
        diffs   = members - centroids[j]          # (n_j, d)
        wcss   += float(np.sum(diffs ** 2))
    return wcss


def compute_bcss(X, assignments, centroids):
    """
    BCSS — Between-Cluster Sum of Squares.
    Measures separation: weighted sum of squared distances from each centroid
    to the global mean. Higher BCSS → clusters are further apart (better).
    """
    global_mean = X.mean(axis=0)
    bcss = 0.0
    for j in range(len(centroids)):
        n_j   = int(np.sum(assignments == j))
        diff  = centroids[j] - global_mean
        bcss += n_j * float(np.dot(diff, diff))
    return bcss


def compute_dunn_index(X, assignments, centroids):
    """
    Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter).
    Higher Dunn Index → well-separated, compact clusters.

    Inter-cluster distance : centroid-to-centroid Euclidean distance.
    Intra-cluster diameter : maximum pairwise distance within a cluster.
    """
    k = len(centroids)

    # ── minimum inter-cluster distance ──────────────────────────────────────
    min_inter = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
            if d < min_inter:
                min_inter = d

    # ── maximum intra-cluster diameter (vectorised per cluster) ─────────────
    max_intra = 0.0
    for j in range(k):
        members = X[assignments == j]
        n_m     = len(members)
        if n_m < 2:
            continue
        # compute pairwise distances with broadcasting
        diffs      = members[:, np.newaxis, :] - members[np.newaxis, :, :]
        pair_dists = np.sqrt(np.sum(diffs ** 2, axis=2))     # (n_m, n_m)
        diameter   = float(pair_dists.max())
        if diameter > max_intra:
            max_intra = diameter

    if max_intra == 0:
        return float('inf')
    return min_inter / max_intra


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESULT FILE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_result_file(assignments, k, wcss, bcss, dunn, filepath=RESULT_FILE):
    """
    Write result.txt with:
      - One line per record: "Record N: Cluster M"
      - Per-cluster record counts
      - WCSS, BCSS, Dunn Index
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        n = len(assignments)

        # individual record assignments
        for i in range(n):
            f.write(f"Record {i + 1}: Cluster {int(assignments[i]) + 1}\n")

        f.write("\n")

        # cluster size summary
        for j in range(k):
            count = int(np.sum(assignments == j))
            f.write(f"Cluster {j + 1}: {count} records\n")

        f.write("\n")

        # quality metrics
        f.write(f"WCSS:        {wcss:.4f}\n")
        f.write(f"BCSS:        {bcss:.4f}\n")
        f.write(f"Dunn Index:  {dunn:.4f}\n")

    print(f"\n[Saved] {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLUSTER VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _draw_scatter(ax, X_norm, assignments, centroids, k, xi, yi, col_names):
    """
    Draw a scatter plot of cluster assignments on axes (xi, yi).
    Points are coloured by cluster; centroids shown as black-edged ×-markers.
    """
    ax.clear()

    for j in range(k):
        mask = (assignments == j)
        ax.scatter(
            X_norm[mask, xi], X_norm[mask, yi],
            c=PALETTE[j % len(PALETTE)],
            label=f'Cluster {j + 1}',
            alpha=0.72, edgecolors='k', linewidths=0.35, s=55, zorder=2,
        )
        # centroid marker
        ax.scatter(
            centroids[j, xi], centroids[j, yi],
            c=PALETTE[j % len(PALETTE)],
            marker='X', s=250, edgecolors='black', linewidths=1.3, zorder=5,
        )

    x_label = col_names[xi]
    y_label = col_names[yi]
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        f'K-Means Clustering  (k = {k})\n'
        f'X-axis: {x_label}   |   Y-axis: {y_label}',
        fontsize=12, pad=7,
    )
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.28)


def save_all_pair_plots(X_norm, assignments, centroids, k, col_names):
    """
    Save a PNG scatter plot for every pair of the 3 variables.
    These are included as mandatory cluster visualisation samples.
    """
    n_cols  = X_norm.shape[1]
    pairs   = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    saved   = []

    for xi, yi in pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_scatter(ax, X_norm, assignments, centroids, k, xi, yi, col_names)
        plt.tight_layout()

        safe_x = col_names[xi].replace(' ', '_').replace('/', '_')
        safe_y = col_names[yi].replace(' ', '_').replace('/', '_')
        fname  = f"cluster_k{k}_{safe_x}_vs_{safe_y}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] {fname}")
        saved.append(fname)

    return saved


def show_interactive_viz(X_norm, assignments, centroids, k, col_names):
    """
    Interactive matplotlib window.
    Shows a 'data / cluster visualization' section with one button per
    variable-pair; clicking a button redraws the scatter plot accordingly.
    Also saves the currently displayed plot when a button is clicked.
    """
    n_cols = X_norm.shape[1]
    pairs  = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.22, top=0.92)

    # draw default view (first pair)
    _draw_scatter(ax, X_norm, assignments, centroids, k,
                  pairs[0][0], pairs[0][1], col_names)

    # ── buttons (one per variable pair) ─────────────────────────────────────
    btn_objects = []
    n_pairs     = len(pairs)
    btn_w       = 0.22
    gap         = 0.03
    total_w     = n_pairs * btn_w + (n_pairs - 1) * gap
    start_x     = (1.0 - total_w) / 2.0

    for idx, (xi, yi) in enumerate(pairs):
        btn_ax  = fig.add_axes([
            start_x + idx * (btn_w + gap),
            0.05, btn_w, 0.09
        ])
        label   = f"{col_names[xi][:10]}\nvs {col_names[yi][:10]}"
        btn     = mwidgets.Button(btn_ax, label,
                                  color='#d0e8ff', hovercolor='#5599dd')

        def on_click(event, _xi=xi, _yi=yi):
            _draw_scatter(ax, X_norm, assignments, centroids, k,
                          _xi, _yi, col_names)
            fig.canvas.draw_idle()
            # also save the selected view as PNG
            safe_x = col_names[_xi].replace(' ', '_').replace('/', '_')
            safe_y = col_names[_yi].replace(' ', '_').replace('/', '_')
            fname  = f"cluster_k{k}_{safe_x}_vs_{safe_y}.png"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"[Saved] {fname}")

        btn.on_clicked(on_click)
        btn_objects.append(btn)     # keep reference so buttons aren't garbage-collected

    fig.suptitle("data / cluster visualization", fontsize=13, fontweight='bold')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load and normalise data ───────────────────────────────────────────────
    print("Loading dataset...")
    X_norm, col_names = load_and_normalize(DATA_PATH)
    print(f"  {X_norm.shape[0]} records  |  {X_norm.shape[1]} features: {col_names}")
    print("  Min-max normalisation applied.\n")

    # ── Get k from the user ───────────────────────────────────────────────────
    while True:
        try:
            k = int(input("Enter the number of clusters (k ≥ 2): "))
            if k < 2:
                print("  k must be at least 2. Try again.")
                continue
            if k > X_norm.shape[0]:
                print("  k cannot exceed the number of data points. Try again.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")

    # ── Run K-Means ───────────────────────────────────────────────────────────
    print(f"\nRunning K-Means++ with k = {k}  (seed={RANDOM_SEED})...\n")
    centroids, assignments, n_iter = run_kmeans(X_norm, k)

    # ── Compute quality metrics ───────────────────────────────────────────────
    print("\nComputing cluster quality metrics...")
    wcss = compute_wcss(X_norm, assignments, centroids)
    bcss = compute_bcss(X_norm, assignments, centroids)
    dunn = compute_dunn_index(X_norm, assignments, centroids)

    print(f"  WCSS       : {wcss:.4f}")
    print(f"  BCSS       : {bcss:.4f}")
    print(f"  Dunn Index : {dunn:.4f}")

    # ── Print cluster summary ─────────────────────────────────────────────────
    print()
    for j in range(k):
        count = int(np.sum(assignments == j))
        print(f"  Cluster {j + 1}: {count} records")

    # ── Write result.txt ──────────────────────────────────────────────────────
    write_result_file(assignments, k, wcss, bcss, dunn)

    # ── Save all pair-wise cluster PNGs ───────────────────────────────────────
    print("\nSaving cluster visualisation PNG files...")
    saved_pngs = save_all_pair_plots(X_norm, assignments, centroids, k, col_names)

    # ── Show interactive visualisation window ─────────────────────────────────
    print("\nOpening interactive visualisation window...")
    print("  Click a button to switch between variable pairs.")
    print("  Clicking a button also saves that view as a PNG.\n")
    try:
        show_interactive_viz(X_norm, assignments, centroids, k, col_names)
    except Exception as e:
        # Fallback: if no display is available (e.g. headless server),
        # the PNG files already saved are sufficient.
        print(f"  [Note] Interactive window could not be opened: {e}")
        print("  PNG files have been saved successfully.")

    print("\nDone.  Deliverables: result.txt  +  cluster PNG files")
