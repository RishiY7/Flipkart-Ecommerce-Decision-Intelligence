"""Part 3: Segmentation and dimensionality reduction.

Produces:
- outputs/part3_results.txt
- outputs/part3_segments.csv
- outputs/part3_hierarchical.csv
"""
import csv
import os
import numpy as np


def run_part3():
    # -------------------------------------------------------------------------
    # Load dataset and extract X1-X5 (continuous predictors)
    # -------------------------------------------------------------------------
    records_path = os.path.join("data", "records.csv")
    with open(records_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    feat_cols = [
        "customer_tenure_months",      # X1
        "complaint_count_90d",         # X2
        "discount_order_share",        # X3
        "avg_order_value_inr_hundreds",# X4
        "loyalty_score"                # X5
    ]

    x_raw = np.array([[float(r[col]) for col in feat_cols] for r in rows])
    outcome = np.array([int(r["churned"]) for r in rows])
    value = np.array([float(r["order_value_inr_hundreds"]) for r in rows])

    n = len(x_raw)
    mean_x = np.mean(x_raw, axis=0)
    sd_x = np.std(x_raw, axis=0, ddof=1)
    x_std = (x_raw - mean_x) / sd_x

    results = []

    # -------------------------------------------------------------------------
    # Task 3.1: K-means from scratch (K in 2, 3, 4, 5, 6)
    # -------------------------------------------------------------------------
    wcss = {}
    assignments_k = {}

    for k in [2, 3, 4, 5, 6]:
        step = n // k
        init_indices = [i * step for i in range(k)]
        centroids = x_std[init_indices].copy()

        current_clusters = np.full(n, -1)
        while True:
            # Squared Euclidean distance to centroids
            dists = np.sum((x_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            # Ties go to lower-numbered centroid (np.argmin preserves first/lowest index)
            new_clusters = np.argmin(dists, axis=1)

            if np.array_equal(new_clusters, current_clusters):
                break
            current_clusters = new_clusters.copy()

            for k_idx in range(k):
                members = x_std[current_clusters == k_idx]
                if len(members) > 0:
                    centroids[k_idx] = np.mean(members, axis=0)

        final_dists = np.sum((x_std - centroids[current_clusters]) ** 2, axis=1)
        wcss[k] = float(np.sum(final_dists))
        assignments_k[k] = (current_clusters, centroids)
        results.append((f"wcss_k{k}", f"{wcss[k]:.4f}"))

    # -------------------------------------------------------------------------
    # Task 3.2: Choose K with the elbow
    # -------------------------------------------------------------------------
    # smallest K in {3, 4, 5} where WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = None
    for k in [3, 4, 5]:
        drop_curr = wcss[k] - wcss[k + 1]
        drop_prev = wcss[k - 1] - wcss[k]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = k
            break

    results.append(("elbow_k", f"{elbow_k}"))

    # -------------------------------------------------------------------------
    # Task 3.3: Profile the K = 3 segments
    # -------------------------------------------------------------------------
    k3_clusters, _ = assignments_k[3]
    # Order clusters by ascending mean of X3 (discount_order_share, index 2)
    x3_raw = x_raw[:, 2]
    mean_x3_per_cluster = [float(np.mean(x3_raw[k3_clusters == k_idx])) for k_idx in range(3)]
    sorted_cluster_indices = np.argsort(mean_x3_per_cluster)  # Segments A, B, C

    os.makedirs("outputs", exist_ok=True)
    segments_csv_path = os.path.join("outputs", "part3_segments.csv")
    segment_names = ["A", "B", "C"]
    segment_masks = {}

    with open(segments_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment", "size", "mean_X1", "mean_X2", "mean_X3",
            "mean_X4", "mean_X5", "outcome_rate", "mean_value"
        ])

        for seg_name, c_idx in zip(segment_names, sorted_cluster_indices):
            mask = (k3_clusters == c_idx)
            segment_masks[seg_name] = mask
            size = int(np.sum(mask))
            results.append((f"size_segment_{seg_name}", f"{size}"))

            means = [float(np.mean(x_raw[mask, j])) for j in range(5)]
            out_rate = float(np.mean(outcome[mask]))
            m_val = float(np.mean(value[mask]))

            writer.writerow([
                seg_name,
                size,
                f"{means[0]:.4f}",
                f"{means[1]:.4f}",
                f"{means[2]:.4f}",
                f"{means[3]:.4f}",
                f"{means[4]:.4f}",
                f"{out_rate:.4f}",
                f"{m_val:.4f}"
            ])

    # -------------------------------------------------------------------------
    # Task 3.4: Hierarchical clustering on supplied dissimilarity table
    # -------------------------------------------------------------------------
    cohort_labels = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])

    def parse_member_num(name):
        return int(name[1:])

    def format_cluster(cluster_members):
        sorted_members = sorted(cluster_members, key=parse_member_num)
        return "+".join(sorted_members), min(parse_member_num(m) for m in sorted_members)

    hier_csv_path = os.path.join("outputs", "part3_hierarchical.csv")
    hier_rows = []

    for linkage in ["single", "complete", "average"]:
        clusters = [[lbl] for lbl in cohort_labels]
        merges = []

        for merge_idx in range(1, 6):
            min_dist = float("inf")
            best_pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    pairwise = [
                        dist_matrix[cohort_labels.index(m1), cohort_labels.index(m2)]
                        for m1 in c1 for m2 in c2
                    ]
                    if linkage == "single":
                        d = min(pairwise)
                    elif linkage == "complete":
                        d = max(pairwise)
                    elif linkage == "average":
                        d = sum(pairwise) / len(pairwise)

                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)

            i, j = best_pair
            c1, c2 = clusters[i], clusters[j]
            name1, min1 = format_cluster(c1)
            name2, min2 = format_cluster(c2)

            if min1 < min2:
                ca, cb = name1, name2
            else:
                ca, cb = name2, name1

            merges.append((merge_idx, ca, cb, min_dist))
            merged = sorted(c1 + c2, key=parse_member_num)
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(merged)

            hier_rows.append([linkage, merge_idx, ca, cb, f"{min_dist:.4f}"])

        # Determine cut based on maximum gap between consecutive merge heights
        heights = [m[3] for m in merges]
        gaps = [heights[g + 1] - heights[g] for g in range(4)]
        max_gap_idx = int(np.argmax(gaps))
        merges_below_cut = max_gap_idx + 1
        num_clusters_at_cut = 6 - merges_below_cut
        results.append((f"{linkage}_clusters_at_cut", f"{num_clusters_at_cut}"))

    with open(hier_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        writer.writerows(hier_rows)

    # -------------------------------------------------------------------------
    # Task 3.5: PCA by the seven steps
    # -------------------------------------------------------------------------
    # (1) Mean vector: all zeros
    # (2) Centre data: already centred
    # (3) Covariance matrix S = (1/(n-1)) X^T X
    cov_s = (1.0 / (n - 1)) * (x_std.T @ x_std)

    # (4) Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_s)

    # (5) Sort descending by eigenvalue
    sort_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]

    # Checkpoint: eigenvalue_sum = 5.0000
    eig_sum = float(np.sum(eigvals))
    results.append(("eigenvalue_sum", f"{eig_sum:.4f}"))

    for i in range(5):
        results.append((f"eigenvalue_{i+1}", f"{eigvals[i]:.4f}"))

    explained_ratios = eigvals / eig_sum
    for i in range(5):
        results.append((f"explained_ratio_{i+1}", f"{explained_ratios[i]:.4f}"))

    cum_ratios = np.cumsum(explained_ratios)
    for i in range(5):
        results.append((f"cumulative_ratio_{i+1}", f"{cum_ratios[i]:.4f}"))

    comp_80 = int(np.where(cum_ratios >= 0.8)[0][0] + 1)
    results.append(("components_for_80pct", f"{comp_80}"))

    # Sign convention: flip each eigenvector so entry of largest absolute value is positive
    for j in range(5):
        col = eigvecs[:, j]
        max_abs_idx = int(np.argmax(np.abs(col)))
        if col[max_abs_idx] < 0:
            eigvecs[:, j] = -col

    pc1_w = eigvecs[:, 0]
    pc2_w = eigvecs[:, 1]

    for i in range(5):
        results.append((f"pc1_loading_X{i+1}", f"{pc1_w[i]:.4f}"))

    for i in range(5):
        results.append((f"pc2_loading_X{i+1}", f"{pc2_w[i]:.4f}"))

    # (6) Keep top two eigenvectors; (7) Project Z = X W
    w_top2 = eigvecs[:, :2]
    z_proj = x_std @ w_top2

    # Projections of first three data rows
    results.append(("row1_pc1", f"{z_proj[0, 0]:.4f}"))
    results.append(("row1_pc2", f"{z_proj[0, 1]:.4f}"))
    results.append(("row2_pc1", f"{z_proj[1, 0]:.4f}"))
    results.append(("row2_pc2", f"{z_proj[1, 1]:.4f}"))
    results.append(("row3_pc1", f"{z_proj[2, 0]:.4f}"))
    results.append(("row3_pc2", f"{z_proj[2, 1]:.4f}"))

    # Mean PC1 score for each Task 3.3 segment
    for seg_name in segment_names:
        mask = segment_masks[seg_name]
        mean_pc1 = float(np.mean(z_proj[mask, 0]))
        results.append((f"mean_pc1_segment_{seg_name}", f"{mean_pc1:.4f}"))

    # -------------------------------------------------------------------------
    # Write part3_results.txt
    # -------------------------------------------------------------------------
    part3_res_path = os.path.join("outputs", "part3_results.txt")
    with open(part3_res_path, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 3 complete. Results written to {part3_res_path}, {segments_csv_path}, and {hier_csv_path}.")


if __name__ == "__main__":
    run_part3()
