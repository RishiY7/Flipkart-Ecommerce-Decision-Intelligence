"""Part 2: Classification and evaluation.

Produces:
- outputs/part2_results.txt
- outputs/part2_thresholds.csv
- outputs/part2_knn_cv.csv
"""
import csv
import os
import numpy as np
from shared_utils import (
    load_dataset,
    make_stratified_split,
    extract_features_and_targets,
    train_logistic_regression,
    bce_loss,
    sigmoid,
    compute_confusion_and_metrics,
    knn_predict_query,
    knn_predict
)


def run_part2():
    # -------------------------------------------------------------------------
    # Task 2.1: Split and standardise
    # -------------------------------------------------------------------------
    rows = load_dataset("data/records.csv")
    train_rows, test_rows = make_stratified_split(rows)

    (
        x_tr_raw, y_train,
        x_te_raw, y_test,
        x_train, x_test,
        tr_mean, tr_sd
    ) = extract_features_and_targets(train_rows, test_rows)

    train_count = len(train_rows)
    train_pos = int(np.sum(y_train == 1))
    test_count = len(test_rows)
    test_pos = int(np.sum(y_test == 1))

    results = []
    results.append(("train_rows", f"{train_count}"))
    results.append(("train_positives", f"{train_pos}"))
    results.append(("test_rows", f"{test_count}"))
    results.append(("test_positives", f"{test_pos}"))

    # -------------------------------------------------------------------------
    # Task 2.2: Logistic regression from scratch (eta=0.1, 1000 updates)
    # -------------------------------------------------------------------------
    w_final, b_final, loss_hist = train_logistic_regression(
        x_train, y_train, eta=0.1, epochs=1000
    )

    results.append(("loss_epoch_0", f"{loss_hist[0]:.4f}"))
    results.append(("loss_epoch_1", f"{loss_hist[1]:.4f}"))
    results.append(("loss_epoch_10", f"{loss_hist[10]:.4f}"))
    results.append(("loss_epoch_100", f"{loss_hist[100]:.4f}"))
    results.append(("loss_epoch_1000", f"{loss_hist[1000]:.4f}"))

    for i in range(7):
        results.append((f"weight_X{i+1}", f"{w_final[i]:.4f}"))
    results.append(("bias", f"{b_final:.4f}"))

    z_test = x_test @ w_final + b_final
    p_test = sigmoid(z_test)
    test_bce = bce_loss(y_test, p_test)
    results.append(("test_bce", f"{test_bce:.4f}"))

    # -------------------------------------------------------------------------
    # Task 2.3: Learning-rate effect (eta in {0.01, 0.1, 1.0}, 200 updates)
    # -------------------------------------------------------------------------
    _, _, hist_001 = train_logistic_regression(x_train, y_train, eta=0.01, epochs=200)
    _, _, hist_01 = train_logistic_regression(x_train, y_train, eta=0.1, epochs=200)
    _, _, hist_1 = train_logistic_regression(x_train, y_train, eta=1.0, epochs=200)

    results.append(("loss_epoch_200_eta_0.01", f"{hist_001[200]:.4f}"))
    results.append(("loss_epoch_200_eta_0.1", f"{hist_01[200]:.4f}"))
    results.append(("loss_epoch_200_eta_1.0", f"{hist_1[200]:.4f}"))

    # -------------------------------------------------------------------------
    # Task 2.4: Odds ratios (exp(weight))
    # -------------------------------------------------------------------------
    odds_ratios = np.exp(w_final)
    for i in range(7):
        results.append((f"odds_ratio_X{i+1}", f"{odds_ratios[i]:.4f}"))

    # -------------------------------------------------------------------------
    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    # -------------------------------------------------------------------------
    m_05 = compute_confusion_and_metrics(y_test, p_test, 0.5)
    results.append(("tau_0.5_tp", f"{m_05['tp']}"))
    results.append(("tau_0.5_fp", f"{m_05['fp']}"))
    results.append(("tau_0.5_tn", f"{m_05['tn']}"))
    results.append(("tau_0.5_fn", f"{m_05['fn']}"))
    results.append(("tau_0.5_accuracy", f"{m_05['accuracy']:.4f}"))
    results.append(("tau_0.5_precision", f"{m_05['precision']:.4f}"))
    results.append(("tau_0.5_recall", f"{m_05['recall']:.4f}"))
    results.append(("tau_0.5_f1", f"{m_05['f1']:.4f}"))
    results.append(("tau_0.5_specificity", f"{m_05['specificity']:.4f}"))

    m_03 = compute_confusion_and_metrics(y_test, p_test, 0.3)
    results.append(("tau_0.3_tp", f"{m_03['tp']}"))
    results.append(("tau_0.3_fp", f"{m_03['fp']}"))
    results.append(("tau_0.3_tn", f"{m_03['tn']}"))
    results.append(("tau_0.3_fn", f"{m_03['fn']}"))
    results.append(("tau_0.3_accuracy", f"{m_03['accuracy']:.4f}"))
    results.append(("tau_0.3_precision", f"{m_03['precision']:.4f}"))
    results.append(("tau_0.3_recall", f"{m_03['recall']:.4f}"))
    results.append(("tau_0.3_f1", f"{m_03['f1']:.4f}"))
    results.append(("tau_0.3_specificity", f"{m_03['specificity']:.4f}"))

    # Blind accuracy (predict 0 for every record)
    blind_acc = float(np.sum(y_test == 0)) / len(y_test)
    results.append(("blind_accuracy", f"{blind_acc:.4f}"))

    # -------------------------------------------------------------------------
    # Task 2.6: Threshold sweep (tau = 0.1, 0.2, ..., 0.9)
    # -------------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    sweep_path = os.path.join("outputs", "part2_thresholds.csv")
    with open(sweep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        for k in range(1, 10):
            tau = round(k * 0.1, 1)
            m = compute_confusion_and_metrics(y_test, p_test, tau)
            tpr = m["recall"]
            fpr = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) > 0 else 0.0
            writer.writerow([
                f"{tau:.1f}",
                m["tp"],
                m["fp"],
                m["tn"],
                m["fn"],
                f"{tpr:.4f}",
                f"{fpr:.4f}"
            ])

    # -------------------------------------------------------------------------
    # Task 2.7: K-nearest neighbours with 5-fold cross-validation
    # -------------------------------------------------------------------------
    k_candidates = list(range(1, 23, 2))  # 1, 3, 5, ..., 21
    folds = np.array([i % 5 for i in range(train_count)])
    knn_cv_path = os.path.join("outputs", "part2_knn_cv.csv")

    cv_scores = {}
    with open(knn_cv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])

        for k in k_candidates:
            fold_f1s = []
            for fold_id in range(5):
                val_mask = (folds == fold_id)
                ref_mask = ~val_mask

                x_ref = x_train[ref_mask]
                y_ref = y_train[ref_mask]
                x_val = x_train[val_mask]
                y_val = y_train[val_mask]

                val_preds, _ = knn_predict(x_val, x_ref, y_ref, k)
                tp = np.sum((val_preds == 1) & (y_val == 1))
                fp = np.sum((val_preds == 1) & (y_val == 0))
                fn = np.sum((val_preds == 0) & (y_val == 1))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                fold_f1s.append(f1)

            mean_f1 = float(np.mean(fold_f1s))
            cv_scores[k] = mean_f1
            writer.writerow([k, f"{mean_f1:.4f}"])

    # Highest mean F1; ties go to smaller K
    knn_best_k = max(k_candidates, key=lambda k: (cv_scores[k], -k))
    results.append(("knn_best_k", f"{knn_best_k}"))

    # Refit with all 560 training rows as reference
    test_knn_preds, _ = knn_predict(x_test, x_train, y_train, knn_best_k)
    k_tp = int(np.sum((test_knn_preds == 1) & (y_test == 1)))
    k_fp = int(np.sum((test_knn_preds == 1) & (y_test == 0)))
    k_tn = int(np.sum((test_knn_preds == 0) & (y_test == 0)))
    k_fn = int(np.sum((test_knn_preds == 0) & (y_test == 1)))
    k_acc = (k_tp + k_tn) / len(y_test)
    k_prec = k_tp / (k_tp + k_fp) if (k_tp + k_fp) > 0 else 0.0
    k_rec = k_tp / (k_tp + k_fn) if (k_tp + k_fn) > 0 else 0.0
    k_f1 = (2.0 * k_prec * k_rec / (k_prec + k_rec)) if (k_prec + k_rec) > 0 else 0.0

    results.append(("knn_tp", f"{k_tp}"))
    results.append(("knn_fp", f"{k_fp}"))
    results.append(("knn_tn", f"{k_tn}"))
    results.append(("knn_fn", f"{k_fn}"))
    results.append(("knn_accuracy", f"{k_acc:.4f}"))
    results.append(("knn_precision", f"{k_prec:.4f}"))
    results.append(("knn_recall", f"{k_rec:.4f}"))
    results.append(("knn_f1", f"{k_f1:.4f}"))

    # -------------------------------------------------------------------------
    # Task 2.8: Choose the alert threshold with money
    # -------------------------------------------------------------------------
    z_train = x_train @ w_final + b_final
    p_train = sigmoid(z_train)

    best_train_cost = float("inf")
    best_tau = None

    for step in range(1, 20):
        tau = step / 20.0
        preds_tr = (p_train >= tau).astype(int)
        fp_tr = int(np.sum((preds_tr == 1) & (y_train == 0)))
        fn_tr = int(np.sum((preds_tr == 0) & (y_train == 1)))
        cost_tr = 50 * fp_tr + 400 * fn_tr
        if cost_tr < best_train_cost:
            best_train_cost = cost_tr
            best_tau = tau

    results.append(("cost_tau", f"{best_tau:.4f}"))
    results.append(("cost_train_at_tau", f"{best_train_cost}"))

    # Evaluate on test rows
    preds_te_tau = (p_test >= best_tau).astype(int)
    fp_te_tau = int(np.sum((preds_te_tau == 1) & (y_test == 0)))
    fn_te_tau = int(np.sum((preds_te_tau == 0) & (y_test == 1)))
    cost_test_at_tau = 50 * fp_te_tau + 400 * fn_te_tau

    preds_te_05 = (p_test >= 0.5).astype(int)
    fp_te_05 = int(np.sum((preds_te_05 == 1) & (y_test == 0)))
    fn_te_05 = int(np.sum((preds_te_05 == 0) & (y_test == 1)))
    cost_test_at_05 = 50 * fp_te_05 + 400 * fn_te_05

    cost_test_flag_nobody = 400 * int(np.sum(y_test == 1))

    results.append(("cost_test_at_tau", f"{cost_test_at_tau}"))
    results.append(("cost_test_at_0.5", f"{cost_test_at_05}"))
    results.append(("cost_test_flag_nobody", f"{cost_test_flag_nobody}"))

    # Write part2_results.txt
    res_path = os.path.join("outputs", "part2_results.txt")
    with open(res_path, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 2 complete. Results written to {res_path}, {sweep_path}, and {knn_cv_path}.")


if __name__ == "__main__":
    run_part2()
