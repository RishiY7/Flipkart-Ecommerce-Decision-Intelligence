"""Part 4: Model-choice and architecture memo.

Produces:
- outputs/part4_results.txt
- outputs/part4_splits.csv
- outputs/part4_voting.csv
- outputs/part4_adaboost.csv
- outputs/part4_architecture.csv
"""
import csv
import math
import os
import numpy as np
from shared_utils import (
    load_dataset,
    make_stratified_split,
    extract_features_and_targets,
    train_logistic_regression,
    sigmoid,
    knn_predict
)


def run_part4():
    # -------------------------------------------------------------------------
    # Load dataset and prepare splits from Task 2.1
    # -------------------------------------------------------------------------
    rows = load_dataset("data/records.csv")
    train_rows, test_rows = make_stratified_split(rows)

    (
        x_tr_raw, y_tr,
        x_te_raw, y_te,
        x_train, x_test,
        tr_mean, tr_sd
    ) = extract_features_and_targets(train_rows, test_rows)

    n_tr = len(y_tr)
    pos_tr = int(np.sum(y_tr == 1))

    results = []
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------------------------------------------------
    # Task 4.1: Which split is purest?
    # -------------------------------------------------------------------------
    def node_entropy(positives, total):
        if total == 0 or positives == 0 or positives == total:
            return 0.0
        p1 = positives / total
        p0 = 1.0 - p1
        return -(p1 * math.log2(p1) + p0 * math.log2(p0))

    def node_gini(positives, total):
        if total == 0:
            return 0.0
        p1 = positives / total
        p0 = 1.0 - p1
        return 1.0 - (p1 ** 2 + p0 ** 2)

    root_entropy = node_entropy(pos_tr, n_tr)
    root_gini = node_gini(pos_tr, n_tr)

    results.append(("root_entropy", f"{root_entropy:.4f}"))
    results.append(("root_gini", f"{root_gini:.4f}"))

    # Splits on unstandardised training rows:
    # S1: X6 == 1 (cod_flag, index 5)
    # S2: X7 == 1 (plus_member_flag, index 6)
    # S3: X3 > 0.5 (discount_order_share, index 2)
    # S4: X1 <= 24 (customer_tenure_months, index 0)
    split_conditions = {
        "S1": x_tr_raw[:, 5] == 1,
        "S2": x_tr_raw[:, 6] == 1,
        "S3": x_tr_raw[:, 2] > 0.5,
        "S4": x_tr_raw[:, 0] <= 24
    }

    splits_csv_path = os.path.join("outputs", "part4_splits.csv")
    split_records = []

    with open(splits_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])

        for s_name in ["S1", "S2", "S3", "S4"]:
            cond = split_conditions[s_name]
            n_yes = int(np.sum(cond))
            pos_yes = int(np.sum(y_tr[cond] == 1))
            n_no = n_tr - n_yes
            pos_no = int(np.sum(y_tr[~cond] == 1))

            ent_yes = node_entropy(pos_yes, n_yes)
            ent_no = node_entropy(pos_no, n_no)
            weighted_ent = (n_yes / n_tr) * ent_yes + (n_no / n_tr) * ent_no
            info_gain = root_entropy - weighted_ent

            gini_yes = node_gini(pos_yes, n_yes)
            gini_no = node_gini(pos_no, n_no)
            weighted_gini = (n_yes / n_tr) * gini_yes + (n_no / n_tr) * gini_no
            gini_gain = root_gini - weighted_gini

            writer.writerow([
                s_name,
                n_yes,
                pos_yes,
                n_no,
                pos_no,
                f"{info_gain:.4f}",
                f"{gini_gain:.4f}"
            ])
            split_records.append((s_name, info_gain, gini_gain, cond))

    best_split_info = max(split_records, key=lambda x: x[1])[0]
    best_split_gini = max(split_records, key=lambda x: x[2])[0]

    results.append(("best_split_by_info_gain", best_split_info))
    results.append(("best_split_by_gini_gain", best_split_gini))

    # -------------------------------------------------------------------------
    # Task 4.2: Stump and Voting Ensembles
    # -------------------------------------------------------------------------
    # Stump is based on best split by information gain
    best_cond_tr = split_conditions[best_split_info]
    stump_rate_yes = float(np.sum(y_tr[best_cond_tr] == 1) / np.sum(best_cond_tr))
    stump_rate_no = float(np.sum(y_tr[~best_cond_tr] == 1) / np.sum(~best_cond_tr))

    results.append(("stump_rate_yes", f"{stump_rate_yes:.4f}"))
    results.append(("stump_rate_no", f"{stump_rate_no:.4f}"))

    # 1. Logistic regression (Task 2.2 model)
    w_lr, b_lr, _ = train_logistic_regression(x_train, y_tr, eta=0.1, epochs=1000)
    p_lr_te = sigmoid(x_test @ w_lr + b_lr)
    pred_lr = (p_lr_te >= 0.3).astype(int)

    # 2. KNN (best K = 5)
    pred_knn, p_knn_te = knn_predict(x_test, x_train, y_tr, k=5)

    # 3. Stump applied to test rows
    if best_split_info == "S4":
        cond_te = x_te_raw[:, 0] <= 24
    elif best_split_info == "S3":
        cond_te = x_te_raw[:, 2] > 0.5
    elif best_split_info == "S1":
        cond_te = x_te_raw[:, 5] == 1
    else:
        cond_te = x_te_raw[:, 6] == 1

    p_stump_te = np.where(cond_te, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump_te >= 0.3).astype(int)

    # 4. Hard vote: at least 2 of 3 predict positive
    pred_hard = ((pred_lr + pred_knn + pred_stump) >= 2).astype(int)

    # 5. Soft vote: mean of three probabilities >= 0.3
    p_soft = (p_lr_te + p_knn_te + p_stump_te) / 3.0
    pred_soft = (p_soft >= 0.3).astype(int)

    voting_csv_path = os.path.join("outputs", "part4_voting.csv")
    models_to_eval = [
        ("logistic", pred_lr),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft)
    ]

    with open(voting_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])

        for m_name, preds in models_to_eval:
            tp = int(np.sum((preds == 1) & (y_te == 1)))
            fp = int(np.sum((preds == 1) & (y_te == 0)))
            tn = int(np.sum((preds == 0) & (y_te == 0)))
            fn = int(np.sum((preds == 0) & (y_te == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            writer.writerow([
                m_name,
                tp,
                fp,
                tn,
                fn,
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}"
            ])

    # -------------------------------------------------------------------------
    # Task 4.3: Two rounds of AdaBoost by hand
    # -------------------------------------------------------------------------
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    adaboost_csv_path = os.path.join("outputs", "part4_adaboost.csv")
    ada_rows = []

    # Round 1
    w_r1_before = np.full(8, 1.0 / 8.0)
    err1 = float(np.sum(w_r1_before[h1 != y_ada]))
    alpha1 = 0.5 * math.log((1.0 - err1) / err1)
    unnorm_w1 = w_r1_before * np.exp(-alpha1 * y_ada * h1)
    w_r1_after = unnorm_w1 / np.sum(unnorm_w1)

    for pt in range(8):
        ada_rows.append([
            1,
            pt + 1,
            f"{w_r1_before[pt]:.4f}",
            f"{unnorm_w1[pt]:.4f}",
            f"{w_r1_after[pt]:.4f}"
        ])

    results.append(("round1_error", f"{err1:.4f}"))
    results.append(("round1_alpha", f"{alpha1:.4f}"))

    # Round 2
    w_r2_before = w_r1_after.copy()
    err2 = float(np.sum(w_r2_before[h2 != y_ada]))
    alpha2 = 0.5 * math.log((1.0 - err2) / err2)
    unnorm_w2 = w_r2_before * np.exp(-alpha2 * y_ada * h2)
    w_r2_after = unnorm_w2 / np.sum(unnorm_w2)

    for pt in range(8):
        ada_rows.append([
            2,
            pt + 1,
            f"{w_r2_before[pt]:.4f}",
            f"{unnorm_w2[pt]:.4f}",
            f"{w_r2_after[pt]:.4f}"
        ])

    results.append(("round2_error", f"{err2:.4f}"))
    results.append(("round2_alpha", f"{alpha2:.4f}"))

    with open(adaboost_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        writer.writerows(ada_rows)

    ens_preds = np.sign(alpha1 * h1 + alpha2 * h2)
    ensemble_acc = float(np.mean(ens_preds == y_ada))
    results.append(("ensemble_accuracy", f"{ensemble_acc:.4f}"))

    # -------------------------------------------------------------------------
    # Task 4.4: Out-of-bag share
    # -------------------------------------------------------------------------
    oob_frac = (1.0 - (1.0 / 560.0)) ** 560
    results.append(("oob_fraction", f"{oob_frac:.4f}"))

    # -------------------------------------------------------------------------
    # Task 4.5: Size the neural networks
    # -------------------------------------------------------------------------
    # Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_h1 = (784 + 1) * 16
    mlp_h2 = (16 + 1) * 16
    mlp_out = (16 + 1) * 10
    mlp_total = mlp_h1 + mlp_h2 + mlp_out

    # Option 2: CNN
    # conv1: 5x5, 6 filters, in=1 -> (5*5*1 + 1)*6 = 156
    cnn_c1 = (5 * 5 * 1 + 1) * 6
    # conv2: 5x5, 16 filters, in=6 -> (5*5*6 + 1)*16 = 2416
    cnn_c2 = (5 * 5 * 6 + 1) * 16
    # dense1: flattened 5*5*16 = 400 -> 32 neurons -> (400 + 1)*32 = 12832
    cnn_d1 = (400 + 1) * 32
    # dense2: 32 -> 10 neurons -> (32 + 1)*10 = 330
    cnn_d2 = (32 + 1) * 10
    cnn_total = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

    arch_csv_path = os.path.join("outputs", "part4_architecture.csv")
    with open(arch_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        writer.writerow(["mlp_hidden1", mlp_h1])
        writer.writerow(["mlp_hidden2", mlp_h2])
        writer.writerow(["mlp_output", mlp_out])
        writer.writerow(["cnn_conv1", cnn_c1])
        writer.writerow(["cnn_conv2", cnn_c2])
        writer.writerow(["cnn_dense1", cnn_d1])
        writer.writerow(["cnn_dense2", cnn_d2])

    results.append(("mlp_total", f"{mlp_total}"))
    results.append(("cnn_total", f"{cnn_total}"))

    # Scaling comparison: 128x128x3 input
    # FC: (128*128*3 + 1) * 1024
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    # Conv: (3*3*3 + 1) * 64
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    results.append(("fc_layer_params", f"{fc_layer_params}"))
    results.append(("conv_layer_params", f"{conv_layer_params}"))

    # Model B: RNN
    # W_xh: 128 x 64 = 8192
    # W_hh: 128 x 128 = 16384
    # b_h: 128
    # W_hy: 16 x 128 = 2048
    # b_y: 16
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16
    results.append(("rnn_total", f"{rnn_total}"))

    # -------------------------------------------------------------------------
    # Write part4_results.txt
    # -------------------------------------------------------------------------
    res_path = os.path.join("outputs", "part4_results.txt")
    with open(res_path, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 4 complete. Results written to {res_path} and CSVs.")


if __name__ == "__main__":
    run_part4()
