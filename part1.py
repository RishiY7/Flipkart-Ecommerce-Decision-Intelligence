"""Part 1: Estimation, inference, and a regression baseline.

Outputs written to outputs/part1_results.txt.
"""
import csv
import math
import os
import numpy as np


def run_part1():
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    records_path = os.path.join("data", "records.csv")
    with open(records_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    # Target value: order_value_inr_hundreds
    value = np.array([float(r["order_value_inr_hundreds"]) for r in rows])
    # Predictors
    x1 = np.array([float(r["customer_tenure_months"]) for r in rows])
    x2 = np.array([float(r["complaint_count_90d"]) for r in rows])
    x3 = np.array([float(r["discount_order_share"]) for r in rows])
    x4 = np.array([float(r["avg_order_value_inr_hundreds"]) for r in rows])
    x5 = np.array([float(r["loyalty_score"]) for r in rows])
    x6 = np.array([float(r["cod_flag"]) for r in rows])
    x7 = np.array([float(r["plus_member_flag"]) for r in rows])

    results = []

    # -------------------------------------------------------------------------
    # Task 1.1: Estimate with an interval
    # -------------------------------------------------------------------------
    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    results.append(("n", f"{n}"))
    results.append(("mean_value", f"{mean_value:.4f}"))
    results.append(("sd_value", f"{sd_value:.4f}"))
    results.append(("se_value", f"{se_value:.4f}"))
    results.append(("ci95_low", f"{ci95_low:.4f}"))
    results.append(("ci95_high", f"{ci95_high:.4f}"))

    # -------------------------------------------------------------------------
    # Task 1.2: Test a planning claim (H0: mu = 8.0 vs H1: mu != 8.0)
    # -------------------------------------------------------------------------
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    results.append(("z_stat", f"{z_stat:.4f}"))
    results.append(("reject_h0", reject_h0))

    # -------------------------------------------------------------------------
    # Task 1.3: Size a future sample (E = 0.5)
    # -------------------------------------------------------------------------
    e_margin = 0.5
    n_required = math.ceil((1.96 * sd_value / e_margin) ** 2)

    results.append(("n_required", f"{n_required}"))

    # -------------------------------------------------------------------------
    # Task 1.4: Simple regression by closed form (value on X4)
    # -------------------------------------------------------------------------
    mean_x4 = float(np.mean(x4))
    cov_x4_val = float(np.sum((x4 - mean_x4) * (value - mean_value)) / (n - 1))
    var_x4 = float(np.sum((x4 - mean_x4) ** 2) / (n - 1))

    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_value - reg_slope * mean_x4
    fitted_val = reg_intercept + reg_slope * x4

    tss = float(np.sum((value - mean_value) ** 2))
    ess = float(np.sum((fitted_val - mean_value) ** 2))
    rss = float(np.sum((value - fitted_val) ** 2))

    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < (1e-6 * tss) else "no"
    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(value - fitted_val)))

    results.append(("reg_slope", f"{reg_slope:.4f}"))
    results.append(("reg_intercept", f"{reg_intercept:.4f}"))
    results.append(("tss", f"{tss:.4f}"))
    results.append(("ess", f"{ess:.4f}"))
    results.append(("rss", f"{rss:.4f}"))
    results.append(("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss))
    results.append(("r2", f"{r2:.4f}"))
    results.append(("rmse", f"{rmse:.4f}"))
    results.append(("mae", f"{mae:.4f}"))

    # -------------------------------------------------------------------------
    # Task 1.5: Fit the same line by gradient descent
    # -------------------------------------------------------------------------
    sd_x4 = math.sqrt(var_x4)
    z_x4 = (x4 - mean_x4) / sd_x4

    m = 0.0
    c = 0.0
    eta = 0.1
    gd_costs = {}

    for step in range(501):
        y_hat = m * z_x4 + c
        e = value - y_hat
        cost_j = float(np.mean(e ** 2))

        if step in [0, 1, 10, 100, 500]:
            gd_costs[step] = cost_j

        if step == 500:
            break

        # Gradients: dJ/dm = -(2/n)*sum(z*e), dJ/dc = -(2/n)*sum(e)
        dj_dm = -2.0 * float(np.mean(z_x4 * e))
        dj_dc = -2.0 * float(np.mean(e))

        m -= eta * dj_dm
        c -= eta * dj_dc

    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - reg_slope)

    results.append(("gd_J0", f"{gd_costs[0]:.4f}"))
    results.append(("gd_J1", f"{gd_costs[1]:.4f}"))
    results.append(("gd_J10", f"{gd_costs[10]:.4f}"))
    results.append(("gd_J100", f"{gd_costs[100]:.4f}"))
    results.append(("gd_J500", f"{gd_costs[500]:.4f}"))
    results.append(("gd_slope", f"{gd_slope:.4f}"))
    results.append(("gd_intercept", f"{gd_intercept:.4f}"))
    results.append(("gd_slope_gap", f"{gd_slope_gap:.4f}"))

    # -------------------------------------------------------------------------
    # Task 1.6: Check for a funnel, then apply the log remedy
    # -------------------------------------------------------------------------
    raw_residuals = value - fitted_val
    order_raw = np.argsort(fitted_val, kind="stable")
    lower_raw_idx = order_raw[:400]
    upper_raw_idx = order_raw[400:]

    resid_sd_lower_raw = float(np.std(raw_residuals[lower_raw_idx], ddof=1))
    resid_sd_upper_raw = float(np.std(raw_residuals[upper_raw_idx], ddof=1))
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log regression: ln(value) on ln(X4)
    ln_val = np.log(value)
    ln_x4 = np.log(x4)
    mean_ln_x4 = float(np.mean(ln_x4))
    mean_ln_val = float(np.mean(ln_val))

    cov_log = float(np.sum((ln_x4 - mean_ln_x4) * (ln_val - mean_ln_val)) / (n - 1))
    var_log_x4 = float(np.sum((ln_x4 - mean_ln_x4) ** 2) / (n - 1))

    loglog_slope = cov_log / var_log_x4
    loglog_intercept = mean_ln_val - loglog_slope * mean_ln_x4
    fitted_log = loglog_intercept + loglog_slope * ln_x4

    tss_log = float(np.sum((ln_val - mean_ln_val) ** 2))
    rss_log = float(np.sum((ln_val - fitted_log) ** 2))
    loglog_r2 = 1.0 - (rss_log / tss_log)
    loglog_rmse = math.sqrt(rss_log / n)

    log_residuals = ln_val - fitted_log
    order_log = np.argsort(fitted_log, kind="stable")
    lower_log_idx = order_log[:400]
    upper_log_idx = order_log[400:]

    resid_sd_lower_log = float(np.std(log_residuals[lower_log_idx], ddof=1))
    resid_sd_upper_log = float(np.std(log_residuals[upper_log_idx], ddof=1))
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    results.append(("resid_sd_lower_raw", f"{resid_sd_lower_raw:.4f}"))
    results.append(("resid_sd_upper_raw", f"{resid_sd_upper_raw:.4f}"))
    results.append(("resid_sd_ratio_raw", f"{resid_sd_ratio_raw:.4f}"))
    results.append(("loglog_intercept", f"{loglog_intercept:.4f}"))
    results.append(("loglog_slope", f"{loglog_slope:.4f}"))
    results.append(("loglog_r2", f"{loglog_r2:.4f}"))
    results.append(("loglog_rmse", f"{loglog_rmse:.4f}"))
    results.append(("resid_sd_lower_log", f"{resid_sd_lower_log:.4f}"))
    results.append(("resid_sd_upper_log", f"{resid_sd_upper_log:.4f}"))
    results.append(("resid_sd_ratio_log", f"{resid_sd_ratio_log:.4f}"))

    # -------------------------------------------------------------------------
    # Task 1.7: Multiple regression by the normal equation
    # -------------------------------------------------------------------------
    # Order: constant 1, ln(X4), X1, X2, X3, X5, X6, X7
    x_mat = np.column_stack([
        np.ones(n),
        ln_x4,
        x1,
        x2,
        x3,
        x5,
        x6,
        x7
    ])
    y_vec = ln_val

    # Normal equation: (X^T X) beta = X^T Y
    xtx = x_mat.T @ x_mat
    xty = x_mat.T @ y_vec
    beta = np.linalg.solve(xtx, xty)

    for i in range(8):
        results.append((f"mr_beta_{i}", f"{beta[i]:.4f}"))

    fitted_mr = x_mat @ beta
    rss_mr = float(np.sum((y_vec - fitted_mr) ** 2))
    mr_r2 = 1.0 - (rss_mr / tss_log)

    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)

    p_log = 1
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - p_log - 1)

    results.append(("mr_r2", f"{mr_r2:.4f}"))
    results.append(("mr_adj_r2", f"{mr_adj_r2:.4f}"))
    results.append(("loglog_adj_r2", f"{loglog_adj_r2:.4f}"))

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "part1_results.txt")
    with open(out_file, "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 1 complete. Results written to {out_file}.")


if __name__ == "__main__":
    run_part1()
