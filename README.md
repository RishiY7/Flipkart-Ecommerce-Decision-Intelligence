Domain: E-commerce

## How to run

From a clean clone of the repository:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

All outputs and CSV files will be generated in the `outputs/` directory.

## Part 1 findings

- **R1.1 — Planning Claim Test & Hypothesis**:
  - The null hypothesis $H_0: \mu = 8.0$ is **rejected** at significance level $\alpha = 0.05$ (two-tailed), as the calculated test statistic `z_stat = 2.8930` strictly exceeds the critical threshold of $1.96$ ($p \approx 0.0038$).
  - In domain terms, a **Type I error** would occur if the planning team mistakenly concludes that true average order value differs from INR 800 (and consequently misallocates category targets, margins, or vendor terms) when the true population mean was in fact INR 800.
  - The gap `mean_value - 8.0` is `0.9382` column units, representing INR 93.82 per order (~11.7% above assumption). In e-commerce operations, a ~12% deviation in average order value materially impacts supply chain budgeting, warehouse unit economics, and quarterly gross merchandise value (GMV) forecasting, making this difference practically significant.

- **R1.2 — Sample Sizing for Category Planning**:
  - Sizing for a margin of error $E = 0.5$ units (INR 50) at 95% confidence requires `n_required = 1293` records.
  - Compared to the 800 available sample rows, the dataset is currently underpowered to guarantee the desired ±INR 50 precision (the current standard error yields a margin of $\pm 1.96 \times 0.3243 = \text{INR } 63.56$). The planning team must collect at least 493 additional transactions before freezing financial tolerances at the ±INR 50 tier.

- **R1.3 — Gradient Descent Convergence**:
  - Gradient descent converged cleanly with `gd_slope_gap = 0.0000`, matching the closed-form OLS slope (`reg_slope = 0.6701`).
  - Because standardized linear regression yields a strictly convex mean squared error surface with a unique global minimum, gradient descent with a learning rate of $\eta = 0.1$ across 500 epochs smoothly and monotonically reaches the exact closed-form solution without overshoot or stagnation.

- **R1.4 — Residual Funnel & Log Remedy**:
  - The raw linear model exhibits severe heteroscedasticity, with `resid_sd_ratio_raw = 3.7525` (`resid_sd_lower_raw = 2.3155` vs `resid_sd_upper_raw = 8.6891`), indicating that residual variance widens dramatically as fitted order value grows (classic funnel effect).
  - In contrast, the log-log specification yields `resid_sd_ratio_log = 1.0396` (`resid_sd_lower_log = 0.5165` vs `resid_sd_upper_log = 0.5369`), effectively eliminating the funnel and restoring homoscedastic residual behavior.

- **R1.5 — Multiple Regression & Predictor Selection**:
  - The predictors with absolute coefficient above 0.05 (excluding intercept `mr_beta_0 = -1.8613`) are:
    - $\ln(X_4)$ (`avg_order_value_inr_hundreds`, `mr_beta_1 = 1.1576`)
    - $X_2$ (`complaint_count_90d`, `mr_beta_3 = 0.0940`)
    - $X_3$ (`discount_order_share`, `mr_beta_4 = 0.1067`)
    - $X_6$ (`cod_flag`, `mr_beta_6 = 0.1518`)
    *(Note: $X_1$ tenure with $\beta_2 = -0.0074$, $X_5$ loyalty with $\beta_5 = 0.0028$, and $X_7$ plus member with $\beta_7 = 0.0172$ did not exceed the 0.05 threshold).*
  - The multiple regression adjusted $R^2$ is `mr_adj_r2 = 0.7403`, compared to `loglog_adj_r2 = 0.6561` for the bivariate log-log model. The ~8.4 percentage point gain in explained variance confirms that the additional predictors earn their place after penalizing for model parameter complexity.

## Part 2 findings

- **R2.1 — Learning Rate Dynamics**:
  - The training losses after 200 epochs ordered from lowest to highest are: $\eta = 1.0$ (`0.3703`) < $\eta = 0.1$ (`0.3776`) < $\eta = 0.01$ (`0.5091`).
  - A learning rate that is too small ($\eta = 0.01$) exhibits sluggish convergence and remains underfitted after 200 steps, whereas a larger learning rate ($\eta = 1.0$) rapidly descends the convex binary cross-entropy surface without numerical instability or overshoot.

- **R2.2 — Key Predictors & Odds Ratio Interpretation**:
  - The three predictors with the largest absolute weights in the model are:
    1. $X_3$ (`discount_order_share`, weight = `0.9013`, odds ratio = `2.4629`)
    2. $X_1$ (`customer_tenure_months`, weight = `-0.5554`, odds ratio = `0.5739`)
    3. $X_2$ (`complaint_count_90d`, weight = `0.5505`, odds ratio = `1.7341`)
  - *Domain interpretation of largest odds ratio:* Holding other factors constant, each one-standard-deviation increase in a customer's discount order share multiplies the odds of churn by approximately 2.46 (a ~146% surge in churn odds), signaling that deal-seeking shoppers have low brand stickiness and depart once promotional incentives subside.

- **R2.3 — Evaluation Metric Realities & Threshold Tuning**:
  - The naive `blind_accuracy` is `0.7875` (achieved by predicting zero churn for every customer), closely matching the model's accuracy at $\tau = 0.5$ (`tau_0.5_accuracy = 0.8042`), which illustrates why overall accuracy is fundamentally misleading on imbalanced datasets where missing rare positive churners carries grave business consequences.
  - Lowering $\tau$ from 0.5 to 0.3 boosts recall from `0.4706` to `0.7451` (catching 38 of 51 churners instead of only 24), while precision shifts from `0.5455` to `0.4634`, an acceptable trade-off given the asymmetric cost structure.

- **R2.4 — Logistic Regression vs K-Nearest Neighbours**:
  - Comparing the logistic regression model at $\tau = 0.3$ against the cross-validated KNN model ($K = 5$):
    - **Logistic ($\tau = 0.3$):** `f1 = 0.5714`, `recall = 0.7451`
    - **KNN ($K = 5$):** `f1 = 0.3696`, `recall = 0.3333`
  - The parametric logistic model establishes a coherent global boundary and dramatically outperforms KNN, which fails to overcome local density imbalances and misses two out of every three churners.

- **R2.5 — Financial Deployment Decision & Leakage Prevention**:
  - We recommend deploying the logistic regression model at the cost-optimal alert threshold of $\tau = 0.15$, which incurs a test cost of `INR 6050`, representing a 48.7% cost reduction compared to $\tau = 0.5$ (`INR 11800`) and a 70.3% savings relative to no intervention (`cost_test_flag_nobody = INR 20400`).
  - The alert threshold was optimized strictly on the training set to prevent data leakage and ensure that reported financial savings reflect genuine generalization to new, unseen customer cohorts.

## Part 3 findings

## Part 4 findings

## Model-choice memo

## Limitations
