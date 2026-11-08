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

- **R3.1 — Elbow Analysis & Deterministic WCSS Drops**:
  - The successive WCSS drops across $K \in \{2, 3, 4, 5, 6\}$ are:
    - $\text{WCSS}(2) - \text{WCSS}(3) = 2262.6753 - 1602.8471 = 659.8282$
    - $\text{WCSS}(3) - \text{WCSS}(4) = 1602.8471 - 1494.7732 = 108.0739$
    - $\text{WCSS}(4) - \text{WCSS}(5) = 1494.7732 - 1323.7157 = 171.0575$
    - $\text{WCSS}(5) - \text{WCSS}(6) = 1323.7157 - 1278.1691 = 45.5466$
  - The elbow criterion selects `elbow_k = 3` because the drop from $K=3$ to $K=4$ (`108.0739`) is strictly less than half the preceding reduction ($0.5 \times 659.8282 = 329.9141$). The non-monotonic drop rebound from $K=4$ to $K=5$ occurs because the pinned, uniform index initializations provide a fully deterministic and reproducible benchmark rather than guarantees of the global minimum at each intermediate $K$.

- **R3.2 — Customer Segmentation & Strategic Actions**:
  - **Segment A (Size: 322) — "Premium Loyal"**:
    - *Profile:* High tenure (47.50 months), low friction (1.95 complaints), low discount dependence (22.50%), highest loyalty score (62.16), and minimal churn (`outcome_rate = 0.0186`).
    - *Action:* Fast-track to Flipkart Plus / VIP perks, offer exclusive early access to annual sales, and retain brand affinity without diluting margin via aggressive discounting.
  - **Segment B (Size: 255) — "High-AOV / Service-Sensitive Spenders"**:
    - *Profile:* Highest past and present spending (`mean_X4 = 28.72` hundred INR, `mean_value = 17.12`), moderate tenure (28.61 months), highest friction (`mean_X2 = 6.07` complaints), and moderate churn (`outcome_rate = 0.2353`).
    - *Action:* Route complaints to dedicated white-glove customer support desks and prioritize expedited logistics to safeguard high-margin revenue streams.
  - **Segment C (Size: 223) — "Bargain-Focused Deal Chasers"**:
    - *Profile:* Heavily promotional (`mean_X3 = 0.7118` discount share), short tenure (14.39 months), depressed loyalty scores (38.52), and highest churn (`outcome_rate = 0.4664`).
    - *Action:* Deploy targeted flash promotions with minimum-order thresholds and app push reminders to stem attrition.
  - *Highest churn segment:* **Segment C** records the highest churn outcome rate at `0.4664` (46.64%).

- **R3.3 — Hierarchical Clustering & Linkage Disagreement**:
  - *Single linkage clusters at cut (4 clusters):* `{E2, E3, E6}`, `{E1}`, `{E4}`, `{E5}`.
  - *Complete linkage clusters at cut (2 clusters):* `{E1, E4, E5}`, `{E2, E3, E6}`.
  - *Explanation of disagreement:* Single linkage assesses cluster proximity using the minimum distance between any single pair of points, which creates an elongated chaining effect that delays the consolidation of E1, E4, and E5. In contrast, complete linkage uses the maximum pairwise distance, penalizing loose chains and enforcing compact, globally coherent groupings that cleanly bifurcate the cohorts into two distinct macro-clusters (`{E1, E4, E5}` and `{E2, E3, E6}`).

- **R3.4 — PCA Loadings, Dimensionality Sizing & Segment Projection**:
  - The standardized feature matrix has a mean vector of all zeros, which was centred by construction.
  - The first principal component (52.27% of variance) is driven positively by customer tenure ($X_1$, loading `0.5317`) and loyalty score ($X_5$, loading `0.4418`), and strongly negatively by discount order share ($X_3$, loading `-0.5187`) and complaint counts ($X_2$, loading `-0.4100`).
  - Sizing for at least 80% cumulative variance requires 3 components (`components_for_80pct = 3`), capturing 86.16% of total variance.
  - Projecting the segments onto PC1 shows clear structural alignment: `mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, and `mean_pc1_segment_C = -1.6709`. PC1 functions as a comprehensive customer relationship health score, separating loyal advocates from dissatisfied spenders and discount chasers.

## Part 4 findings

- **R4.1 — Split Purity, Overfitting & Tree Depth Controls**:
  - Split $S_4$ ($X_1 \le 24$, customer tenure $\le 24$ months) is the purest split, yielding the highest information gain (`info_gain = 0.1252`) and highest Gini gain (`gini_gain = 0.0606`) by partitioning the 560 training records into a high-churn subset of 188 newer accounts (86 positives, 45.74% churn rate) and 372 mature accounts (33 positives, 8.87% churn rate).
  - Growing a decision tree until every leaf is pure causes the model to memorize random training noise, outliers, and sample-specific idiosyncrasies, creating excessively fragmented decision boundaries that fail to generalize to unseen test distributions.
  - Setting a maximum tree depth constraint (`max_depth`) or requiring a minimum number of samples per leaf node (`min_samples_leaf`) serves as an effective stopping rule to curtail recursive partitioning before variance inflates.

- **R4.2 — Voting Ensembles & Model Selection**:
  - We recommend deploying the single **logistic regression** model (`tau = 0.3`).
  - Combining models did not improve performance: logistic regression achieved the top F1 score (`f1 = 0.5714`, `recall = 0.7451`), surpassing both hard voting (`f1 = 0.5344`, `recall = 0.6863`) and soft voting (`f1 = 0.5507`, `recall = 0.7451`), as well as the decision stump (`f1 = 0.5532`, `recall = 0.7647`) and KNN (`f1 = 0.3696`, `recall = 0.3333`).
  - Ensembling weakened performance because the lower predictive quality and uncalibrated probabilities of the KNN model degraded the smooth, well-calibrated decision boundary established by logistic regression.

- **R4.3 — AdaBoost Sample Reweighting Dynamics**:
  - The total weight of the two misclassified points (Points 3 and 7) after Round 1 normalisation is `0.2500 + 0.2500 = 0.5000` (exactly half of the total ensemble weight distribution).
  - Round 2 assigns greater trust to $h_2$ than Round 1 assigned to $h_1$ ($\alpha_2 = 0.8047$ vs $\alpha_1 = 0.5493$) because $h_2$ successfully classifies both of the heavily reweighted prior error cases (Points 3 and 7) while maintaining a lower weighted error rate ($\epsilon_2 = 0.1667$ vs $\epsilon_1 = 0.2500$).

- **R4.4 — Neural Architecture Sizing, Activation Choices & Residual Highways**:
  - **Model A (Option 2 — CNN) Layer Sizing Table**:
    | Layer | Output Shape ($H \times W \times D$) | Parameter Count |
    |---|---|---|
    | `conv1` ($5 \times 5$, 6 filters, pad 2, stride 1) + ReLU | $28 \times 28 \times 6$ | 156 |
    | `pool1` ($2 \times 2$ max-pool, stride 2) | $14 \times 14 \times 6$ | 0 |
    | `conv2` ($5 \times 5$, 16 filters, pad 0, stride 1) + ReLU | $10 \times 10 \times 16$ | 2,416 |
    | `pool2` ($2 \times 2$ max-pool, stride 2) | $5 \times 5 \times 16$ | 0 |
    | `flatten` | 400 | 0 |
    | `dense1` (32 neurons) + ReLU | 32 | 12,832 |
    | `dense2` (10 neurons, output) | 10 | 330 |
    | **Total CNN Parameters** | — | **15,734** |
  - **Model B (RNN Shapes & Sequence Invariance)**:
    - Weight matrix shapes: $W_{xh} \in \mathbb{R}^{128 \times 64}$, $W_{hh} \in \mathbb{R}^{128 \times 128}$, $W_{hy} \in \mathbb{R}^{16 \times 128}$, with bias vectors $b_h \in \mathbb{R}^{128}$ and $b_y \in \mathbb{R}^{16}$.
    - Parameter count invariance: `rnn_total = 26768` remains strictly constant regardless of input sequence length because the recurrent cell shares the exact same projection matrices and biases across every unrolled time step.
  - **Activation & Loss Specifications (10 Classes)**:
    - Hidden layers: ReLU ($\max(0, x)$) for fast non-saturating gradients.
    - Output layer: Softmax ($\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$) paired with Categorical Cross-Entropy loss ($-\sum_{k=1}^{10} y_k \ln \hat{y}_k$).
  - **Vanishing Gradients & Skip Connections**:
    - In deep plain networks, repeated backpropagation multiplications by weight matrices and fractional activation derivatives cause gradient magnitudes to diminish exponentially as they travel toward the input, leaving initial layers virtually untrained.
    - A residual skip connection ($x + \mathcal{F}(x)$) introduces an uninterrupted identity pathway with gradient derivative $1 + \mathcal{F}'(x)$, guaranteeing that error signals flow directly to early layers without attenuation.

## Model-choice memo

**To:** Head of Customer Retention, Flipkart  
**From:** Decision Intelligence Analytics Team  
**Subject:** Production Churn Alert Model Recommendation  

We recommend deploying the **logistic regression classifier** for live churn alerts. In test evaluations (Task 4.2), logistic regression attained the highest F1 score (`0.5714`) and tied for highest recall (`0.7451`, catching 38 of 51 churners) with precision of `0.4634`. It decisively outperformed K-nearest neighbours (`f1 = 0.3696`), the decision stump (`f1 = 0.5532`), hard voting (`f1 = 0.5344`), and soft voting (`f1 = 0.5507`). Ensembling degraded performance because KNN's weaker predictions compromised logistic regression's calibrated probability surface.

For production, we recommend the cost-optimal alert threshold of **$\tau = 0.15$**, chosen by minimizing business expense on training data. Given Flipkart's asymmetric cost reality—INR 50 per false retention coupon versus INR 400 per missed churning customer—operating at $\tau = 0.15$ incurs a test cost of **INR 6,050**. In comparison, the conventional $\tau = 0.5$ threshold incurs **INR 11,800**, while flagging nobody costs **INR 20,400**. Operating at $\tau = 0.15$ reduces costs by 48.7% versus $\tau = 0.5$ and saves INR 14,350 over inaction on this 240-customer test cohort.

**Operational Risk:** Our evidence relies on a single synthetic 70/30 split with only 51 test churners; real customer variance may alter coupon spending.

**Unstructured Model Priority:** Neither Model A (image sorter) nor Model B (review topic RNN) is worth building before the structured churn model is deployed, as churn intervention directly defends revenue while visual and text processing are secondary enhancements.

## Limitations

1. **Synthetic Data Generation**: All records were synthetically generated through parametric mathematical formulas; they lack real-world e-commerce complexities such as multi-channel browsing touchpoints, festive Diwali seasonality spikes, regional delivery logistics bottlenecks, or fraudulent account abuse.
2. **Single Static Train-Test Partition**: The analytical pipeline relies on a single deterministic 70/30 stratified split (560 training / 240 test records). Without out-of-time temporal validation or repeated nested cross-validation, the reported performance metrics carry unavoidable sampling variance.
3. **Small Positive Holdout Sample (51 Records)**: The holdout test set contains only 51 churning customers (`test_positives = 51`), meaning each individual customer prediction alters recall and precision by approximately 2.0 percentage points, making precision-recall estimates sensitive to small boundary perturbations.
4. **Absence of Regularization and Dynamic Feature Engineering**: Predictors were used strictly in linear/standardized form without interaction terms (e.g., tenure $\times$ discount share), polynomial transformations, or L1/L2 shrinkage penalties (Ridge/Lasso), which could further suppress variance in high-dimensional production environments.
