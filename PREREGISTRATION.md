# DACHP-v2 confirmatory experiment — pre-registration

Written 28 September 2026 (IST), **before** any run of this experiment.
Configuration: `config_v2.yaml` (committed together with this file).

## Why this experiment exists
The first held-out evaluation (splits 0–4, `config.yaml`) showed DACHP significantly
better than all graph-based / classical methods, but about 0.01 below SupCon,
Proxy-Anchor and CE-MLP (same encoder) in test 5-NN accuracy. Its ablation and
sensitivity analysis suggested two changes: removing the auxiliary loss (λ = 0) and a
larger margin α. **These changes were therefore chosen after seeing test results of
splits 0–4.** To avoid selecting on test data, DACHP-v2 is evaluated here only on new
splits that no method has been evaluated on.

## Fixed in advance
* **Splits:** seeds 5, 6, 7, 8, 9 (stratified 60/20/20; grouped by mouse for Mice Protein). Never used before.
* **Datasets:** the same 11 as the main study. None may be added or removed.
* **DACHP-v2 search space (6 configurations):** λ ∈ {0, 0.3} × α ∈ {1.2, 1.5, 2.0};
  τ₀ = 0.2 when λ > 0; everything else identical to DACHP (Table 4 of the manuscript).
* **Equal budget for baselines (6 configurations each):** SupCon τ, Triplet margin,
  Proxy-Anchor δ, ArcFace m, CosFace m, CE label smoothing (3 values each, as before)
  × learning rate ∈ {0.005, 0.002}. Original DACHP keeps its 3-value α grid (reference).
  Supervised UMAP keeps its 24-configuration grid.
* **Selection:** for every method, the configuration with the highest validation 5-NN
  accuracy on split 5 (ties: validation Silhouette), then fixed for splits 5–9.
* **Methods evaluated:** DACHP, DACHP-v2, SupCon, Triplet, Proxy-Anchor, ArcFace,
  CosFace, CE-MLP, Supervised UMAP (default / tuned / tuned-16-D), LDA, PCA.
  (t-SNE, UMAP, PaCMAP, TriMap are omitted: they ranked last in the main study.)
* **Primary endpoint:** mean test 5-NN accuracy per dataset.
  **Secondary:** test linear-probe accuracy, test native Silhouette.

## Decision rule (stated before the run)
DACHP-v2 will be described as **superior** to a method only if, across the 11 datasets,
the two-sided Wilcoxon signed-rank test on the primary endpoint has Holm-adjusted
p < 0.05 **and** the mean difference is positive. If the adjusted p ≥ 0.05 the result is
reported as **no significant difference (parity)**; if significant and negative, as
**inferior**. The same wording rule applies to the secondary endpoints, reported separately.
All outcomes will be reported, whatever they are, together with this document.
