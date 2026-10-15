# Changes from the code of the first submission

The experiments of the revised paper were run again from the beginning with the code in this repository. The table lists what changed compared with the notebook used for the first submission, and the reviewer comment (R1 to R4) that each change answers.

| # | Original notebook | Revision | Reviewer |
|---|---|---|---|
| 1 | Embeddings and metrics computed on the training data (no held-out data); scaler fitted on all data | Stratified 60/20/20 train/val/test split per seed; imputer, scaler and every PCA fitted on **train only**; all metrics on **test** | R2-9, R4-7…12 |
| 2 | Primary metric = Silhouette of 2-D PCA projection | Primary = test **5-NN / linear-probe / nearest-class-mean accuracy**; Silhouette in native dim. **and** 2-D; train Silhouette reported separately to show the gap | R2-8, R4-10…12 |
| 3 | Baselines: PCA, t-SNE, UMAP, Sup-UMAP (default) | + SupCon, Triplet (batch-hard), Proxy-Anchor, ArcFace, CosFace, CE-MLP — **same encoder, optimiser, schedule, epochs**; + LDA, PaCMAP, TriMap; Sup-UMAP tuned (n_neighbors × min_dist × PCA pre-reduction), also at 16-D | R1-1, R2-7, R3, R4-1 |
| 4 | Only DACHP tuned | Every tunable method gets one grid tuned on the split-0 **validation** set with the same criterion (val 5-NN acc, tie-break val Silhouette) | R2-7 |
| 5 | "COIL-20" = `fetch_openml(data_id=2)` = **UCI Annealing** (898×38, 5 classes) | True COIL-20 (1440×1024, 20 classes) **and** Annealing under its real name | integrity |
| 6 | τ initialised as `log_tau=2.659` → τ = 14.3 (auxiliary loss ~flat) | τ = 0.07, learnable, clamped to [0.01, 1]; the old value is kept as ablation `legacy_tau` | R3 (τ) |
| 7 | Ablation: 2 datasets, 3 runs, different schedule; "no hard-neg" variant *minimised* negative distances | All datasets × 5 splits × 9 variants, same schedule as main runs; "no hard-neg" now penalises **all** negatives towards the antipode | R1-2, R2-4, R3 |
| 8 | Sensitivity table had no code | α, d, λ, τ on all datasets × 3 splits | R1-2, R3 |
| 9 | Radar & graphical abstract used hard-coded numbers | Removed / regenerated from results only | R2-3, R2-6 |
| 10 | Mice Protein NaN → 0; random split mixes replicates of the same mouse | Median imputation (train); **split grouped by mouse** | R1-6, leakage |
| 11 | Single baseline run, Wilcoxon on 8 means, Cohen's d | 5 seeds for **every** method; 95% CIs; per-dataset paired tests + Holm; Friedman + Nemenyi; Wilcoxon + Holm; rank-biserial r; bootstrap CIs | R1-3, R2-10, R3, R4-4 |
| 12 | — | Label-permutation control; collapse monitor (resultant length, cosine variance, effective rank per epoch); wall-clock time and memory; input-space k-NN purity / hubness / distance concentration vs. gain | R1-3, R1-4, R1-5, R4-2 |

