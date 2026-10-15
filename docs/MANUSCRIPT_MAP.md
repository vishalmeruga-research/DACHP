# Where each table and figure of the paper comes from

`S1` = `results/experiment1/report/`, `S2` = `results/experiment2/report/`. Tables are in `tables/` (CSV and Markdown), figures in `figures/` (PNG and PDF). All numbers quoted in the text are also in `summary.json`.

| Paper | Content | Report file(s) |
|---|---|---|
| Table 3 | Datasets and split sizes | S1 `T2_datasets` |
| Table 5 | Experiment 1, test 5-NN accuracy | S1 `T_main_knn_acc`, `T_avg_rank_knn_acc` |
| Table 6 | Experiment 1, test Silhouette (native dimension) | S1 `T_main_sil_native`, `T_avg_rank_sil_native` |
| Table 7 | Experiment 1, DACHP against each method across datasets | S1 `T_across_tests_knn_acc`, `T_across_tests_sil_native` |
| Table 8 | Input-space measures and gain over Supervised UMAP | S1 `T_input_diagnostics`, `T_diagnostic_correlation` |
| Table 9 | Training time and memory | S1 `T_cost` |
| Table 10 | Sensitivity analysis | S1 `T3_sensitivity` |
| Table 11 | Experiment 2, mean of each metric | S2 `T_main_knn_acc`, `T_main_linear_acc`, `T_main_sil_native`, `T_main_sil_native_train`, `T_main_db_native`, `T_avg_rank_*` |
| Table 12 | Experiment 2, DACHP-v2 against each method | S2 `T_across_tests_knn_acc`, `T_across_tests_sil_native` |
| Table 13 | Ablation, test Silhouette | S1 `T_ablation_sil_native` (accuracy: `T_ablation_knn_acc`) |
| Fig. 2 | Experiment 1, test 5-NN accuracy per dataset | S1 `F_bars_knn_acc` |
| Fig. 3 | Experiment 1, average ranks and critical difference | S1 `F_cd_knn_acc` (left) and `F_cd_sil_native` (right) |
| Fig. 4 | Gain against input-space measures | S1 `F_gain_vs_diagnostics` |
| Fig. 5 | 2-D views of test embeddings | S1 `F_embeddings_test` |
| Fig. 6 | Collapse measures during training | S1 `F_collapse_dynamics` |
| Fig. 7 | Sensitivity | S1 `F_sensitivity` |
| Fig. 8 | Experiment 2, average ranks and critical difference | S2 `F_cd_knn_acc` (left) and `F_cd_sil_native` (right) |
| Fig. 9 | Ablation, change in test Silhouette | S1 `F_ablation_heatmap_sil_native` |

The per-dataset tests, confidence intervals, trustworthiness, training-set Silhouette, permutation control, collapse values and convergence curves in the Supplementary Material come from the remaining files in the same folders (`T_perdataset_tests_*`, `T_main_*`, `T_permutation`, `T_collapse_final`, `F_convergence`, `F_dimensionality_*`, `F_bars_sil_native`). Fig. 1 (architecture) is a drawing and is not produced by the code.
