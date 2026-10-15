# DACHP: Deep Hyperspherical Contrastive Embeddings for High-Dimensional Image and Signal Representation

Code, configuration files, pre-registration and complete run records for the paper

> V.B. Meruga, L. Gudala, C.S.K. Reddy, L. Rudraksha, *Deep Hyperspherical Contrastive Embeddings for High-Dimensional Image and Signal Representation: A Leakage-Free Evaluation*, Digital Signal Processing (under review, manuscript DSP-D-26-01640).

One script (`make_report.py`) produces every table, figure and number in the paper from the run records in `results/`. No value in the paper is entered by hand.

## Repository contents

```
dachp/                  method and evaluation code
  core.py               encoder, DACHP loss, baseline losses, training loop, collapse measures
  data.py               dataset loaders, stratified/grouped splits, preprocessing fitted on training data
  baselines.py          PCA, LDA, t-SNE, UMAP, Supervised UMAP, PaCMAP, TriMap
  metrics.py            test-set probes (5-NN, linear, nearest class mean), cluster indices, trustworthiness
run_experiments.py      runs all experiments; resumable (finished runs are skipped)
make_report.py          builds all tables, figures and statistics from the run records
config.yaml             Experiment 1 settings (splits 0-4)
config_v2.yaml          Experiment 2 settings (pre-registered, splits 5-9)
PREREGISTRATION.md      Experiment 2 protocol and decision rule, written before the run
results/experiment1/         run records (runs.jsonl), logs and generated report of Experiment 1
results/experiment2/         the same for Experiment 2
docs/MANUSCRIPT_MAP.md  which report file gives which table or figure of the paper
docs/CHANGES.md         changes from the code of the first submission
```

## Installation

Python 3.10 or later. All computation runs on the CPU.

```bash
pip install -r requirements.txt
```

## 1. Rebuild all tables and figures from the recorded runs (a few minutes)

```bash
python make_report.py --results results/experiment1/results_a results/experiment1/results_b --out report_experiment1
python make_report.py --results results/experiment2/results_v2_a results/experiment2/results_v2_b --out report_experiment2 --focal dachp_v2
```

Each command writes `tables/` (CSV and Markdown), `figures/` (PNG and PDF), `REPORT.md` and `summary.json`. The output is the same as the reports already stored in `results/experiment1/report` and `results/experiment2/report`.

## 2. Run the experiments again

```bash
# quick test with a very small number of epochs (a few minutes; the numbers are not meaningful)
python run_experiments.py --quick --datasets breast_cancer --seeds 0 1 --out smoke_test

# Experiment 1: all datasets, all stages
python run_experiments.py --out results_experiment1

# Experiment 2: pre-registered confirmatory experiment
python run_experiments.py --config config_v2.yaml --out results_experiment2 --stages diagnostics tune main
```

* Stages run in the order `diagnostics, tune, main, ablation, permutation, sensitivity`; `--stages` selects a subset and `--datasets` a subset of datasets, so the work can be split across several processes or machines.
* Every finished run is appended to `<out>/runs.jsonl`. If a run is interrupted, the same command continues where it stopped.
* The full Experiment 1 takes many hours on a CPU; the larger datasets (MNIST, Fashion-MNIST, HAR, ISOLET) take most of the time.
* The results in `results/` were produced with this code, with the deep models trained on an accelerator. Running on a CPU follows the same procedure, but numbers can differ slightly in the last digits because of floating-point differences between devices. Timing values depend on the hardware.

## Datasets

All datasets are public and are downloaded automatically on first use (scikit-learn or OpenML), except COIL-20, which is downloaded from the Columbia University CAVE website (processed version, resized to 32x32). If a download is blocked, place `coil-20-proc.zip` in a folder and set `DACHP_DATA=/path/to/folder`. Downloaded files are cached in `data_cache/`.

| Key | Dataset | Source |
|---|---|---|
| breast_cancer | Breast Cancer (Wisconsin diagnostic) | scikit-learn |
| digits | Digits (optical recognition) | scikit-learn |
| mice_protein | Mice Protein Expression | OpenML 40966 |
| anneal | Annealing | OpenML 2 |
| coil20 | COIL-20 (processed) | Columbia CAVE |
| cnae9 | CNAE-9 | OpenML 1468 |
| mnist | MNIST (10,000 samples, 1,000 per class) | OpenML 554 |
| fashion_mnist | Fashion-MNIST (10,000 samples, 1,000 per class) | OpenML 40996 |
| har | Human Activity Recognition (smartphone) | OpenML 1478 |
| isolet | ISOLET | OpenML 300 |
| olivetti | Olivetti Faces | scikit-learn |

## Evaluation protocol (summary)

Five stratified 60/20/20 train/validation/test splits per experiment (grouped by mouse for Mice Protein). Imputation, scaling and any PCA are fitted on the training split only. Hyper-parameters are selected on the validation set of the first split and then fixed. All reported metrics are computed on the test split. Full details are in Section 4 of the paper and in the configuration files.

## Pre-registration (Experiment 2)

`PREREGISTRATION.md` and `config_v2.yaml` were written before Experiment 2 was run (the document calls Experiment 1 the "main study"). Their SHA-256 hashes are:

```
PREREGISTRATION.md  adf48efe6980fcc6bdf89d427b75c37429d409fc9b87757abb1ea671553453de
config_v2.yaml      d8d823dc8729f10aff1d33a7775a0c7ec0517ec63afb51c57bd9255212297bf3
```

Check with `sha256sum PREREGISTRATION.md config_v2.yaml`.

## Run records

`runs.jsonl` holds one JSON record per run (dataset, method, split, settings, all metrics, training history and timing). Experiment 1 contains 2,211 distinct runs and Experiment 2 contains 1,485. The Experiment 1 files also contain 220 repeated records of runs that were written more than once; their results are identical, and `make_report.py` keeps one record per run key.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT License (see `LICENSE`).
