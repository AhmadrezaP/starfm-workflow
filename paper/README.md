# Reproducing the article's tables and figures

`analysis/build_results.py` regenerates every table, figure and in-text number of the article from the
campaign records. It needs Python with pandas, NumPy, SciPy and matplotlib; no satellite data or GDAL is required.

1. Download the campaign records archive (version 1.1 or later; https://doi.org/10.5281/zenodo.24542147
   resolves to the latest version) and extract it so that this directory
   contains `data/`:

   ```text
   paper/data/DATE_PLANS/final_main_v4/results_merged.csv   one row per planned prediction
   paper/data/DATE_PLANS/final_main_v4/failures.csv         cause of each excluded prediction
   paper/data/TESTS/<test_id>/                              manifests, summaries and validation metrics
   paper/data/BASELINES/final_main/results/<test_id>.json   STARFM and temporal baselines on common pixels
   paper/data/BASELINES/examples/<test_id>.npz              crops used in the example figure
   paper/data/development/...                               development (parameter-selection) campaign
   paper/data/COMPUTE/                                      Slurm accounting and file sizes (cost table)
   ```

2. From this directory, run:

   ```bash
   python analysis/build_results.py
   ```

Outputs are written to `tables/`, `figures/`, `numbers.tex` and `analysis/summary.json`.
