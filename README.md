# STARFM workflow: Landsat-like reflectance from Landsat 8/9 with MODIS or VIIRS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542146.svg)](https://doi.org/10.5281/zenodo.24542146)

An open, provenance-tracked workflow that automates the spatial and temporal adaptive reflectance fusion
model (STARFM; Gao et al. 2006) for Landsat 8/9 Collection 2 surface reflectance with daily reflectance from
either MODIS or VIIRS, and evaluates every prediction against a withheld Landsat image and simple temporal
baselines (nearest-date persistence and linear interpolation) on identical pixels.

It automates:

- discovery and download of Landsat 8/9, MOD09GA or VNP09GA, MCD43 or VNP43 BRDF parameters and MCD12Q1
  land cover, with validated, atomic downloads and retries for transient service errors;
- daily nadir-view BRDF normalization of MODIS or VIIRS reflectance, with a land-cover fallback for missing
  BRDF coefficients and automatic substitution of unpublished land-cover years;
- registration of the coarse images to Landsat (moving-window correlation, AROSICS, or none);
- preparation and execution of the original STARFM program (USDA-ARS, v1.2.2, unmodified);
- held-out Landsat validation and temporal baselines, with a manifest recording every input and setting;
- campaigns of hundreds of predictions as Slurm job arrays, with cross-node cache locks, resumable retries and
  recorded failure causes.

The workflow and its multi-region evaluation are described in the accompanying article (see
[Citation](#citation)).

## Requirements

- **Operating system:** Linux. Campaigns were run on the University of Nebraska–Lincoln Holland Computing
  Center *Swan* cluster. Slurm is needed only for campaigns; single predictions run on any Linux machine.
- **Python 3.11** with GDAL built with the HDF4 and HDF5 drivers. `environment.yml` creates a suitable conda
  environment.
- **A C compiler with OpenMP** (e.g. gcc) to build STARFM.
- **Accounts:** a NASA Earthdata login token (for MODIS, VIIRS and land-cover products) and a USGS ERS
  username with an M2M application token (for Landsat).
- **Hardware (per prediction, one band):** 2 CPU cores; 32 GB of memory was requested in the published
  campaigns. Runs took 11–34 minutes (median about 21) on Swan once inputs were cached. Downloaded inputs are
  cached and shared between predictions; plan for tens of gigabytes per Landsat scene and season.

## Installation

```bash
git clone <repository URL> starfm-workflow
cd starfm-workflow
conda env create -f environment.yml
conda activate thesis_paper_env
make -C StarFM_original/source        # builds StarFM_original/source/StarFM.exe
cp config.example.yaml config.yaml    # then add your credentials; config.yaml is ignored by Git
```

Check that GDAL has the HDF drivers:

```bash
python -c 'from osgeo import gdal; assert gdal.GetDriverByName("HDF4") and gdal.GetDriverByName("HDF5")'
```

## A single prediction

Predict Landsat B4 for one date from two bracketing Landsat pairs, with VIIRS as the coarse sensor, and
evaluate it against the Landsat image acquired on the prediction date:

```bash
python run_nearest_pair_pipeline.py \
  --prediction-date 2023-09-18 --path 29 --row 32 --band B4 \
  --pair-dates 2023-09-02 2023-09-26 \
  --coarse-sensor VIIRS --coregistration-method moving-window \
  --starfm-max-search-distance 950 --starfm-uncertainty 35 \
  --validate-held-out-landsat --test-id example_b4
```

Results are written to `COREGISTRATION_FILES/OUTPUTS/TESTS/<test-id>/`: the prediction GeoTIFF, the
validation rasters and metrics, `run_manifest.json` (every input granule, setting and diagnostic) and
`test_summary.json`. Run `python run_nearest_pair_pipeline.py --help` for all options, including
`--coarse-sensor MODIS`, `--pair-count 1` (single nearest pair) and `--all-bands`.

## Campaigns on a Slurm cluster

A campaign is defined by an Excel workbook with one sheet per scene, named `path N - row N`, and the columns
`Pair 1`, `Prediction date`, `Pair 2`. Preview it, then submit it as a job array:

```bash
bash slurm/run_date_plan.sh --workbook dates.xlsx --mode final --campaign my_campaign \
  --max-concurrent 20 --remove-unnecessary            # preview
bash slurm/run_date_plan.sh --workbook dates.xlsx --mode final --campaign my_campaign \
  --max-concurrent 20 --remove-unnecessary --submit   # submit
```

Re-run only incomplete predictions after the array finishes with `--retry-failed` (optionally
`--only-tasks 3,17`), and collect the results with
`python slurm/date_plan.py --collect COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/my_campaign/plan.json`.
Set your e-mail address in the `#SBATCH --mail-user` lines and adjust partitions and module names to your
cluster. See [slurm/README.md](slurm/README.md) for details, including the shared-cache safety rules.

## Temporal baselines

`evaluate_baselines.py` scores nearest-date persistence and linear interpolation of the Landsat pair images
against the held-out target on each completed prediction's own comparison pixels, and rescores STARFM on the
same pixels. On a cluster, list the completed `run_manifest.json` files and submit:

```bash
sbatch --array=0-<N-1>%30 --export=ALL,PROJECT_DIR=$PWD,MANIFEST_LIST=manifests.txt,OUTPUT_DIR=baselines \
  slurm/baselines_array.sbatch
```

## Reproducing the article's tables and figures

`paper/analysis/build_results.py` regenerates every table, figure and number in the article from the
published campaign records. See [paper/README.md](paper/README.md).

## Tests

```bash
python -m unittest -v
```

Tests that need GDAL are skipped or fail without it; run them in the conda environment.

## Citation

If you use this workflow, please cite the article (citation to be added on publication) and the software
release:

Pourghodrat A, Akasheh SZ, Yu H, Neale CMU (2026) STARFM workflow: Landsat-like reflectance from Landsat 8/9
with MODIS or VIIRS, with built-in temporal baselines (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.24542146

The campaign records behind the article's evaluation are archived separately (CC BY 4.0):
https://doi.org/10.5281/zenodo.24542148

STARFM itself should be cited as:

Gao F, Masek J, Schwaller M, Hall F (2006) On the blending of the Landsat and MODIS surface reflectance:
predicting daily Landsat surface reflectance. IEEE Trans Geosci Remote Sens 44(8):2207–2218.
https://doi.org/10.1109/TGRS.2006.872081

## Licence

The workflow code is released under the MIT licence (see [LICENSE](LICENSE)). The STARFM source in
`StarFM_original/` is the unmodified USDA-ARS release, dedicated to the public domain under CC0 1.0; see
[StarFM_original/NOTICE.md](StarFM_original/NOTICE.md).
