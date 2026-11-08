# SAW vs EDAS on the water potability dataset: reproduction code

Reproduces every number, table and figure in the manuscript
"Agreement Between SAW and EDAS in Ranking Water Potability Samples and the Effect of Criterion Direction".

## Data
`water_potability.csv`: public Kaggle dataset (A. Kadiwal, 2021),
https://www.kaggle.com/datasets/adityakadiwal/water-potability (3,276 samples, 9 criteria + Potability label).

## Requirements
Python 3.10+ with numpy, pandas, scipy, scikit-learn, matplotlib
(tested: Python 3.13.7, numpy 2.3.3, pandas 2.3.2, scipy 1.16.2, scikit-learn 1.7.2, matplotlib 3.10.7).

## Run
    python analysis.py   # writes results.json (Tables 2-8, Monte Carlo, noise, direction check)
    python figures.py    # writes figures/figure2..5 (.png at 300 dpi and .pdf)

## Files
- `core.py`: data loading (column-mean imputation), SAW (max / min-max normalisation), EDAS, ranking, Overlap@K, entropy weights
- `analysis.py`: all statistics; random seed 42
- `figures.py`: Figures 2-5
- `results.json`: output of analysis.py as used in the manuscript

Notes: raw importance scores 0.15/0.10 sum to 1.05 and are normalised to 1 (0.1429/0.0952);
sample IDs are zero-based row indices of the CSV.
