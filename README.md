# SAW vs EDAS on the water potability dataset: reproduction code

Reproduces every number, table and figure in the manuscript
"Agreement Between SAW and EDAS in Ranking Water Potability Samples and the Effect of Criterion Direction".

## Repository layout
```
code/       core.py      SAW, EDAS, ranking, Overlap@K, entropy weights, data loading
            analysis.py  all statistics (Tables 2-8, Monte Carlo, noise, direction check); seed 42
            figures.py   Figures 2-5 (PNG at 300 dpi + PDF)
dataSBP/    water_potability.csv   input dataset
results/    results.json           numbers reported in the manuscript
figures/    figure2..5 (.png, .pdf)
```

## Data
`dataSBP/water_potability.csv`: public Kaggle dataset (A. Kadiwal, 2021),
https://www.kaggle.com/datasets/adityakadiwal/water-potability (3,276 samples, 9 criteria + Potability label).

## Requirements
Python 3.10+ with numpy, pandas, scipy, scikit-learn, matplotlib
(tested: Python 3.13.7, numpy 2.3.3, pandas 2.3.2, scipy 1.16.2, scikit-learn 1.7.2, matplotlib 3.10.7).

## Run (from the repository root)
    python code/analysis.py   # writes results/results.json
    python code/figures.py    # writes figures/figure2..5

Notes: raw importance scores 0.15/0.10 sum to 1.05 and are normalised to 1 (0.1429/0.0952);
sample IDs are zero-based row indices of the CSV.
