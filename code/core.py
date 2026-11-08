"""SAW and EDAS core routines for the water potability ranking study."""
import os
import numpy as np
import pandas as pd
from scipy import stats

CRITERIA = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataSBP", "water_potability.csv")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def load(path=DATA):
    df = pd.read_csv(path)
    X = df[CRITERIA].copy()
    X = X.fillna(X.mean())          # column-mean imputation
    return X.to_numpy(float), df["Potability"].to_numpy(int), df


def saw(X, w, benefit, minmax=False):
    X = np.asarray(X, float)
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        if minmax:
            rng = col.max() - col.min()
            R[:, j] = (col - col.min()) / rng if benefit[j] else (col.max() - col) / rng
        else:
            R[:, j] = col / col.max() if benefit[j] else col.min() / col
    return R @ np.asarray(w)


def edas(X, w, benefit):
    X = np.asarray(X, float)
    AV = X.mean(axis=0)
    PDA = np.where(benefit, np.maximum(0, X - AV), np.maximum(0, AV - X)) / AV
    NDA = np.where(benefit, np.maximum(0, AV - X), np.maximum(0, X - AV)) / AV
    SP, SN = PDA @ w, NDA @ w
    NSP = SP / SP.max()
    NSN = 1 - SN / SN.max()
    return 0.5 * (NSP + NSN)


def ranks(score):
    # rank 1 = best (highest score)
    return stats.rankdata(-score, method="ordinal")


def overlap(a, b, k):
    ta = set(np.argsort(-a, kind="stable")[:k])
    tb = set(np.argsort(-b, kind="stable")[:k])
    return len(ta & tb)


def entropy_weights(X):
    P = X / X.sum(axis=0)
    P = np.where(P > 0, P, 1e-12)
    E = -(P * np.log(P)).sum(axis=0) / np.log(X.shape[0])
    d = 1 - E
    return d / d.sum()
