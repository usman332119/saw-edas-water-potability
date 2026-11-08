"""Reproduces every number reported in the manuscript (Tables 2-8, Monte Carlo,
noise and criterion-direction checks). Run:  python analysis.py
Outputs: results.json and printed tables."""
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import core

SEED = 42
RAW_W = np.array([.15, .15, .10, .10, .15, .10, .10, .10, .10])   # raw importance scores
W = RAW_W / RAW_W.sum()                                            # normalised (sum = 1)
K_LIST = [10, 20, 50, 100, 200, 500, 1000]


def agreement(s, e):
    return dict(spearman=stats.spearmanr(s, e)[0], kendall=stats.kendalltau(s, e)[0],
                pearson=stats.pearsonr(s, e)[0])


def main(path="water_potability.csv"):
    X, y, df = core.load(path)
    out = {"raw_weight_sum": float(RAW_W.sum()), "weights": W.round(4).tolist()}
    out["missing"] = df[core.CRITERIA].isna().sum().to_dict()
    out["descriptives"] = {c: dict(mean=X[:, j].mean(), sd=X[:, j].std(ddof=1), min=X[:, j].min(),
                                   max=X[:, j].max()) for j, c in enumerate(core.CRITERIA)}
    ben = np.ones(9, bool)
    s, e = core.saw(X, W, ben), core.edas(X, W, ben)
    rs_, re_ = core.ranks(s), core.ranks(e)
    out["base"] = agreement(s, e)
    wil = stats.wilcoxon(rs_, re_)
    out["wilcoxon_ranks"] = dict(W=float(wil.statistic), p=float(wil.pvalue))
    out["wilcoxon_scores_p"] = float(stats.wilcoxon(s, e).pvalue)
    top = np.argsort(-s, kind="stable")[:10]
    out["top10"] = [dict(id=int(i), ph=X[i, 0], hardness=X[i, 1], solids=X[i, 2], chloramines=X[i, 3],
                         thm=X[i, 7], saw=s[i], saw_rank=int(rs_[i]), edas=e[i], edas_rank=int(re_[i]))
                    for i in top]
    out["overlap"] = {k: core.overlap(s, e, k) for k in K_LIST}
    # weighting schemes
    schemes = {"expert": W, "equal": np.full(9, 1 / 9), "entropy": core.entropy_weights(X)}
    out["entropy_w"] = schemes["entropy"].round(4).tolist()
    out["schemes"] = {k: agreement(core.saw(X, w, ben), core.edas(X, w, ben)) for k, w in schemes.items()}
    # Monte Carlo Dirichlet weights
    rng = np.random.default_rng(SEED)
    mc = np.array([stats.spearmanr(core.saw(X, w, ben), core.edas(X, w, ben))[0]
                   for w in rng.dirichlet(np.ones(9), 1000)])
    out["mc"] = dict(mean=mc.mean(), sd=mc.std(ddof=1), min=mc.min(), max=mc.max(),
                     share_above_expert=float((mc > out["base"]["spearman"]).mean()))
    np.save("mc_spearman.npy", mc)
    # multiplicative noise
    rng = np.random.default_rng(SEED)
    out["noise"] = {}
    for p in [.05, .10, .15, .20]:
        v = []
        for _ in range(100):
            Xn = X * (1 + rng.uniform(-p, p, X.shape))
            v.append(stats.spearmanr(core.saw(Xn, W, ben), core.edas(Xn, W, ben))[0])
        out["noise"][f"{int(p*100)}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
    # criterion direction: WHO-oriented
    Xw = X.copy(); Xw[:, 0] = np.abs(X[:, 0] - 7.5)
    benw = np.array([False, True] + [False] * 7)
    sw, ew = core.saw(Xw, W, benw, minmax=True), core.edas(Xw, W, benw)
    np.save("scores.npy", np.vstack([s, e, sw, ew]))
    phbad = lambda idx: int(((X[idx, 0] < 6.5) | (X[idx, 0] > 8.5)).sum())
    t = lambda a, k: np.argsort(-a, kind="stable")[:k]
    out["direction"] = {
        "who": agreement(sw, ew),
        "overlap_who": {k: core.overlap(sw, ew, k) for k in K_LIST},
        "ph_out_top10": dict(saw=phbad(t(s, 10)), edas=phbad(t(e, 10)), saw_who=phbad(t(sw, 10)), edas_who=phbad(t(ew, 10))),
        "potable_top100": dict(saw=y[t(s, 100)].mean(), edas=y[t(e, 100)].mean(), saw_who=y[t(sw, 100)].mean(), edas_who=y[t(ew, 100)].mean()),
        "base_rate": y.mean(),
        "auc": dict(saw=roc_auc_score(y, s), edas=roc_auc_score(y, e), saw_who=roc_auc_score(y, sw), edas_who=roc_auc_score(y, ew)),
        "saw_orig_vs_who": stats.spearmanr(s, sw)[0],
        "saw_top100_shared": len(set(t(s, 100)) & set(t(sw, 100))),
        "top10_mean_chloramines": X[t(s, 10), 3].mean(), "top10_mean_thm": X[t(s, 10), 7].mean(),
        "single_criterion_auc": [roc_auc_score(y, X[:, j]) for j in range(9)],
    }
    conv = lambda o: {k: conv(v) for k, v in o.items()} if isinstance(o, dict) else [conv(v) for v in o] if isinstance(o, list) else (float(o) if isinstance(o, (np.floating, np.integer)) else o)
    json.dump(conv(out), open("results.json", "w"), indent=1)
    print(json.dumps(conv(out), indent=1))


if __name__ == "__main__":
    main()
