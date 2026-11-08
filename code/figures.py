"""Draws Figures 2-5 at 300 dpi from the outputs of analysis.py.
Run from the repository root after analysis.py:  python code/figures.py  -> figures/*.png, *.pdf"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import core

DPI = 300
BLUE, TEAL, RED, GOLD, GREY, INK = "#2160A8", "#1F8079", "#C8473B", "#C88A12", "#8A93A3", "#262B36"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "Times New Roman"],
                     "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#D9DCE3", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": DPI,
                     "mathtext.fontset": "stix"})
SIZE = (6.8, 2.75)
os.makedirs(core.FIGURES, exist_ok=True)
X, y, _ = core.load()
R = json.load(open(os.path.join(core.RESULTS, "results.json")))
s, e, sw, ew = np.load(os.path.join(core.RESULTS, "scores.npy"))
mc = np.load(os.path.join(core.RESULTS, "mc_spearman.npy"))
Ks = np.unique(np.round(np.logspace(1, 3, 60)).astype(int))
thou = FuncFormatter(lambda v, _: f"{int(v):,}")


def panel(ax, letter):
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=10.5)


def save(fig, name):
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(core.FIGURES, f"{name}.{ext}"), dpi=DPI, metadata={"Author": None, "Creator": None} if ext == "pdf" else None)
    plt.close(fig)


# Figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
a.axvspan(6.5, 8.5, color="#DFE2E8", label="WHO 6.5–8.5")
a.hist(X[:, 0], bins=np.arange(0, 14.6, 0.5), color=BLUE, edgecolor="white", linewidth=0.6)
a.set(xlabel="pH (C1)", ylabel="Number of samples", xlim=(-0.5, 14.5)); a.legend(loc="upper left", fontsize=8)
b.hist(X[:, 2] / 1000, bins=40, color=TEAL, edgecolor="white", linewidth=0.6)
b.set(xlabel="Total dissolved solids, C3 (×1,000 ppm)", ylabel="Number of samples")
for ax in (a, b): ax.grid(axis="x", visible=False)
panel(a, "a"); panel(b, "b"); save(fig, "figure2_distribution")

# Figure 3
fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
a.scatter(s[y == 0], e[y == 0], s=5, color=GREY, alpha=0.45, lw=0, label="Not potable (label 0)")
a.scatter(s[y == 1], e[y == 1], s=5, color=BLUE, alpha=0.55, lw=0, label="Potable (label 1)")
a.set(xlabel="SAW score (V)", ylabel="EDAS score (AS)")
a.legend(loc="upper left", fontsize=8, markerscale=2.5)
a.text(0.97, 0.05, f"$r_s$ = {R['base']['spearman']:.4f}", transform=a.transAxes, ha="right")
ov = [100 * core.overlap(s, e, k) / k for k in Ks]
b.plot(Ks, ov, color=BLUE, lw=1.6)
kt = [int(k) for k in R["overlap"]]
b.scatter(kt, [100 * R["overlap"][str(k)] / k for k in kt], color=RED, s=22, zorder=3, label="K in Table 5")
b.set(xscale="log", xlabel="Cut-off K (top samples)", ylabel="Overlap@K (%)", ylim=(40, 100))
b.set_xticks([10, 20, 50, 100, 200, 500, 1000]); b.xaxis.set_major_formatter(thou)
b.legend(loc="lower right", fontsize=8)
panel(a, "a"); panel(b, "b"); save(fig, "figure3_scatter_overlap")

# Figure 4
fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
a.hist(mc, bins=30, color=BLUE, edgecolor="white", linewidth=0.6)
a.axvline(mc.mean(), ymax=0.74, color=RED, ls="--", lw=1.4, label=f"mean = {mc.mean():.4f}")
a.axvline(R["base"]["spearman"], ymax=0.74, color=INK, ls=":", lw=1.4, label=f"expert weights = {R['base']['spearman']:.4f}")
a.set(xlabel="Spearman SAW–EDAS ($r_s$)", ylabel="Frequency"); a.set_ylim(0, a.get_ylim()[1] * 1.38); a.legend(loc="upper left", fontsize=8)
a.grid(axis="x", visible=False)
names = ["expert", "equal", "entropy"]; xs = np.arange(3); wd = 0.26
for k, (m, c, lab) in enumerate([("spearman", BLUE, "Spearman"), ("kendall", TEAL, "Kendall"), ("pearson", GOLD, "Pearson")]):
    v = [R["schemes"][n][m] for n in names]
    bars = b.bar(xs + (k - 1) * wd, v, wd * 0.92, color=c, label=lab)
    for rect, val in zip(bars, v):
        b.text(rect.get_x() + rect.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", fontsize=7)
b.set_xticks(xs, ["Expert", "Equal", "Entropy"]); b.set(ylim=(0.80, 1.0), ylabel="Correlation coefficient")
b.grid(axis="x", visible=False); b.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.58, 1.15), fontsize=8, handlelength=1.2, columnspacing=1.0)
panel(a, "a"); panel(b, "b"); save(fig, "figure4_montecarlo_weights")

# Figure 5
fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
rng = np.random.default_rng(0)
a.axhspan(6.5, 8.5, color="#DFE2E8", label="WHO pH 6.5–8.5")
for i, (sc, col) in enumerate([(s, RED), (e, RED), (sw, TEAL), (ew, TEAL)]):
    top = np.argsort(-sc, kind="stable")[:100]
    a.scatter(i + rng.uniform(-0.18, 0.18, 100), X[top, 0], s=9, color=col, alpha=0.6, lw=0)
    a.hlines(np.median(X[top, 0]), i - 0.3, i + 0.3, color=INK, lw=1.6)
a.set_xticks(range(4), ["SAW\nall benefit", "EDAS\nall benefit", "SAW\nWHO-based", "EDAS\nWHO-based"], fontsize=8)
a.set(ylabel="pH of the top-100 samples"); a.grid(axis="x", visible=False); a.legend(loc="upper right", fontsize=8)
b.plot(Ks, [100 * core.overlap(s, e, k) / k for k in Ks], color=RED, lw=1.6, label="SAW–EDAS, all benefit")
b.plot(Ks, [100 * core.overlap(sw, ew, k) / k for k in Ks], color=TEAL, lw=1.6, label="SAW–EDAS, WHO-oriented")
b.plot(Ks, [100 * core.overlap(s, sw, k) / k for k in Ks], color=GREY, lw=1.4, ls="--", label="SAW: all benefit vs WHO-oriented")
b.set(xscale="log", xlabel="Cut-off K (top samples)", ylabel="Overlap@K (%)", ylim=(0, 100))
b.set_xticks([10, 20, 50, 100, 200, 500, 1000]); b.xaxis.set_major_formatter(thou)
b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=7.5)
panel(a, "a"); panel(b, "b"); save(fig, "figure5_criterion_direction")
print("done")
