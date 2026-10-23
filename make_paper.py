"""Paper figures (PDF) and LaTeX tables, rebuilt from the stored results.

    python make_paper.py        -> paper/figures/*.pdf, paper/tables/*.tex

Needs the result folders (results_heart, results_tcga, results_isic and the
results_*_loo controls) and the CSVs written by analyze_heart.py,
analyze_tcga.py, analyze_isic.py, validity_table.py (--weights) and the
reproduction scripts. Every number uses the same definitions as those
scripts (docs/PROTOCOL.md): points = metric x 100; means and 95% t CIs over seeds;
"mean of real attacks" = per seed, the mean over every non-degenerate
attack and every attacker identity. Leave-out tables: attack damage
without the ALIE variants is the main column (2026-10-19).

Tables are bare `tabular` environments (needs booktabs and amssymb);
captions are in paper/README.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_loo
from analyze_heart import HOSPITALS, collect, mean_ci
from analyze_isic import CENTRES
from analyze_tcga import REGIONS
from bench.plotstyle import GRID, INK, MUTED, RULE_COLORS, RULE_LABELS, plt

TAB, RT = Path("results_tables"), Path("paper")
FIGS, TABS = RT / "figures", RT / "tables"

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "sans-serif",
                     "axes.titlelocation": "left"})
WIDTH = 6.5   # inches, full text width
MAX_ROWS = 46  # longest table that fits one page; longer tables are split

DS = {
    "heart": dict(name="Fed-Heart-Disease", short="Heart", folder="results_heart",
                  dataset="fed_heart_disease", clients=HOSPITALS, metric="Accuracy",
                  real={1: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1"]},
                  largest=0),
    "tcga": dict(name="Fed-TCGA-BRCA", short="TCGA", folder="results_tcga",
                 dataset="fed_tcga_brca", clients=REGIONS, metric="C-index",
                 real={1: ["sign_flip", "gaussian", "ipm", "alie_z1"],
                       2: ["sign_flip", "gaussian", "alie", "alie_z1", "ipm"]},
                 largest=0, ineligible=[4, 5]),
    "isic": dict(name="Fed-ISIC2019", short="ISIC", folder="results_isic",
                 dataset="fed_isic2019", clients=CENTRES, metric="Balanced accuracy",
                 real={1: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1"],
                       2: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1", "alie"]},
                 largest=0),
}
RULES = list(RULE_LABELS)


def lc(metric):
    return metric if metric.startswith("C-") else metric.lower()


def variants(rules=RULES):
    """(rule, weighted) rows; weighted Krum is Krum (it ignores weights)."""
    return [(r, w) for r in rules for w in (False, True) if not (r == "krum" and w)]


def vlabel(rule, w):
    return RULE_LABELS[rule] + (" (w)" if w else "")


_cache = {}


def runs(key):
    if key not in _cache:
        d = DS[key]
        _cache[key] = collect(Path(d["folder"]), dataset=d["dataset"])
    return _cache[key]


# ---------------------------------------------------------------- numbers

def clean_stats(df, rule, w, fa):
    """Clean runs (server assumes fa): per-seed avg and worst client; the
    per-client accuracies (mean over seeds)."""
    g = df[(df.attack == "none") & (df.f_assumed == fa) & (df.aggregator == rule) &
           (df.weighted == w) & (df.status == "ok")]
    if g.empty:
        return None
    acc = [c for c in g if c.startswith("acc_")]
    return dict(avg=mean_ci(g.avg_acc * 100), worst=mean_ci(g.worst_acc * 100),
                clients=(g[acc].mean() * 100).to_numpy())


def attacked_stats(df, rule, w, f, real, attacker=None):
    """Mean of real attacks at f (per seed over attacks x attackers); the
    attacker set can be a predicate on the byz_ids string."""
    g = df[(df.f == f) & df.attack.isin(real) & (df.aggregator == rule) &
           (df.weighted == w) & (df.status == "ok")]
    if attacker is not None:
        g = g[g.attacker.astype(str).map(attacker)]
    if g.empty:
        return None
    s = g.groupby("seed")[["avg_acc", "worst_honest_acc"]].mean() * 100
    return dict(avg=mean_ci(s.avg_acc), worst=mean_ci(s.worst_honest_acc), n_seeds=len(s))


def _matched(path):
    """A clean run scored as if each attacker set had attacked: worst
    client over the same honest clients (summarize.run_metrics with byz_ids
    set), so clean and attacked compare the same clients."""
    import json
    from bench.design import attacker_sets
    from summarize import run_metrics
    r = json.loads(Path(path).read_text())
    if r["status"] != "ok":
        return []
    c, rows = r["config"], []
    n = len(r["client_sizes"])
    for f in (1, 2):
        for ids in attacker_sets(n, f):
            r["byz_ids"] = [int(i) for i in ids.split(",")]
            rows.append(dict(rule=c["aggregator"], weighted=c["weighted"], seed=c["seed"],
                             fa=r["f_assumed"], f=f, attacker=ids,
                             worst=run_metrics(r)["worst_honest_acc"] * 100))
    return rows


def matched_clean(key):
    if ("matched", key) not in _cache:
        from multiprocessing import Pool
        files = [p for p in Path(DS[key]["folder"]).glob("*_none_*_fa[12]_*.json")]
        with Pool() as pool:
            rows = [x for xs in pool.map(_matched, files, chunksize=8) for x in xs]
        _cache[("matched", key)] = pd.DataFrame(rows)
    return _cache[("matched", key)]


def clean_worst_matched(key, rule, w, f, attacker=None):
    m = matched_clean(key)
    g = m[(m.rule == rule) & (m.weighted == w) & (m.fa == f) & (m.f == f)]
    if attacker is not None:
        g = g[g.attacker.map(attacker)]
    if g.empty:
        return None
    return mean_ci(g.groupby("seed").worst.mean())


def big_delta(key, rule, w, f):
    """Attacker sets containing the largest client: per-seed change vs the
    clean run (average; worst over the same honest clients)."""
    df, d = runs(key), DS[key]
    pick = has(d["largest"])
    g = df[(df.f == f) & df.attack.isin(d["real"][f]) & (df.aggregator == rule) &
           (df.weighted == w) & (df.status == "ok") & df.attacker.astype(str).map(pick)]
    c = df[(df.attack == "none") & (df.f_assumed == f) & (df.aggregator == rule) &
           (df.weighted == w) & (df.status == "ok")].set_index("seed")
    m = matched_clean(key)
    m = m[(m.rule == rule) & (m.weighted == w) & (m.fa == f) & (m.f == f) & m.attacker.map(pick)]
    if g.empty or c.empty:
        return None
    a = g.groupby("seed")[["avg_acc", "worst_honest_acc"]].mean() * 100
    seeds = sorted(set(a.index) & set(c.index))
    mw = m.groupby("seed").worst.mean()
    return dict(avg=mean_ci(a.avg_acc[seeds] - 100 * c.avg_acc[seeds]),
                worst=mean_ci(a.worst_honest_acc[seeds] - mw[seeds]))


def has(i):
    return lambda a: str(i) in a.split(",")


def summary(key, f):
    """One row per rule variant: clean (fa = f), attacked, attacked by sets
    containing the largest client."""
    if ("summary", key, f) in _cache:
        return _cache[("summary", key, f)]
    df, d = runs(key), DS[key]
    out = {}
    for rule, w in variants():
        c = clean_stats(df, rule, w, f)
        a = attacked_stats(df, rule, w, f, d["real"][f])
        if c is None or a is None:
            continue
        out[(rule, w)] = dict(clean=c, att=a,
                              clean_worst=clean_worst_matched(key, rule, w, f),
                              big=attacked_stats(df, rule, w, f, d["real"][f], has(d["largest"])),
                              big_delta=big_delta(key, rule, w, f))
    _cache[("summary", key, f)] = out
    return out


def order(key):
    """Client indices sorted by training size, largest first (ties keep the
    dataset's order). Every table and figure lists clients in this order."""
    t = pd.read_csv(TAB / f"{key}_exclusion_fa1.csv")
    sz = (t[(t.aggregator == "fedavg") & ~t.weighted].set_index("hospital").train_size
          .reindex(DS[key]["clients"]).to_numpy())
    return [int(i) for i in np.argsort(-sz, kind="stable")]


def ordered_clients(key):
    return [DS[key]["clients"][i] for i in order(key)]


def influence(key):
    """Relative influence per client (clean, server assumes f = 1)."""
    t = pd.read_csv(TAB / (f"{key}_exclusion_fa1.csv"))
    return {(r, w): g.set_index("hospital").rel_influence_mean.reindex(DS[key]["clients"]).to_numpy()
            for (r, w), g in t.groupby(["aggregator", "weighted"])}


def loo_seeds(key, f):
    return analyze_loo.decompose(key, f, per_seed=True)[4]


def group_decomp(p, rule, w, sets):
    """Mean +- CI over seeds of the parts, averaged per seed over left-out sets."""
    g = p[(p.rule == rule) & (p.weighted == w) & p.left_out.isin(sets)]
    if g.empty:
        return None
    s = g.groupby("seed").mean(numeric_only=True)
    return {c: mean_ci(s[c]) for c in s if c.startswith(("avg_", "worst_"))}


# ---------------------------------------------------------------- LaTeX

def pm(x, digits=1, sign=False):
    if x is None or x[0] is None or np.isnan(x[0]):
        return "--"
    m, c = x
    s = (f"{m:+.{digits}f}" if sign else f"{m:.{digits}f}").replace("-", "$-$")
    if c is None or np.isnan(c):
        return s
    return f"{s}{{\\scriptsize\\,$\\pm${c:.{digits}f}}}"


def num(x, digits=1, sign=False):
    if x is None or np.isnan(x):
        return "--"
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "$-$")


def write_tab(name, colspec, header, body, note=None):
    """header: list of LaTeX header lines (each ending without \\\\); body:
    list of rows (lists of cells) or the string 'MID' / ('GROUP', text)."""
    ncol = len(colspec.replace("@{}", "").replace("|", "").replace(" ", ""))
    lines = [f"% generated by make_paper.py; do not edit by hand",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += [h if h.startswith("\\cmidrule") or h.rstrip().endswith("\\\\") else h + r" \\"
              for h in header]
    lines.append("\\midrule")
    for row in body:
        if row == "MID":
            lines.append("\\midrule")
        elif isinstance(row, tuple) and row[0] == "GROUP":
            lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{row[1]}}}}} \\\\")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.insert(0, f"% {note}")
    (TABS / f"{name}.tex").write_text("\n".join(lines) + "\n")


def save_fig(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})   # same data -> same bytes
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")   # preview only
    plt.close(fig)


def style(ax, xgrid=True):
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)


# ---------------------------------------------------------------- settings

MODEL = {"heart": "logistic regression", "tcga": "linear Cox model",
         "isic": "linear head on frozen EfficientNet-B0 features"}
METRIC = {"heart": "accuracy (test-size weighted)", "tcga": "C-index (stratified)",
          "isic": "balanced accuracy (pooled)"}
SEEDS = {"heart": "20", "tcga": "20 ($f=2$: 5)", "isic": "20"}


def train_sizes(key):
    t = pd.read_csv(TAB / f"{key}_exclusion_fa1.csv")
    return t[(t.aggregator == "fedavg") & ~t.weighted].set_index("hospital").train_size


def tab_settings():
    from bench.tasks import PRESETS
    shares = {}
    for key in DS:
        g = train_sizes(key)
        sh = g.reindex(ordered_clients(key)) / g.sum() * 100
        shares[key] = ", ".join(f"{s:.0f}" for s in sh)
    body = []
    for key in DS:
        p = PRESETS[DS[key]["dataset"]]
        body.append([DS[key]["name"], str(p["n_clients"]), shares[key], MODEL[key], METRIC[key],
                     f"{p['lr']:g}", str(p["rounds"]), SEEDS[key]])
    write_tab("tab_settings", "@{}lrp{2.6cm}p{2.6cm}p{2.4cm}rrr@{}",
              ["Dataset & $n$ & Training share per client (\\%, largest first) & Model & Main metric & lr & Rounds & Seeds"],
              body)


def tab_datasets():
    """Settings with every client's training share and train/test sizes (test
    sizes from the data, summarize.client_test_sizes) and CC's tau
    (results_tune/cc_tau_<dataset>.json)."""
    import json
    from bench.tasks import PRESETS
    from summarize import client_test_sizes
    body = []
    for key in DS:
        d, p = DS[key], PRESETS[DS[key]["dataset"]]
        train = train_sizes(key).reindex(d["clients"])
        test = dict(zip(d["clients"], client_test_sizes(d["dataset"])))
        tau = json.loads(Path(f"results_tune/cc_tau_{d['dataset']}.json").read_text())["cc_tau"]
        clients = ", ".join(f"{c.replace('centre ', 'Centre ')} {100 * train[c] / train.sum():.1f}\\% "
                            f"({train[c]}/{test[c]})" for c in ordered_clients(key))
        body.append([d["name"], str(p["n_clients"]), clients, MODEL[key], METRIC[key],
                     f"{p['lr']:g}", str(p["rounds"]), f"{tau:g}", SEEDS[key]])
    write_tab("tab_datasets", "@{}lrp{5.2cm}p{2.4cm}p{2.2cm}rrrr@{}",
              ["Dataset & $n$ & Client: training share (train/test size), largest first & Model & "
               "Main metric & lr & Rounds & $\\tau$ (CC) & Seeds"],
              body)


# ---------------------------------------------------------------- validity

def validity_cells():
    v = pd.read_csv(TAB / "validity.csv")
    w = pd.read_csv(TAB / "validity_weighted.csv")
    w = w[w.clients == "all clients"].set_index(["dataset", "rule"])
    count = {}
    for (n, f, rule), g in v.set_index(["n", "f", "rule"]).iterrows():
        if g.status != "valid":
            count[(n, f, rule)] = "invalid"
        elif isinstance(g.degenerate, str) and g.degenerate.startswith("="):
            count[(n, f, rule)] = g.degenerate          # '= mean', '= median'
        else:
            count[(n, f, rule)] = "valid"
    weight = {}
    for key in DS:
        for rule in RULES:
            try:
                r = w.loc[(DS[key]["dataset"], rule)]
            except KeyError:
                weight[(key, rule)] = []
                continue
            flags = []
            if str(r.dictator) == "True":           # bool or object column (NaN elsewhere)
                flags.append("D")
            for f in (1, 2):
                b = r.get(f"breakdown_f{f}")
                ex = r.get(f"breakdown_exact_f{f}")
                if str(b) == "True":
                    approx = rule == "geomed" and str(ex) == "False"
                    flags.append(f"B{f}" + ("*" if approx else ""))
            weight[(key, rule)] = flags
    return count, weight


def tab_validity():
    count, weight = validity_cells()
    sym = {"valid": "$\\checkmark$", "invalid": "$\\times$", "= mean": "= mean",
           "= median": "= median"}
    share = {"heart": "40.9", "tcga": "28.6", "isic": "53.4"}
    body = []
    for rule in RULES:
        row = [RULE_LABELS[rule]]
        for n, f in ((4, 1), (4, 2), (6, 1), (6, 2)):
            row.append("--" if rule == "fedavg" else sym.get(count[(n, f, rule)], count[(n, f, rule)]))
        for key in DS:
            fl = weight[(key, rule)]
            n = 4 if key == "heart" else 6
            if rule == "krum":
                row.append("ignores $w$")
            elif rule == "fedavg":
                row.append("--")
            elif count[(n, 1, rule)] == "invalid":
                row.append("n/a")
            else:
                row.append(", ".join(x.replace("*", "$^*$") for x in fl) if fl else "$\\checkmark$")
        body.append(row)
    write_tab("tab_validity", "@{}l cccc ccc@{}",
              ["& \\multicolumn{4}{c}{By client count} & \\multicolumn{3}{c}{Weighted, by data share} \\\\",
               "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
               "& \\multicolumn{2}{c}{$n=4$ (Heart)} & \\multicolumn{2}{c}{$n=6$ (TCGA, ISIC)} "
               f"& Heart & TCGA & ISIC \\\\",
               f"Rule & $f=1$ & $f=2$ & $f=1$ & $f=2$ & ({share['heart']}\\%) & "
               f"({share['tcga']}\\%) & ({share['isic']}\\%)"],
              body)
    return count, weight


def fig_validity():
    count, weight = validity_cells()
    cols = [("Heart\n$n$=4, $f$=1", 4, 1, "heart"), ("Heart\n$n$=4, $f$=2", 4, 2, "heart"),
            ("TCGA\n$n$=6, $f$=1", 6, 1, "tcga"), ("TCGA\n$n$=6, $f$=2", 6, 2, "tcga"),
            ("ISIC\n$n$=6, $f$=1", 6, 1, "isic"), ("ISIC\n$n$=6, $f$=2", 6, 2, "isic")]
    colors = {"invalid": "#d9d9d6", "reduces": "#f3e9c6", "valid": "#ffffff",
              "dict": "#e34948", "breaks": "#f4a6a5"}
    rows = variants()
    fig, ax = plt.subplots(figsize=(WIDTH * 0.66, 3.3))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.88, bottom=0.03)
    for i, (rule, w) in enumerate(rows):
        for j, (_, n, f, key) in enumerate(cols):
            c = count[(n, f, rule)]
            txt, col = "", colors["valid"]
            if c == "invalid":
                txt, col = "invalid", colors["invalid"]
            elif rule == "fedavg":
                txt = "baseline"
            else:
                if c.startswith("="):
                    txt, col = c, colors["reduces"]
                if w and rule not in ("krum", "fedavg"):
                    fl = weight[(key, rule)]
                    if f"B{f}" in fl or f"B{f}*" in fl:
                        dic = "D" in fl
                        txt = ("dictator\n+ breaks" if dic else "breaks") + ("*" if f"B{f}*" in fl else "")
                        col = colors["dict"] if dic else colors["breaks"]
                    elif "D" in fl:
                        txt, col = "dictator", colors["dict"]
                if not txt:
                    txt = "valid"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor="white", lw=1.5))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.5 if "\n" in txt else 6,
                    linespacing=0.9,
                    color=INK if txt not in ("valid", "baseline") else MUTED)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, [c[0] for c in cols])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5, [vlabel(*r) for r in rows])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.02, 0.0, "invalid: fails the count condition; = mean / = median: the rule reduces to it; "
             "dictator: clean output = largest client's update;\nbreaks: the heaviest f clients "
             "control the output by weight (* only with 10 Weiszfeld iterations, not at convergence).",
             fontsize=5.8, color=MUTED, va="top")
    save_fig(fig, "fig_validity")


# ---------------------------------------------------------------- clean

def tab_clean():
    body = []
    for key in DS:
        s = {v: clean_stats(runs(key), *v, 1) for v in variants()}
        inf = influence(key)
        corr = pd.read_csv(TAB / (f"{key}_exclusion_size_corr_fa1.csv" if key != "heart"
                                  else "heart_exclusion_size_corr.csv")).set_index(["aggregator", "weighted"])
        body.append(("GROUP", f"{DS[key]['name']} ({lc(DS[key]['metric'])})"))
        fed = {w: s[("fedavg", w)]["avg"][0] for w in (False, True)}
        for rule in RULES:
            row = [RULE_LABELS[rule]]
            for w in (False, True):
                v = (rule, w)
                if rule == "krum" and w:
                    row += ["\\multicolumn{4}{c}{= Krum (ignores weights)}"]
                    continue
                c = s.get(v)
                if c is None:
                    row += ["invalid", "", "", ""]
                    continue
                rel = inf.get(v)
                low = ("--" if rule == "fedavg" or rel is None else
                       f"{DS[key]['clients'][int(np.nanargmin(rel))]} ({np.nanmin(rel):.2f})")
                rho = corr.loc[v].rho_size_vs_influence if v in corr.index else np.nan
                row += [pm(c["avg"]), pm(c["worst"]), low.replace("centre ", "c"),
                        num(rho, 2) if rule != "fedavg" else "--"]
            body.append(row)
        body.append("MID")
    body.pop()
    write_tab("tab_clean", "@{}l cccc cccc@{}",
              ["& \\multicolumn{4}{c}{Unweighted} & \\multicolumn{4}{c}{Weighted (w)} \\\\",
               "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
               "Rule & Avg & Worst & Least influence & $\\rho$ & Avg & Worst & Least influence & $\\rho$"],
              body)


def fig_clean():
    """Clean cost: average and worst client per rule variant, per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.0), sharey=True)
    rows = variants()
    for ax, key in zip(axes, DS):
        for i, v in enumerate(rows):
            c = clean_stats(runs(key), *v, 1)
            if c is None:
                ax.text(0.02, i, "invalid", transform=ax.get_yaxis_transform(), fontsize=6,
                        color=MUTED, va="center")
                continue
            col = RULE_COLORS[v[0]]
            (a, ac), (b, bc) = c["avg"], c["worst"]
            ax.plot([b, a], [i, i], color=col, lw=1, alpha=0.5)
            ax.errorbar(a, i, xerr=ac, fmt="o", color=col, ms=4, mec="white", mew=0.5, elinewidth=0.8)
            ax.errorbar(b, i, xerr=bc, fmt="o", color=col, ms=4, mfc="white", elinewidth=0.8)
        f0 = clean_stats(runs(key), "fedavg", False, 1)["avg"][0]
        ax.axvline(f0, color=MUTED, lw=0.6, ls=":")
        ax.set_title(DS[key]["name"])
        ax.set_xlabel(f"{DS[key]['metric']} (points)")
        style(ax)
    axes[0].set_yticks(range(len(rows)), [vlabel(*v) for v in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="average"),
         plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4, label="worst client"),
         plt.Line2D([], [], color=MUTED, ls=":", lw=0.6, label="FedAvg average")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save_fig(fig, "fig_clean")


def fig_who_pays():
    """Top: relative influence per client (clean); bottom: per-client change
    vs FedAvg of the same weighting."""
    from matplotlib.colors import TwoSlopeNorm
    rows = [v for v in variants() if v[0] not in ("fedavg", "bulyan")]   # Bulyan: invalid at f = 1
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 6.8),
                             gridspec_kw=dict(width_ratios=[4, 6, 6]))
    fig.subplots_adjust(left=0.13, right=0.995, top=0.94, bottom=0.11, wspace=0.06, hspace=0.42)
    cm = plt.get_cmap("RdBu")
    for j, key in enumerate(DS):
        d, df = DS[key], runs(key)
        t = pd.read_csv(TAB / f"{key}_exclusion_fa1.csv")
        sz = t[(t.aggregator == "fedavg") & ~t.weighted].set_index("hospital").train_size
        sz = sz.reindex(d["clients"])
        idx = order(key)
        names = [f"{d['clients'][i].replace('centre ', 'centre ')} ({100 * sz.iloc[i] / sz.sum():.0f}%)"
                 + ("$^\\dagger$" if i in d.get("ineligible", []) else "") for i in idx]
        inf = influence(key)
        V = np.full((len(rows), len(idx)), np.nan)      # true values (printed)
        D = np.full_like(V, np.nan)
        for i, v in enumerate(rows):
            if v in inf:
                V[i] = inf[v][idx]
            c = clean_stats(df, *v, 1)
            if c is not None:
                D[i] = (c["clients"] - clean_stats(df, "fedavg", v[1], 1)["clients"])[idx]
        # colour: log2 of the influence, floored at 1/16 for the colour scale only
        C = np.log2(np.clip(V, 1 / 16, 16))
        for ax, X, col, norm, fmt in ((axes[0, j], V, C, TwoSlopeNorm(0, -3, 3), "inf"),
                                      (axes[1, j], D, D, TwoSlopeNorm(0, -20, 20), "acc")):
            ax.imshow(col, cmap=cm, norm=norm, aspect="auto")
            for (r, c), x in np.ndenumerate(X):
                if np.isnan(x):
                    lab = "inv."
                    ax.text(c, r, lab, ha="center", va="center", fontsize=5, color=MUTED)
                else:
                    lab = f"{x:.2f}" if fmt == "inf" else f"{x:+.0f}"
                    ax.text(c, r, lab.replace("-", "−"), ha="center", va="center", fontsize=5,
                            color="white" if abs(norm(col[r, c]) - 0.5) > 0.35 else INK)
                if fmt == "inf":
                    WHO_PAYS_LABELS[(key, vlabel(*rows[r]), d["clients"][idx[c]])] = lab
            ax.set_xticks(range(len(idx)), names, fontsize=5.8, rotation=40, ha="right",
                          rotation_mode="anchor")
            ax.set_yticks(range(len(rows)), [vlabel(*v) for v in rows] if j == 0 else [])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, j].set_title(d["name"])
    fig.text(0.005, 0.975, "(a) Relative influence, clean: effective / nominal share "
             "(< 1 = discounted). Clients sorted by training share.", fontsize=7.5)
    fig.text(0.005, 0.505, "(b) Per-client accuracy, clean: change vs FedAvg of the same "
             "weighting (points).", fontsize=7.5)
    for j in range(3):
        axes[1, j].set_title("")
    save_fig(fig, "fig_who_pays")


WHO_PAYS_LABELS = {}


def check_who_pays():
    """Every printed influence label in fig_who_pays equals the cell of
    tab_influence_<ds>.tex for the same rule and client; 'inv.' cells have no
    table row. Raises on any mismatch."""
    import re
    table = {}
    for key in DS:
        lines = (TABS / f"tab_influence_{key}.tex").read_text().splitlines()
        head = next(l for l in lines if l.startswith("Rule &"))
        names = [c.strip().rstrip("\\").strip() for c in head.split("&")[1:]]
        clients = {c.replace("centre ", "Centre "): c for c in DS[key]["clients"]}
        for l in lines:
            if not l.endswith("\\\\") or l.startswith(("Rule &", "\\", "%")):
                continue
            cells = [c.strip() for c in l[:-2].split("&")]
            for n, v in zip(names, cells[1:]):
                table[(key, cells[0], clients[n])] = re.sub(r"\$-\$", "-", v)
    assert WHO_PAYS_LABELS, "fig_who_pays has not been drawn"
    bad = []
    for k, lab in WHO_PAYS_LABELS.items():
        want = table.get(k, "inv.")
        if lab != want:
            bad.append((k, lab, want))
    assert not bad, f"fig_who_pays labels differ from the tables: {bad[:5]}"
    print(f"check: {len(WHO_PAYS_LABELS)} influence labels in fig_who_pays match tables 9-11")


def tab_influence():
    for key in DS:
        inf, d = influence(key), DS[key]
        body = []
        for v in variants():
            if v[0] == "fedavg" or v not in inf:
                continue
            body.append([vlabel(*v)] + [num(inf[v][i], 2) for i in order(key)])
        names = [c.replace("centre ", "Centre ") for c in ordered_clients(key)]
        write_tab(f"tab_influence_{key}", "@{}l" + "r" * len(names) + "@{}",
                  ["Rule & " + " & ".join(names)], body)


# ---------------------------------------------------------------- attacks

def tab_attacked(f, keys, name):
    body = []
    for key in keys:
        s = summary(key, f)
        body.append(("GROUP", f"{DS[key]['name']}, $f={f}$"))
        for v in variants():
            if v not in s:
                continue
            x = s[v]
            da = x["att"]["avg"][0] - x["clean"]["avg"][0]
            body.append([vlabel(*v), pm(x["clean"]["avg"]), pm(x["att"]["avg"]), num(da, 1, True),
                         pm(x["clean_worst"]), pm(x["att"]["worst"]),
                         pm(x["big_delta"]["avg"], sign=True) if x["big_delta"] else "--",
                         pm(x["big_delta"]["worst"], sign=True) if x["big_delta"] else "--"])
        body.append("MID")
    body.pop()
    big = "/".join(f"{DS[k]['clients'][DS[k]['largest']]}".replace("centre ", "centre ")
                   for k in keys)
    write_tab(name, "@{}l ccc cc cc@{}",
              ["& \\multicolumn{3}{c}{Average} & \\multicolumn{2}{c}{Worst honest client} & "
               "\\multicolumn{2}{c}{Largest client attacks: $\\Delta$} \\\\",
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
               "Rule & Clean & Attacked & $\\Delta$ & Clean & Attacked & Avg & Worst"],
              body, note=f"largest client: {big}")


def fig_attacked(f, keys, name):
    fig, axes = plt.subplots(2, len(keys), figsize=(WIDTH if len(keys) == 3 else WIDTH * 0.7, 5.6),
                             sharey=True, squeeze=False)
    rows = [v for v in variants() if any(v in summary(k, f) for k in keys)]
    for j, key in enumerate(keys):
        s = summary(key, f)
        for r, metric in enumerate(("avg", "worst")):
            ax = axes[r, j]
            for i, v in enumerate(rows):
                if v not in s:
                    ax.text(0.02, i, "invalid", transform=ax.get_yaxis_transform(), fontsize=6,
                            color=MUTED, va="center")
                    continue
                x, col = s[v], RULE_COLORS[v[0]]
                c0 = (x["clean"]["avg"] if metric == "avg" else x["clean_worst"])[0]
                a0 = x["att"][metric][0]
                ax.annotate("", xy=(a0, i), xytext=(c0, i),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=2, shrinkB=2,
                                            mutation_scale=6))
                ax.plot(c0, i, "o", mfc="white", mec=col, ms=4)
                ax.errorbar(a0, i, xerr=x["att"][metric][1], fmt="o", color=col, ms=4,
                            mec="white", mew=0.5, elinewidth=0.8)
                if x["big"] and metric == "avg":
                    ax.plot(x["big"][metric][0], i, "x", color=col, ms=4, mew=1)
            style(ax)
            if r == 0:
                ax.set_title(f"{DS[key]['name']} ({lc(DS[key]['metric'])})")
            ax.set_xlabel(("Average" if metric == "avg" else "Worst honest client") + " (points)")
        axes[0, 0].set_yticks(range(len(rows)), [vlabel(*v) for v in rows])
        axes[0, 0].set_ylim(len(rows) - 0.5, -0.5)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4, label="clean"),
         plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4,
                    label=f"mean of real attacks, $f$ = {f}"),
         plt.Line2D([], [], marker="x", ls="", color=MUTED, ms=4,
                    label="attacker set includes the largest client (average only)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save_fig(fig, name)


# ---------------------------------------------------------------- leave-out

LOO_PANELS_F1 = [("heart", "Cleveland"), ("heart", "Hungary"), ("tcga", "Northeast"),
                 ("isic", "centre 0")]


def loo_parts():
    out = {}
    for key in DS:
        p1 = loo_seeds(key, 1)
        for (k, client) in LOO_PANELS_F1:
            if k == key:
                out[(key, 1, client)] = {v: group_decomp(p1, *v, [client]) for v in variants()}
        if 2 in DS[key]["real"]:
            p2 = loo_seeds(key, 2)
            big = DS[key]["clients"][DS[key]["largest"]]
            sets = sorted(set(p2.left_out))
            w_big = [s for s in sets if big in s.split("+")]
            wo_big = [s for s in sets if big not in s.split("+")]
            out[(key, 2, "with " + big)] = {v: group_decomp(p2, *v, w_big) for v in variants()}
            out[(key, 2, "without " + big)] = {v: group_decomp(p2, *v, wo_big) for v in variants()}
    return out


def tab_loo(parts):
    def block(panels, name, rules):
        body = []
        for v in variants(rules):
            row = [vlabel(*v)]
            any_ = False
            for p in panels:
                x = parts[p].get(v)
                if x is None:
                    row += ["--", "--"]
                else:
                    any_ = True
                    row += [pm(x["avg_data_loss"], sign=True), pm(x["avg_attack_damage_no_alie"], sign=True)]
            if any_:
                body.append(row)
        heads = [f"{DS[k]['short']}: {c}" + (" (pairs)" if f == 2 else "") for k, f, c in panels]
        cm = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(panels)))
        write_tab(name, "@{}l" + " cc" * len(panels) + "@{}",
                  ["& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in heads) + " \\\\", cm,
                   "Rule" + " & Data loss & Attack" * len(panels)], body)
    block([(k, 1, c) for k, c in LOO_PANELS_F1], "tab_loo_f1", RULES)
    block([("tcga", 2, "with Northeast"), ("tcga", 2, "without Northeast"),
           ("isic", 2, "with centre 0"), ("isic", 2, "without centre 0")], "tab_loo_f2",
          ["fedavg", "median", "geomed", "cc"])


def tab_loo_appendix():
    """Every left-out client: data loss, attack damage without ALIE (main),
    with all real attacks (appendix column), worst-honest damage, own test set."""
    for key in DS:
        for f in sorted(DS[key]["real"]):
            t = pd.read_csv(TAB / f"loo_{key}_f{f}.csv")
            rank = {c: i for i, c in enumerate(ordered_clients(key))}
            groups = sorted(t.groupby("left_out", sort=False),
                            key=lambda kv: sorted(rank[c] for c in kv[0].split("+")))
            blocks = []
            for lo, g in groups:
                lo = "+".join(sorted(lo.split("+"), key=rank.get))
                body = [("GROUP", f"Left out / attacking: {lo}")]
                g = g.set_index(["rule", "weighted"])
                for v in variants():
                    if v not in g.index:
                        continue
                    r = g.loc[v]
                    body.append([vlabel(*v) + ("$^\\ddagger$" if r.dictator else ""),
                                 pm((r.avg_data_loss, r.avg_data_loss_ci95), sign=True),
                                 pm((r.avg_attack_damage_no_alie, r.avg_attack_damage_no_alie_ci95), sign=True),
                                 pm((r.avg_attack_damage, r.avg_attack_damage_ci95), sign=True),
                                 pm((r.worst_data_loss, r.worst_data_loss_ci95), sign=True),
                                 pm((r.worst_attack_damage_no_alie, r.worst_attack_damage_no_alie_ci95), sign=True)])
                blocks.append(body)
            # split at group boundaries into parts that fit one page
            parts, cur = [], []
            for b in blocks:
                if cur and len(cur) + len(b) > MAX_ROWS:
                    parts.append(cur)
                    cur = []
                cur += b
            parts.append(cur)
            for k, body in enumerate(parts, 1):
                name = f"tab_loo_{key}_f{f}_all" + (f"_{k}" if len(parts) > 1 else "")
                write_tab(name, "@{}l ccc cc@{}",
                          ["& \\multicolumn{3}{c}{Average} & \\multicolumn{2}{c}{Worst honest client} \\\\",
                           "\\cmidrule(lr){2-4}\\cmidrule(l){5-6}",
                           "Rule & Data loss & Attack (no ALIE) & Attack (all) & Data loss & Attack (no ALIE)"],
                          body, note=f"part {k} of {len(parts)}" if len(parts) > 1 else None)
        o = pd.read_csv(TAB / f"loo_{key}_own.csv")
        piv = {(r.rule, r.weighted, r.client): (r.change, r.ci95) for r in o.itertuples()}
        cl = ordered_clients(key)
        body = [[vlabel(*v)] + [pm(piv.get((v[0], v[1], c)), sign=True) for c in cl]
                for v in variants() if any((v[0], v[1], c) in piv for c in cl)]
        write_tab(f"tab_loo_{key}_own", "@{}l" + "c" * len(cl) + "@{}",
                  ["Rule & " + " & ".join(c.replace("centre ", "Centre ") for c in cl)], body)


def fig_loo(parts, f, panels, name, rules):
    rows = variants(rules)
    rows = [v for v in rows if any(parts[p].get(v) for p in panels)]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 0.2 * len(rows) + 1.2), sharey=True)
    for ax, p in zip(axes, panels):
        key, _, client = p
        for i, v in enumerate(rows):
            x = parts[p].get(v)
            if x is None:
                ax.text(0.5, i, "invalid", transform=ax.get_yaxis_transform(), fontsize=6,
                        color=MUTED, va="center", ha="center")
                continue
            dl, dlc = x["avg_data_loss"]
            ad, adc = x["avg_attack_damage_no_alie"]
            ax.barh(i - 0.19, dl, height=0.36, color="#b9b8b3", xerr=dlc,
                    error_kw=dict(elinewidth=0.6, ecolor=MUTED, capsize=0))
            ax.barh(i + 0.19, ad, height=0.36, color=RULE_COLORS[v[0]], alpha=0.95 if v[1] else 0.6,
                    hatch="////" if v[1] else None, edgecolor="white", lw=0, xerr=adc,
                    error_kw=dict(elinewidth=0.6, ecolor=INK, capsize=0))
        ax.axvline(0, color=INK, lw=0.6)
        style(ax)
        lab = client if f == 1 else f"pairs {client}"
        ax.set_title(f"{DS[key]['short']}:\n{lab}" if f == 2 else f"{DS[key]['short']}: {lab}",
                     fontsize=7.5)
        ax.set_xlabel("Points")
    axes[0].set_yticks(range(len(rows)), [vlabel(*v) for v in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    h = [plt.Rectangle((0, 0), 1, 1, color="#b9b8b3", label="data loss (control − clean)"),
         plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.6,
                       label="attack damage, unweighted (attacked − control, without ALIE)"),
         plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch="////", edgecolor="white",
                       label="attack damage, weighted")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08 if f == 1 else -0.12))
    fig.tight_layout()
    save_fig(fig, name)


# ---------------------------------------------------------------- reproductions

def repro_rows(key, fn):
    """Reproduction table rows: pooled first, then clients by training share."""
    t = pd.read_csv(TAB / fn)
    names = ["pooled"] + [c.lower().replace(" ", "_") for c in ordered_clients(key)]
    return t.set_index("test_set").loc[names].reset_index()


def tab_repro():
    body = []
    labels = {"heart": ("Fed-Heart-Disease, accuracy, 5 seeds", "flamby_heart_repro.csv"),
              "tcga": ("Fed-TCGA-BRCA, C-index, 5 seeds", "flamby_tcga_repro.csv"),
              "isic": ("Fed-ISIC2019, balanced accuracy, FLamby's code, 2 seeds (ours) vs 5",
                       "flamby_isic_repro.csv")}
    for key, (lab, fn) in labels.items():
        t = repro_rows(key, fn)
        body.append(("GROUP", lab))
        for r in t.itertuples():
            ours = (f"{r.ours_mean:.3f} ({r.ours_seeds})" if key == "isic"
                    else f"{r.ours_mean:.3f} $\\pm$ {r.ours_std:.3f}")
            body.append([{"va": "VA"}.get(r.test_set, r.test_set.replace("_", " ").capitalize()), ours,
                         f"{r.flamby_mean:.3f} $\\pm$ {r.flamby_std:.3f}",
                         num(r.diff, 3, True), "yes" if r.within_1_std else "no"])
        body.append("MID")
    body.pop()
    write_tab("tab_repro", "@{}l l l r c@{}",
              ["Test set & Ours & FLamby (published) & Difference & Within 1 s.d."], body)


def fig_repro():
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), gridspec_kw=dict(width_ratios=[5, 7, 7]))
    for ax, (key, fn) in zip(axes, (("heart", "flamby_heart_repro.csv"),
                                    ("tcga", "flamby_tcga_repro.csv"),
                                    ("isic", "flamby_isic_repro.csv"))):
        t = repro_rows(key, fn)
        x = np.arange(len(t))
        top = 1.0 if key == "tcga" else np.inf    # a C-index cannot exceed 1

        def bars(mean, sd):
            up = np.minimum(mean + sd, top) - mean
            return np.vstack([sd, up]), mean + sd > top
        yerr, cut = bars(t.flamby_mean.to_numpy(), t.flamby_std.to_numpy())
        ax.errorbar(x + 0.12, t.flamby_mean, yerr=yerr, fmt="s", color=MUTED, ms=3.5,
                    elinewidth=0.8, label="FLamby (published, mean ± s.d.)")
        ax.plot((x + 0.12)[cut], [top] * cut.sum(), "^", color=MUTED, ms=3, clip_on=False)
        if key == "isic":
            for i, s in enumerate(t.ours_seeds):
                ax.plot([i - 0.12] * 2, [float(v) for v in s.split(",")], "_", color=RULE_COLORS["fedavg"],
                        ms=5, label="ours, each seed (ISIC: 2 seeds)" if i == 0 else None)
            ax.plot(x - 0.12, t.ours_mean, "o", color=RULE_COLORS["fedavg"], ms=3.5)
        else:
            yerr, cut = bars(t.ours_mean.to_numpy(), t.ours_std.to_numpy())
            ax.errorbar(x - 0.12, t.ours_mean, yerr=yerr, fmt="o", color=RULE_COLORS["fedavg"],
                        ms=3.5, elinewidth=0.8, label="ours (mean ± s.d.)")
            ax.plot((x - 0.12)[cut], [top] * cut.sum(), "^", color=RULE_COLORS["fedavg"], ms=3,
                    clip_on=False)
        if key == "tcga":
            ax.set_ylim(top=1.0)
            ax.text(1.0, 1.02, "▲ s.d. bar cut at 1", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, color=MUTED)
        names = {"va": "VA", "pooled": "Pooled"}
        ax.set_xticks(x, [names.get(s, s.replace("centre_", "c").capitalize()) for s in t.test_set],
                      rotation=45, ha="right", fontsize=6.5)
        ax.set_title(DS[key]["name"])
        ax.set_ylabel(DS[key]["metric"])
        style(ax, xgrid=False)
    h = axes[0].get_legend_handles_labels()
    h2 = axes[2].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.995, top=0.9, bottom=0.36, wspace=0.32)
    fig.legend(h[0] + h2[0][:1], h[1] + h2[1][:1], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, "fig_repro")


def tab_isic_leak():
    t = pd.read_csv(TAB / "isic_noleak_check.csv")
    body = []
    for v in variants():
        g = t[(t.aggregator == v[0]) & (t.weighted == v[1])].set_index("f_assumed")
        if g.empty:
            continue
        body.append([vlabel(*v)] + [f"{num(g.loc[fa].d_avg_mean, 2, True)} / {num(g.loc[fa].d_worst_mean, 2, True)}"
                                    if fa in g.index else "--" for fa in (0, 1, 2)])
    write_tab("tab_isic_leak", "@{}l ccc@{}",
              ["Rule & $f_\\text{assumed}=0$ & $f_\\text{assumed}=1$ & $f_\\text{assumed}=2$"], body)


def tab_mnist():
    t = pd.read_csv(TAB / "summary_results_pooled.csv")
    t = t[t.dataset == "mnist"]
    attacks = ["none", "label_flip", "sign_flip", "gaussian", "alie", "ipm"]
    names = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip",
             "gaussian": "Gaussian", "alie": "ALIE", "ipm": "IPM"}
    body = []
    for rule in RULES:
        g = t[t.aggregator == rule].set_index("attack")
        body.append([RULE_LABELS[rule]] + [pm((100 * g.loc[a].final_acc_mean, 100 * g.loc[a].final_acc_ci95))
                                           for a in attacks])
    write_tab("tab_mnist", "@{}l" + "c" * len(attacks) + "@{}",
              ["Rule & " + " & ".join(names[a] for a in attacks)], body)


FIG_ORDER = ["fig_validity", "fig_clean", "fig_who_pays", "fig_f1", "fig_f2", "fig_loo_f1",
             "fig_loo_f2", "fig_repro"]
TAB_ORDER = ["tab_settings", "tab_datasets", "tab_validity", "tab_clean", "tab_f1", "tab_f2", "tab_loo_f1",
             "tab_loo_f2", "tab_repro", "tab_influence_heart", "tab_influence_tcga",
             "tab_influence_isic", "tab_loo_heart_f1_all", "tab_loo_tcga_f1_all",
             "tab_loo_isic_f1_all", "tab_loo_tcga_f2_all", "tab_loo_isic_f2_all",
             "tab_loo_heart_own", "tab_loo_tcga_own", "tab_loo_isic_own", "tab_isic_leak",
             "tab_mnist"]


def write_preview():
    """paper/preview.tex: every figure and table (split tables part by part)
    and the acknowledgments, to check the layout on Overleaf."""
    tabs = []
    for t in TAB_ORDER:
        tabs += [t] if (TABS / f"{t}.tex").exists() else \
            sorted((q.stem for q in TABS.glob(f"{t}_[0-9].tex")), key=lambda x: int(x.rsplit("_", 1)[1]))
    made = {q.stem for q in TABS.glob("*.tex")}
    assert set(tabs) == made, f"tables missing from the preview list: {made - set(tabs)}"
    esc = lambda x: x.replace("_", "\\_")
    L = [r"% Preview of every generated figure and table (generated by make_paper.py;",
         r"% compile on Overleaf after uploading paper/).",
         r"\documentclass[10pt]{article}", r"\usepackage[margin=2cm]{geometry}",
         r"\usepackage{booktabs,amssymb,amsmath,graphicx}", r"\begin{document}"]
    for f in FIG_ORDER:
        L += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=\linewidth]{{figures/{f}.pdf}}",
              rf"\caption{{\texttt{{{esc(f)}}}}}", r"\end{figure}"]
    for t in tabs:
        L += [r"\begin{table}[p]\centering\small", rf"\caption{{\texttt{{{esc(t)}}}}}",
              rf"\resizebox{{\linewidth}}{{!}}{{\input{{tables/{t}}}}}", r"\end{table}"]
    L += [r"\clearpage", r"\input{acknowledgments}", r"\end{document}"]
    (RT / "preview.tex").write_text("\n".join(L) + "\n")
    return tabs


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABS.mkdir(parents=True, exist_ok=True)
    for q in list(FIGS.glob("*")) + list(TABS.glob("*.tex")):   # no stale outputs
        q.unlink()
    tab_settings()
    tab_datasets()
    tab_validity()
    fig_validity()
    tab_clean()
    fig_clean()
    fig_who_pays()
    tab_influence()
    tab_attacked(1, list(DS), "tab_f1")
    fig_attacked(1, list(DS), "fig_f1")
    tab_attacked(2, ["tcga", "isic"], "tab_f2")
    fig_attacked(2, ["tcga", "isic"], "fig_f2")
    parts = loo_parts()
    tab_loo(parts)
    tab_loo_appendix()
    fig_loo(parts, 1, [(k, 1, c) for k, c in LOO_PANELS_F1], "fig_loo_f1", RULES)
    fig_loo(parts, 2, [("tcga", 2, "with Northeast"), ("tcga", 2, "without Northeast"),
                       ("isic", 2, "with centre 0"), ("isic", 2, "without centre 0")],
            "fig_loo_f2", ["fedavg", "median", "geomed", "cc"])
    tab_repro()
    fig_repro()
    tab_isic_leak()
    tab_mnist()
    check_who_pays()
    tabs = write_preview()
    print("wrote", len(list(FIGS.glob("*.pdf"))), "figures and", len(tabs), "tables (preview.tex)")


if __name__ == "__main__":
    main()
