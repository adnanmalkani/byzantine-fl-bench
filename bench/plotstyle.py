"""Shared figure style: one fixed colour per rule (the same in every figure),
recessive axes and grid. Colours: a validated categorical palette, assigned in
the fixed order of AGGREGATORS (never by rank)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

RULE_COLORS = {
    "fedavg": "#2a78d6", "median": "#eb6834", "trimmed_mean": "#1baf7a",
    "krum": "#eda100", "multi_krum": "#e87ba4", "bulyan": "#008300",
    "geomed": "#4a3aa7", "cc": "#e34948",
}
RULE_LABELS = {
    "fedavg": "FedAvg", "median": "Median", "trimmed_mean": "Trimmed mean",
    "krum": "Krum", "multi_krum": "Multi-Krum", "bulyan": "Bulyan",
    "geomed": "GeoMed", "cc": "CC",
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, path_stem):
    """Write <stem>.png (200 dpi) and <stem>.pdf."""
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
