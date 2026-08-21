"""Render the three publication figures for the OMTR paper.

Reads results/analysis_<tag>.json and writes results/figures/*.png at 300 dpi.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"

INDIGO = "#3E4DA8"
AMBER = "#A66908"
INK = "#1B2130"
MUTED = "#5B6274"
SURFACE = "#FFFFFF"

# display name, analysis file tag, structural ceiling (n_layers-1)/n_layers, color
MODELS = [
    ("Pythia-410M", "EleutherAI_pythia-410m", 23 / 24, INDIGO),
    ("Pythia-1B", "EleutherAI_pythia-1b", 15 / 16, INDIGO),
    ("Pythia-1.4B", "EleutherAI_pythia-1.4b", 23 / 24, INDIGO),
    ("Pythia-2.8B", "EleutherAI_pythia-2.8b", 31 / 32, INDIGO),
    ("OLMo-2-1B", "allenai_OLMo-2-0425-1B", 15 / 16, AMBER),
]

LEVELS = ["L0", "L0N", "L1", "L2", "L3", "L4", "L5"]

# Pythia parameter counts in billions, matching MODELS order
PYTHIA_PARAMS = {"Pythia-410M": 0.41, "Pythia-1B": 1.0, "Pythia-1.4B": 1.4, "Pythia-2.8B": 2.8}

MINUS = "−"  # typographic minus


def load(tag):
    with open(RESULTS / f"analysis_{tag}.json", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_signed(v, places=2):
    """Format with a typographic minus rather than a hyphen."""
    return f"{v:.{places}f}".replace("-", MINUS)


def fmt_p(p):
    """Two significant digits, switching to scientific notation below 0.001."""
    if p < 1e-3:
        exponent = 0
        mantissa = p
        while mantissa < 1:
            mantissa *= 10
            exponent += 1
        sup = str(exponent).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{mantissa:.1f}×10⁻{sup}"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig1():
    data = {tag: load(tag) for _, tag, _, _ in MODELS}

    fig, axes = plt.subplots(1, 5, figsize=(13.0, 3.7), sharex=True, sharey=True)

    for ax, (name, tag, ceiling, color) in zip(axes, MODELS):
        d = data[tag]
        pts = [q for q in d["dose_response"]["points"] if q["level"] == "L0"]
        xs = [q["gold"] for q in pts]
        ys = [q["depth"] for q in pts]
        stat = d["dose_response"]["within_condition"]["L0"]

        despine(ax)
        ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.7, linestyle="-")
        ax.set_axisbelow(True)

        # structural ceiling: depth cannot exceed (n_layers-1)/n_layers
        ax.axhline(ceiling, color=MUTED, alpha=0.75, linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)

        ax.scatter(xs, ys, s=60, color=color, alpha=0.85, linewidths=0.9,
                   edgecolors=SURFACE, zorder=3, clip_on=False)

        # stats ride the header rather than a data corner: the point clouds reach
        # into every corner of at least one panel, and a header line also lets the
        # five correlations be read down the row
        ax.set_title(name, color=INK, pad=20)
        ax.annotate(
            f"ρ = {fmt_signed(stat['rho'])}  ·  p = {fmt_p(stat['p'])}",
            xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 6),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8.5, color=MUTED,
        )

    # honesty marker named once, in the first panel only
    axes[0].annotate(
        "ceiling", xy=(0.97, MODELS[0][2] + 0.006), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=8.5, color=MUTED,
    )

    axes[0].set_ylabel("Depth τ = 0.1", color=INK, labelpad=8)
    axes[0].set_ylim(0.655, 1.012)
    axes[0].set_xlim(-3.9, 0.4)
    axes[0].set_xticks([-3, -2, -1, 0])
    axes[0].set_yticks([0.7, 0.8, 0.9, 1.0])

    fig.supxlabel("Gold log-probability per token", color=INK, fontsize=10, y=0.04)
    fig.subplots_adjust(left=0.058, right=0.988, top=0.83, bottom=0.185, wspace=0.16)
    fig.savefig(FIGDIR / "fig1_dose_response.png", dpi=300)
    plt.close(fig)


def fig2():
    series = [
        ("Pythia-1.4B", "EleutherAI_pythia-1.4b", INDIGO),
        ("OLMo-2-1B", "allenai_OLMo-2-0425-1B", AMBER),
    ]
    vals = {}
    for name, tag, _ in series:
        d = load(tag)
        vals[name] = [d["levels"][lv]["final_entropy_mean"]["mean"] for lv in LEVELS]

    # design order runs top-to-bottom, so L0 gets the highest y position
    ypos = {lv: len(LEVELS) - 1 - i for i, lv in enumerate(LEVELS)}
    # the two series sit within a nats of each other at L3 and L4, so dodge them
    # off the row centre rather than let one dot hide the other
    DODGE = 0.20
    yoff = {"Pythia-1.4B": DODGE, "OLMo-2-1B": -DODGE}

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    despine(ax)
    ax.grid(axis="x", color=MUTED, alpha=0.18, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    # banding keeps each dodged pair reading as one condition
    for lv in LEVELS[::2]:
        ax.axhspan(ypos[lv] - 0.5, ypos[lv] + 0.5, color=MUTED, alpha=0.055,
                   linewidth=0, zorder=0)

    for i, lv in enumerate(LEVELS):
        a, b = vals["Pythia-1.4B"][i], vals["OLMo-2-1B"][i]
        ax.plot([a, b], [ypos[lv] + DODGE, ypos[lv] - DODGE], color=MUTED, alpha=0.30,
                linewidth=1.4, zorder=2)

    for name, _, color in series:
        ax.scatter(vals[name], [ypos[lv] + yoff[name] for lv in LEVELS], s=110,
                   color=color, alpha=0.85, linewidths=1.4, edgecolors=SURFACE,
                   zorder=3, label=name)

    # direct value labels on the extreme conditions only, placed outward from each pair
    for lv in ("L0", "L5"):
        i = LEVELS.index(lv)
        pair = sorted(
            [(vals["Pythia-1.4B"][i], "Pythia-1.4B"), (vals["OLMo-2-1B"][i], "OLMo-2-1B")],
            key=lambda t: t[0],
        )
        (lo, lo_name), (hi, hi_name) = pair
        ax.annotate(f"{lo:.2f}", xy=(lo, ypos[lv] + yoff[lo_name]), xytext=(-11, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=INK)
        ax.annotate(f"{hi:.2f}", xy=(hi, ypos[lv] + yoff[hi_name]), xytext=(11, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK)

    ax.set_yticks([ypos[lv] for lv in LEVELS])
    ax.set_yticklabels(LEVELS)
    ax.set_ylim(-0.62, len(LEVELS) - 0.38)
    ax.set_xlim(0.6, 5.05)
    ax.set_xlabel("Mean final-layer entropy (nats)", color=INK, labelpad=8)
    ax.set_ylabel("Condition", color=INK, labelpad=8)
    ax.tick_params(axis="y", length=0)

    # sits just above the plot so it never reads as belonging to the L0 band
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9.5,
              labelcolor=INK, handletextpad=0.4, columnspacing=1.6,
              borderaxespad=0.35, scatterpoints=1)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_entropy_ladder.png", dpi=300)
    plt.close(fig)


def fig3():
    xs, ys, names = [], [], []
    for name, tag, _, _ in MODELS[:4]:
        xs.append(PYTHIA_PARAMS[name])
        ys.append(load(tag)["levels"]["L0"]["gold_logprob"]["mean"])
        names.append(name)
    olmo_y = load("allenai_OLMo-2-0425-1B")["levels"]["L0"]["gold_logprob"]["mean"]
    olmo_x = 1.0

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    despine(ax)
    ax.set_xscale("log")
    ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    ax.plot(xs, ys, color=INDIGO, linewidth=2.0, solid_capstyle="round",
            solid_joinstyle="round", zorder=2, label="Pythia")
    # markers sit on top of the line, so full opacity keeps them the same
    # value as the stroke instead of reading as a lighter halo
    ax.scatter(xs, ys, s=80, color=INDIGO, linewidths=1.4,
               edgecolors=SURFACE, zorder=3)
    ax.scatter([olmo_x], [olmo_y], s=140, marker="D", color=AMBER,
               linewidths=1.4, edgecolors=SURFACE, zorder=3, label="OLMo-2")

    # five values total, so annotating every point stays readable
    offsets = {"Pythia-410M": (0, 11), "Pythia-1B": (0, -14),
               "Pythia-1.4B": (0, 11), "Pythia-2.8B": (0, 11)}
    for x, y, name in zip(xs, ys, names):
        dx, dy = offsets[name]
        ax.annotate(fmt_signed(y), xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top", fontsize=9, color=INK)

    ax.annotate(fmt_signed(olmo_y), xy=(olmo_x, olmo_y), xytext=(14, -4),
                textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
    ax.annotate("OLMo-2-1B\n(different corpus)", xy=(olmo_x, olmo_y), xytext=(0, 14),
                textcoords="offset points", ha="center", va="bottom", fontsize=9,
                color=INK, linespacing=1.4)

    ax.set_xticks([0.41, 1.0, 1.4, 2.8])
    ax.set_xticklabels(["0.41B", "1.0B", "1.4B", "2.8B"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.33, 3.6)
    ax.set_ylim(-1.92, -0.28)
    ax.set_yticks([-1.8, -1.5, -1.2, -0.9, -0.6, -0.3])
    ax.set_yticklabels([fmt_signed(v, 1) for v in [-1.8, -1.5, -1.2, -0.9, -0.6, -0.3]])
    ax.set_xlabel("Parameter count", color=INK, labelpad=8)
    ax.set_ylabel("Mean L0 gold log-probability per token", color=INK, labelpad=8)

    handles = [
        # no surface ring on the key itself, or it breaks the line and reads as dashed
        Line2D([0], [0], color=INDIGO, linewidth=2.0, marker="o", markersize=7,
               markeredgecolor=INDIGO, markeredgewidth=0, label="Pythia"),
        Line2D([0], [0], color="none", marker="D", markersize=8, markerfacecolor=AMBER,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label="OLMo-2"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5, labelcolor=INK,
              handletextpad=0.5, borderaxespad=0.4)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_memorization_scaling.png", dpi=300)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    style()
    fig1()
    fig2()
    fig3()
    print("wrote:", *(str(p) for p in sorted(FIGDIR.glob("*.png"))), sep="\n  ")


if __name__ == "__main__":
    main()
