"""
visualize.py - Generates visualizations for RIF analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), "../results/rif_results.csv"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "../results/figures"
)

TOP_JOURNALS = [
    "New England Journal of Medicine",
    "Nature",
    "The Lancet",
    "JAMA",
    "Nature Medicine",
]


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    return df


# ─────────────────────────────────────────────
# Graph 1: Stability Score Histogram (2018)
# ─────────────────────────────────────────────

def plot_stability_histogram(stability_scores, year=2018):
    """
    Plots histogram of stability scores for a representative year.
    stability_scores: list of float values between 0 and 1.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(stability_scores, bins=20, color="#4C72B0",
            edgecolor="white", linewidth=0.6)

    ax.axvline(x=0.5, color="red", linestyle="--",
               linewidth=1.5, label="Threshold = 0.5")

    ax.set_title(f"Stability Score Distribution ({year})", fontsize=14)
    ax.set_xlabel("Stability Score", fontsize=12)
    ax.set_ylabel("Number of Edges", fontsize=12)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"stability_histogram_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Graph 2: IF vs RIF Line Chart (top journals)
# ─────────────────────────────────────────────

def plot_if_vs_rif(df, journals=TOP_JOURNALS):
    """
    Plots Baseline IF, Filtered RIF, and Weighted RIF over years
    for a selected set of journals.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_filtered = df[df["journal"].isin(journals)]

    fig, axes = plt.subplots(
        len(journals), 1,
        figsize=(10, 4 * len(journals)),
        sharex=True
    )

    if len(journals) == 1:
        axes = [axes]

    colors = {
        "baseline_if":  "#4C72B0",
        "filtered_rif": "#DD8452",
        "weighted_rif": "#55A868",
    }
    labels = {
        "baseline_if":  "Baseline IF",
        "filtered_rif": "Filtered RIF",
        "weighted_rif": "Weighted RIF",
    }

    for ax, journal in zip(axes, journals):
        df_j = df_filtered[df_filtered["journal"] == journal].sort_values("year")

        if df_j.empty:
            ax.set_title(f"{journal} (no data)", fontsize=11)
            continue

        for col, color in colors.items():
            ax.plot(
                df_j["year"], df_j[col],
                marker="o", color=color,
                linewidth=2, label=labels[col]
            )

        ax.set_title(journal, fontsize=12)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Year", fontsize=12)
    fig.suptitle("Baseline IF vs Filtered RIF vs Weighted RIF", fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "if_vs_rif_top_journals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Graph 3: Heatmap — RIF reduction rate
# ─────────────────────────────────────────────

def plot_heatmap(df, top_n=10):
    """
    Plots a heatmap of RIF reduction rate per journal per year.
    Reduction rate = (baseline_if - filtered_rif) / baseline_if * 100
    Higher value = more citations filtered = less stable journal.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()
    df["reduction_rate"] = (
        (df["baseline_if"] - df["filtered_rif"])
        / df["baseline_if"].replace(0, float("nan"))
        * 100
    ).round(2)

    # pick top_n journals by average reduction rate across years
    top_journals = (
        df.groupby("journal")["reduction_rate"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    df_top  = df[df["journal"].isin(top_journals)]
    pivot   = df_top.pivot_table(
        index="journal", columns="year", values="reduction_rate"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Reduction Rate (%)"}
    )

    ax.set_title(
        f"RIF Reduction Rate (%) — Top {top_n} Journals by Instability",
        fontsize=13
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Journal", fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "heatmap_reduction_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_results()
    print(f"Loaded {len(df)} rows, years: {sorted(df['year'].unique())}")
    print(f"Journals: {df['journal'].nunique()}")

    # Graph 2 and 3 — from CSV
    plot_if_vs_rif(df)
    plot_heatmap(df, top_n=10)

    print("\nAll figures saved to:", OUTPUT_DIR)