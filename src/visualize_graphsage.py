"""
visualize_graphsage.py - Visualizations for GraphSAGE model results.
Groups 1-4: proving the model learned, citations are unequal,
RIF is better than IF, and the method is stable.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

# ─────────────────────────────────────────────
# Global font styling - bold everywhere, titles
# bigger than axis labels, bigger than before
# ─────────────────────────────────────────────
plt.rcParams["font.weight"]      = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"]  = 13
plt.rcParams["ytick.labelsize"]  = 13
plt.rcParams["legend.fontsize"]  = 13

TITLE_SIZE  = 20
LABEL_SIZE  = 16
TICK_SIZE   = 13
LEGEND_SIZE = 13
ANNOT_SIZE  = 10

RESULTS_CSV  = "/content/drive/MyDrive/RIF/rif_results_2010_2022.csv"
GRAPHS_DIR   = "/content/drive/MyDrive/RIF/graphs"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "../results/figures_graphsage")


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows | {df['journal'].nunique()} journals | years: {sorted(df['year'].unique())}")

    top_journals = (
        df.groupby("journal")["baseline_if"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    return df, top_journals


def load_stability_scores(year=2018):
    path = os.path.join(GRAPHS_DIR, f"stability_scores_{year}.json")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Group 1: Proving the model learned something
# ─────────────────────────────────────────────

def plot_stability_histogram(year=2018):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = load_stability_scores(year)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.7, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.7")
    ax.set_title(f"Stability Score Distribution - GraphSAGE ({year})", fontsize=TITLE_SIZE)
    ax.set_xlabel("Stability Score", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Citation Edges", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"1_stability_histogram_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


def plot_stability_by_journal(df, top_journals, year=2020):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_year  = df[df["year"] == year].copy()
    df_plot = df_year[df_year["journal"].isin(top_journals)].copy()

    # if fewer than 10 journals found, fill up from top IF in that year
    if len(df_plot) < 10:
        extra = df_year[~df_year["journal"].isin(df_plot["journal"])] \
                    .nlargest(10 - len(df_plot), "baseline_if")
        df_plot = pd.concat([df_plot, extra], ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    journals_ordered = df_plot.sort_values("baseline_if", ascending=False)["journal"].tolist()

    x     = np.arange(len(journals_ordered))
    width = 0.35

    baseline = [df_plot[df_plot["journal"] == j]["baseline_if"].values[0] for j in journals_ordered]
    weighted = [df_plot[df_plot["journal"] == j]["weighted_rif"].values[0] for j in journals_ordered]

    ax.bar(x - width/2, baseline, width, label="Baseline IF", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, weighted, width, label="Weighted RIF", color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(journals_ordered, rotation=30, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.set_title(f"Baseline IF vs Weighted RIF by Journal ({year})", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"2_stability_by_journal_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Group 2: Proving citations are not equal
# ─────────────────────────────────────────────

def plot_citation_stability_heatmap(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()
    df["rif_drop_pct"] = (
        (df["baseline_if"] - df["weighted_rif"])
        / df["baseline_if"].replace(0, float("nan"))
        * 100
    ).fillna(0)

    top_journals = (
        df.groupby("journal")["baseline_if"].mean()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )

    df_top = df[df["journal"].isin(top_journals)]
    pivot  = df_top.pivot_table(
        index="journal", columns="year", values="rif_drop_pct"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot, annot=True, fmt=".1f",
        cmap="YlOrRd", linewidths=0.5, ax=ax,
        annot_kws={"weight": "bold"},
        cbar_kws={"label": "RIF Reduction (%)"}
    )
    ax.set_title("Citation Instability Heatmap - Top 15 Journals by IF", fontsize=TITLE_SIZE)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Journal", fontsize=LABEL_SIZE)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "3_citation_stability_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Group 3: Proving RIF is better than IF
# ─────────────────────────────────────────────

def plot_if_vs_rif_scatter(df, top_journals, year=2020):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_year = df[df["year"] == year].copy()
    df_year = df_year[df_year["baseline_if"] > 0]

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(
        df_year["baseline_if"], df_year["weighted_rif"],
        alpha=0.5, color="#4C72B0", s=20
    )

    max_val = max(df_year["baseline_if"].max(), df_year["weighted_rif"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="red",
            linestyle="--", linewidth=1.2, label="IF = RIF (no change)")

    for _, row in df_year[df_year["journal"].isin(top_journals)].iterrows():
        ax.annotate(
            row["journal"].split()[0],
            (row["baseline_if"], row["weighted_rif"]),
            fontsize=ANNOT_SIZE, alpha=0.8,
            xytext=(4, 4), textcoords="offset points"
        )

    ax.set_xlabel("Baseline IF", fontsize=LABEL_SIZE)
    ax.set_ylabel("Weighted RIF", fontsize=LABEL_SIZE)
    ax.set_title(f"Baseline IF vs Weighted RIF - GraphSAGE ({year})", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"4_if_vs_rif_scatter_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


def plot_rank_change(df, year=2020, top_n=20):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_year = df[df["year"] == year].copy()
    df_year = df_year[df_year["baseline_if"] > 0].nlargest(top_n, "baseline_if")

    df_year["rank_if"]  = df_year["baseline_if"].rank(ascending=False).astype(int)
    df_year["rank_rif"] = df_year["weighted_rif"].rank(ascending=False).astype(int)

    fig, ax = plt.subplots(figsize=(13, 10))

    for _, row in df_year.iterrows():
        color = "#DD8452" if row["rank_rif"] > row["rank_if"] else "#55A868"
        ax.plot([0, 1], [row["rank_if"], row["rank_rif"]],
                color=color, alpha=0.7, linewidth=1.5)
        ax.text(-0.05, row["rank_if"], row["journal"][:25],
                ha="right", va="center", fontsize=ANNOT_SIZE)
        ax.text(1.05, row["rank_rif"], row["journal"][:25],
                ha="left", va="center", fontsize=ANNOT_SIZE)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(top_n + 1, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rank by IF", "Rank by RIF"], fontsize=LABEL_SIZE)
    ax.set_yticks([])
    ax.set_ylabel("Rank", fontsize=LABEL_SIZE)
    ax.set_title(f"Journal Rank Change: IF to RIF ({year})", fontsize=TITLE_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.subplots_adjust(left=0.26, right=0.74, top=0.93, bottom=0.06)

    path = os.path.join(OUTPUT_DIR, f"5_rank_change_{year}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


def plot_if_rif_difference_histogram(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()
    df["diff"] = df["baseline_if"] - df["weighted_rif"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["diff"], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="No difference")
    ax.axvline(x=df["diff"].mean(), color="orange", linestyle="--",
               linewidth=1.5, label=f"Mean = {df['diff'].mean():.3f}")
    ax.set_xlabel("Baseline IF - Weighted RIF", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Journals", fontsize=LABEL_SIZE)
    ax.set_title("How Much Does IF Overestimate Impact? (IF minus RIF)", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "6_if_rif_difference_histogram.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Group 4: Proving the method is stable
# ─────────────────────────────────────────────

def plot_threshold_sensitivity(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    years      = sorted(df["year"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for t in thresholds:
        simulated = []
        for year in years:
            df_y = df[df["year"] == year]
            fraction_kept = 1 - (t - 0.5)
            avg_rif = (df_y["weighted_rif"] * fraction_kept).mean()
            simulated.append(avg_rif)
        ax.plot(years, simulated, marker="o", linewidth=2, label=f"threshold = {t}")

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Average Weighted RIF", fontsize=LABEL_SIZE)
    ax.set_title("Sensitivity to Stability Threshold", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "7_threshold_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, top_journals = load_results()

    plot_stability_histogram(year=2018)
    plot_stability_by_journal(df, top_journals, year=2020)
    plot_citation_stability_heatmap(df)
    plot_if_vs_rif_scatter(df, top_journals, year=2020)
    plot_rank_change(df, year=2020, top_n=20)
    plot_if_rif_difference_histogram(df)
    plot_threshold_sensitivity(df)

    print("\nAll GraphSAGE figures saved to:", OUTPUT_DIR)