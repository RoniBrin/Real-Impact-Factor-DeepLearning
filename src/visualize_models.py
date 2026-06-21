"""
visualize_models.py - Model comparison visualizations (Group 5).
Compares GraphSAGE, VGAE, and Node2Vec results.
Run after all three pipelines have completed.
"""

import pandas as pd
import matplotlib.pyplot as plt
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

TITLE_SIZE      = 20
SUPTITLE_SIZE   = 22
SUBPLOT_TITLE   = 18
LABEL_SIZE      = 16
TICK_SIZE       = 13
LEGEND_SIZE     = 13

RESULTS_GRAPHSAGE = "/content/drive/MyDrive/RIF/rif_results_v2.csv"
RESULTS_VGAE      = "/content/drive/MyDrive/RIF/rif_results_vgae.csv"
RESULTS_NODE2VEC  = "/content/drive/MyDrive/RIF/rif_results_node2vec.csv"
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "../results/figures_models")


def load_all():
    df_gs  = pd.read_csv(RESULTS_GRAPHSAGE)
    df_vg  = pd.read_csv(RESULTS_VGAE)
    df_n2v = pd.read_csv(RESULTS_NODE2VEC)

    df_gs["model"]  = "GraphSAGE"
    df_vg["model"]  = "VGAE"
    df_n2v["model"] = "Node2Vec"

    df = pd.concat([df_gs, df_vg, df_n2v], ignore_index=True)
    print(f"Loaded: GraphSAGE {len(df_gs)} | VGAE {len(df_vg)} | Node2Vec {len(df_n2v)} rows")
    return df, df_gs, df_vg, df_n2v


def plot_avg_stability_by_model(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    years  = sorted(df["year"].unique())
    models = ["GraphSAGE", "VGAE", "Node2Vec"]
    x      = np.arange(len(years))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            df[(df["model"] == model) & (df["year"] == y)]["weighted_rif"].mean()
            for y in years
        ]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(years, fontsize=TICK_SIZE)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Average Weighted RIF", fontsize=LABEL_SIZE)
    ax.set_title("Average Weighted RIF per Model and Year", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "8_avg_stability_by_model.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


def plot_rif_correlation(df_gs, df_vg, df_n2v, year=2020):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merge_cols = ["journal", "year", "weighted_rif"]

    df_year_gs  = df_gs[df_gs["year"] == year][merge_cols].rename(
        columns={"weighted_rif": "rif_graphsage"})
    df_year_vg  = df_vg[df_vg["year"] == year][merge_cols].rename(
        columns={"weighted_rif": "rif_vgae"})
    df_year_n2v = df_n2v[df_n2v["year"] == year][merge_cols].rename(
        columns={"weighted_rif": "rif_node2vec"})

    df_merged = df_year_gs.merge(df_year_vg,  on="journal") \
                           .merge(df_year_n2v, on="journal")
    df_merged = df_merged.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, [
        ("rif_graphsage", "rif_vgae",     "GraphSAGE RIF", "VGAE RIF"),
        ("rif_graphsage", "rif_node2vec", "GraphSAGE RIF", "Node2Vec RIF"),
    ]):
        ax.scatter(df_merged[xcol], df_merged[ycol],
                   alpha=0.5, color="#4C72B0", s=15)
        max_val = max(df_merged[xcol].max(), df_merged[ycol].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], color="red",
                linestyle="--", linewidth=1.2, label="Perfect agreement")
        corr = df_merged[[xcol, ycol]].corr().iloc[0, 1]
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(f"{xlabel} vs {ylabel}\n(r = {corr:.3f})", fontsize=SUBPLOT_TITLE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle(f"How Well Do the Models Agree? RIF Correlation ({year})", fontsize=SUPTITLE_SIZE)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"9_rif_correlation_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


def plot_rif_reduction_by_model(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()
    df["reduction_pct"] = (
        (df["baseline_if"] - df["weighted_rif"])
        / df["baseline_if"].replace(0, float("nan"))
        * 100
    ).fillna(0)

    years  = sorted(df["year"].unique())
    models = ["GraphSAGE", "VGAE", "Node2Vec"]
    x      = np.arange(len(years))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            df[(df["model"] == model) & (df["year"] == y)]["reduction_pct"].mean()
            for y in years
        ]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(years, fontsize=TICK_SIZE)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Average IF Reduction (%)", fontsize=LABEL_SIZE)
    ax.set_title("How Much Does Each Model Reduce IF? (2018-2022)", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "10_rif_reduction_by_model.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


def plot_if_vs_rif_per_model(df_gs, df_vg, df_n2v, year=2020):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (df_m, model, color) in zip(axes, [
        (df_gs,  "GraphSAGE", "#4C72B0"),
        (df_vg,  "VGAE",      "#DD8452"),
        (df_n2v, "Node2Vec",  "#55A868"),
    ]):
        df_year = df_m[df_m["year"] == year].copy()
        df_year = df_year[df_year["baseline_if"] > 0]

        ax.scatter(
            df_year["baseline_if"], df_year["weighted_rif"],
            alpha=0.5, color=color, s=15
        )

        max_val = max(df_year["baseline_if"].max(),
                      df_year["weighted_rif"].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], color="red",
                linestyle="--", linewidth=1.2, label="IF = RIF")

        ax.set_xlabel("Baseline IF", fontsize=LABEL_SIZE)
        ax.set_ylabel("Weighted RIF", fontsize=LABEL_SIZE)
        ax.set_title(f"{model} ({year})", fontsize=SUBPLOT_TITLE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle(f"IF vs Weighted RIF - Comparing All Three Models ({year})", fontsize=SUPTITLE_SIZE)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"11_if_vs_rif_all_models_{year}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, df_gs, df_vg, df_n2v = load_all()

    plot_avg_stability_by_model(df)
    plot_rif_correlation(df_gs, df_vg, df_n2v, year=2020)
    plot_rif_reduction_by_model(df)
    plot_if_vs_rif_per_model(df_gs, df_vg, df_n2v, year=2020)

    print("\nAll model comparison figures saved to:", OUTPUT_DIR)