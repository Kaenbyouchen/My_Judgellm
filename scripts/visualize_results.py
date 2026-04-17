#!/usr/bin/env python3
"""Generate visualization charts from results_summary.csv."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "results_summary.csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

MODEL_SHORT = {
    "gpt52":                      "GPT-5.2",
    "vllmserve_biomistral_7b":    "BioMistral-7B",
    "vllmserve_medgemma_4b":      "MedGemma-4B",
    "vllmserve_prometheus2_7b":   "Prometheus2-7B",
}

MODEL_ORDER = ["GPT-5.2", "BioMistral-7B", "MedGemma-4B", "Prometheus2-7B"]

DATASET_SHORT = {
    "counselbench":         "CounselBench",
    "medaesqa":             "MedAESQA",
    "medical_eval_sphere":  "MedEvalSphere",
    "mediq_askdocs":        "MedIQ-AskDocs",
    "medval_bench":         "MedValBench",
}
DATASET_ORDER = ["CounselBench", "MedAESQA", "MedEvalSphere", "MedIQ-AskDocs", "MedValBench"]

BIAS_SHORT = {
    "authority_style":      "Authority\nStyle",
    "clinical_formatting":  "Clinical\nFormatting",
    "empathy_tone":         "Empathy\nTone",
    "fake_citation":        "Fake\nCitation",
    "jargon_overloading":   "Jargon\nOverloading",
    "language_fluency":     "Language\nFluency",
    "plain_language":       "Plain\nLanguage",
}

PALETTE_MODEL = {
    "GPT-5.2":          "#4C72B0",
    "BioMistral-7B":    "#DD8452",
    "MedGemma-4B":      "#55A868",
    "Prometheus2-7B":   "#C44E52",
}

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["model"] = df["judge llm"].map(MODEL_SHORT)
    df["ds"] = df["dataset"].map(DATASET_SHORT)
    df["bias"] = df["bias type"].map(BIAS_SHORT)
    return df


def fig1_model_avg_metrics(df):
    """Bar chart: per-model average ACC, ACC_biased, RR, CR."""
    metrics = ["acc", "acc_biased", "rr", "cr"]
    labels  = ["ACC (original)", "ACC (biased)", "Robustness Rate", "Consistency Rate"]
    agg = df.groupby("model")[metrics].mean().reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        bars = ax.bar(x + (i - 1.5) * width, agg[m], width, label=lab, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Per-Model Average Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_model_avg_metrics.png"), dpi=200)
    plt.close(fig)
    print("  [1] fig1_model_avg_metrics.png")


def fig2_rr_per_model_per_dataset(df):
    """Grouped bar: RR per model, grouped by dataset."""
    pivot = df.groupby(["ds", "model"])["rr"].mean().unstack("model").reindex(
        index=DATASET_ORDER, columns=MODEL_ORDER)

    x = np.arange(len(DATASET_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = pivot[m].values
        mask = ~np.isnan(vals)
        bars = ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
                      label=m, color=PALETTE_MODEL[m])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER, fontsize=10)
    ax.set_ylabel("Robustness Rate (%)", fontsize=11)
    ax.set_title("Robustness Rate by Dataset and Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_rr_by_dataset_model.png"), dpi=200)
    plt.close(fig)
    print("  [2] fig2_rr_by_dataset_model.png")


def fig3_rr_per_bias_per_model(df):
    """Grouped bar: average RR per bias type, per model."""
    pivot = df.groupby(["bias", "model"])["rr"].mean().unstack("model").reindex(
        columns=MODEL_ORDER)
    bias_order = list(BIAS_SHORT.values())
    pivot = pivot.reindex(bias_order)

    x = np.arange(len(bias_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = pivot[m].values
        mask = ~np.isnan(vals)
        bars = ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
                      label=m, color=PALETTE_MODEL[m])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_order, fontsize=9)
    ax.set_ylabel("Robustness Rate (%)", fontsize=11)
    ax.set_title("Robustness Rate by Bias Type and Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_rr_by_bias_model.png"), dpi=200)
    plt.close(fig)
    print("  [3] fig3_rr_by_bias_model.png")


def fig4_heatmap_dataset_model_rr(df):
    """Heatmap: average RR — dataset × model."""
    pivot = df.groupby(["ds", "model"])["rr"].mean().unstack("model").reindex(
        index=DATASET_ORDER, columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=40, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "RR (%)"})
    ax.set_title("Robustness Rate Heatmap (Dataset × Model)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_heatmap_rr_dataset_model.png"), dpi=200)
    plt.close(fig)
    print("  [4] fig4_heatmap_rr_dataset_model.png")


def fig5_heatmap_bias_model_rr(df):
    """Heatmap: average RR — bias × model."""
    bias_order = list(BIAS_SHORT.values())
    pivot = df.groupby(["bias", "model"])["rr"].mean().unstack("model").reindex(
        index=bias_order, columns=MODEL_ORDER)
    bias_labels = [b.replace("\n", " ") for b in bias_order]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=40, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "RR (%)"}, yticklabels=bias_labels)
    ax.set_title("Robustness Rate Heatmap (Bias × Model)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_heatmap_rr_bias_model.png"), dpi=200)
    plt.close(fig)
    print("  [5] fig5_heatmap_rr_bias_model.png")


def fig6_acc_vs_rr_scatter(df):
    """Scatter plot: ACC (original) vs RR, colored by model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        ax.scatter(sub["acc"], sub["rr"], label=m, color=PALETTE_MODEL[m],
                   alpha=0.65, s=40, edgecolors="white", linewidth=0.4)
    ax.plot([20, 100], [20, 100], ls="--", color="grey", lw=0.8, alpha=0.5)
    ax.axhline(50, color="red", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("ACC (original) %", fontsize=11)
    ax.set_ylabel("Robustness Rate %", fontsize=11)
    ax.set_title("ACC vs Robustness Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_acc_vs_rr.png"), dpi=200)
    plt.close(fig)
    print("  [6] fig6_acc_vs_rr.png")


def fig7_acc_biased_delta_by_bias(df):
    """Bar chart: average (acc_biased - acc) per bias, showing vulnerability."""
    df2 = df.copy()
    df2["delta"] = df2["acc_biased"] - df2["acc"]
    bias_order = list(BIAS_SHORT.values())
    pivot = df2.groupby(["bias", "model"])["delta"].mean().unstack("model").reindex(
        index=bias_order, columns=MODEL_ORDER)

    x = np.arange(len(bias_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = pivot[m].values
        mask = ~np.isnan(vals)
        bars = ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
                      label=m, color=PALETTE_MODEL[m])
        for bar in bars:
            h = bar.get_height()
            offset = 0.5 if h >= 0 else -1.8
            ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_order, fontsize=9)
    ax.set_ylabel("ACC Change (biased − original) pp", fontsize=11)
    ax.set_title("Accuracy Shift After Bias Injection", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_acc_delta_by_bias.png"), dpi=200)
    plt.close(fig)
    print("  [7] fig7_acc_delta_by_bias.png")


def fig8_cr_per_model_per_dataset(df):
    """Grouped bar: CR per model, grouped by dataset."""
    pivot = df.groupby(["ds", "model"])["cr"].mean().unstack("model").reindex(
        index=DATASET_ORDER, columns=MODEL_ORDER)

    x = np.arange(len(DATASET_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = pivot[m].values
        mask = ~np.isnan(vals)
        bars = ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
                      label=m, color=PALETTE_MODEL[m])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER, fontsize=10)
    ax.set_ylabel("Consistency Rate (%)", fontsize=11)
    ax.set_title("Consistency Rate by Dataset and Model", fontsize=13, fontweight="bold")
    ax.set_ylim(85, 102)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_cr_by_dataset_model.png"), dpi=200)
    plt.close(fig)
    print("  [8] fig8_cr_by_dataset_model.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data from {CSV_PATH}")
    df = load_data()
    print(f"  {len(df)} rows loaded\n")
    print("Generating figures...")
    fig1_model_avg_metrics(df)
    fig2_rr_per_model_per_dataset(df)
    fig3_rr_per_bias_per_model(df)
    fig4_heatmap_dataset_model_rr(df)
    fig5_heatmap_bias_model_rr(df)
    fig6_acc_vs_rr_scatter(df)
    fig7_acc_biased_delta_by_bias(df)
    fig8_cr_per_model_per_dataset(df)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
