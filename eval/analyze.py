import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 10

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CIPHER_TYPES = ["caesar", "atbash", "vigenere", "rot13", "rail_fence", "reverse", "base64", "morse", "bacon"]
CORE_PROMPTS = ["baseline", "choices", "cot"]
CATEGORIES   = {
    "caesar": "substitution", "atbash": "substitution",
    "vigenere": "substitution", "rot13": "substitution",
    "rail_fence": "transposition", "reverse": "transposition",
    "base64": "encoding", "morse": "encoding", "bacon": "encoding",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {name}.pdf / .png")



def figure1_heatmap(summary: pd.DataFrame) -> None:
    core = summary[summary["prompt"].isin(CORE_PROMPTS)]
    best = core.loc[core.groupby("model")["overall"].idxmax()]
    cipher_cols = [c for c in summary.columns if c.startswith("cipher_")]
    labels = [c.replace("cipher_", "") for c in cipher_cols]
    data = best.set_index("model")[cipher_cols].rename(columns=dict(zip(cipher_cols, labels))) * 100

    fig, ax = plt.subplots(figsize=(12, max(3, len(data) * 0.7)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("Cipher Identification Accuracy by Model and Cipher Type", pad=12)
    ax.set_ylabel("Model")
    ax.set_xlabel("Cipher Type")
    save_fig(fig, "fig1_heatmap")



def figure2_confusion_all(summary: pd.DataFrame) -> None:
    models = summary["model"].unique().tolist()
    for model in models:
        path = RESULTS / f"confusion_{model}.csv"
        if not path.exists():
            print(f"No confusion matrix for {model}, skipping.")
            continue
        cm = pd.read_csv(path, index_col=0)
        cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Proportion"})
        ax.set_title(f"Confusion Matrix — {model} (row-normalized)", pad=12)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        save_fig(fig, f"fig2_confusion_{model}")



def figure3_memorization(df_raw: pd.DataFrame) -> None:
    df = df_raw[df_raw["prompt_variant"].isin(CORE_PROMPTS)]
    targets = {"caesar_3": ("caesar", 3), "caesar_7": ("caesar", 7),
               "caesar_19": ("caesar", 19), "rot13": ("rot13", None)}

    rows = []
    for label, (ctype, shift) in targets.items():
        sub = df[df["cipher_type_true"] == ctype]
        if shift is not None:
            sub = sub[sub["id"].str.contains(f"_{shift}$")]
        if sub.empty:
            continue
        for model, msub in sub.groupby("model"):
            rows.append({"cipher": label, "model": model, "accuracy": msub["correct"].mean() * 100})

    if not rows:
        print("No data for Fig 3, skipping.")
        return

    plot_df = pd.DataFrame(rows)
    models = sorted(plot_df["model"].unique().tolist())
    ciphers = list(targets.keys())
    x = np.arange(len(ciphers))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [plot_df[(plot_df["model"] == model) & (plot_df["cipher"] == c)]["accuracy"].values
                for c in ciphers]
        heights = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + i * width - 0.4 + width / 2, heights, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(["Caesar-3", "Caesar-7", "Caesar-19", "ROT13"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ROT13 vs Caesar Accuracy (Memorization Test)")
    ax.legend()
    save_fig(fig, "fig3_memorization")



def figure4_by_length(df_raw: pd.DataFrame, dataset: pd.DataFrame) -> None:
    df = df_raw[df_raw["prompt_variant"].isin(CORE_PROMPTS)].copy()

    # Join plaintext_length from dataset
    length_map = dataset.set_index("id")["plaintext_length"]
    df["plaintext_length"] = df["id"].map(length_map)
    df = df.dropna(subset=["plaintext_length"])

    # Split into terciles: short / medium / long
    terciles = df["plaintext_length"].quantile([1/3, 2/3]).values
    def bin_length(l):
        if l <= terciles[0]:
            return "Short"
        elif l <= terciles[1]:
            return "Medium"
        else:
            return "Long"

    df["length_bin"] = df["plaintext_length"].map(bin_length)

    rows = []
    for (model, length_bin), grp in df.groupby(["model", "length_bin"]):
        rows.append({
            "model": model,
            "length_bin": length_bin,
            "accuracy": grp["correct"].mean() * 100,
        })

    plot_df = pd.DataFrame(rows)
    models = sorted(plot_df["model"].unique().tolist())
    bins = ["Short", "Medium", "Long"]
    x = np.arange(len(bins))
    width = 0.8 / max(len(models), 1)

    # Annotate bins with length ranges
    short_max  = int(terciles[0])
    medium_max = int(terciles[1])
    bin_labels = [
        f"Short\n(≤{short_max} chars)",
        f"Medium\n({short_max+1}–{medium_max} chars)",
        f"Long\n(>{medium_max} chars)",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [plot_df[(plot_df["model"] == model) & (plot_df["length_bin"] == b)]["accuracy"].values
                for b in bins]
        heights = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + i * width - 0.4 + width / 2, heights, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cipher Identification Accuracy by Plaintext Length")
    ax.legend()
    save_fig(fig, "fig4_by_length")



def table1_main_results(summary: pd.DataFrame) -> None:
    df = summary[summary["prompt"].isin(CORE_PROMPTS)].copy()
    df = df[["model", "prompt", "overall", "substitution", "transposition", "encoding"]].copy()
    df.columns = ["Model", "Prompt", "Overall", "Substitution", "Transposition", "Encoding"]

    # Percentages
    for col in ["Overall", "Substitution", "Transposition", "Encoding"]:
        df[col] = (df[col] * 100).round(1)

    df = df.sort_values(["Model", "Prompt"])
    df.to_csv(RESULTS / "table1_main_results.csv", index=False)

    # LaTeX
    latex = df.to_latex(index=False, float_format="%.1f",
                        caption="Overall and per-category accuracy (\\%) by model and prompt variant.",
                        label="tab:main_results")
    (RESULTS / "table1_main_results.tex").write_text(latex)
    print("Saved table1_main_results.csv / .tex")



def table2_per_cipher(summary: pd.DataFrame) -> None:
    core = summary[summary["prompt"].isin(CORE_PROMPTS)]
    best = core.loc[core.groupby("model")["overall"].idxmax()].copy()

    cipher_cols = [f"cipher_{c}" for c in CIPHER_TYPES]
    display_cols = [c.replace("cipher_", "").replace("_", " ").title() for c in cipher_cols]

    df = best[["model", "prompt"] + cipher_cols].copy()
    df.columns = ["Model", "Best Prompt"] + display_cols

    for col in display_cols:
        df[col] = (df[col] * 100).round(1)

    df = df.sort_values("Model")
    df.to_csv(RESULTS / "table2_per_cipher.csv", index=False)

    latex = df.to_latex(index=False, float_format="%.1f",
                        caption="Per-cipher accuracy (\\%) for each model's best prompt variant.",
                        label="tab:per_cipher")
    (RESULTS / "table2_per_cipher.tex").write_text(latex)
    print("Saved table2_per_cipher.csv / .tex")



def main() -> None:
    summary_path = RESULTS / "summary_table.csv"
    if not summary_path.exists():
        print("Run eval/score.py first to generate summary_table.csv.")
        return

    summary = pd.read_csv(summary_path)

    # Load raw results
    records = []
    for path in RESULTS.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    df_raw = pd.DataFrame(records) if records else pd.DataFrame()

    # Load dataset for plaintext lengths
    dataset_path = ROOT / "data" / "dataset.jsonl"
    dataset_records = [json.loads(l) for l in dataset_path.read_text().splitlines() if l.strip()]
    dataset = pd.DataFrame(dataset_records)

    figure1_heatmap(summary)
    figure2_confusion_all(summary)

    if not df_raw.empty:
        df_core = df_raw[df_raw["prompt_variant"].isin(CORE_PROMPTS)]
        figure3_memorization(df_core)
        figure4_by_length(df_core, dataset)

    table1_main_results(summary)
    table2_per_cipher(summary)

    print("\nAll figures and tables saved to results/")


if __name__ == "__main__":
    main()
