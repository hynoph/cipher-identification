import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"

CIPHER_TYPES = ["caesar", "atbash", "vigenere", "rot13",
                "rail_fence", "reverse", "base64", "morse", "bacon"]
CATEGORIES = {
    "caesar": "substitution", "atbash": "substitution",
    "vigenere": "substitution", "rot13": "substitution",
    "rail_fence": "transposition", "reverse": "transposition",
    "base64": "encoding", "morse": "encoding", "bacon": "encoding",
}

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 10_000



def load_results() -> pd.DataFrame:
    records = []
    for path in RESULTS.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["correct"] = df["correct"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(correct: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for mean(correct)."""
    means = np.array([
        correct[RNG.integers(0, len(correct), len(correct))].mean()
        for _ in range(n)
    ])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def compute_balanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, prompt), grp in df.groupby(["model", "prompt_variant"]):
        support = grp["cipher_type_true"].value_counts()
        majority_class = support.idxmax()
        majority_baseline = support.max() / len(grp)

        per_class_acc = {}
        for ct in CIPHER_TYPES:
            sub = grp[grp["cipher_type_true"] == ct]
            per_class_acc[ct] = float(sub["correct"].mean()) if len(sub) else float("nan")

        valid = [v for v in per_class_acc.values() if not np.isnan(v)]
        macro_avg = float(np.mean(valid)) if valid else float("nan")

        row = {
            "model": model, "prompt": prompt,
            "macro_avg": round(macro_avg, 4),
            "majority_class": majority_class,
            "majority_baseline": round(majority_baseline, 4),
            **{f"support_{ct}": int(support.get(ct, 0)) for ct in CIPHER_TYPES},
            **{
                f"recall_{ct}": round(per_class_acc[ct], 4)
                if not np.isnan(per_class_acc.get(ct, float("nan"))) else float("nan")
                for ct in CIPHER_TYPES
            },
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "prompt"])


def compute_bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, prompt), grp in df.groupby(["model", "prompt_variant"]):
        correct = grp["correct"].values
        acc = correct.mean()
        lo, hi = bootstrap_ci(correct)
        rows.append({
            "model": model, "prompt": prompt,
            "accuracy": round(acc, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "n": len(correct),
        })
    return pd.DataFrame(rows).sort_values(["model", "prompt"])


# ---------------------------------------------------------------------------
# McNemar helpers
# ---------------------------------------------------------------------------

def mcnemar_test(a_correct: pd.Series, b_correct: pd.Series) -> tuple[float, float, int, int, float]:
    b01 = int(((~a_correct) & b_correct).sum())
    b10 = int((a_correct & (~b_correct)).sum())
    table = np.array([[0, b01], [b10, 0]])
    n_disc = b01 + b10
    if n_disc == 0:
        return 0.0, 1.0, 0, 0, float("nan")
    exact = n_disc < 25
    result = mcnemar(table, exact=exact, correction=not exact)
    odds_ratio = b01 / b10 if b10 > 0 else (float("inf") if b01 > 0 else float("nan"))
    return float(result.statistic), float(result.pvalue), b01, b10, round(odds_ratio, 3)


def apply_holm_correction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    reject, p_corr, _, _ = multipletests(df["p_value"].values, method="holm", alpha=0.05)
    df["p_holm"] = np.round(p_corr, 4)
    df["significant_holm"] = reject
    return df


# ---------------------------------------------------------------------------
# 1. Pairwise model comparisons (per prompt variant)
# ---------------------------------------------------------------------------

def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prompt_variants = df["prompt_variant"].unique()
    models = df["model"].unique()

    for prompt in prompt_variants:
        pdf = df[df["prompt_variant"] == prompt]
        # Build per-sample correctness per model, aligned on sample id
        pivot = pdf.pivot_table(index="id", columns="model", values="correct", aggfunc="first")
        pivot = pivot.dropna()  # only samples present for all models

        for m1, m2 in itertools.combinations(models, 2):
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            stat, pval, b01, b10, oddsratio = mcnemar_test(pivot[m1], pivot[m2])
            acc1 = pivot[m1].mean()
            acc2 = pivot[m2].mean()
            rows.append({
                "prompt": prompt,
                "model_a": m1, "model_b": m2,
                "acc_a": round(acc1, 4), "acc_b": round(acc2, 4),
                "b01": b01, "b10": b10, "odds_ratio": oddsratio,
                "mcnemar_stat": round(stat, 4),
                "p_value": round(pval, 4),
                "significant_p05": pval < 0.05,
                "n_matched": len(pivot),
            })
    return apply_holm_correction(pd.DataFrame(rows).sort_values(["prompt", "p_value"]))


# ---------------------------------------------------------------------------
# 2. Pairwise prompt comparisons (per model)
# ---------------------------------------------------------------------------

def compare_prompts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = df["model"].unique()
    prompt_variants = df["prompt_variant"].unique()

    for model in models:
        mdf = df[df["model"] == model]
        pivot = mdf.pivot_table(index="id", columns="prompt_variant", values="correct", aggfunc="first")
        pivot = pivot.dropna()

        for p1, p2 in itertools.combinations(prompt_variants, 2):
            if p1 not in pivot.columns or p2 not in pivot.columns:
                continue
            stat, pval, b01, b10, oddsratio = mcnemar_test(pivot[p1], pivot[p2])
            acc1 = pivot[p1].mean()
            acc2 = pivot[p2].mean()
            rows.append({
                "model": model,
                "prompt_a": p1, "prompt_b": p2,
                "acc_a": round(acc1, 4), "acc_b": round(acc2, 4),
                "b01": b01, "b10": b10, "odds_ratio": oddsratio,
                "mcnemar_stat": round(stat, 4),
                "p_value": round(pval, 4),
                "significant_p05": pval < 0.05,
                "n_matched": len(pivot),
            })
    return apply_holm_correction(pd.DataFrame(rows).sort_values(["model", "p_value"]))


# ---------------------------------------------------------------------------
# 3. ROT13 vs Caesar shifts (memorization test)
# ---------------------------------------------------------------------------

def compare_memorization(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model (best prompt): McNemar's test comparing ROT13 accuracy
    against each Caesar shift variant (3, 7, 19).
    """
    rows = []
    # Use baseline prompt for clean comparison; fall back to best
    for model, mdf in df.groupby("model"):
        best_prompt = mdf.groupby("prompt_variant")["correct"].mean().idxmax()
        sub = mdf[mdf["prompt_variant"] == best_prompt]

        rot13_ids  = sub[sub["cipher_type_true"] == "rot13"].set_index("id")["correct"]
        caesar_ids = sub[sub["cipher_type_true"] == "caesar"].set_index("id")["correct"]

        # Split Caesar by shift via id suffix
        for shift in [3, 7, 19]:
            caesar_shift = caesar_ids[caesar_ids.index.str.endswith(f"_{shift}")]
            if caesar_shift.empty or rot13_ids.empty:
                continue
            # Matched test requires same sample ids — use chi2 on totals instead
            c_correct = int(caesar_shift.sum())
            c_total   = len(caesar_shift)
            r_correct = int(rot13_ids.sum())
            r_total   = len(rot13_ids)
            table = np.array([
                [r_correct,         r_total - r_correct],
                [c_correct,         c_total - c_correct],
            ])
            if table.min() == 0:
                # Add 0.5 continuity correction
                table = table + 0.5
            chi2, pval, _, _ = chi2_contingency(table, correction=False)
            n_total = r_total + c_total
            phi = float(np.sqrt(chi2 / n_total)) if n_total > 0 else float("nan")
            rows.append({
                "model": model, "prompt": best_prompt,
                "comparison": f"rot13_vs_caesar_{shift}",
                "rot13_acc": round(r_correct / r_total, 4),
                "caesar_acc": round(c_correct / c_total, 4),
                "chi2_stat": round(chi2, 4),
                "phi": round(phi, 3),
                "p_value": round(pval, 4),
                "significant_p05": pval < 0.05,
            })
    return pd.DataFrame(rows).sort_values(["model", "comparison"])


# ---------------------------------------------------------------------------
# 4. Encoding vs substitution (per model, best prompt)
# ---------------------------------------------------------------------------

def compare_categories(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, mdf in df.groupby("model"):
        best_prompt = mdf.groupby("prompt_variant")["correct"].mean().idxmax()
        sub = mdf[mdf["prompt_variant"] == best_prompt].copy()
        sub["category"] = sub["cipher_type_true"].map(CATEGORIES)

        enc  = sub[sub["category"] == "encoding"]["correct"]
        subs = sub[sub["category"] == "substitution"]["correct"]
        trns = sub[sub["category"] == "transposition"]["correct"]

        for cat_a_name, cat_a, cat_b_name, cat_b in [
            ("encoding", enc, "substitution", subs),
            ("encoding", enc, "transposition", trns),
            ("substitution", subs, "transposition", trns),
        ]:
            n_a, n_b = len(cat_a), len(cat_b)
            if n_a == 0 or n_b == 0:
                continue
            a_correct = int(cat_a.sum())
            b_correct = int(cat_b.sum())
            table = np.array([
                [a_correct,     n_a - a_correct],
                [b_correct,     n_b - b_correct],
            ]) + 0.5
            chi2, pval, _, _ = chi2_contingency(table, correction=False)
            n_total = n_a + n_b
            phi = float(np.sqrt(chi2 / n_total)) if n_total > 0 else float("nan")
            rows.append({
                "model": model, "prompt": best_prompt,
                "cat_a": cat_a_name, "cat_b": cat_b_name,
                "acc_a": round(cat_a.mean(), 4),
                "acc_b": round(cat_b.mean(), 4),
                "chi2_stat": round(chi2, 4),
                "phi": round(phi, 3),
                "p_value": round(pval, 4),
                "significant_p05": pval < 0.05,
            })
    return pd.DataFrame(rows).sort_values(["model", "cat_a"])


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def write_summary(ci_df, model_df, prompt_df, mem_df, cat_df, bal_df) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL TEST SUMMARY")
    lines.append(f"Bootstrap CIs: {N_BOOTSTRAP:,} iterations, seed=42")
    lines.append("Multiple comparisons: Holm-Bonferroni correction (p_holm column)")
    lines.append("Effect sizes: odds ratio (McNemar), phi (chi-squared)")
    lines.append("=" * 70)

    lines.append("\n--- Balanced Accuracy and Class Support ---")
    for _, r in bal_df.iterrows():
        lines.append(f"  {r['model']:20s} [{r['prompt']:10s}]  "
                     f"macro={r['macro_avg']:.1%}  majority_baseline={r['majority_baseline']:.1%} ({r['majority_class']})")

    lines.append("\n--- Accuracy with 95% Bootstrap CIs ---")
    for _, r in ci_df.iterrows():
        lines.append(f"  {r['model']:20s} [{r['prompt']:10s}]  "
                     f"{r['accuracy']:.1%}  ({r['ci_lower']:.1%} – {r['ci_upper']:.1%})  n={r['n']}")

    lines.append("\n--- Pairwise Model Comparisons (McNemar, Holm-corrected) ---")
    for _, r in model_df.iterrows():
        sig = "**" if r["significant_holm"] else "  "
        lines.append(f"  {sig} [{r['prompt']:10s}] {r['model_a']:20s} vs {r['model_b']:20s} "
                     f"p={_fmt_p(r['p_value'])}  p_holm={_fmt_p(r['p_holm'])}  "
                     f"OR={r['odds_ratio']:.2f}  b01={r['b01']} b10={r['b10']}")

    lines.append("\n--- Pairwise Prompt Comparisons (McNemar, Holm-corrected) ---")
    for _, r in prompt_df.iterrows():
        sig = "**" if r["significant_holm"] else "  "
        lines.append(f"  {sig} [{r['model']:20s}] {r['prompt_a']:10s} vs {r['prompt_b']:10s} "
                     f"p={_fmt_p(r['p_value'])}  p_holm={_fmt_p(r['p_holm'])}  "
                     f"OR={r['odds_ratio']:.2f}  b01={r['b01']} b10={r['b10']}")

    lines.append("\n--- ROT13 vs Caesar Memorization (chi-squared) ---")
    for _, r in mem_df.iterrows():
        sig = "**" if r["significant_p05"] else "  "
        lines.append(f"  {sig} [{r['model']:20s}] {r['comparison']:25s} "
                     f"p={_fmt_p(r['p_value'])}  phi={r['phi']:.3f}  "
                     f"(rot13={r['rot13_acc']:.1%} vs caesar={r['caesar_acc']:.1%})")

    lines.append("\n--- Category Accuracy Differences (chi-squared) ---")
    for _, r in cat_df.iterrows():
        sig = "**" if r["significant_p05"] else "  "
        lines.append(f"  {sig} [{r['model']:20s}] {r['cat_a']:15s} vs {r['cat_b']:15s} "
                     f"p={_fmt_p(r['p_value'])}  phi={r['phi']:.3f}  ({r['acc_a']:.1%} vs {r['acc_b']:.1%})")

    lines.append("\n** = significant after Holm-Bonferroni correction")
    text = "\n".join(lines)
    (RESULTS / "stats_summary.txt").write_text(text)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_results()
    if df.empty:
        print("No results found. Run eval/run_eval.py first.")
        return

    # Filter to baseline/choices/cot only (exclude expert / partial runs)
    df = df[df["prompt_variant"].isin(["baseline", "choices", "cot"])]

    print("Computing bootstrap CIs...")
    ci_df = compute_bootstrap_ci(df)
    ci_df.to_csv(RESULTS / "stats_bootstrap_ci.csv", index=False)
    print("Saved stats_bootstrap_ci.csv")

    print("Running model comparison tests...")
    model_df = compare_models(df)
    model_df.to_csv(RESULTS / "stats_mcnemar_models.csv", index=False)
    print("Saved stats_mcnemar_models.csv")

    print("Running prompt comparison tests...")
    prompt_df = compare_prompts(df)
    prompt_df.to_csv(RESULTS / "stats_mcnemar_prompts.csv", index=False)
    print("Saved stats_mcnemar_prompts.csv")

    print("Running memorization tests...")
    mem_df = compare_memorization(df)
    mem_df.to_csv(RESULTS / "stats_memorization.csv", index=False)
    print("Saved stats_memorization.csv")

    print("Running category tests...")
    cat_df = compare_categories(df)
    cat_df.to_csv(RESULTS / "stats_categories.csv", index=False)
    print("Saved stats_categories.csv")

    print("Computing balanced metrics...")
    bal_df = compute_balanced_metrics(df)
    bal_df.to_csv(RESULTS / "stats_balanced_metrics.csv", index=False)
    print("Saved stats_balanced_metrics.csv")

    write_summary(ci_df, model_df, prompt_df, mem_df, cat_df, bal_df)
    print("\nAll stats saved to results/")


if __name__ == "__main__":
    main()
