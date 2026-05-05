import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

CIPHER_TYPES = ["caesar", "atbash", "vigenere", "rot13", "rail_fence", "reverse", "base64", "morse", "bacon"]
CATEGORIES   = {"caesar": "substitution", "atbash": "substitution", "vigenere": "substitution",
                "rot13": "substitution", "rail_fence": "transposition", "reverse": "transposition",
                "base64": "encoding", "morse": "encoding", "bacon": "encoding"}


def load_results() -> pd.DataFrame:
    records = []
    for path in RESULTS.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> None:
    summary_rows = []

    for (model, prompt), grp in df.groupby(["model", "prompt_variant"]):
        parseable_rate = grp["parseable"].mean()
        overall_acc    = grp["correct"].mean()

        per_cipher = {}
        for ct in CIPHER_TYPES:
            sub = grp[grp["cipher_type_true"] == ct]
            per_cipher[ct] = sub["correct"].mean() if len(sub) else float("nan")

        per_cat = {}
        for cat in ["substitution", "transposition", "encoding"]:
            sub = grp[grp["cipher_type_true"].map(CATEGORIES) == cat]
            per_cat[cat] = sub["correct"].mean() if len(sub) else float("nan")

        row = {"model": model, "prompt": prompt,
               "overall": overall_acc, "parseable_rate": parseable_rate,
               **{f"cipher_{k}": v for k, v in per_cipher.items()},
               **per_cat}
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary_table.csv", index=False)
    print("Saved summary_table.csv")

    # Confusion matrices per model (best prompt by overall accuracy, restricted to baseline/choices/cot)
    for model, mdf in df.groupby("model"):
        core_prompts = mdf[mdf["prompt_variant"].isin(["baseline", "choices", "cot"])]
        best_prompt = core_prompts.groupby("prompt_variant")["correct"].mean().idxmax()
        sub = mdf[mdf["prompt_variant"] == best_prompt]
        true_labels = sub["cipher_type_true"].tolist()
        pred_labels = [p if p in CIPHER_TYPES else "unknown" for p in sub["cipher_type_pred"].tolist()]
        labels = CIPHER_TYPES
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(RESULTS / f"confusion_{model}.csv")
        print(f"Saved confusion_{model}.csv (prompt={best_prompt})")

    # Overall metrics JSON
    metrics = {
        "total_samples":  len(df),
        "models":         df["model"].unique().tolist(),
        "prompt_variants": df["prompt_variant"].unique().tolist(),
        "overall_by_model_prompt": summary[["model", "prompt", "overall"]].to_dict(orient="records"),
    }
    (RESULTS / "overall_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved overall_metrics.json")


def main() -> None:
    df = load_results()
    if df.empty:
        print("No results found in results/. Run eval/run_eval.py first.")
        return
    compute_metrics(df)


if __name__ == "__main__":
    main()
