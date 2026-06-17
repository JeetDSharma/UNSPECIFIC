"""Driver: compute the 12 x 4 copy-paste-seriousness table.

Enumerates all 48 (setting, method, model, eval_mode) cells, resolves the
response CSV + response column + which constraints to compare against, runs
CopyPasteScorer, and writes:
  - per_row.csv      (one row per sample x cell -> 1200 rows when complete)
  - cells_long.csv   (one row per cell -> 48 rows)
  - table_12x4.md    (pivoted to match the headline table layout)

Outputs are co-located with the source CSVs under:
  data/direct_blog_(25th_March_Meet)/news/copy_paste_analysis/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cs4.analysis.copy_paste import CellSpec, CopyPasteScorer


NEWS_DIR = Path("data/direct_blog_(25th_March_Meet)/news").resolve()
OUT_DIR = NEWS_DIR / "copy_paste_analysis"


# ---- file-path templates per (setting, method, model, eval_mode) ----------

MODEL_TAGS = {
    "GPT5-mini": "gpt-5-mini",
    "Llama-8B": "Llama-3-8B-Lite",
}

# (method_label, response_csv_template, response_column_raw, response_column_summ,
#  constraints_csv_template, constraints_column)
# Templates use {model_tag} and {scope_tag} where {scope_tag} = "news" | "single_news".
# Response CSVs:
#   - raw: <basename>.csv
#   - summ: <basename>_summarized_25pct_evaluated_claude-sonnet-4-20250514.csv
# We store the raw basename and append the summ suffix at lookup time.
METHODS = [
    {
        "method": "Direct - original constraints",
        "kind": "direct",
        "response_basename": "direct_evaluated_{scope_tag}_25_{model_tag}",
        "response_subdir": "",
        "constraints_basename": "direct_evaluated_{scope_tag}_25_{model_tag}",  # same CSV
        "constraints_subdir": "",
        "constraints_column": "constraints",
    },
    {
        "method": "Direct - blind revised",
        "kind": "direct",
        "response_basename": "direct_evaluated_blind_{scope_tag}_25_{model_tag}",
        "response_subdir": "",
        "constraints_basename": "revised_constraints_blind_{scope_tag}_25_gpt-4.1-mini",
        "constraints_subdir": "",
        "constraints_column": "revised_constraints",
    },
    {
        "method": "Direct - eval revised",
        "kind": "direct",
        "response_basename": "direct_evaluated_eval_revised_{scope_tag}_25_{model_tag}",
        "response_subdir": "",
        "constraints_basename": "revised_constraints_eval_{scope_tag}_25_gpt-4.1-mini",
        "constraints_subdir": "",
        "constraints_column": "revised_constraints",
    },
    {
        "method": "Base Revision - normal constraints",
        "kind": "fitted",
        "response_basename": "fitted_normal_constraints_{model_tag_dashed}_evaluated_claude-sonnet-4-20250514_{scope_tag}_25",
        "response_subdir": "base_revision",
        "constraints_basename": "fitted_normal_constraints_{model_tag_dashed}_evaluated_claude-sonnet-4-20250514_{scope_tag}_25",
        "constraints_subdir": "base_revision",
        "constraints_column": "constraints",
    },
    {
        "method": "Base Revision - blind revised",
        "kind": "fitted",
        "response_basename": "fitted_blind_revised_{model_tag_dashed}_evaluated_claude-sonnet-4-20250514_{scope_tag}_25",
        "response_subdir": "base_revision",
        "constraints_basename": "revised_constraints_blind_{scope_tag}_25_gpt-4.1-mini",
        "constraints_subdir": "",
        "constraints_column": "revised_constraints",
    },
    {
        "method": "Base Revision - eval revised",
        "kind": "fitted",
        "response_basename": "fitted_eval_revised_{model_tag_dashed}_evaluated_claude-sonnet-4-20250514_{scope_tag}_25",
        "response_subdir": "base_revision",
        "constraints_basename": "revised_constraints_eval_{scope_tag}_25_gpt-4.1-mini",
        "constraints_subdir": "",
        "constraints_column": "revised_constraints",
    },
]

SETTINGS = [
    {"setting": "Common Constraints News", "scope_tag": "news"},
    {"setting": "single_news", "scope_tag": "single_news"},
]


def _model_tag_dashed(model_tag: str) -> str:
    # base_revision filenames use the long Together model id with `/` -> `-`
    if model_tag == "Llama-3-8B-Lite":
        return "meta-llama-Meta-Llama-3-8B-Instruct-Lite"
    return model_tag  # gpt-5-mini stays as-is


def build_specs() -> list[CellSpec]:
    specs: list[CellSpec] = []
    for s in SETTINGS:
        scope_root = NEWS_DIR / ("single_news" if s["scope_tag"] == "single_news" else ".")
        scope_root = scope_root.resolve()
        for m in METHODS:
            for model_label, model_tag in MODEL_TAGS.items():
                fmt = {
                    "scope_tag": s["scope_tag"],
                    "model_tag": model_tag,
                    "model_tag_dashed": _model_tag_dashed(model_tag),
                }
                resp_subdir = m["response_subdir"]
                const_subdir = m["constraints_subdir"]
                # The "revised_constraints" CSVs live under single_news/ (not single_news/base_revision/)
                # so const_subdir stays "" even for fitted blind/eval methods.
                response_base = m["response_basename"].format(**fmt)
                constraints_base = m["constraints_basename"].format(**fmt)
                raw_csv = (scope_root / resp_subdir / f"{response_base}.csv") if resp_subdir else (scope_root / f"{response_base}.csv")
                summ_csv = (scope_root / resp_subdir / f"{response_base}_summarized_25pct_evaluated_claude-sonnet-4-20250514.csv") if resp_subdir else (scope_root / f"{response_base}_summarized_25pct_evaluated_claude-sonnet-4-20250514.csv")
                const_csv = (scope_root / const_subdir / f"{constraints_base}.csv") if const_subdir else (scope_root / f"{constraints_base}.csv")

                # raw response column: direct_content (direct) or fitted_content (fitted)
                resp_col_raw = "direct_content" if m["kind"] == "direct" else "fitted_content"
                # summarized response column: always summarized_content
                resp_col_summ = "summarized_content"

                for eval_mode, resp_csv, resp_col in [
                    ("raw", raw_csv, resp_col_raw),
                    ("summ", summ_csv, resp_col_summ),
                ]:
                    specs.append(CellSpec(
                        setting=s["setting"],
                        method=m["method"],
                        model=model_label,
                        eval_mode=eval_mode,
                        response_csv=str(resp_csv),
                        response_column=resp_col,
                        constraints_csv=str(const_csv),
                        constraints_column=m["constraints_column"],
                    ))
    return specs


# ---- column-label mapping for the 12 x 4 table -----------------------------

COL_HEADERS = [
    ("GPT5-mini", "raw"),
    ("Llama-8B", "raw"),
    ("summ (GPT5-mini)", "summ-GPT5-mini"),
    ("Summ (Llama-8B)", "summ-Llama-8B"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    assert len(specs) == 48, f"expected 48 cells, got {len(specs)}"

    # Verify all source files exist before we waste GPU time
    missing = [s for s in specs if not Path(s.response_csv).exists() or not Path(s.constraints_csv).exists()]
    if missing:
        for s in missing:
            print(f"MISSING: setting={s.setting} method={s.method} model={s.model} eval={s.eval_mode}")
            if not Path(s.response_csv).exists():
                print(f"  response_csv -> {s.response_csv}")
            if not Path(s.constraints_csv).exists():
                print(f"  constraints_csv -> {s.constraints_csv}")
        raise SystemExit(f"{len(missing)} cells have missing source files")

    scorer = CopyPasteScorer()

    per_row_frames: list[pd.DataFrame] = []
    cell_rows: list[dict] = []

    for i, spec in enumerate(specs, start=1):
        print(f"[{i:2d}/{len(specs)}] {spec.setting} | {spec.method} | {spec.model} | {spec.eval_mode}")
        result = scorer.score_cell(spec)
        per_row_frames.append(result.per_row)
        cell_rows.append({
            "setting": spec.setting,
            "method": spec.method,
            "model": spec.model,
            "eval_mode": spec.eval_mode,
            "response_csv": spec.response_csv,
            "constraints_csv": spec.constraints_csv,
            "response_column": spec.response_column,
            "constraints_column": spec.constraints_column,
            "n_rows": int(len(result.per_row)),
            "cell_score": result.cell_score,
            "row_score_median": float(result.per_row["row_score"].median()) if len(result.per_row) else float("nan"),
            "row_score_std": float(result.per_row["row_score"].std(ddof=0)) if len(result.per_row) else float("nan"),
            "mean_n_constraints": float(result.per_row["n_constraints"].mean()) if len(result.per_row) else float("nan"),
            "mean_n_sentences": float(result.per_row["n_sentences"].mean()) if len(result.per_row) else float("nan"),
        })
        print(f"        -> cell_score = {result.cell_score:.4f} over {len(result.per_row)} rows")

    per_row_df = pd.concat(per_row_frames, ignore_index=True)
    cells_long_df = pd.DataFrame(cell_rows)

    per_row_path = OUT_DIR / "per_row.csv"
    cells_long_path = OUT_DIR / "cells_long.csv"
    table_path = OUT_DIR / "table_12x4.md"

    per_row_df.to_csv(per_row_path, index=False)
    cells_long_df.to_csv(cells_long_path, index=False)

    # ---- build the 12 x 4 markdown table -----------------------------------
    method_order = [m["method"] for m in METHODS]
    setting_order = [s["setting"] for s in SETTINGS]

    def cell_lookup(setting: str, method: str, model: str, eval_mode: str) -> float:
        mask = (
            (cells_long_df["setting"] == setting)
            & (cells_long_df["method"] == method)
            & (cells_long_df["model"] == model)
            & (cells_long_df["eval_mode"] == eval_mode)
        )
        sub = cells_long_df.loc[mask, "cell_score"]
        return float(sub.iloc[0]) if len(sub) else float("nan")

    lines = []
    lines.append("# Copy-Paste Seriousness Table (12 x 4)")
    lines.append("")
    lines.append("Each cell = mean over 25 rows of (mean over 39 constraints of max cosine similarity")
    lines.append("between the constraint and any sentence in the row's response).")
    lines.append("Embedding model: `sentence-transformers/all-mpnet-base-v2`.")
    lines.append("Higher score => more copy-paste of the prompted constraint text into the response.")
    lines.append("")
    header = "| Setting | Method | GPT5-mini | Llama-8B | summ (GPT5-mini) | Summ (Llama-8B) |"
    sep =    "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for setting in setting_order:
        for method in method_order:
            cells = [
                cell_lookup(setting, method, "GPT5-mini", "raw"),
                cell_lookup(setting, method, "Llama-8B", "raw"),
                cell_lookup(setting, method, "GPT5-mini", "summ"),
                cell_lookup(setting, method, "Llama-8B", "summ"),
            ]
            cell_strs = [f"{v:.4f}" if v == v else "NaN" for v in cells]  # NaN check via v == v
            lines.append(f"| {setting} | {method} | {cell_strs[0]} | {cell_strs[1]} | {cell_strs[2]} | {cell_strs[3]} |")

    table_path.write_text("\n".join(lines) + "\n")

    print()
    print(f"Wrote {per_row_path} ({len(per_row_df)} rows)")
    print(f"Wrote {cells_long_path} ({len(cells_long_df)} rows)")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
