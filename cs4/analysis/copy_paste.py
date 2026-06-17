"""Constraint copy-paste scoring.

For each (constraint, response) pair we find the most similar sentence in the
response by cosine similarity over `all-mpnet-base-v2` embeddings. The per-row
score is the mean of those max-similarities across the row's constraints; the
per-cell score is the mean across rows in that cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


_NUMBERED_ITEM_RE = re.compile(r"(?ms)^\s*\d+\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)")


def parse_numbered_constraints(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    items = [m.group(1).strip() for m in _NUMBERED_ITEM_RE.finditer(text)]
    return [c for c in (re.sub(r"\s+", " ", c).strip() for c in items) if c]


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = [s.strip() for s in sent_tokenize(text)]
    return [s for s in sents if s]


@dataclass
class CellSpec:
    setting: str           # "Common Constraints News" | "single_news"
    method: str            # e.g. "Direct - blind revised"
    model: str             # "GPT5-mini" | "Llama-8B"
    eval_mode: str         # "raw" | "summ"
    response_csv: str      # absolute path to evaluated CSV
    response_column: str   # direct_content | fitted_content | summarized_content
    constraints_csv: str   # absolute path to CSV holding the constraints to compare against
    constraints_column: str  # "constraints" | "revised_constraints"


@dataclass
class CellResult:
    spec: CellSpec
    per_row: pd.DataFrame   # one row per sample in the cell
    cell_score: float       # mean over per_row.row_score


class CopyPasteScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str | None = None, batch_size: int = 256):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        new = [t for t in texts if t not in self._cache]
        if new:
            vecs = self.model.encode(
                new,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            for t, v in zip(new, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def score_row(self, constraints: list[str], response: str) -> dict:
        sents = split_sentences(response)
        n_c, n_s = len(constraints), len(sents)
        if n_c == 0 or n_s == 0:
            return {"row_score": np.nan, "n_constraints": n_c, "n_sentences": n_s, "per_constraint_max": []}
        c_emb = self.embed(constraints)        # (n_c, d), L2-normalized
        s_emb = self.embed(sents)              # (n_s, d), L2-normalized
        sims = c_emb @ s_emb.T                 # cosine since both normalized
        max_per_constraint = sims.max(axis=1)  # (n_c,)
        return {
            "row_score": float(max_per_constraint.mean()),
            "n_constraints": n_c,
            "n_sentences": n_s,
            "per_constraint_max": max_per_constraint.tolist(),
        }

    def score_cell(self, spec: CellSpec) -> CellResult:
        resp_df = pd.read_csv(spec.response_csv)
        if spec.constraints_csv == spec.response_csv:
            const_df = resp_df
        else:
            const_df = pd.read_csv(spec.constraints_csv)

        merge_key = "instruction_number"
        if merge_key not in resp_df.columns or merge_key not in const_df.columns:
            raise ValueError(f"missing {merge_key} in {spec.response_csv} or {spec.constraints_csv}")

        joined = resp_df[[merge_key, spec.response_column]].merge(
            const_df[[merge_key, spec.constraints_column]],
            on=merge_key,
            how="inner",
            validate="one_to_one",
        )

        rows = []
        for _, row in joined.iterrows():
            constraints = parse_numbered_constraints(row[spec.constraints_column])
            r = self.score_row(constraints, row[spec.response_column])
            rows.append({
                "setting": spec.setting,
                "method": spec.method,
                "model": spec.model,
                "eval_mode": spec.eval_mode,
                "instruction_number": int(row[merge_key]),
                "row_score": r["row_score"],
                "n_constraints": r["n_constraints"],
                "n_sentences": r["n_sentences"],
            })

        per_row_df = pd.DataFrame(rows)
        cell_score = float(np.nanmean(per_row_df["row_score"].values)) if len(per_row_df) else float("nan")
        return CellResult(spec=spec, per_row=per_row_df, cell_score=cell_score)
