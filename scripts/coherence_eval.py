#!/usr/bin/env python3
#This script fits the base content to set of 23 constraints and then evaluates the revised content against another article based on grammar, coherence, likeability etc.
#to run the script, first first the articles need to be revised(on 23 constraints) using --revise-input and --revise-output arguments
#then run the evaluation phase using --base-path, --article2-path and --eval-output
from http import client
import re
import os
import argparse
import logging
from pathlib import Path
from time import sleep
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm

from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config

REVISION_PROMPT = """You are an expert news editor.

You will be given:
1) A news article.
2) A list of 23 high–level narrative constraints.

Task:
Your job is to revise and expand the base content to satisfy ALL 23 constraints.

Instructions:
- Keep the core ideas from the base content
- Integrate all 23 constraints seamlessly
- Maintain a natural, engaging writing style
- Ensure the content flows logically
- Do not mention the constraints explicitly in the content
- Aim for completeness - the content should feel finished and polish
- Keep it around 500 words.
- Maintain professional journalistic tone.

Return only your revised article text.
"""

SEGMENT_RE = re.compile(r"\b\d+\.\s*(.*?)(?=\s*\d+\.\s*|$)", re.S)
def constraints_to_list(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    items = [re.sub(r"\s+", " ", s).strip() for s in SEGMENT_RE.findall(text)]
    return [s for s in items if s]


def call_llm(client: OpenAIClient, prompt: str, model: str):
    """
    Minimal wrapper around the project's OpenAIClient.chat_completion interface.
    Returns (text, raw_response)
    """
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        model=model,
    )
    text = response.choices[0].message.content.strip()
    tokens = 0
    try:
        
        tokens = int(response.usage.total_tokens)
    except Exception:
        try:
            
            tokens = int(response.get("usage", {}).get("total_tokens", 0))
        except Exception:
            tokens = 0
    return text, response, tokens


def revise_article(client: OpenAIClient, original_article: str, constraints: list[str], model: str) -> tuple[str, object, int]:
    constraint_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(constraints)])
    prompt = (
        REVISION_PROMPT
        + "\n\nOriginal Article\n"
        + (original_article or "").strip()
        + "\n\nConstraints (1–{})\n".format(len(constraints))
        + constraint_block
    )
    out_text, out_resp, tokens = call_llm(client, prompt, model=model)
    return out_text.strip(), out_resp, tokens


def chat_eval(client: OpenAIClient, instruction: str, system_prompt: str, model: str):
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        model=model
    )
    text = response.choices[0].message.content
    tokens = 0
    try:
        tokens = int(response.usage.total_tokens)
    except Exception:
        try:
            tokens = int(response.get("usage", {}).get("total_tokens", 0))
        except Exception:
            tokens = 0
    return text, response, tokens

def parse_evaluation(evaluation: str):
    """
    Robust parser for LLM pairwise evaluation outputs.
    Returns a dict with numeric scores and recomputed prefs, or None on failure.
    """
    if not evaluation or (isinstance(evaluation, float) and pd.isna(evaluation)):
        return None

    try:
        txt = evaluation.replace('\r', '\n')
        txt_lower = txt.lower()

        def extract_scores_for(category_name: str):
            start = txt_lower.find(category_name.lower())
            if start == -1:
                return None, None
            # end at next category or end of text
            next_idxs = [txt_lower.find(k, start+1) for k in ("coherence", "likability", "grammar") if k != category_name.lower()]
            next_idxs = [i for i in next_idxs if i != -1]
            end = min(next_idxs) if next_idxs else len(txt)
            seg = txt[start:end]

            # pattern "A ... number ... B ... number"
            m = re.search(r'A[^0-9\r\n\-:]*([0-9]+(?:[.,][0-9]+)?)\D+B[^0-9\r\n\-:]*([0-9]+(?:[.,][0-9]+)?)', seg, re.IGNORECASE)
            if m:
                a = m.group(1).replace(',', '.')
                b = m.group(2).replace(',', '.')
                return float(a), float(b)

            # lines like "A - 4/5", "A: 4/5", "A 4.5"
            ma = re.search(r'(^|\n)\s*A[^0-9A-Za-z\r\n\-:]{0,3}([0-9]+(?:[.,][0-9]+)?)(?:\s*/\s*[0-9]+)?', seg, re.IGNORECASE)
            mb = re.search(r'(^|\n)\s*B[^0-9A-Za-z\r\n\-:]{0,3}([0-9]+(?:[.,][0-9]+)?)(?:\s*/\s*[0-9]+)?', seg, re.IGNORECASE)
            if ma and mb:
                a = ma.group(2).replace(',', '.')
                b = mb.group(2).replace(',', '.')
                return float(a), float(b)

            # fallback: first two numeric values in segment
            nums = re.findall(r'([0-9]+(?:[.,][0-9]+)?)', seg)
            nums = [n.replace(',', '.') for n in nums]
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            return None, None

        gA, gB = extract_scores_for("Grammar")
        cA, cB = extract_scores_for("Coherence")
        lA, lB = extract_scores_for("Likability")

        # if no numeric scores found, give up
        if all(v is None for v in (gA, gB, cA, cB, lA, lB)):
            logging.error("Parse failure: no numeric scores found")
            logging.debug("Raw eval:\n%s", evaluation)
            return None

        def f(x): return float(x) if (x is not None) else 0.0
        parsed = {
            'grammar_score_A':    f(gA),
            'grammar_score_B':    f(gB),
            'coherence_score_A':  f(cA),
            'coherence_score_B':  f(cB),
            'likability_score_A': f(lA),
            'likability_score_B': f(lB),
        }

        # recompute prefs and overall winner
        a_wins = 0
        b_wins = 0

        if parsed['grammar_score_A'] >= parsed['grammar_score_B']:
            parsed['grammar_pref'] = "A"; a_wins += 1
        else:
            parsed['grammar_pref'] = "B"; b_wins += 1

        if parsed['coherence_score_A'] >= parsed['coherence_score_B']:
            parsed['coherence_pref'] = "A"; a_wins += 1
        else:
            parsed['coherence_pref'] = "B"; b_wins += 1

        if parsed['likability_score_A'] >= parsed['likability_score_B']:
            parsed['likability_pref'] = "A"; a_wins += 1
        else:
            parsed['likability_pref'] = "B"; b_wins += 1

        parsed['overall_pref'] = "A" if a_wins >= b_wins else "B"
        return parsed

    except Exception as e:
        logging.error("Parse error: %s\nRaw eval:\n%s", e, evaluation)
        return None


PAIRWISE_SYSTEM_PROMPT = """
You are an English writing expert and you can compare and evaluate story essays on these metrics with the following definitions -
    1. Grammar: Which story has better writing and grammar comparitively?
    2. Coherence: Which story has a better logical flow and the writing fits together with respect to the plot?
    3. Likability: Which story do you find more enjoyable to read?
You will be given two Stories - Story A and Story B.
Add a rating out of 5 for each category, specify which story you prefer for each metric by responding with just the letter "A" or "B" followed by a hyphen and one line reasoning for your preference.
For each category provide a category winner story as the letter "A" or "B", based on the category ratings.
Finally, assign an overall winner story as the letter "A" or "B" based on the ratings and category wins.

IMPORTANT - DO NOT GIVE ANY OTHER TEXT APART FROM THE SCORE, METRICS AND PREFERENCE. FOLLOW THE EXACT FORMAT AS GIVEN IN THE EXAMPLES.
"""


def pairwise_eval(client: OpenAIClient, story1: str, story2: str, model: str) -> str:
    prompt0 = f"Story A:\n{story1}\n\nStory B:\n{story2}\n"
    return chat_eval(client, prompt0, system_prompt=PAIRWISE_SYSTEM_PROMPT, model=model)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    client = OpenAIClient(log_usage=True)  
    MODEL_REVISION = args.revision_model or Config.DEFAULT_FITTING_MODEL
    MODEL_EVAL = args.eval_model or Config.DEFAULT_EVALUATION_MODEL
    MAX_REDO = 3


    if args.revise_input and args.revise_output:
        logging.info("Starting revision phase")
        df = pd.read_csv(args.revise_input, dtype=str, keep_default_na=False)
        revised_col = "revised_article"
        base_col = args.base_col
        cons_col = args.cons_col
        MAX_CONSTRAINTS = args.max_constraints or 23

        revised_list = []
        revision_tokens = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            base_article = row.get(base_col, "").strip()
            cons_text = row.get(cons_col, "").strip()
            cons_list = constraints_to_list(cons_text)

            if not base_article:
                revised_list.append("")
                revision_tokens.append(0)
                continue

            selected_constraints = cons_list[:MAX_CONSTRAINTS]
            try:
                out_text, out_resp, tokens = revise_article(client, base_article, selected_constraints, MODEL_REVISION)
                revised_list.append(out_text)
                revision_tokens.append(int(tokens or 0))
            except Exception as e:
                logging.exception("Revision error")
                revised_list.append("")
                revision_tokens.append(0)
                

        df[revised_col] = revised_list
        df["revision_tokens"] = revision_tokens
        df.to_csv(args.revise_output, index=False, encoding="utf-8-sig")
        logging.info("Saved revised CSV to %s", args.revise_output)


    if args.base_path and args.eval_output:
        logging.info("Starting evaluation phase")
        base_df = pd.read_csv(args.base_path, dtype=str, keep_default_na=False)
        
        # Single CSV mode: use same CSV for both articles
        if args.article2_path is None:
            logging.info("Single CSV mode: using same CSV for both articles")
            article_df = base_df.copy()
        else:
            article_df = pd.read_csv(args.article2_path, dtype=str, keep_default_na=False)
        
        # Apply subset filters if specified
        if args.subset_a is not None:
            if 'subset_size' not in base_df.columns:
                raise ValueError("--subset-a requires 'subset_size' column in base CSV")
            base_df = base_df[base_df['subset_size'].astype(str) == str(args.subset_a)].copy()
            logging.info(f"Filtered base_df to subset_size={args.subset_a}, {len(base_df)} rows")
        
        if args.subset_b is not None:
            if 'subset_size' not in article_df.columns:
                raise ValueError("--subset-b requires 'subset_size' column in article CSV")
            article_df = article_df[article_df['subset_size'].astype(str) == str(args.subset_b)].copy()
            logging.info(f"Filtered article_df to subset_size={args.subset_b}, {len(article_df)} rows")
        
        # Align by instruction_number if available in both DataFrames
        if 'instruction_number' in base_df.columns and 'instruction_number' in article_df.columns:
            logging.info("Aligning by instruction_number")
            base_df = base_df.sort_values('instruction_number').reset_index(drop=True)
            article_df = article_df.sort_values('instruction_number').reset_index(drop=True)
            
            # Merge on instruction_number to ensure proper alignment
            base_df['_idx'] = base_df['instruction_number']
            article_df['_idx'] = article_df['instruction_number']
            merged = pd.merge(base_df, article_df[['_idx', args.eval_article]], 
                            on='_idx', suffixes=('', '_article2'))
            base_df2 = merged.drop('_idx', axis=1)
            n = len(base_df2)
            logging.info(f"Aligned {n} rows by instruction_number")
        else:
            # Fallback to index-based alignment
            logging.info("Aligning by index (no instruction_number found)")
            n = min(len(base_df), len(article_df))
            base_df2 = base_df.iloc[:n].reset_index(drop=True).copy()
            article_df = article_df.iloc[:n].reset_index(drop=True).copy()
            base_df2[args.eval_article] = article_df[args.eval_article].values
        
        results_df = base_df2.copy()

        cols = [
            "grammar_score_A", "grammar_score_B",
            "coherence_score_A", "coherence_score_B",
            "likability_score_A", "likability_score_B",
            "grammar_pref", "coherence_pref",
            "likability_pref", "overall_pref",
            "order", "needs_parsing", "evaluations_raw",
        ]
        for c in cols:
            if c not in results_df.columns:
                results_df[c] = None

        evaluation_tokens = []
        for idx, row in results_df.iterrows():
            story_a_orig = row[args.base_col_eval]
            story_b_orig = row[args.eval_article]

            rand_order = np.random.randint(2)
            if rand_order == 0:
                a_text, b_text = story_a_orig, story_b_orig
            else:
                a_text, b_text = story_b_orig, story_a_orig

            eval_text = None
            eval_resp = None
            eval_tokens = 0
            parsed = None
            needs_parsing = 0

            for attempt in range(MAX_REDO):
                try:
                    eval_text, eval_resp, eval_tokens = pairwise_eval(client, a_text, b_text, model=MODEL_EVAL)
                    parsed = parse_evaluation(eval_text)
                    if parsed is not None:
                        break
                except Exception as e:
                    logging.exception("Evaluation error, retrying...")
                    sleep(1)

            if parsed is None:
                needs_parsing = 1

            results_df.at[idx, "evaluations_raw"] = eval_text
            results_df.at[idx, "needs_parsing"] = needs_parsing
            results_df.at[idx, "order"] = rand_order
            evaluation_tokens.append(int(eval_tokens or 0))

            if parsed is not None:
                for k, v in parsed.items():
                    results_df.at[idx, k] = v

            logging.info("Row %d/%d done | Coherence A/B: %s/%s",
                         idx, n-1,
                         parsed['coherence_score_A'] if parsed else 'NA',
                         parsed['coherence_score_B'] if parsed else 'NA')

        results_df["evaluation_tokens"] = evaluation_tokens
        results_df.to_csv(args.eval_output, index=False, encoding="utf-8-sig")
        logging.info("Saved evaluation results to %s", args.eval_output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Revise articles and optionally run pairwise evaluation")
    # revision inputs
    p.add_argument("--revise-input", type=str, help="CSV to read for revision phase")
    p.add_argument("--revise-output", type=str, help="CSV to write revised articles")
    p.add_argument("--base-col", type=str)
    p.add_argument("--cons-col", type=str)
    p.add_argument("--max-constraints", type=int, default=23)
    p.add_argument("--revision-model", type=str, default=None)

    # evaluation inputs
    p.add_argument("--base-path", type=str, help="CSV with base/revised articles")
    p.add_argument("--article2-path", type=str, help="CSV with articles to compare (optional if using single CSV mode)")
    p.add_argument("--eval-output", type=str, help="CSV to write pairwise evaluation results")
    p.add_argument("--eval-article", type=str, help="Column name for article B")
    p.add_argument("--base-col-eval", type=str, help="Column name for article A")
    p.add_argument("--eval-model", type=str, default="gpt-4o-mini")
    
    # single CSV mode options
    p.add_argument("--subset-a", type=int, help="Filter base_path to subset_size value for article A")
    p.add_argument("--subset-b", type=int, help="Filter article2_path (or base_path) to subset_size value for article B")

    args = p.parse_args()
    main(args)