"""
Pairwise quality evaluation module for comparing content quality.
"""

import pandas as pd
import numpy as np
import logging
import re
from time import sleep
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from cs4.core.prompts import get_pairwise_quality_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class QualityEvaluator:
    """Evaluate content quality through pairwise comparisons."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize quality evaluator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_EVALUATION_MODEL
        self.retry_attempts = retry_attempts
        self.delay = delay
        
        self.logger = logging.getLogger("CS4QualityEvaluator")
    
    def evaluate_pair(
        self,
        content_a: str,
        content_b: str,
        log: bool = True
    ) -> Tuple[Optional[Dict], str, int]:
        """
        Evaluate a pair of content for quality comparison.
        
        Args:
            content_a: First content to compare
            content_b: Second content to compare
            log: Whether to log token usage
            
        Returns:
            Tuple of (parsed_results_dict, raw_response, tokens_used)
        """
        prompt = get_pairwise_quality_prompt(content_a, content_b)
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model
                    )
                    raw_response = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    raw_response = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Parse the response
                parsed = self._parse_evaluation(raw_response)
                
                if parsed is None:
                    if attempt < self.retry_attempts:
                        self.logger.warning(f"Parse failed, retrying ({attempt}/{self.retry_attempts})")
                        sleep(self.delay)
                        continue
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return parsed, raw_response, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to evaluate content pair")
    
    def _parse_evaluation(self, evaluation: str) -> Optional[Dict]:
        """
        Parse LLM evaluation output into structured scores.
        
        Returns:
            Dict with scores and preferences, or None if parsing fails
        """
        if not evaluation or (isinstance(evaluation, float) and pd.isna(evaluation)):
            return None
        
        try:
            txt = evaluation.replace('\r', '\n')
            # Strip markdown emphasis so verdict letters aren't masked by **bold**/##headers
            # (e.g. "## Overall Winner: **B**"); the score numbers/letters are unaffected.
            txt = re.sub(r'[*#`]+', '', txt)
            txt_lower = txt.lower()
            
            def extract_scores_for(category_name: str):
                """Extract A and B scores for a given category."""
                start = txt_lower.find(category_name.lower())
                if start == -1:
                    return None, None
                
                next_categories = ["grammar:", "coherence:", "likability:", "overall winner:"]
                next_categories = [c for c in next_categories if not c.startswith(category_name.lower())]
                next_idxs = [txt_lower.find(c, start+1) for c in next_categories]
                next_idxs = [i for i in next_idxs if i != -1]
                end = min(next_idxs) if next_idxs else len(txt)
                seg = txt[start:end]
                
                # Pattern: "A - 4/5" or "A: 4/5" or "A - 4" or "A - 3.5/5"
                # Look for lines that are ONLY the score (not "Issues in A:")
                # Match: newline/start, optional whitespace, A/B, whitespace, dash/colon, number
                lines = seg.split('\n')
                score_a = None
                score_b = None
                
                for line in lines:
                    line_stripped = line.strip()
                    # Match lines like "A - 4/5" or "A: 3.5/5" but not "Issues in A:"
                    if re.match(r'^A\s*[-:]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+)?\s*$', line_stripped, re.IGNORECASE):
                        match = re.match(r'^A\s*[-:]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+)?\s*$', line_stripped, re.IGNORECASE)
                        if match:
                            score_a = float(match.group(1))
                    elif re.match(r'^B\s*[-:]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+)?\s*$', line_stripped, re.IGNORECASE):
                        match = re.match(r'^B\s*[-:]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+)?\s*$', line_stripped, re.IGNORECASE)
                        if match:
                            score_b = float(match.group(1))
                
                if score_a is not None and score_b is not None:
                    return score_a, score_b
                
                return None, None
            
            # gA, gB = extract_scores_for("Grammar")
            cA, cB = extract_scores_for("Coherence")
            lA, lB = extract_scores_for("Likability")
            
            def extract_pref(category_name: str):
                pattern = rf'{category_name}.*?Preference:\s*([AB])'
                match = re.search(pattern, txt, re.IGNORECASE | re.DOTALL)
                return match.group(1).upper() if match else None
            
            # grammar_pref = extract_pref("Grammar")
            coherence_pref = extract_pref("Coherence")
            likability_pref = extract_pref("Likability")
            
            # Tolerate "Overall Winner: B", "Overall Winner\n\nB", optional colon.
            overall_match = re.search(r'Overall\s+Winner\s*:?\s*([AB])\b', txt, re.IGNORECASE)
            overall_pref = overall_match.group(1).upper() if overall_match else None
            
            if all(v is None for v in (cA, cB, lA, lB)):
                self.logger.error("Parse failure: no numeric scores found")
                return None
            
            parsed = {
                # 'grammar_score_a': float(gA) if gA is not None else 0.0,
                # 'grammar_score_b': float(gB) if gB is not None else 0.0,
                'coherence_score_a': float(cA) if cA is not None else 0.0,
                'coherence_score_b': float(cB) if cB is not None else 0.0,
                'likability_score_a': float(lA) if lA is not None else 0.0,
                'likability_score_b': float(lB) if lB is not None else 0.0,
                # 'grammar_pref': grammar_pref or '',
                'coherence_pref': coherence_pref or '',
                'likability_pref': likability_pref or '',
                'overall_pref': overall_pref or ''
            }
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return None
    
    def evaluate_batch_pairwise(
        self,
        df: pd.DataFrame,
        content_column: str = "revised_base",
        baseline_subset: int = 23,
        comparison_subsets: Optional[List[int]] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate content quality by comparing baseline subset against others.
        
        Args:
            df: Input DataFrame with multiple subsets per instruction_number
            content_column: Column name containing content to evaluate
            baseline_subset: Subset size to use as baseline (default: 23)
            comparison_subsets: List of subsets to compare against (default: all others)
            output_path: Optional path to save results (saved incrementally)
            
        Returns:
            DataFrame in long format with one row per comparison
        """
        if "instruction_number" not in df.columns:
            raise ValueError("DataFrame must have 'instruction_number' column")
        if "subset_size" not in df.columns:
            raise ValueError("DataFrame must have 'subset_size' column")
        if content_column not in df.columns:
            raise ValueError(f"DataFrame must have '{content_column}' column")
        
        instruction_numbers = sorted(df["instruction_number"].unique())
        
        all_subsets = sorted(df["subset_size"].unique())
        if baseline_subset not in all_subsets:
            raise ValueError(f"Baseline subset {baseline_subset} not found in data")
        
        if comparison_subsets is None:
            comparison_subsets = [s for s in all_subsets if s != baseline_subset]
        else:
            for cs in comparison_subsets:
                if cs not in all_subsets:
                    raise ValueError(f"Comparison subset {cs} not found in data")
        
        self.logger.info(f"Baseline subset: {baseline_subset}")
        self.logger.info(f"Comparison subsets: {comparison_subsets}")
        self.logger.info(f"Processing {len(instruction_numbers)} instructions")
        
        results = []
        total_comparisons = len(instruction_numbers) * len(comparison_subsets)
        comparison_count = 0
        
        for instruction_num in instruction_numbers:
            baseline_rows = df[
                (df["instruction_number"] == instruction_num) &
                (df["subset_size"] == baseline_subset)
            ]
            
            if len(baseline_rows) == 0:
                self.logger.warning(
                    f"No baseline subset {baseline_subset} found for instruction {instruction_num}, skipping"
                )
                continue
            
            baseline_row = baseline_rows.iloc[0]
            baseline_content = baseline_row[content_column]
            
            constraints_col = "selected_constraints" if "selected_constraints" in df.columns else "constraints"
            baseline_constraints = baseline_row.get(constraints_col, "")
            
            for comp_subset in comparison_subsets:
                comparison_count += 1
                
                comp_rows = df[
                    (df["instruction_number"] == instruction_num) &
                    (df["subset_size"] == comp_subset)
                ]
                
                if len(comp_rows) == 0:
                    self.logger.warning(
                        f"No subset {comp_subset} found for instruction {instruction_num}, skipping"
                    )
                    continue
                
                comp_row = comp_rows.iloc[0]
                comp_content = comp_row[content_column]
                comp_constraints = comp_row.get(constraints_col, "")
                
                self.logger.info(
                    f"Evaluating instruction #{instruction_num}: "
                    f"subset {baseline_subset} vs {comp_subset} "
                    f"({comparison_count}/{total_comparisons})"
                )
                
                order = np.random.randint(2)
                if order == 0:
                    content_a = baseline_content
                    content_b = comp_content
                else:
                    content_a = comp_content
                    content_b = baseline_content
                
                try:
                    parsed, raw_response, tokens = self.evaluate_pair(
                        content_a=content_a,
                        content_b=content_b,
                        log=True
                    )
                    
                    result_row = {
                        "instruction_number": instruction_num,
                        "blog1": baseline_row.get("blog1", ""),
                        "blog2": baseline_row.get("blog2", ""),
                        "main_task": baseline_row.get("main_task", ""),
                        "baseline_subset": baseline_subset,
                        "comparison_subset": comp_subset,
                        "content_baseline": baseline_content,
                        "constraints_baseline": baseline_constraints,
                        "content_comparison": comp_content,
                        "constraints_comparison": comp_constraints,
                        "order": order,
                        "evaluation_raw": raw_response,
                        "eval_tokens": tokens,
                        "eval_model": self.model,
                        "eval_timestamp": datetime.now().isoformat()
                    }
                    
                    if parsed:
                        result_row.update(parsed)
                    else:
                        result_row.update({
                            # "grammar_score_a": 0.0,
                            # "grammar_score_b": 0.0,
                            "coherence_score_a": 0.0,
                            "coherence_score_b": 0.0,
                            "likability_score_a": 0.0,
                            "likability_score_b": 0.0,
                            # "grammar_pref": "",
                            "coherence_pref": "",
                            "likability_pref": "",
                            "overall_pref": ""
                        })
                    
                    results.append(result_row)
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to evaluate instruction {instruction_num} "
                        f"({baseline_subset} vs {comp_subset}): {e}"
                    )
                    results.append({
                        "instruction_number": instruction_num,
                        "blog1": baseline_row.get("blog1", ""),
                        "blog2": baseline_row.get("blog2", ""),
                        "main_task": baseline_row.get("main_task", ""),
                        "baseline_subset": baseline_subset,
                        "comparison_subset": comp_subset,
                        "content_baseline": baseline_content,
                        "constraints_baseline": baseline_constraints,
                        "content_comparison": comp_content,
                        "constraints_comparison": comp_constraints,
                        "order": 0,
                        # "grammar_score_a": 0.0,
                        # "grammar_score_b": 0.0,
                        "coherence_score_a": 0.0,
                        "coherence_score_b": 0.0,
                        "likability_score_a": 0.0,
                        "likability_score_b": 0.0,
                        # "grammar_pref": "",
                        "coherence_pref": "",
                        "likability_pref": "",
                        "overall_pref": "",
                        "evaluation_raw": "",
                        "eval_tokens": 0,
                        "eval_model": self.model,
                        "eval_timestamp": datetime.now().isoformat()
                    })
                
                if output_path and len(results) > 0:
                    result_df = pd.DataFrame(results)
                    result_df.to_csv(output_path, index=False, encoding="utf-8")
                    self.logger.debug(
                        f"Progress saved ({comparison_count}/{total_comparisons})"
                    )
        
        result_df = pd.DataFrame(results)

        if output_path:
            self.logger.info(f"All quality evaluations saved to {output_path}")

        return result_df

    def evaluate_batch_base_vs_revised(
        self,
        df: pd.DataFrame,
        revised_column: str = "revised_base",
        base_column: str = "base_content",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare each row's constraint-fitted content against its original base story.

        Unlike evaluate_batch_pairwise (which pairs two rows at different subset
        sizes), this is a per-row comparison: revised_column vs base_column on the
        SAME row. Every row that has both is evaluated; subset_size is carried
        through unchanged. A/B order is randomized per row and de-randomized into
        explicit base/revised winners.

        Returns a DataFrame with one row per input row.
        """
        for col in (revised_column, base_column):
            if col not in df.columns:
                raise ValueError(f"DataFrame must have '{col}' column")

        results = []
        total = len(df)

        for count, (_, row) in enumerate(df.iterrows(), start=1):
            base_content = self._clean_text(row.get(base_column))
            revised_content = self._clean_text(row.get(revised_column))
            instruction_num = row.get("instruction_number", count)

            if not base_content or not revised_content:
                self.logger.warning(
                    f"Row {count}/{total} (instruction {instruction_num}): "
                    f"missing base or revised content, skipping"
                )
                continue

            constraints_col = "selected_constraints" if "selected_constraints" in df.columns else "constraints"
            revised_constraints = row.get(constraints_col, "")

            # Randomize slot assignment; remember which slot held the revised text.
            order = int(np.random.randint(2))
            if order == 0:
                content_a, content_b = revised_content, base_content
            else:
                content_a, content_b = base_content, revised_content

            self.logger.info(
                f"Evaluating instruction #{instruction_num} "
                f"(subset {row.get('subset_size', 'NA')}) base vs revised "
                f"({count}/{total})"
            )

            def _winner(pref: str) -> str:
                """Map an A/B preference back to base/revised given the order."""
                if pref not in ("A", "B"):
                    return ""
                revised_slot = "A" if order == 0 else "B"
                return "revised" if pref == revised_slot else "base"

            base_row = {
                "instruction_number": instruction_num,
                "blog1": row.get("blog1", ""),
                "blog2": row.get("blog2", ""),
                "main_task": row.get("main_task", ""),
                "subset_size": row.get("subset_size", ""),
                "content_base": base_content,
                "content_revised": revised_content,
                "constraints_revised": revised_constraints,
                "order": order,
                "eval_model": self.model,
                "eval_timestamp": datetime.now().isoformat(),
            }

            try:
                parsed, raw_response, tokens = self.evaluate_pair(
                    content_a=content_a, content_b=content_b, log=True
                )
                base_row["evaluation_raw"] = raw_response
                base_row["eval_tokens"] = tokens
                if parsed:
                    base_row.update(parsed)
                else:
                    base_row.update({
                        "coherence_score_a": 0.0, "coherence_score_b": 0.0,
                        "likability_score_a": 0.0, "likability_score_b": 0.0,
                        "coherence_pref": "", "likability_pref": "", "overall_pref": "",
                    })
            except Exception as e:
                self.logger.error(
                    f"Failed instruction {instruction_num} (base vs revised): {e}"
                )
                base_row.update({
                    "coherence_score_a": 0.0, "coherence_score_b": 0.0,
                    "likability_score_a": 0.0, "likability_score_b": 0.0,
                    "coherence_pref": "", "likability_pref": "", "overall_pref": "",
                    "evaluation_raw": "", "eval_tokens": 0,
                })

            # Derived, de-randomized winners (base/revised) for direct reading.
            base_row["coherence_winner"] = _winner(base_row.get("coherence_pref", ""))
            base_row["likability_winner"] = _winner(base_row.get("likability_pref", ""))
            base_row["winner"] = _winner(base_row.get("overall_pref", ""))
            base_row["revised_win"] = base_row["winner"] == "revised"

            # De-randomized numeric scores so downstream means aren't slot-mixed.
            base_row["coherence_base"], base_row["coherence_revised"] = self._role_scores(
                base_row.get("coherence_score_a", 0.0), base_row.get("coherence_score_b", 0.0), order)
            base_row["likability_base"], base_row["likability_revised"] = self._role_scores(
                base_row.get("likability_score_a", 0.0), base_row.get("likability_score_b", 0.0), order)

            results.append(base_row)

            if output_path and results:
                pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")

        result_df = pd.DataFrame(results)
        if output_path:
            self.logger.info(f"All base-vs-revised evaluations saved to {output_path}")
        return result_df

    @staticmethod
    def _clean_text(val) -> str:
        """Coerce a cell value to a stripped string; NaN/None -> '' (no crash)."""
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        return str(val).strip()

    @staticmethod
    def _winner_from_pref(pref: str, order: int) -> str:
        """Map an A/B preference back to base/revised given the slot order.

        order==0 -> slot A held the revised text; order==1 -> slot A held base.
        """
        if pref not in ("A", "B"):
            return ""
        revised_slot = "A" if order == 0 else "B"
        return "revised" if pref == revised_slot else "base"

    @staticmethod
    def _role_scores(score_a, score_b, order):
        """De-randomize (slot A, slot B) scores into (base, revised) given order.

        order==0 -> slot A held revised; order==1 -> slot A held base.
        """
        if order == 0:
            return score_b, score_a   # (base, revised)
        return score_a, score_b

    def evaluate_batch_base_vs_revised_anthropic(
        self,
        df: pd.DataFrame,
        revised_column: str = "revised_base",
        base_column: str = "base_content",
        output_path: Optional[str] = None,
        poll_seconds: int = 30,
    ) -> pd.DataFrame:
        """Base-vs-revised pairwise quality eval via the Anthropic Message Batches API.

        One request per row (custom_id = instruction_number), ~50% cheaper, async.
        The randomized A/B order chosen at submit time is stored per custom_id so the
        returned A/B preferences de-randomize into base/revised winners correctly.
        Results arrive at the end (no incremental save); the batch_id is logged so a
        crashed poll is recoverable. Same output schema as evaluate_batch_base_vs_revised.
        """
        import time
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        if not isinstance(self.llm_client, AnthropicClient):
            raise ValueError("evaluate_batch_base_vs_revised_anthropic requires an AnthropicClient")
        for col in (revised_column, base_column):
            if col not in df.columns:
                raise ValueError(f"DataFrame must have '{col}' column")

        constraints_col = "selected_constraints" if "selected_constraints" in df.columns else "constraints"

        # custom_id -> per-row metadata (idx + randomized order). Guard collisions.
        meta = {}
        requests = []
        for idx, row in df.iterrows():
            base_content = self._clean_text(row.get(base_column))
            revised_content = self._clean_text(row.get(revised_column))
            if not base_content or not revised_content:
                self.logger.warning(
                    f"Row idx {idx}: missing base or revised content, skipping"
                )
                continue
            cid = str(row.get("instruction_number", idx))
            if cid in meta:
                raise ValueError(f"Duplicate custom_id '{cid}' — instruction_number must be unique")

            order = int(np.random.randint(2))
            if order == 0:
                content_a, content_b = revised_content, base_content
            else:
                content_a, content_b = base_content, revised_content

            meta[cid] = {"idx": idx, "order": order,
                         "base_content": base_content, "revised_content": revised_content}
            requests.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": get_pairwise_quality_prompt(content_a, content_b)}],
                ),
            ))

        if not requests:
            raise ValueError("No rows had both base and revised content")

        batch = self.llm_client.create_batch(requests)
        self.logger.info(f"Submitted quality batch {batch.id} with {len(requests)} requests "
                         f"(recover with this id if interrupted)")

        while True:
            b = self.llm_client.retrieve_batch(batch.id)
            if b.processing_status == "ended":
                break
            rc = b.request_counts
            self.logger.info(f"batch {batch.id}: processing={rc.processing} "
                             f"succeeded={rc.succeeded} errored={rc.errored}")
            time.sleep(poll_seconds)

        empty_scores = {
            "coherence_score_a": 0.0, "coherence_score_b": 0.0,
            "likability_score_a": 0.0, "likability_score_b": 0.0,
            "coherence_pref": "", "likability_pref": "", "overall_pref": "",
        }

        rows_by_cid = {}
        n_ok = 0
        for res in self.llm_client.batch_results(batch.id):
            m = meta.get(res.custom_id)
            if m is None:
                self.logger.error(f"Result custom_id {res.custom_id} not in input — skipped")
                continue
            idx, order = m["idx"], m["order"]
            src = df.loc[idx]
            out = {
                "instruction_number": src.get("instruction_number", idx),
                "blog1": src.get("blog1", ""),
                "blog2": src.get("blog2", ""),
                "main_task": src.get("main_task", ""),
                "subset_size": src.get("subset_size", ""),
                "content_base": m["base_content"],
                "content_revised": m["revised_content"],
                "constraints_revised": src.get(constraints_col, ""),
                "order": order,
                "eval_model": self.model,
                "eval_timestamp": datetime.now().isoformat(),
            }
            if res.result.type == "succeeded":
                msg = res.result.message
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                parsed = self._parse_evaluation(text)
                out["evaluation_raw"] = text
                out["eval_tokens"] = msg.usage.input_tokens + msg.usage.output_tokens
                out.update(parsed if parsed else empty_scores)
                n_ok += 1
            else:
                self.logger.error(f"custom_id {res.custom_id}: result.type={res.result.type}")
                out["evaluation_raw"] = ""
                out["eval_tokens"] = 0
                out.update(empty_scores)

            out["coherence_winner"] = self._winner_from_pref(out.get("coherence_pref", ""), order)
            out["likability_winner"] = self._winner_from_pref(out.get("likability_pref", ""), order)
            out["winner"] = self._winner_from_pref(out.get("overall_pref", ""), order)
            out["revised_win"] = out["winner"] == "revised"
            out["coherence_base"], out["coherence_revised"] = self._role_scores(
                out.get("coherence_score_a", 0.0), out.get("coherence_score_b", 0.0), order)
            out["likability_base"], out["likability_revised"] = self._role_scores(
                out.get("likability_score_a", 0.0), out.get("likability_score_b", 0.0), order)
            rows_by_cid[res.custom_id] = out

        # Emit a zeroed placeholder for any submitted request with no returned result,
        # so the output row count always matches the submitted set (no silent drops).
        for cid, m in meta.items():
            if cid in rows_by_cid:
                continue
            self.logger.error(f"No batch result returned for custom_id {cid} — emitting zeroed row")
            order = m["order"]
            src = df.loc[m["idx"]]
            ph = {
                "instruction_number": src.get("instruction_number", m["idx"]),
                "blog1": src.get("blog1", ""), "blog2": src.get("blog2", ""),
                "main_task": src.get("main_task", ""), "subset_size": src.get("subset_size", ""),
                "content_base": m["base_content"], "content_revised": m["revised_content"],
                "constraints_revised": src.get(constraints_col, ""), "order": order,
                "eval_model": self.model, "eval_timestamp": datetime.now().isoformat(),
                "evaluation_raw": "", "eval_tokens": 0,
                **empty_scores,
                "coherence_winner": "", "likability_winner": "", "winner": "", "revised_win": False,
                "coherence_base": 0.0, "coherence_revised": 0.0,
                "likability_base": 0.0, "likability_revised": 0.0,
            }
            rows_by_cid[cid] = ph

        # Preserve input order (results may arrive out of order).
        results = [rows_by_cid[cid] for cid in meta if cid in rows_by_cid]
        self.logger.info(f"Quality batch complete: {n_ok}/{len(requests)} succeeded, "
                         f"{len(results)} rows emitted")

        result_df = pd.DataFrame(results)
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"All base-vs-revised batch evaluations saved to {output_path}")
        return result_df

    def evaluate_batch_base_vs_revised_openai(
        self,
        df: pd.DataFrame,
        revised_column: str = "revised_base",
        base_column: str = "base_content",
        output_path: Optional[str] = None,
        poll_seconds: int = 30,
    ) -> pd.DataFrame:
        """Base-vs-revised pairwise quality eval via the OpenAI Batch API (~50% cheaper).

        Mirrors evaluate_batch_base_vs_revised_anthropic: one request per row
        (custom_id = instruction_number), randomized A/B order stored per custom_id and
        de-randomized into base/revised winners. OpenAI mechanics differ: a JSONL of
        /v1/chat/completions requests is uploaded and submitted, then the output file is
        downloaded and mapped by custom_id. Same output schema. OpenAI provider only.
        """
        import time

        if not isinstance(self.llm_client, OpenAIClient):
            raise ValueError("evaluate_batch_base_vs_revised_openai requires an OpenAIClient")
        for col in (revised_column, base_column):
            if col not in df.columns:
                raise ValueError(f"DataFrame must have '{col}' column")

        constraints_col = "selected_constraints" if "selected_constraints" in df.columns else "constraints"

        meta = {}
        requests = []
        for idx, row in df.iterrows():
            base_content = self._clean_text(row.get(base_column))
            revised_content = self._clean_text(row.get(revised_column))
            if not base_content or not revised_content:
                self.logger.warning(f"Row idx {idx}: missing base or revised content, skipping")
                continue
            cid = str(row.get("instruction_number", idx))
            if cid in meta:
                raise ValueError(f"Duplicate custom_id '{cid}' — instruction_number must be unique")

            order = int(np.random.randint(2))
            if order == 0:
                content_a, content_b = revised_content, base_content
            else:
                content_a, content_b = base_content, revised_content

            meta[cid] = {"idx": idx, "order": order,
                         "base_content": base_content, "revised_content": revised_content}
            requests.append({
                "custom_id": cid,
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": get_pairwise_quality_prompt(content_a, content_b)}],
                },
            })

        if not requests:
            raise ValueError("No rows had both base and revised content")

        batch = self.llm_client.create_batch(requests)
        self.logger.info(f"Submitted OpenAI quality batch {batch.id} with {len(requests)} requests "
                         f"(recover with this id if interrupted)")

        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            b = self.llm_client.retrieve_batch(batch.id)
            if b.status in terminal:
                break
            rc = getattr(b, "request_counts", None)
            self.logger.info(f"batch {batch.id}: status={b.status} "
                             f"counts={getattr(rc,'completed',0)}/{getattr(rc,'total',0)} "
                             f"failed={getattr(rc,'failed',0)}")
            time.sleep(poll_seconds)

        if b.status != "completed":
            self.logger.error(f"OpenAI batch ended with status={b.status}; no output expected")

        empty_scores = {
            "coherence_score_a": 0.0, "coherence_score_b": 0.0,
            "likability_score_a": 0.0, "likability_score_b": 0.0,
            "coherence_pref": "", "likability_pref": "", "overall_pref": "",
        }

        rows_by_cid = {}
        n_ok = 0
        for res in self.llm_client.batch_results(b):
            cid = res.get("custom_id")
            m = meta.get(cid)
            if m is None:
                self.logger.error(f"Result custom_id {cid} not in input — skipped")
                continue
            idx, order = m["idx"], m["order"]
            src = df.loc[idx]
            out = {
                "instruction_number": src.get("instruction_number", idx),
                "blog1": src.get("blog1", ""), "blog2": src.get("blog2", ""),
                "main_task": src.get("main_task", ""), "subset_size": src.get("subset_size", ""),
                "content_base": m["base_content"], "content_revised": m["revised_content"],
                "constraints_revised": src.get(constraints_col, ""), "order": order,
                "eval_model": self.model, "eval_timestamp": datetime.now().isoformat(),
            }
            err = res.get("error")
            resp = res.get("response") or {}
            body = resp.get("body") or {}
            if err is None and resp.get("status_code") == 200 and body.get("choices"):
                text = (body["choices"][0]["message"].get("content") or "")
                parsed = self._parse_evaluation(text)
                out["evaluation_raw"] = text
                out["eval_tokens"] = (body.get("usage") or {}).get("total_tokens", 0)
                out.update(parsed if parsed else empty_scores)
                if parsed:
                    n_ok += 1
            else:
                self.logger.error(f"custom_id {cid}: error={err} status={resp.get('status_code')}")
                out["evaluation_raw"] = ""
                out["eval_tokens"] = 0
                out.update(empty_scores)

            out["coherence_winner"] = self._winner_from_pref(out.get("coherence_pref", ""), order)
            out["likability_winner"] = self._winner_from_pref(out.get("likability_pref", ""), order)
            out["winner"] = self._winner_from_pref(out.get("overall_pref", ""), order)
            out["revised_win"] = out["winner"] == "revised"
            out["coherence_base"], out["coherence_revised"] = self._role_scores(
                out.get("coherence_score_a", 0.0), out.get("coherence_score_b", 0.0), order)
            out["likability_base"], out["likability_revised"] = self._role_scores(
                out.get("likability_score_a", 0.0), out.get("likability_score_b", 0.0), order)
            rows_by_cid[cid] = out

        for cid, m in meta.items():
            if cid in rows_by_cid:
                continue
            self.logger.error(f"No batch result returned for custom_id {cid} — emitting zeroed row")
            order = m["order"]
            src = df.loc[m["idx"]]
            rows_by_cid[cid] = {
                "instruction_number": src.get("instruction_number", m["idx"]),
                "blog1": src.get("blog1", ""), "blog2": src.get("blog2", ""),
                "main_task": src.get("main_task", ""), "subset_size": src.get("subset_size", ""),
                "content_base": m["base_content"], "content_revised": m["revised_content"],
                "constraints_revised": src.get(constraints_col, ""), "order": order,
                "eval_model": self.model, "eval_timestamp": datetime.now().isoformat(),
                "evaluation_raw": "", "eval_tokens": 0, **empty_scores,
                "coherence_winner": "", "likability_winner": "", "winner": "", "revised_win": False,
                "coherence_base": 0.0, "coherence_revised": 0.0,
                "likability_base": 0.0, "likability_revised": 0.0,
            }

        results = [rows_by_cid[cid] for cid in meta if cid in rows_by_cid]
        self.logger.info(f"OpenAI quality batch complete: {n_ok}/{len(requests)} parsed, "
                         f"{len(results)} rows emitted")

        result_df = pd.DataFrame(results)
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"All base-vs-revised batch evaluations saved to {output_path}")
        return result_df
