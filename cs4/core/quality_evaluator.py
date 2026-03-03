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
            
            overall_match = re.search(r'Overall\s+Winner:\s*([AB])', txt, re.IGNORECASE)
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
