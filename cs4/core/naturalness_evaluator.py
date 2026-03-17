"""
Naturalness evaluation module for comparing constraint naturalness.
"""

import pandas as pd
import numpy as np
import logging
import re
from time import sleep
from typing import Optional, Dict, Tuple
from datetime import datetime

from cs4.core.prompts import get_naturalness_evaluation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class NaturalnessEvaluator:
    """Evaluate constraint naturalness through pairwise comparisons."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize naturalness evaluator.
        
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
        
        self.logger = logging.getLogger("CS4NaturalnessEvaluator")
    
    def evaluate_pair(
        self,
        prompt: str,
        constraint_a: str,
        constraint_b: str,
        log: bool = True
    ) -> Tuple[Optional[Dict], str, int]:
        """
        Evaluate a pair of constraints for naturalness comparison.
        
        Args:
            prompt: Writing prompt/main task
            constraint_a: First constraint to compare
            constraint_b: Second constraint to compare
            log: Whether to log token usage
            
        Returns:
            Tuple of (parsed_results_dict, raw_response, tokens_used)
        """
        evaluation_prompt = get_naturalness_evaluation_prompt(prompt, constraint_a, constraint_b)
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        model=self.model
                    )
                    raw_response = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": evaluation_prompt}],
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
        
        raise RuntimeError("Failed to evaluate constraint pair")
    
    def _parse_evaluation(self, evaluation: str) -> Optional[Dict]:
        """
        Parse LLM evaluation output into structured scores.
        
        Returns:
            Dict with preference and scores, or None if parsing fails
        """
        if not evaluation or (isinstance(evaluation, float) and pd.isna(evaluation)):
            return None
        
        try:
            txt = evaluation.replace('\r', '\n')
            txt_lower = txt.lower()
            
            # Extract preference (A or B)
            preference = None
            pref_match = re.search(r'preference:\s*([AB])', txt, re.IGNORECASE)
            if pref_match:
                preference = pref_match.group(1).upper()
            else:
                # Fallback: look for "prefer A" or "prefer B"
                fallback_match = re.search(r'prefer\s+([AB])', txt, re.IGNORECASE)
                if fallback_match:
                    preference = fallback_match.group(1).upper()
            
            # Extract Score A (1-5)
            score_a = None
            score_a_match = re.search(r'score\s*a:\s*([1-5])', txt, re.IGNORECASE)
            if score_a_match:
                score_a = int(score_a_match.group(1))
            else:
                # Fallback: look for "A - 4" or "A: 4" patterns
                fallback_a = re.search(r'(?:^|\n)\s*a\s*[-:]\s*([1-5])', txt, re.IGNORECASE | re.MULTILINE)
                if fallback_a:
                    score_a = int(fallback_a.group(1))
            
            # Extract Score B (1-5)
            score_b = None
            score_b_match = re.search(r'score\s*b:\s*([1-5])', txt, re.IGNORECASE)
            if score_b_match:
                score_b = int(score_b_match.group(1))
            else:
                # Fallback: look for "B - 3" or "B: 3" patterns
                fallback_b = re.search(r'(?:^|\n)\s*b\s*[-:]\s*([1-5])', txt, re.IGNORECASE | re.MULTILINE)
                if fallback_b:
                    score_b = int(fallback_b.group(1))
            
            # Validate that we got all required fields
            if preference is None or score_a is None or score_b is None:
                self.logger.error(
                    f"Parse failure: preference={preference}, score_a={score_a}, score_b={score_b}"
                )
                return None
            
            # Validate scores are in range
            if not (1 <= score_a <= 5 and 1 <= score_b <= 5):
                self.logger.error(f"Scores out of range: score_a={score_a}, score_b={score_b}")
                return None
            
            parsed = {
                'preference': preference,
                'score_a': float(score_a),
                'score_b': float(score_b)
            }
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return None
    
    def evaluate_batch(
        self,
        df: pd.DataFrame,
        prompt_column: str = "main_task",
        constraint_a_column: str = "constraint_a",
        constraint_b_column: str = "constraint_b",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate constraint naturalness for a batch of constraint pairs.
        
        Args:
            df: Input DataFrame with constraint pairs
            prompt_column: Column name containing prompts
            constraint_a_column: Column name for constraint A
            constraint_b_column: Column name for constraint B
            output_path: Optional path to save results (saved incrementally)
            
        Returns:
            DataFrame with evaluation results
        """
        if prompt_column not in df.columns:
            raise ValueError(f"DataFrame must have '{prompt_column}' column")
        if constraint_a_column not in df.columns:
            raise ValueError(f"DataFrame must have '{constraint_a_column}' column")
        if constraint_b_column not in df.columns:
            raise ValueError(f"DataFrame must have '{constraint_b_column}' column")
        
        self.logger.info(f"Processing {len(df)} constraint pairs")
        
        results = []
        
        for idx, row in df.iterrows():
            prompt = row[prompt_column]
            constraint_a = row[constraint_a_column]
            constraint_b = row[constraint_b_column]
            
            self.logger.info(f"Evaluating pair {idx + 1}/{len(df)}")
            
            # Randomize order to prevent position bias
            order = np.random.randint(2)
            if order == 0:
                eval_constraint_a = constraint_a
                eval_constraint_b = constraint_b
            else:
                eval_constraint_a = constraint_b
                eval_constraint_b = constraint_a
            
            try:
                parsed, raw_response, tokens = self.evaluate_pair(
                    prompt=prompt,
                    constraint_a=eval_constraint_a,
                    constraint_b=eval_constraint_b,
                    log=True
                )
                
                # Create result row with all original columns plus evaluation results
                result_row = row.to_dict()
                result_row.update({
                    "order": order,
                    "evaluation_raw": raw_response,
                    "eval_tokens": tokens,
                    "eval_model": self.model,
                    "eval_timestamp": datetime.now().isoformat()
                })
                
                if parsed:
                    # Adjust preference and scores based on randomization
                    if order == 1:
                        # We swapped A and B, so swap back the results
                        adjusted_preference = 'B' if parsed['preference'] == 'A' else 'A'
                        result_row.update({
                            "preference": adjusted_preference,
                            "score_a": parsed['score_b'],
                            "score_b": parsed['score_a']
                        })
                    else:
                        result_row.update(parsed)
                else:
                    result_row.update({
                        "preference": "",
                        "score_a": 0.0,
                        "score_b": 0.0
                    })
                
                results.append(result_row)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate pair {idx + 1}: {e}")
                result_row = row.to_dict()
                result_row.update({
                    "order": 0,
                    "preference": "",
                    "score_a": 0.0,
                    "score_b": 0.0,
                    "evaluation_raw": "",
                    "eval_tokens": 0,
                    "eval_model": self.model,
                    "eval_timestamp": datetime.now().isoformat()
                })
                results.append(result_row)
            
            # Save progress incrementally
            if output_path and len(results) > 0:
                result_df = pd.DataFrame(results)
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                self.logger.debug(f"Progress saved ({len(results)}/{len(df)})")
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            self.logger.info(f"All naturalness evaluations saved to {output_path}")
        
        return result_df
