"""
Constraint satisfaction evaluator module.
"""

import pandas as pd
import logging
import re
from time import sleep
from typing import Optional, Tuple
from datetime import datetime

from cs4.core.prompts import get_evaluation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class ConstraintEvaluator:
    """Evaluate constraint satisfaction in generated content."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint evaluator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_EVALUATION_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        
        self.logger = logging.getLogger("CS4Evaluator")
    
    def evaluate_content(
        self,
        content: str,
        constraints: str,
        log: bool = True
    ) -> Tuple[str, int, int]:
        """
        Evaluate content against constraints.
        
        Args:
            content: Generated content to evaluate
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (satisfaction_results, num_satisfied, tokens_used)
        """
        prompt = get_evaluation_prompt(
            content_type=self.content_type,
            content=content,
            constraints=constraints
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model
                    )
                    results = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model    
                    )
                    results = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Extract number of satisfied constraints
                num_satisfied = self._extract_satisfaction_count(results)
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                    self.logger.info(f"Constraints satisfied: {num_satisfied}")
                
                return results, num_satisfied, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to evaluate content")
    
    def _extract_satisfaction_count(self, results: str) -> int:
        """Extract the number of satisfied constraints from evaluation results."""
        # Count actual "Yes" lines - more reliable than LLM's stated count
        yes_count = len(re.findall(r'^\d+\.\s+Yes', results, re.MULTILINE))
        if yes_count > 0:
            return yes_count
        
        # Fallback: use LLM's stated count
        match = re.search(r'Number of constraints satisfied:\s*(\d+)', results)
        if match:
            return int(match.group(1))
        
        return 0
    
    def evaluate_batch(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        constraints_column: str = "constraints",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate constraint satisfaction for a batch of samples.
        Preserves all original columns and adds new evaluation columns.
        
        Args:
            df: Input DataFrame (can be any CSV with content and constraints)
            content_column: Name of column with content to evaluate
            constraints_column: Name of column with constraints
            output_path: Optional path to save results (saved incrementally)
            
        Returns:
            DataFrame with all original columns plus evaluation results
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        if constraints_column not in df.columns:
            raise ValueError(f"Column '{constraints_column}' not found in DataFrame")
        
        # Check for instruction_number
        has_instruction_num = "instruction_number" in df.columns
        if not has_instruction_num:
            self.logger.warning("No 'instruction_number' column found, using index")
        
        # Check for subset_size column (from bucketed constraints)
        has_subset_size = "subset_size" in df.columns
        if has_subset_size:
            self.logger.info("Using 'subset_size' column for constraint count")
        else:
            self.logger.info("No 'subset_size' column found, will parse constraints with regex")
        
        self.logger.info(f"Evaluating {len(df)} samples")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize new columns
        result_df["satisfaction_results"] = ""
        result_df["num_satisfied"] = 0
        result_df["total_constraints"] = 0
        result_df["satisfaction_rate"] = 0.0
        result_df["eval_model"] = ""
        result_df["eval_tokens"] = 0
        result_df["eval_timestamp"] = ""
        
        for idx, row in df.iterrows():
            content = row[content_column]
            constraints = row[constraints_column]
            instruction_num = row["instruction_number"] if has_instruction_num else idx + 1
            
            self.logger.info(f"Evaluating sample #{instruction_num} (row {idx + 1}/{len(df)})")
            
            try:
                satisfaction_results, num_satisfied, tokens = self.evaluate_content(
                    content=content,
                    constraints=constraints,
                    log=True
                )
                
                # Count total constraints - use subset_size if available, otherwise parse
                if has_subset_size:
                    total_constraints = int(row["subset_size"])
                else:
                    total_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                satisfaction_rate = num_satisfied / total_constraints if total_constraints > 0 else 0.0
                
                result_df.at[idx, "satisfaction_results"] = satisfaction_results
                result_df.at[idx, "num_satisfied"] = num_satisfied
                result_df.at[idx, "total_constraints"] = total_constraints
                result_df.at[idx, "satisfaction_rate"] = satisfaction_rate
                result_df.at[idx, "eval_model"] = self.model
                result_df.at[idx, "eval_tokens"] = tokens
                result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate sample {instruction_num}: {e}"
                )
                # Get total_constraints even in error case
                if has_subset_size:
                    error_total_constraints = int(row["subset_size"])
                else:
                    error_total_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                result_df.at[idx, "satisfaction_results"] = ""
                result_df.at[idx, "num_satisfied"] = 0
                result_df.at[idx, "total_constraints"] = error_total_constraints
                result_df.at[idx, "satisfaction_rate"] = 0.0
                result_df.at[idx, "eval_model"] = self.model
                result_df.at[idx, "eval_tokens"] = 0
                result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
            
            # Save incrementally after each row to prevent data loss
            if output_path:
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                self.logger.debug(f"Progress saved (row {idx + 1}/{len(df)})")
        
        if output_path:
            self.logger.info(f"All evaluation results saved to {output_path}")
        
        return result_df


# Legacy function for backward compatibility
def evaluate_constraints(
    input_path: str,
    output_path: str,
    model: str = None,
    content_type: str = "blog"
):
    """
    Legacy interface for constraint evaluation.
    
    Args:
        input_path: Path to CSV with generated content
        output_path: Path to save evaluation results
        model: LLM model identifier
        content_type: Type of content (blog, story, news)
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    evaluator = ConstraintEvaluator(model=model, content_type=content_type)
    df = pd.read_csv(input_path, encoding="utf-8")
    
    result_df = evaluator.evaluate_batch(df, output_path=output_path)
    
    return result_df
