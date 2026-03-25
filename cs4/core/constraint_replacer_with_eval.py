"""
Constraint replacement module that uses evaluation results to replace satisfied constraints.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cs4.core.prompts import get_constraint_replacement_with_eval_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class ConstraintReplacerWithEval:
    """Replace satisfied constraints with harder ones based on evaluation results."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint replacer.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_CONSTRAINT_MODEL
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.logger = logging.getLogger("CS4Generator")
    
    def replace_constraints(
        self,
        main_task: str,
        original_constraints: str,
        base_content: str,
        satisfaction_results: str,
        log: bool = True
    ) -> Tuple[str, str, int]:
        """
        Replace satisfied constraints with harder ones.
        
        Args:
            main_task: Main task description
            original_constraints: Original 39 constraints
            base_content: Base story/blog content
            satisfaction_results: Formatted string of which constraints were satisfied
            log: Whether to log token usage
            
        Returns:
            Tuple of (main_task, revised_constraints, tokens_used)
        """
        prompt = get_constraint_replacement_with_eval_prompt(
            main_task=main_task,
            original_constraints=original_constraints,
            base_content=base_content,
            satisfaction_results=satisfaction_results
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    response_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    response_text = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                revised_task, revised_constraints = self._parse_response(response_text)
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return revised_task, revised_constraints, tokens
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to replace constraints")
    
    def _parse_response(self, response_text: str) -> Tuple[str, str]:
        """Parse LLM response to extract main task and revised constraints."""
        lines = response_text.strip().split('\n')
        main_task = ""
        constraints_lines = []
        
        in_constraints = False
        for line in lines:
            line = line.strip()
            if line.startswith("Main Task:"):
                main_task = line.replace("Main Task:", "").strip()
            elif line.startswith("Constraints:") or line.startswith("Revised Constraints:"):
                in_constraints = True
            elif in_constraints and line:
                constraints_lines.append(line)
        
        constraints = "\n".join(constraints_lines)
        return main_task, constraints
    
    def _format_eval_results(self, eval_dict: Dict[str, Any]) -> str:
        """Format evaluation dictionary into a string for the prompt."""
        lines = []
        # Support different formats of evaluation results
        if "details" in eval_dict:
            details = eval_dict["details"]
        elif "constraint_details" in eval_dict:
            details = eval_dict["constraint_details"]
        else:
            return "No detailed evaluation results found."
            
        for i, detail in enumerate(details, 1):
            status = "Yes" if detail.get("satisfied", False) else "No"
            lines.append(f"{i}. {status} - {detail.get('reasoning', '')}")
            
        return "\n".join(lines)
    
    def replace_batch_parallel(
        self,
        constraints_df: pd.DataFrame,
        base_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        output_path: Optional[str] = None,
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Replace constraints for a batch of samples using parallel processing.
        
        Args:
            constraints_df: Original constraints (from common_constraints.csv)
            base_df: Base content (from base_generated.csv)
            eval_df: Evaluation results (from base_evaluated.csv)
            output_path: Optional path to save results
            max_workers: Maximum number of parallel workers (default: 5)
            
        Returns:
            DataFrame with revised constraints
        """
        import json
        
        # Merge the three dataframes
        merged = pd.merge(
            constraints_df, 
            base_df[["instruction_number", "base_content"]], 
            on="instruction_number", 
            how="left"
        )
        
        merged = pd.merge(
            merged,
            eval_df[["instruction_number", "satisfaction_results"]],
            on="instruction_number",
            how="left"
        )
        
        self.logger.info(f"Replacing constraints for {len(merged)} samples with {max_workers} parallel workers")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {}
            for idx, row in merged.iterrows():
                # Get the satisfaction string directly from the evaluation DataFrame
                satisfaction_str = row.get("satisfaction_results", "No evaluation results available.")
                
                future = executor.submit(
                    self.replace_constraints,
                    row.get("main_task", ""),
                    row.get("constraints", ""),
                    row.get("base_content", ""),
                    satisfaction_str,
                    False  # Don't log each individual sample
                )
                future_to_data[future] = (idx, row)
            
            # Collect results as they complete
            for future in as_completed(future_to_data):
                idx, row = future_to_data[future]
                instruction_num = row["instruction_number"]
                main_task = row.get("main_task", "")
                original_constraints = row.get("constraints", "")
                base_content = row.get("base_content", "")
                
                try:
                    revised_task, revised_constraints, tokens = future.result()
                    
                    results.append({
                        "instruction_number": instruction_num,
                        "main_task": revised_task,
                        "original_constraints": original_constraints,
                        "revised_constraints": revised_constraints,
                        "base_content": base_content,
                        "model_used": self.model,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    completed_count += 1
                    self.logger.info(f"Completed sample #{instruction_num} ({completed_count}/{len(merged)}) - {tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Failed to replace constraints for sample {instruction_num}: {e}")
                    results.append({
                        "instruction_number": instruction_num,
                        "main_task": main_task,
                        "original_constraints": original_constraints,
                        "revised_constraints": "",
                        "base_content": base_content,
                        "model_used": self.model,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat()
                    })
                    completed_count += 1
        
        # Sort results by instruction_number
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("instruction_number").reset_index(drop=True)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Revised constraints saved to {output_path}")
        
        return result_df
