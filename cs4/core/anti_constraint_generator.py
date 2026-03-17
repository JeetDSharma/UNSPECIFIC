"""
Anti-constraint base generation - generates content that intentionally violates specified constraints.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional, Tuple
from datetime import datetime

from cs4.core.prompts import get_anti_constraint_generation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class AntiConstraintGenerator:
    """Generate base content that intentionally avoids satisfying easy constraints."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize anti-constraint generator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_BASE_GEN_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.logger = logging.getLogger("CS4Generator")
    
    def generate_avoiding_constraints(
        self,
        task: str,
        constraints_to_avoid: str,
        log: bool = True
    ) -> Tuple[str, int]:
        """
        Generate base content that avoids satisfying specified constraints.
        
        Args:
            task: Main task description
            constraints_to_avoid: Constraints to intentionally NOT satisfy
            log: Whether to log token usage
            
        Returns:
            Tuple of (generated_content, tokens_used)
        """
        prompt = get_anti_constraint_generation_prompt(
            content_type=self.content_type,
            task=task,
            constraints_to_avoid=constraints_to_avoid
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    content = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    content = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return content, tokens
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to generate anti-constraint base")
    
    def generate_batch(
        self,
        constraints_df: pd.DataFrame,
        evaluation_df: pd.DataFrame,
        task_column: str = "main_task",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate base content avoiding satisfied constraints for a batch.
        
        Args:
            constraints_df: Original constraints (from common_constraints.csv)
            evaluation_df: Evaluation results identifying satisfied constraints
            task_column: Column containing task descriptions
            output_path: Optional path to save results
            
        Returns:
            DataFrame with anti-constraint base content
        """
        merged = pd.merge(constraints_df, evaluation_df, on="instruction_number", suffixes=("_orig", "_eval"))
        
        self.logger.info(f"Generating anti-constraint base for {len(merged)} samples")
        
        results = []
        for idx, row in merged.iterrows():
            instruction_num = row["instruction_number"]
            task = row[task_column]
            all_constraints = row["constraints_orig"]
            satisfaction_results = row["satisfaction_results"]
            
            satisfied_constraints = self._extract_satisfied_constraints(
                all_constraints, satisfaction_results
            )
            
            self.logger.info(f"Processing sample #{instruction_num}")
            self.logger.info(f"  Found {len(satisfied_constraints.split(chr(10)))} satisfied constraints to avoid")
            
            try:
                content, tokens = self.generate_avoiding_constraints(
                    task=task,
                    constraints_to_avoid=satisfied_constraints,
                    log=True
                )
                
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints_to_avoid": satisfied_constraints,
                    "all_constraints": all_constraints,
                    "anti_base_content": content,
                    "content_length": len(content),
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Failed to generate for sample {instruction_num}: {e}")
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints_to_avoid": satisfied_constraints,
                    "all_constraints": all_constraints,
                    "anti_base_content": "",
                    "content_length": 0,
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Anti-constraint base content saved to {output_path}")
        
        return result_df
    
    def _extract_satisfied_constraints(
        self, 
        all_constraints: str, 
        satisfaction_results: str
    ) -> str:
        """Extract only the constraints marked as satisfied (Yes)."""
        constraint_lines = [c for c in all_constraints.split('\n') if c.strip()]
        result_lines = satisfaction_results.split('\n')
        
        satisfied = []
        for line in result_lines:
            if '. Yes' in line:
                try:
                    num = int(line.split('.')[0].strip())
                    if 1 <= num <= len(constraint_lines):
                        satisfied.append(constraint_lines[num - 1])
                except:
                    continue
        
        return '\n'.join(satisfied)
