"""
Base content generation module - generates content from task descriptions only.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime

from cs4.core.prompts import get_base_generation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class BaseGenerator:
    """Generate base content from task descriptions (without constraints)."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize base content generator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_BASE_GEN_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.system_prompt = get_base_generation_prompt(content_type)
        
        self.logger = logging.getLogger("CS4Generator")
    
    def generate_base_content(
        self,
        task: str,
        log: bool = True
    ) -> tuple[str, int]:
        """
        Generate base content for a single task.
        
        Args:
            task: Task description
            log: Whether to log token usage
            
        Returns:
            Tuple of (generated_content, tokens_used)
        """
        user_input = f"Task: {task}"
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        model=self.model,
                    )
                    content = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": user_input}],
                        model=self.model,
                        system=self.system_prompt,
                        )
                    content = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return content, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to generate base content")
    
    def generate_batch(
        self,
        df: pd.DataFrame,
        task_column: str = "main_task",
        output_path: Optional[str] = None,
        deduplicate_by_instruction: bool = True
    ) -> pd.DataFrame:
        """
        Generate base content for a batch of tasks.
        
        Args:
            df: Input DataFrame with tasks (typically from constraints.csv)
            task_column: Name of column containing task descriptions
            output_path: Optional path to save results
            deduplicate_by_instruction: If True and instruction_number exists with duplicates,
                                       generate base content once per unique instruction_number
                                       and replicate across all rows (for expanded constraints)
            
        Returns:
            DataFrame with generated base content
        """
        if task_column not in df.columns:
            raise ValueError(f"Column '{task_column}' not found in DataFrame")
        
        # Check if instruction_number exists
        if "instruction_number" in df.columns:
            has_instruction_num = True
        else:
            has_instruction_num = False
            self.logger.warning("No 'instruction_number' column found, using index")
        
        # Check if deduplication is needed (expanded constraints with subset_size)
        needs_deduplication = (
            deduplicate_by_instruction and 
            has_instruction_num and 
            df["instruction_number"].duplicated().any()
        )
        
        if needs_deduplication:
            self.logger.info(f"Detected {df['instruction_number'].duplicated().sum()} duplicate instruction_numbers")
            self.logger.info("Will generate base content once per unique instruction_number")
            
            # Get unique tasks by instruction_number
            unique_df = df.drop_duplicates(subset=["instruction_number"]).copy()
            self.logger.info(f"Generating base content for {len(unique_df)} unique tasks (from {len(df)} total rows)")
            
            # Generate for unique tasks
            base_results = []
            for idx, row in unique_df.iterrows():
                task = row[task_column]
                instruction_num = row["instruction_number"]
                
                self.logger.info(f"Processing task #{instruction_num}")
                
                try:
                    content, tokens = self.generate_base_content(task, log=True)
                    
                    base_results.append({
                        "instruction_number": instruction_num,
                        "main_task": task,
                        "base_content": content,
                        "content_length": len(content),
                        "model_used": self.model,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate base content for task {instruction_num}: {e}"
                    )
                    base_results.append({
                        "instruction_number": instruction_num,
                        "main_task": task,
                        "base_content": "",
                        "content_length": 0,
                        "model_used": self.model,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat()
                    })
            
            base_df = pd.DataFrame(base_results)
            
            # Merge back to original df to replicate base content across all rows
            result_df = df.merge(
                base_df[["instruction_number", "base_content", "content_length", 
                         "model_used", "tokens_used", "timestamp"]],
                on="instruction_number",
                how="left"
            )
            
            self.logger.info(f"Replicated base content across {len(result_df)} rows")
            
        else:
            # Original behavior for backward compatibility - preserves all input columns
            self.logger.info(f"Generating base content for {len(df)} tasks")
            
            base_results = []
            for idx, row in df.iterrows():
                task = row[task_column]
                instruction_num = row["instruction_number"] if has_instruction_num else idx + 1
                
                self.logger.info(f"Processing task #{instruction_num}")
                
                try:
                    content, tokens = self.generate_base_content(task, log=True)
                    
                    base_results.append({
                        "instruction_number": instruction_num,
                        "base_content": content,
                        "content_length": len(content),
                        "model_used": self.model,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate base content for task {instruction_num}: {e}"
                    )
                    base_results.append({
                        "instruction_number": instruction_num,
                        "base_content": "",
                        "content_length": 0,
                        "model_used": self.model,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat()
                    })
            
            base_df = pd.DataFrame(base_results)
            
            # Merge to preserve all original columns from input df
            if has_instruction_num:
                result_df = df.merge(
                    base_df[["instruction_number", "base_content", "content_length", 
                             "model_used", "tokens_used", "timestamp"]],
                    on="instruction_number",
                    how="left"
                )
            else:
                # If no instruction_number, add columns directly (assumes same order)
                result_df = df.copy()
                result_df["base_content"] = base_df["base_content"].values
                result_df["content_length"] = base_df["content_length"].values
                result_df["model_used"] = base_df["model_used"].values
                result_df["tokens_used"] = base_df["tokens_used"].values
                result_df["timestamp"] = base_df["timestamp"].values
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Base content saved to {output_path}")
        
        return result_df
