"""
Direct content generation module - generates final content directly from task and constraints.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cs4.core.prompts import get_direct_generation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, TogetherAIClient
from cs4.config import Config


class DirectGenerator:
    """Generate final content directly from task and constraints without base generation."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize direct generator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_FITTING_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.logger = logging.getLogger("CS4Generator")
    
    def generate_content(
        self,
        task: str,
        constraints: str,
        log: bool = True
    ) -> tuple[str, int]:
        """
        Generate final content directly from task and constraints.
        
        Args:
            task: Task description
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (generated_content, tokens_used)
        """
        prompt = get_direct_generation_prompt(
            content_type=self.content_type,
            task=task,
            constraints=constraints
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model,
                    )
                    content = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                    )
                    content = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                elif isinstance(self.llm_client, TogetherAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                    )
                    message = response.choices[0].message
                    content = message.content.strip() if message.content else ""
                    
                    # Some models (like Qwen reasoning models) output to 'reasoning' field
                    if not content and hasattr(message, 'reasoning') and message.reasoning:
                        content = message.reasoning.strip()
                    
                    tokens = response.usage.total_tokens
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
        
        raise RuntimeError("Failed to generate content directly")
    
    def generate_batch(
        self,
        constraints_df: pd.DataFrame,
        output_path: Optional[str] = None,
        constraint_column: str = "constraints",
        task_column: str = "main_task"
    ) -> pd.DataFrame:
        """
        Generate final content directly for a batch of samples.
        
        Args:
            constraints_df: DataFrame containing tasks and constraints
            output_path: Optional path to save results
            constraint_column: Column name containing constraints
            task_column: Column name containing the main task
            
        Returns:
            DataFrame with generated content
        """
        if constraint_column not in constraints_df.columns:
            raise ValueError(f"DataFrame must have '{constraint_column}' column")
        if task_column not in constraints_df.columns:
            raise ValueError(f"DataFrame must have '{task_column}' column")
        
        self.logger.info(f"Generating content directly for {len(constraints_df)} samples")
        
        results = []
        for idx, row in constraints_df.iterrows():
            instruction_num = row.get("instruction_number", idx)
            task = row[task_column]
            constraints = row[constraint_column]
            
            self.logger.info(f"Processing sample #{instruction_num}")
            
            try:
                generated_content, tokens = self.generate_content(
                    task=task,
                    constraints=constraints,
                    log=True
                )
                
                import re
                num_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "direct_content": generated_content,
                    "content_length": len(generated_content),
                    "num_constraints": num_constraints,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to generate content for sample {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "direct_content": "",
                    "content_length": 0,
                    "num_constraints": 0,
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Generated content saved to {output_path}")
        
        return result_df
    
    def generate_batch_parallel(
        self,
        constraints_df: pd.DataFrame,
        output_path: Optional[str] = None,
        constraint_column: str = "constraints",
        task_column: str = "main_task",
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Generate final content directly for a batch using parallel processing.
        
        Args:
            constraints_df: DataFrame containing tasks and constraints
            output_path: Optional path to save results
            constraint_column: Column name containing constraints
            task_column: Column name containing the main task
            max_workers: Maximum number of parallel workers
            
        Returns:
            DataFrame with generated content
        """
        if constraint_column not in constraints_df.columns:
            raise ValueError(f"DataFrame must have '{constraint_column}' column")
        if task_column not in constraints_df.columns:
            raise ValueError(f"DataFrame must have '{task_column}' column")
        
        self.logger.info(f"Generating content directly for {len(constraints_df)} samples with {max_workers} parallel workers")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {}
            for idx, row in constraints_df.iterrows():
                instruction_num = row.get("instruction_number", idx)
                task = row[task_column]
                constraints = row[constraint_column]
                
                future = executor.submit(
                    self.generate_content,
                    task,
                    constraints,
                    False
                )
                future_to_data[future] = (instruction_num, task, constraints)
            
            for future in as_completed(future_to_data):
                instruction_num, task, constraints = future_to_data[future]
                
                try:
                    generated_content, tokens = future.result()
                    
                    import re
                    num_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                    
                    results.append({
                        "instruction_number": instruction_num,
                        "main_task": task,
                        "constraints": constraints,
                        "direct_content": generated_content,
                        "content_length": len(generated_content),
                        "num_constraints": num_constraints,
                        "model_used": self.model,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    completed_count += 1
                    self.logger.info(f"Completed sample #{instruction_num} ({completed_count}/{len(constraints_df)}) - {tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate content for sample {instruction_num}: {e}")
                    results.append({
                        "instruction_number": instruction_num,
                        "main_task": task,
                        "constraints": constraints,
                        "direct_content": "",
                        "content_length": 0,
                        "num_constraints": 0,
                        "model_used": self.model,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat()
                    })
                    completed_count += 1
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("instruction_number").reset_index(drop=True)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Generated content saved to {output_path}")
        
        return result_df
