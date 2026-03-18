"""
Common constraint generation module - extracts constraints common to two pieces of content.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cs4.core.prompts import get_common_constraint_generation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class CommonConstraintGenerator:
    """Generate constraints common to two pieces of content."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize common constraint generator.
        
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
        self.system_prompt = get_common_constraint_generation_prompt()
        
        self.logger = logging.getLogger("CS4Generator")
    
    def generate_constraints_for_pair(
        self,
        blog1: str,
        blog2: str,
        log: bool = True
    ) -> tuple[str, str, int]:
        """
        Generate common constraints for a pair of blogs.
        
        Args:
            blog1: First blog content
            blog2: Second blog content
            log: Whether to log token usage
            
        Returns:
            Tuple of (main_task, constraints, tokens_used)
        """
        # Format the prompt with both blogs
        user_input = self.system_prompt.format(blog1=blog1, blog2=blog2)
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": user_input}
                        ],
                        model=self.model
                    )
                    response_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": user_input}],
                        model=self.model
                    )
                    response_text = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Parse response to extract main task and constraints
                main_task, constraints = self._parse_response(response_text)

                if not constraints:
                    self.logger.error(
                        f"""
                        EMPTY CONSTRAINTS PARSED
                        ------------------------
                        Raw LLM response:
                        {response_text}
                        ------------------------
                        """
                    )
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return main_task, constraints, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to generate common constraints")
    
    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """
        Parse LLM response to extract main task and constraints.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (main_task, constraints)
        """
        lines = response_text.strip().split('\n')
        main_task = ""
        constraints_lines = []

        in_constraints = False

        for line in lines:
            clean_line = line.strip().strip("*").strip()

            lower = clean_line.lower()

            if "main task" in lower:
                if ":" in clean_line:
                    main_task = clean_line.split(":", 1)[1].strip()
                else:
                    main_task = clean_line.replace("main task", "").strip()

            elif "constraints" in lower:
                in_constraints = True

            elif in_constraints and clean_line:
                constraints_lines.append(clean_line)

        constraints = "\n".join(constraints_lines)
        return main_task, constraints
    
    def generate_constraints_batch(
        self,
        df: pd.DataFrame,
        blog1_column: str = "Blog A",
        blog2_column: str = "Blog B",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate common constraints for a batch of blog pairs.
        
        Args:
            df: Input DataFrame with blog pairs
            blog1_column: Name of column containing first blog
            blog2_column: Name of column containing second blog
            output_path: Optional path to save results
            
        Returns:
            DataFrame with constraints
        """
        if blog1_column not in df.columns:
            raise ValueError(f"Column '{blog1_column}' not found in DataFrame")
        if blog2_column not in df.columns:
            raise ValueError(f"Column '{blog2_column}' not found in DataFrame")
        
        self.logger.info(f"Processing {len(df)} blog pairs")
        
        results = []
        for idx, row in df.iterrows():
            blog1 = row[blog1_column]
            blog2 = row[blog2_column]
            instruction_num = row.get("instruction_number", idx + 1)
            
            self.logger.info(f"Processing pair #{instruction_num}")
            
            try:
                main_task, constraints, tokens = self.generate_constraints_for_pair(
                    blog1, blog2, log=True
                )
                
                results.append({
                    "instruction_number": instruction_num,
                    "blog1": blog1,
                    "blog2": blog2,
                    "main_task": main_task,
                    "constraints": constraints,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to generate constraints for pair {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "blog1": blog1,
                    "blog2": blog2,
                    "main_task": "",
                    "constraints": "",
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Common constraints saved to {output_path}")
        
        return result_df
    
    def generate_constraints_batch_parallel(
        self,
        df: pd.DataFrame,
        blog1_column: str = "Blog A",
        blog2_column: str = "Blog B",
        output_path: Optional[str] = None,
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Generate common constraints for a batch of blog pairs using parallel processing.
        
        Args:
            df: Input DataFrame with blog pairs
            blog1_column: Name of column containing first blog
            blog2_column: Name of column containing second blog
            output_path: Optional path to save results
            max_workers: Maximum number of parallel workers (default: 5)
            
        Returns:
            DataFrame with constraints
        """
        if blog1_column not in df.columns:
            raise ValueError(f"Column '{blog1_column}' not found in DataFrame")
        if blog2_column not in df.columns:
            raise ValueError(f"Column '{blog2_column}' not found in DataFrame")
        
        self.logger.info(f"Processing {len(df)} blog pairs with {max_workers} parallel workers")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(
                    self.generate_constraints_for_pair,
                    row[blog1_column],
                    row[blog2_column],
                    False  # Don't log each individual pair
                ): (idx, row)
                for idx, row in df.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_data):
                idx, row = future_to_data[future]
                instruction_num = row.get("instruction_number", idx + 1)
                
                try:
                    main_task, constraints, tokens = future.result()
                    
                    results.append({
                        "instruction_number": instruction_num,
                        "blog1": row[blog1_column],
                        "blog2": row[blog2_column],
                        "main_task": main_task,
                        "constraints": constraints,
                        "model_used": self.model,
                        "tokens_used": tokens,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    completed_count += 1
                    self.logger.info(f"Completed pair #{instruction_num} ({completed_count}/{len(df)}) - {tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate constraints for pair {instruction_num}: {e}")
                    results.append({
                        "instruction_number": instruction_num,
                        "blog1": row[blog1_column],
                        "blog2": row[blog2_column],
                        "main_task": "",
                        "constraints": "",
                        "model_used": self.model,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat()
                    })
                    completed_count += 1
        
        # Sort results by instruction_number to maintain order
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("instruction_number").reset_index(drop=True)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Common constraints saved to {output_path}")
        
        return result_df
