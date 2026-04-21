"""
Constraint fitting module - fits base content to satisfy constraints.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime

from cs4.core.prompts import get_constraint_fitting_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, TogetherAIClient
from cs4.config import Config


class ConstraintFitter:
    """Fit base content to satisfy multiple constraints."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint fitter.
        
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
    
    def fit_content(
        self,
        task: str,
        base_content: str,
        constraints: str,
        log: bool = True
    ) -> tuple[str, int]:
        """
        Fit base content to satisfy all constraints.
        
        Args:
            task: Task description
            base_content: Base content to fit
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (fitted_content, tokens_used)
        """
        prompt = get_constraint_fitting_prompt(
            content_type=self.content_type,
            task=task,
            base_content=base_content,
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
                    if not content and hasattr(message, "reasoning") and message.reasoning:
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
        
        raise RuntimeError("Failed to fit content to constraints")
    
    def fit_batch(
        self,
        constraints_df: pd.DataFrame,
        base_df: pd.DataFrame,
        output_path: Optional[str] = None,
        base_column: str = "base_content",
        constraint_column: str = "constraints"
    ) -> pd.DataFrame:
        """
        Fit base content to constraints for a batch of samples.
        
        Args:
            constraints_df: Constraints (from constraints.csv)
            base_df: Base content (from base_generated.csv)
            output_path: Optional path to save results
            base_column: Column name containing base content (default: "base_content")
            constraint_column: Column name containing constraints (default: "constraints")
            
        Returns:
            DataFrame with fitted content
        """
        # Merge dataframes on instruction_number
        if "instruction_number" not in constraints_df.columns:
            raise ValueError("constraints_df must have 'instruction_number' column")
        if "instruction_number" not in base_df.columns:
            raise ValueError("base_df must have 'instruction_number' column")
        
        merged = pd.merge(
            constraints_df,
            base_df,
            on="instruction_number",
            suffixes=("_constraint", "_base")
        )
        
        self.logger.info(f"Fitting content for {len(merged)} samples")
        
        results = []
        for idx, row in merged.iterrows():
            instruction_num = row["instruction_number"]
            
            # Extract fields (handle suffix conflicts)
            task = row.get("main_task_constraint", row.get("main_task", ""))
            
            # Try to get constraints with the specified column name
            if constraint_column in row:
                constraints = row[constraint_column]
            elif f"{constraint_column}_constraint" in row:
                constraints = row[f"{constraint_column}_constraint"]
            elif "constraints" in row:
                constraints = row["constraints"]
            elif "constraints_constraint" in row:
                constraints = row["constraints_constraint"]
            else:
                raise ValueError(f"Could not find constraint column '{constraint_column}' in merged dataframe")
            
            # Try to get base content with the specified column name
            if base_column in row:
                base_content = row[base_column]
            elif f"{base_column}_base" in row:
                base_content = row[f"{base_column}_base"]
            elif "base_content" in row:
                base_content = row["base_content"]
            elif "base_content_base" in row:
                base_content = row["base_content_base"]
            else:
                raise ValueError(f"Could not find base content column '{base_column}' in merged dataframe")
            
            self.logger.info(f"Processing sample #{instruction_num}")
            
            try:
                fitted_content, tokens = self.fit_content(
                    task=task,
                    base_content=base_content,
                    constraints=constraints,
                    log=True
                )
                
                # Count constraints
                import re
                num_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "base_content": base_content,
                    "fitted_content": fitted_content,
                    "fitted_length": len(fitted_content),
                    "num_constraints": num_constraints,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to fit content for sample {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "base_content": base_content,
                    "fitted_content": "",
                    "fitted_length": 0,
                    "num_constraints": 0,
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Fitted content saved to {output_path}")
        
        return result_df
    
    def fit_batch_combined(
        self,
        base_constraints_df: pd.DataFrame,
        output_path: Optional[str] = None,
        base_column: str = "base_content",
        constraint_column: str = "selected_constraints",
        task_column: str = "main_task"
    ) -> pd.DataFrame:
        """
        Fit base content to constraints for a batch where both are in the same DataFrame.
        
        This method is designed for CSVs that contain both the base content and constraints
        in a single file (e.g., from constraint subset experiments).
        
        Args:
            base_constraints_df: DataFrame containing both base content and constraints
            output_path: Optional path to save results
            base_column: Column name containing base content (default: "base_content")
            constraint_column: Column name containing constraints (default: "selected_constraints")
            task_column: Column name containing the main task (default: "main_task")
            
        Returns:
            DataFrame with all original columns plus a new "revised_base" column
        """
        # Validate required columns
        if base_column not in base_constraints_df.columns:
            raise ValueError(f"DataFrame must have '{base_column}' column")
        if constraint_column not in base_constraints_df.columns:
            raise ValueError(f"DataFrame must have '{constraint_column}' column")
        if task_column not in base_constraints_df.columns:
            raise ValueError(f"DataFrame must have '{task_column}' column")
        
        self.logger.info(f"Fitting content for {len(base_constraints_df)} samples")
        
        # Create a copy to avoid modifying the original
        result_df = base_constraints_df.copy()
        
        # Initialize new columns
        result_df["revised_base"] = ""
        result_df["revised_length"] = 0
        result_df["fitting_model"] = ""
        result_df["fitting_tokens"] = 0
        result_df["fitting_timestamp"] = ""
        
        for idx, row in base_constraints_df.iterrows():
            task = row[task_column]
            base_content = row[base_column]
            constraints = row[constraint_column]
            
            # Log progress with instruction_number if available
            if "instruction_number" in row:
                self.logger.info(f"Processing sample #{row['instruction_number']} (row {idx + 1}/{len(base_constraints_df)})")
            else:
                self.logger.info(f"Processing row {idx + 1}/{len(base_constraints_df)}")
            
            try:
                fitted_content, tokens = self.fit_content(
                    task=task,
                    base_content=base_content,
                    constraints=constraints,
                    log=True
                )
                
                result_df.at[idx, "revised_base"] = fitted_content
                result_df.at[idx, "revised_length"] = len(fitted_content)
                result_df.at[idx, "fitting_model"] = self.model
                result_df.at[idx, "fitting_tokens"] = tokens
                result_df.at[idx, "fitting_timestamp"] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(
                    f"Failed to fit content for row {idx}: {e}"
                )
                result_df.at[idx, "revised_base"] = ""
                result_df.at[idx, "revised_length"] = 0
                result_df.at[idx, "fitting_model"] = self.model
                result_df.at[idx, "fitting_tokens"] = 0
                result_df.at[idx, "fitting_timestamp"] = datetime.now().isoformat()
            
            # Save incrementally after each row to prevent data loss
            if output_path:
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                self.logger.debug(f"Progress saved (row {idx + 1}/{len(base_constraints_df)})")
        
        if output_path:
            self.logger.info(f"All fitted content saved to {output_path}")
        
        return result_df
