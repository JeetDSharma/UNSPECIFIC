"""
Content summarization module - reduces content length while preserving key insights.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional, Tuple
from datetime import datetime

from cs4.core.prompts import get_summarization_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, TogetherAIClient, UsageTracker
from cs4.config import Config


class ContentSummarizer:
    """Summarize content to a target percentage of original length."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        target_length_pct: float = 0.25,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize content summarizer.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            target_length_pct: Target length as percentage of original (default: 0.25 = 25%)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_MODEL
        self.content_type = content_type
        self.target_length_pct = target_length_pct
        self.retry_attempts = retry_attempts
        self.delay = delay
        
        self.logger = logging.getLogger("CS4Generator")
    
    def summarize_content(
        self,
        content: str,
        log: bool = True
    ) -> Tuple[str, int]:
        """
        Summarize content to target length percentage.
        
        Args:
            content: Content to summarize
            log: Whether to log token usage
            
        Returns:
            Tuple of (summarized_content, tokens_used)
        """
        prompt = get_summarization_prompt(
            content_type=self.content_type,
            content=content,
            target_length_pct=self.target_length_pct
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model
                    )
                    summarized = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, TogetherAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    summarized = self.llm_client.get_response_text(response)
                    tokens = response.usage.total_tokens
                else:  # Anthropic
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model
                    )
                    summarized = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                
                if log:
                    UsageTracker.log_usage(
                        provider=self.llm_client.__class__.__name__.replace("Client", "").lower(),
                        model=self.model,
                        tokens=tokens,
                        metadata=f"summarize ({len(content)} → {len(summarized)} chars)"
                    )
                    self.logger.info(f"Total tokens used: {UsageTracker.get_total_usage()['total_tokens']}")
                
                return summarized, tokens
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to summarize content")
    
    def summarize_batch(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Summarize content for a batch of samples.
        
        Args:
            df: DataFrame with content to summarize
            content_column: Name of column containing content
            output_path: Optional path to save results
            
        Returns:
            DataFrame with summarized content
        """
        if content_column not in df.columns:
            raise ValueError(f"DataFrame must have '{content_column}' column")
        
        self.logger.info(f"Summarizing {len(df)} samples to {int(self.target_length_pct * 100)}% length")
        
        results = []
        for idx, row in df.iterrows():
            instruction_num = row.get("instruction_number", idx + 1)
            content = row[content_column]
            
            self.logger.info(f"Processing sample #{instruction_num}")
            
            try:
                summarized, tokens = self.summarize_content(
                    content=content,
                    log=True
                )
                
                result = row.copy()
                result["summarized_content"] = summarized
                result["summarized_length"] = len(summarized)
                result["original_length"] = len(content)
                result["compression_ratio"] = len(summarized) / len(content) if len(content) > 0 else 0
                result["model_used"] = self.model
                result["tokens_used"] = tokens
                result["timestamp"] = datetime.now().isoformat()
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to summarize sample #{instruction_num}: {e}")
                # Keep original with error marker
                result = row.copy()
                result["summarized_content"] = content  # Keep original
                result["summarized_length"] = len(content)
                result["original_length"] = len(content)
                result["compression_ratio"] = 1.0
                result["error"] = str(e)
                results.append(result)
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Summarized content saved to {output_path}")
        
        return result_df
