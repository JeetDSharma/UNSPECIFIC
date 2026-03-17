#!/usr/bin/env python3
"""
CLI script for summarizing content to a target length percentage.
Based on summarize_blog from notebooks/fit_base_blog.ipynb
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.content_summarizer import ContentSummarizer
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Summarize content to target length percentage"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with content to summarize"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with summarized content"
    )
    parser.add_argument(
        "--content-column",
        default="fitted_content",
        help="Name of column containing content to summarize"
    )
    parser.add_argument(
        "--target-length-pct",
        type=float,
        default=0.25,
        help="Target length as percentage of original (default: 0.25 = 25%%)"
    )
    parser.add_argument(
        "--domain",
        default="blog",
        help="Content domain (blog, story, news)"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts on failure"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Config.LOGS_DIR / "summarization.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting content summarization for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Target length: {int(args.target_length_pct * 100)}% of original")
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    # Initialize LLM client
    if args.provider == "openai":
        client = OpenAIClient(log_usage=True)
    else:
        client = AnthropicClient(log_usage=True)
    
    # Initialize summarizer
    try:
        summarizer = ContentSummarizer(
            llm_client=client,
            model=args.model,
            content_type=args.domain,
            target_length_pct=args.target_length_pct,
            retry_attempts=args.retry_attempts
        )
        
        # Summarize content
        result_df = summarizer.summarize_batch(
            df=df,
            content_column=args.content_column,
            output_path=args.output_path
        )
        
        logger.info(f"Successfully summarized {len(result_df)} samples")
        
        # Print statistics
        if "compression_ratio" in result_df.columns:
            avg_compression = result_df["compression_ratio"].mean()
            logger.info(f"Average compression ratio: {avg_compression:.2%}")
            logger.info(f"Target compression: {args.target_length_pct:.2%}")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Summarization complete!")


if __name__ == "__main__":
    main()
