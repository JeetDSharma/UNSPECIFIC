#!/usr/bin/env python3
"""
Merge blog pairs into single coherent blogs using LLM.
CLI wrapper for BlogMerger class.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.blog_merger import BlogMerger
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Merge blog pairs into single coherent blogs using LLM"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with blog pairs (from find_blog_pairs.py)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with merged blogs"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_MERGE_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_MERGE_MODEL})"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N pairs (default: 10)"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts on failure (default: 3)"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Config.LOGS_DIR / "merge_blogs.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info("Starting blog merging")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        pairs_df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(pairs_df)} blog pairs")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    # Initialize LLM client
    try:
        if args.provider == "openai":
            client = OpenAIClient(log_usage=True)
        else:
            client = AnthropicClient(log_usage=True)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    # Initialize BlogMerger and merge pairs
    try:
        merger = BlogMerger(
            llm_client=client,
            model=args.model,
            retry_attempts=args.retry_attempts
        )
        
        result_df = merger.merge_pairs(
            pairs_df=pairs_df,
            output_path=args.output_path,
            save_interval=args.save_interval
        )
        
        logger.info(f"Successfully merged {len(result_df)} blog pairs")
        
        # Print statistics
        if len(result_df) > 0:
            successful = result_df['merged_length'] > 0
            logger.info(f"Successful merges: {successful.sum()}/{len(result_df)}")
            logger.info(f"Average merged length: {result_df['merged_length'].mean():.0f} characters")
            logger.info(f"Total tokens used: {result_df['tokens_used'].sum():,}")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total API tokens: {usage['total_tokens']:,}")
        
    except Exception as e:
        logger.error(f"Blog merging failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Blog merging complete!")


if __name__ == "__main__":
    main()
