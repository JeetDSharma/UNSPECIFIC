#!/usr/bin/env python3
"""
CLI script for base content generation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.core.base_generator import BaseGenerator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config

def main():
    parser = argparse.ArgumentParser(
        description="Generate base content from task descriptions"
    )
    parser.add_argument(
        "--domain",
        choices=["blog", "story", "news"],
        default="blog",
        help="Content domain"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV (e.g., constraints.csv)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., base_generated.csv)"
    )
    parser.add_argument(
        "--task-column",
        default="main_task",
        help="Name of column containing task descriptions"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_BASE_GEN_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_BASE_GEN_MODEL})"
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
    log_file = Config.LOGS_DIR / "base_generation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting base generation for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} tasks")
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
    
    # Initialize generator
    generator = BaseGenerator(
        llm_client=client,
        model=args.model,
        content_type=args.domain,
        retry_attempts=args.retry_attempts
    )
    
    # Generate base content
    try:
        result_df = generator.generate_batch(
            df=df,
            task_column=args.task_column,
            output_path=args.output_path
        )
        logger.info(f"Successfully generated base content for {len(result_df)} tasks")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Base generation failed: {e}")
        sys.exit(1)
    
    logger.info("Base generation complete!")


if __name__ == "__main__":
    main()
