#!/usr/bin/env python3
"""
CLI script for generating common constraints from two blogs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.core.common_constraint_generator import CommonConstraintGenerator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Generate common constraints from two blogs"
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
        help="Path to input CSV with blog pairs"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., common_constraints.csv)"
    )
    parser.add_argument(
        "--blog1-column",
        default="Blog A",
        help="Name of column containing first blog"
    )
    parser.add_argument(
        "--blog2-column",
        default="Blog B",
        help="Name of column containing second blog"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_CONSTRAINT_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_CONSTRAINT_MODEL})"
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
        "--delay",
        type=float,
        default=1.0,
        help="Delay between retries (seconds)"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Config.LOGS_DIR / "common_constraint_generation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting common constraint generation for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Blog columns: '{args.blog1_column}', '{args.blog2_column}'")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} blog pairs")
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
    generator = CommonConstraintGenerator(
        llm_client=client,
        model=args.model,
        retry_attempts=args.retry_attempts,
        delay=args.delay
    )
    
    # Generate common constraints
    try:
        result_df = generator.generate_constraints_batch(
            df=df,
            blog1_column=args.blog1_column,
            blog2_column=args.blog2_column,
            output_path=args.output_path
        )
        logger.info(f"Successfully generated common constraints for {len(result_df)} pairs")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Common constraint generation failed: {e}")
        sys.exit(1)
    
    logger.info("Common constraint generation complete!")


if __name__ == "__main__":
    main()
