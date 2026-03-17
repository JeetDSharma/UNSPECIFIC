#!/usr/bin/env python3
"""
CLI script for constraint generation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.core.constraint_generator import ConstraintGenerator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config

def main():
    parser = argparse.ArgumentParser(
        description="Generate constraints from existing content"
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
        help="Path to input CSV (e.g., merged_blogs.csv)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., constraints.csv)"
    )
    parser.add_argument(
        "--content-column",
        default="Merged Blog",
        help="Name of column containing content"
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
    log_file = Config.LOGS_DIR / "constraint_generation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting constraint generation for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    # Initialize LLM client
    try:
        client = OpenAIClient(log_usage=True)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    # Initialize generator
    generator = ConstraintGenerator(
        llm_client=client,
        model=args.model,
        retry_attempts=args.retry_attempts,
        delay=args.delay
    )
    
    # Generate constraints
    try:
        result_df = generator.generate_constraints_batch(
            df=df,
            content_column=args.content_column,
            output_path=args.output_path
        )
        logger.info(f"Successfully generated constraints for {len(result_df)} samples")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Constraint generation failed: {e}")
        sys.exit(1)
    
    logger.info("Constraint generation complete!")


if __name__ == "__main__":
    main()
