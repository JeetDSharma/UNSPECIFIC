#!/usr/bin/env python3
"""
CLI script for direct final content generation from task and constraints.
Bypasses base generation and revision steps.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.core.direct_generator import DirectGenerator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, TogetherAIClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Generate final content directly from task and constraints"
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
        help="Path to input CSV containing main_task and constraints columns"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., direct_generated.csv)"
    )
    parser.add_argument(
        "--task-column",
        default="main_task",
        help="Name of column containing task descriptions (default: main_task)"
    )
    parser.add_argument(
        "--constraint-column",
        default="constraints",
        help="Name of column containing constraints (default: constraints)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "together"],
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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for faster execution"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)"
    )
    
    args = parser.parse_args()
    
    log_file = Config.LOGS_DIR / "direct_generation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting direct generation for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        logger.info(f"Max workers: {args.max_workers}")
    
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} samples")
        
        if args.task_column not in df.columns:
            logger.error(f"Task column '{args.task_column}' not found in input CSV")
            sys.exit(1)
        if args.constraint_column not in df.columns:
            logger.error(f"Constraint column '{args.constraint_column}' not found in input CSV")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    try:
        if args.provider == "openai":
            client = OpenAIClient(log_usage=True)
        elif args.provider == "anthropic":
            client = AnthropicClient(log_usage=True)
        elif args.provider == "together":
            client = TogetherAIClient(log_usage=True)
        else:
            logger.error(f"Unknown provider: {args.provider}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    generator = DirectGenerator(
        llm_client=client,
        model=args.model,
        content_type=args.domain,
        retry_attempts=args.retry_attempts
    )
    
    try:
        if args.parallel:
            result_df = generator.generate_batch_parallel(
                constraints_df=df,
                output_path=args.output_path,
                constraint_column=args.constraint_column,
                task_column=args.task_column,
                max_workers=args.max_workers
            )
        else:
            result_df = generator.generate_batch(
                constraints_df=df,
                output_path=args.output_path,
                constraint_column=args.constraint_column,
                task_column=args.task_column
            )
        logger.info(f"Successfully generated content for {len(result_df)} samples")
        
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Direct generation failed: {e}")
        sys.exit(1)
    
    logger.info("Direct generation complete!")


if __name__ == "__main__":
    main()
