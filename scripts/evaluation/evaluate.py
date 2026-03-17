#!/usr/bin/env python3
"""
CLI script for evaluating constraint satisfaction.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.evaluator import ConstraintEvaluator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate constraint satisfaction in generated content"
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
        help="Path to input CSV (e.g., fitted_content.csv)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., evaluation_results.csv)"
    )
    parser.add_argument(
        "--content-column",
        default="fitted_content",
        help="Name of column containing content to evaluate"
    )
    parser.add_argument(
        "--constraints-column",
        default="constraints",
        help="Name of column containing constraints"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_EVALUATION_MODEL,
        help=f"LLM model to use for evaluation (default: {Config.DEFAULT_EVALUATION_MODEL})"
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
    log_file = Config.LOGS_DIR / "evaluation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Evaluator")
    
    logger.info(f"Starting evaluation for domain: {args.domain}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} samples for evaluation")
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
    
    # Initialize evaluator
    evaluator = ConstraintEvaluator(
        llm_client=client,
        model=args.model,
        content_type=args.domain,
        retry_attempts=args.retry_attempts
    )
    
    # Evaluate
    try:
        result_df = evaluator.evaluate_batch(
            df=df,
            content_column=args.content_column,
            constraints_column=args.constraints_column,
            output_path=args.output_path
        )
        logger.info(f"Successfully evaluated {len(result_df)} samples")
        
        # Print statistics
        if len(result_df) > 0:
            avg_satisfaction = result_df["satisfaction_rate"].mean()
            logger.info(f"Average satisfaction rate: {avg_satisfaction:.2%}")
            logger.info(f"Best: {result_df['satisfaction_rate'].max():.2%}")
            logger.info(f"Worst: {result_df['satisfaction_rate'].min():.2%}")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
