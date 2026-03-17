#!/usr/bin/env python3
"""
CLI script for generating base content that avoids easy constraints (Method 2).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.core.anti_constraint_generator import AntiConstraintGenerator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Generate base content that avoids satisfied constraints"
    )
    parser.add_argument(
        "--domain",
        choices=["blog", "story", "news"],
        default="blog",
        help="Content domain"
    )
    parser.add_argument(
        "--constraints-path",
        required=True,
        help="Path to constraints CSV"
    )
    parser.add_argument(
        "--evaluation-path",
        required=True,
        help="Path to base evaluation CSV (identifies easy constraints)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output anti-constraint base CSV"
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
    
    log_file = Config.LOGS_DIR / "anti_constraint_generation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info("Starting anti-constraint base generation (Method 2)")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Constraints: {args.constraints_path}")
    logger.info(f"Evaluation: {args.evaluation_path}")
    logger.info(f"Output: {args.output_path}")
    
    try:
        constraints_df = pd.read_csv(args.constraints_path, encoding="utf-8")
        evaluation_df = pd.read_csv(args.evaluation_path, encoding="utf-8")
        logger.info(f"Loaded {len(constraints_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        sys.exit(1)
    
    try:
        if args.provider == "openai":
            client = OpenAIClient(log_usage=True)
        else:
            client = AnthropicClient(log_usage=True)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    generator = AntiConstraintGenerator(
        llm_client=client,
        model=args.model,
        content_type=args.domain,
        retry_attempts=args.retry_attempts
    )
    
    try:
        result_df = generator.generate_batch(
            constraints_df=constraints_df,
            evaluation_df=evaluation_df,
            output_path=args.output_path
        )
        logger.info(f"Successfully generated anti-constraint base for {len(result_df)} samples")
        
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Anti-constraint generation failed: {e}")
        sys.exit(1)
    
    logger.info("Anti-constraint generation complete!")


if __name__ == "__main__":
    main()
