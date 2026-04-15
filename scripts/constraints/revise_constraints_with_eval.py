#!/usr/bin/env python3
"""
CLI script for replacing easy constraints with harder ones.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from cs4.core.constraint_replacer_with_eval import ConstraintReplacerWithEval
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Replace satisfied (easy) constraints with harder ones"
    )
    parser.add_argument(
        "--constraints-path",
        required=True,
        help="Path to original constraints CSV"
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help="Path to base generated content CSV (must include base_content, instruction_number)"
    )
    parser.add_argument(
        "--eval-path",
        required=True,
        help="Path to base evaluation CSV (must include satisfaction_results, instruction_number)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output revised constraints CSV"
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
    
    log_file = Config.LOGS_DIR / "constraint_replacement.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info("Starting constraint replacement with evaluation results")
    logger.info(f"Constraints: {args.constraints_path}")
    logger.info(f"Base content: {args.base_path}")
    logger.info(f"Base evaluation: {args.eval_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        logger.info(f"Max workers: {args.max_workers}")
    
    try:
        constraints_df = pd.read_csv(args.constraints_path, encoding="utf-8")
        base_df = pd.read_csv(args.base_path, encoding="utf-8")
        eval_df = pd.read_csv(args.eval_path, encoding="utf-8")
        for name, df, col in (
            ("base", base_df, "base_content"),
            ("eval", eval_df, "satisfaction_results"),
        ):
            if col not in df.columns:
                logger.error(f"{name} CSV missing required column: {col}")
                sys.exit(1)
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
    
    replacer = ConstraintReplacerWithEval(
        llm_client=client,
        model=args.model,
        retry_attempts=args.retry_attempts
    )
    
    try:
        if args.parallel:
            result_df = replacer.replace_batch_parallel(
                constraints_df=constraints_df,
                base_df=base_df,
                eval_df=eval_df,
                output_path=args.output_path,
                max_workers=args.max_workers
            )
        else:
            result_df = replacer.replace_batch_parallel(
                constraints_df=constraints_df,
                base_df=base_df,
                eval_df=eval_df,
                output_path=args.output_path,
                max_workers=1
            )
        logger.info(f"Successfully replaced constraints for {len(result_df)} samples")
        
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Constraint replacement failed: {e}")
        sys.exit(1)
    
    logger.info("Constraint replacement complete!")


if __name__ == "__main__":
    main()
