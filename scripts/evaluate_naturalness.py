#!/usr/bin/env python3
"""
CLI script for naturalness evaluation of constraint pairs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.naturalness_evaluator import NaturalnessEvaluator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate constraint naturalness through pairwise comparisons"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with constraint pairs (from prepare_naturalness_data.py)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., naturalness_results_A_vs_B1.csv)"
    )
    parser.add_argument(
        "--prompt-column",
        default="main_task",
        help="Column name containing prompts (default: main_task)"
    )
    parser.add_argument(
        "--constraint-a-column",
        default="constraint_a",
        help="Column name for constraint A (default: constraint_a)"
    )
    parser.add_argument(
        "--constraint-b-column",
        default="constraint_b",
        help="Column name for constraint B (default: constraint_b)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model to use for evaluation (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
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
    
    log_file = Config.LOGS_DIR / "naturalness_evaluation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4NaturalnessEvaluator")
    
    logger.info("Starting naturalness evaluation")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Prompt column: {args.prompt_column}")
    logger.info(f"Constraint A column: {args.constraint_a_column}")
    logger.info(f"Constraint B column: {args.constraint_b_column}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Provider: {args.provider}")
    
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} constraint pairs from input CSV")
        
        if args.prompt_column not in df.columns:
            logger.error(f"Input CSV must have '{args.prompt_column}' column")
            sys.exit(1)
        if args.constraint_a_column not in df.columns:
            logger.error(f"Input CSV must have '{args.constraint_a_column}' column")
            sys.exit(1)
        if args.constraint_b_column not in df.columns:
            logger.error(f"Input CSV must have '{args.constraint_b_column}' column")
            sys.exit(1)
        
        if "comparison_type" in df.columns:
            comparison_type = df["comparison_type"].iloc[0] if len(df) > 0 else "unknown"
            logger.info(f"Comparison type: {comparison_type}")
        
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    try:
        if args.provider == "openai":
            client = OpenAIClient(log_usage=True)
        else:
            client = AnthropicClient(log_usage=True)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    evaluator = NaturalnessEvaluator(
        llm_client=client,
        model=args.model,
        retry_attempts=args.retry_attempts
    )
    
    try:
        result_df = evaluator.evaluate_batch(
            df=df,
            prompt_column=args.prompt_column,
            constraint_a_column=args.constraint_a_column,
            constraint_b_column=args.constraint_b_column,
            output_path=args.output_path
        )
        
        logger.info(f"Successfully completed {len(result_df)} naturalness evaluations")
        
        if len(result_df) > 0 and "preference" in result_df.columns:
            pref_counts = result_df["preference"].value_counts()
            logger.info(f"\nPreference distribution:")
            logger.info(f"  A: {pref_counts.get('A', 0)}")
            logger.info(f"  B: {pref_counts.get('B', 0)}")
            
            if "score_a" in result_df.columns and "score_b" in result_df.columns:
                avg_score_a = result_df["score_a"].mean()
                avg_score_b = result_df["score_b"].mean()
                logger.info(f"\nAverage scores:")
                logger.info(f"  Score A: {avg_score_a:.2f}")
                logger.info(f"  Score B: {avg_score_b:.2f}")
        
        usage = get_total_usage()
        logger.info(f"\nTotal tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Naturalness evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Naturalness evaluation complete!")


if __name__ == "__main__":
    main()
