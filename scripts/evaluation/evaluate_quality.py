#!/usr/bin/env python3
"""
CLI script for pairwise quality evaluation comparing content across different constraint subsets.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.quality_evaluator import QualityEvaluator
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate content quality through pairwise comparisons across constraint subsets"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with multiple subsets (e.g., revised_base_23_39.csv)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV (e.g., quality_eval_results.csv)"
    )
    parser.add_argument(
        "--content-column",
        default="revised_base",
        help="Column name containing content to evaluate (default: revised_base)"
    )
    parser.add_argument(
        "--vs-base",
        action="store_true",
        help="Compare each row's revised content against its original base story "
             "(--base-column) instead of pairing constraint subsets. Per-row, every subset."
    )
    parser.add_argument(
        "--base-column",
        default="base_content",
        help="Column with the original base story for --vs-base (default: base_content)"
    )
    parser.add_argument(
        "--base-path",
        help="For --vs-base: separate CSV holding --base-column, merged onto the input "
             "on instruction_number (use when the base story is not in the input file)."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit as an async Anthropic Message Batch (~50%% cheaper). "
             "Anthropic provider + --vs-base only."
    )
    parser.add_argument(
        "--baseline-subset",
        type=int,
        default=23,
        help="Subset size to use as baseline for comparison (default: 23)"
    )
    parser.add_argument(
        "--comparison-subsets",
        type=str,
        help="Comma-separated list of subsets to compare against baseline (e.g., '7,15,39'). If not provided, compares against all other subsets."
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

    if args.batch and not args.vs_base:
        print("Error: --batch is currently implemented only with --vs-base")
        sys.exit(1)

    # Parse comparison subsets if provided
    comparison_subsets = None
    if args.comparison_subsets:
        try:
            comparison_subsets = [int(s.strip()) for s in args.comparison_subsets.split(",")]
        except ValueError:
            print(f"Error: Invalid comparison_subsets format. Use comma-separated integers (e.g., '7,15,39')")
            sys.exit(1)
    
    log_file = Config.LOGS_DIR / "quality_evaluation.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4QualityEvaluator")
    
    logger.info(f"Starting pairwise quality evaluation")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Content column: {args.content_column}")
    logger.info(f"Baseline subset: {args.baseline_subset}")
    logger.info(f"Comparison subsets: {comparison_subsets or 'all others'}")
    logger.info(f"Model: {args.model}")
    
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} rows from input CSV")

        if args.content_column not in df.columns:
            logger.error(f"Input CSV must have '{args.content_column}' column")
            sys.exit(1)

        if args.vs_base:
            if args.base_path:
                if "instruction_number" not in df.columns:
                    logger.error("--base-path merge requires 'instruction_number' in the input CSV")
                    sys.exit(1)
                base_df = pd.read_csv(args.base_path, encoding="utf-8")
                if args.base_column not in base_df.columns:
                    logger.error(f"--base-path CSV must have '{args.base_column}' column")
                    sys.exit(1)
                if "instruction_number" not in base_df.columns:
                    logger.error("--base-path CSV must have 'instruction_number' column")
                    sys.exit(1)
                if args.base_column in df.columns:
                    df = df.drop(columns=[args.base_column])
                before = len(df)
                df = df.merge(
                    base_df[["instruction_number", args.base_column]],
                    on="instruction_number", how="left"
                )
                merged_ok = df[args.base_column].notna().sum()
                logger.info(f"Merged base story from {args.base_path}: "
                            f"{merged_ok}/{before} rows matched on instruction_number")
                if merged_ok == 0:
                    logger.error("No rows matched on instruction_number — check base-path")
                    sys.exit(1)
            if args.base_column not in df.columns:
                logger.error(f"Input CSV must have '{args.base_column}' column for --vs-base "
                             f"(or pass --base-path)")
                sys.exit(1)
        else:
            if "instruction_number" not in df.columns:
                logger.error("Input CSV must have 'instruction_number' column")
                sys.exit(1)
            if "subset_size" not in df.columns:
                logger.error("Input CSV must have 'subset_size' column")
                sys.exit(1)
            subset_counts = df.groupby("subset_size").size()
            logger.info(f"Subset distribution:\n{subset_counts}")

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
    
    evaluator = QualityEvaluator(
        llm_client=client,
        model=args.model,
        retry_attempts=args.retry_attempts
    )
    
    try:
        if args.vs_base:
            if args.batch:
                if args.provider == "anthropic":
                    result_df = evaluator.evaluate_batch_base_vs_revised_anthropic(
                        df=df,
                        revised_column=args.content_column,
                        base_column=args.base_column,
                        output_path=args.output_path,
                    )
                else:
                    result_df = evaluator.evaluate_batch_base_vs_revised_openai(
                        df=df,
                        revised_column=args.content_column,
                        base_column=args.base_column,
                        output_path=args.output_path,
                    )
            else:
                result_df = evaluator.evaluate_batch_base_vs_revised(
                    df=df,
                    revised_column=args.content_column,
                    base_column=args.base_column,
                    output_path=args.output_path,
                )
            logger.info(f"Successfully completed {len(result_df)} base-vs-revised evaluations")
            if len(result_df) > 0 and "winner" in result_df.columns:
                logger.info(f"\nOverall winners: {result_df['winner'].value_counts().to_dict()}")
                if "subset_size" in result_df.columns:
                    by_subset = result_df.groupby("subset_size")["revised_win"].mean()
                    logger.info(f"Revised win-rate by subset:\n{by_subset}")
        else:
            result_df = evaluator.evaluate_batch_pairwise(
                df=df,
                content_column=args.content_column,
                baseline_subset=args.baseline_subset,
                comparison_subsets=comparison_subsets,
                output_path=args.output_path
            )

            logger.info(f"Successfully completed {len(result_df)} pairwise evaluations")

            if len(result_df) > 0:
                for comp_subset in result_df["comparison_subset"].unique():
                    subset_rows = result_df[result_df["comparison_subset"] == comp_subset]

                    if "overall_pref" in subset_rows.columns:
                        pref_counts = subset_rows["overall_pref"].value_counts()
                        logger.info(
                            f"\nBaseline ({args.baseline_subset}) vs {comp_subset}: "
                            f"{len(subset_rows)} comparisons"
                        )
                        logger.info(f"  Preferences: {pref_counts.to_dict()}")

        usage = get_total_usage()
        logger.info(f"\nTotal tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Quality evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Quality evaluation complete!")


if __name__ == "__main__":
    main()
