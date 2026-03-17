#!/usr/bin/env python3
"""
Filter out samples where constraint satisfaction rate is above threshold.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Filter out samples where satisfaction rate is above threshold"
    )
    parser.add_argument(
        "--constraints-path",
        required=True,
        help="Path to revised constraints CSV"
    )
    parser.add_argument(
        "--evaluation-path",
        required=True,
        help="Path to base evaluation on revised constraints CSV"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output filtered constraints CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Satisfaction rate threshold (default: 0.40 = 40%%)"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    log_file = Config.LOGS_DIR / "filter_constraints.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Filtering samples with satisfaction rate > {args.threshold*100:.1f}%")
    logger.info(f"Constraints: {args.constraints_path}")
    logger.info(f"Evaluation: {args.evaluation_path}")
    logger.info(f"Output: {args.output_path}")
    
    try:
        constraints_df = pd.read_csv(args.constraints_path, encoding="utf-8")
        evaluation_df = pd.read_csv(args.evaluation_path, encoding="utf-8")
        logger.info(f"Loaded {len(constraints_df)} constraint samples")
        logger.info(f"Loaded {len(evaluation_df)} evaluation samples")
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        sys.exit(1)
    
    if 'satisfaction_rate' not in evaluation_df.columns:
        logger.error("Column 'satisfaction_rate' not found in evaluation CSV")
        sys.exit(1)
    
    merged = pd.merge(
        constraints_df,
        evaluation_df[['instruction_number', 'satisfaction_rate']],
        on='instruction_number',
        how='inner'
    )
    
    logger.info(f"Total samples before filtering: {len(merged)}")
    
    filtered_df = merged[merged['satisfaction_rate'] <= args.threshold]
    
    filtered_df = filtered_df[constraints_df.columns]
    
    if len(filtered_df) == 0:
        logger.error(f"All samples were filtered out! No samples have satisfaction_rate <= {args.threshold}")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.info(f"Filtered constraints saved to {args.output_path}")
    logger.info(f"Samples kept: {len(filtered_df)}/{len(merged)} ({len(filtered_df)/len(merged)*100:.1f}%)")
    logger.info(f"Samples removed: {len(merged)-len(filtered_df)} ({(len(merged)-len(filtered_df))/len(merged)*100:.1f}%)")
    logger.info("Constraint filtering complete!")


if __name__ == "__main__":
    main()
