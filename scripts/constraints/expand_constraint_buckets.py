#!/usr/bin/env python3
"""
Expand constraints into progressive subsets (buckets) for testing.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.constraint_expander import ConstraintExpander
from cs4.config import Config
from cs4.utils.log_utils import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Expand constraints into progressive subsets"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with constraints"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with expanded constraint buckets"
    )
    parser.add_argument(
        "--constraint-column",
        default="constraints",
        help="Name of column containing constraints"
    )
    parser.add_argument(
        "--subset-sizes",
        default="7,15,23,31,39",
        help="Comma-separated list of subset sizes (default: 7,15,23,31,39)"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Config.LOGS_DIR / "constraint_expansion.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info("Starting constraint expansion")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    
    # Parse subset sizes
    try:
        subset_sizes = [int(s.strip()) for s in args.subset_sizes.split(",")]
        logger.info(f"Subset sizes: {subset_sizes}")
    except ValueError:
        logger.error(f"Invalid subset sizes: {args.subset_sizes}")
        sys.exit(1)
    
    # Load input data
    try:
        df = pd.read_csv(args.input_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)
    
    # Initialize expander
    try:
        expander = ConstraintExpander(subset_sizes=subset_sizes)
        
        # Expand constraints
        result_df = expander.expand_constraints(
            df=df,
            constraint_column=args.constraint_column,
            output_path=args.output_path
        )
        
        logger.info(f"Successfully expanded {len(df)} rows to {len(result_df)} rows")
        logger.info(f"Expansion factor: {len(result_df) / len(df):.1f}x")
        
        # Print statistics
        if "subset_size" in result_df.columns:
            for size in subset_sizes:
                count = (result_df["subset_size"] == size).sum()
                logger.info(f"  Subset size {size}: {count} rows")
        
    except Exception as e:
        logger.error(f"Constraint expansion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Constraint expansion complete!")


if __name__ == "__main__":
    main()
