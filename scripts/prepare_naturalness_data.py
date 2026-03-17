#!/usr/bin/env python3
"""
CLI script for preparing naturalness evaluation data by sampling constraints.
"""

import argparse
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def parse_constraints(constraint_text: str) -> list:
    """
    Parse numbered constraint list into individual constraints.
    
    Args:
        constraint_text: Newline-separated numbered list of constraints
        
    Returns:
        List of constraint strings (without numbers)
    """
    if not isinstance(constraint_text, str) or not constraint_text.strip():
        return []
    
    # Split by newlines and extract numbered items
    lines = constraint_text.strip().split('\n')
    constraints = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match patterns like "1. constraint text" or "1) constraint text"
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if match:
            constraints.append(match.group(1).strip())
        elif line and not re.match(r'^\d+[\.\)]?\s*$', line):
            # If no number prefix but has content, might be continuation
            if constraints:
                constraints[-1] += ' ' + line
    
    return constraints


def main():
    parser = argparse.ArgumentParser(
        description="Prepare naturalness evaluation data by sampling constraints from datasets"
    )
    parser.add_argument(
        "--a-path",
        required=True,
        help="Path to A dataset (no_revision)"
    )
    parser.add_argument(
        "--b1-path",
        required=True,
        help="Path to B1 dataset (revision without base evaluation)"
    )
    parser.add_argument(
        "--b2-path",
        required=True,
        help="Path to B2 dataset (revision with base evaluation)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output CSVs"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of constraints to sample per main task (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--logging-config",
        default="configs/logging_config.yaml",
        help="Path to logging config"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    log_file = Config.LOGS_DIR / "prepare_naturalness_data.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4NaturalnessPrep")
    
    logger.info("Starting naturalness data preparation")
    logger.info(f"A dataset: {args.a_path}")
    logger.info(f"B1 dataset: {args.b1_path}")
    logger.info(f"B2 dataset: {args.b2_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Samples per task: {args.num_samples}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load datasets
    try:
        df_a = pd.read_csv(args.a_path, encoding="utf-8")
        df_b1 = pd.read_csv(args.b1_path, encoding="utf-8")
        df_b2 = pd.read_csv(args.b2_path, encoding="utf-8")
        
        logger.info(f"Loaded A: {len(df_a)} rows")
        logger.info(f"Loaded B1: {len(df_b1)} rows")
        logger.info(f"Loaded B2: {len(df_b2)} rows")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)
    
    # Validate required columns
    if "instruction_number" not in df_a.columns:
        logger.error("A dataset missing 'instruction_number' column")
        sys.exit(1)
    if "main_task" not in df_a.columns:
        logger.error("A dataset missing 'main_task' column")
        sys.exit(1)
    if "constraints" not in df_a.columns:
        logger.error("A dataset missing 'constraints' column")
        sys.exit(1)
    if "revised_constraints" not in df_b1.columns:
        logger.error("B1 dataset missing 'revised_constraints' column")
        sys.exit(1)
    if "revised_constraints" not in df_b2.columns:
        logger.error("B2 dataset missing 'revised_constraints' column")
        sys.exit(1)
    
    # Merge datasets on instruction_number
    try:
        merged_b1 = df_a.merge(
            df_b1[["instruction_number", "revised_constraints"]],
            on="instruction_number",
            how="inner",
            suffixes=("", "_b1")
        )
        merged_b1.rename(columns={"revised_constraints": "constraints_b1"}, inplace=True)
        
        merged_b2 = df_a.merge(
            df_b2[["instruction_number", "revised_constraints"]],
            on="instruction_number",
            how="inner",
            suffixes=("", "_b2")
        )
        merged_b2.rename(columns={"revised_constraints": "constraints_b2"}, inplace=True)
        
        logger.info(f"Merged A-B1: {len(merged_b1)} rows")
        logger.info(f"Merged A-B2: {len(merged_b2)} rows")
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        sys.exit(1)
    
    # Get unique main tasks
    unique_tasks = df_a["main_task"].unique()
    logger.info(f"Found {len(unique_tasks)} unique main tasks")
    
    # Prepare comparison data
    comparisons_a_b1 = []
    comparisons_a_b2 = []
    
    for task in unique_tasks:
        # Get rows for this task from merged datasets
        task_rows_b1 = merged_b1[merged_b1["main_task"] == task]
        task_rows_b2 = merged_b2[merged_b2["main_task"] == task]
        
        if len(task_rows_b1) == 0:
            logger.warning(f"No B1 data for task: {task[:50]}...")
            continue
        if len(task_rows_b2) == 0:
            logger.warning(f"No B2 data for task: {task[:50]}...")
            continue
        
        # Use first row for this task
        row_b1 = task_rows_b1.iloc[0]
        row_b2 = task_rows_b2.iloc[0]
        
        # Parse constraints
        constraints_a = parse_constraints(row_b1["constraints"])
        constraints_b1 = parse_constraints(row_b1["constraints_b1"])
        constraints_b2 = parse_constraints(row_b2["constraints_b2"])
        
        if len(constraints_a) == 0:
            logger.warning(f"No constraints parsed for A in task: {task[:50]}...")
            continue
        if len(constraints_b1) == 0:
            logger.warning(f"No constraints parsed for B1 in task: {task[:50]}...")
            continue
        if len(constraints_b2) == 0:
            logger.warning(f"No constraints parsed for B2 in task: {task[:50]}...")
            continue
        
        # Sample constraint indices
        max_constraints = min(len(constraints_a), len(constraints_b1), len(constraints_b2))
        if max_constraints < args.num_samples:
            logger.warning(
                f"Task has only {max_constraints} constraints, sampling all instead of {args.num_samples}"
            )
            sampled_indices = list(range(max_constraints))
        else:
            sampled_indices = np.random.choice(max_constraints, args.num_samples, replace=False)
        
        # Create comparison rows
        for idx in sampled_indices:
            # A vs B1
            comparisons_a_b1.append({
                "instruction_number": row_b1["instruction_number"],
                "main_task": task,
                "constraint_index": int(idx + 1),  # 1-indexed
                "constraint_a": constraints_a[idx],
                "constraint_b": constraints_b1[idx],
                "comparison_type": "A_vs_B1"
            })
            
            # A vs B2
            comparisons_a_b2.append({
                "instruction_number": row_b2["instruction_number"],
                "main_task": task,
                "constraint_index": int(idx + 1),  # 1-indexed
                "constraint_a": constraints_a[idx],
                "constraint_b": constraints_b2[idx],
                "comparison_type": "A_vs_B2"
            })
    
    # Create DataFrames
    df_a_b1 = pd.DataFrame(comparisons_a_b1)
    df_a_b2 = pd.DataFrame(comparisons_a_b2)
    
    logger.info(f"Created {len(df_a_b1)} A vs B1 comparisons")
    logger.info(f"Created {len(df_a_b2)} A vs B2 comparisons")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_a_b1 = output_dir / "naturalness_eval_A_vs_B1.csv"
    output_a_b2 = output_dir / "naturalness_eval_A_vs_B2.csv"
    
    try:
        df_a_b1.to_csv(output_a_b1, index=False, encoding="utf-8")
        df_a_b2.to_csv(output_a_b2, index=False, encoding="utf-8")
        
        logger.info(f"Saved A vs B1 to: {output_a_b1}")
        logger.info(f"Saved A vs B2 to: {output_a_b2}")
    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        sys.exit(1)
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
