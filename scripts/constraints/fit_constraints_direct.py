#!/usr/bin/env python3
"""
CLI script for fitting constraints directly to merged blogs.
Skips base generation and fits constraints to the original merged content.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cs4.core.constraint_fitter import ConstraintFitter
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Fit constraints directly to merged blog content"
    )
    parser.add_argument(
        "--constraints-path",
        required=True,
        help="Path to CSV with generated constraints"
    )
    parser.add_argument(
        "--merged-path",
        required=True,
        help="Path to CSV with merged blogs (from merge_blogs.py)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with fitted content"
    )
    parser.add_argument(
        "--constraint-column",
        default="constraints",
        help="Name of column containing constraints (use 'selected_constraints' for bucketed)"
    )
    parser.add_argument(
        "--merged-content-column",
        default="Merged Blog",
        help="Name of column containing merged blog content"
    )
    parser.add_argument(
        "--domain",
        default="blog",
        help="Content domain (blog, story, news)"
    )
    parser.add_argument(
        "--model",
        default=Config.DEFAULT_FITTING_MODEL,
        help=f"LLM model to use (default: {Config.DEFAULT_FITTING_MODEL})"
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
    log_file = Config.LOGS_DIR / "constraint_fitting_direct.log"
    setup_logging(args.logging_config, job_log_file=log_file)
    logger = get_logger("CS4Generator")
    
    logger.info(f"Starting direct constraint fitting for domain: {args.domain}")
    logger.info(f"Constraints: {args.constraints_path}")
    logger.info(f"Merged blogs: {args.merged_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Model: {args.model}")
    
    # Load constraints
    try:
        constraints_df = pd.read_csv(args.constraints_path, encoding="utf-8")
        logger.info(f"Loaded {len(constraints_df)} constraint sets")
    except Exception as e:
        logger.error(f"Failed to load constraints file: {e}")
        sys.exit(1)
    
    # Load merged blogs
    try:
        merged_df = pd.read_csv(args.merged_path, encoding="utf-8")
        logger.info(f"Loaded {len(merged_df)} merged blogs")
    except Exception as e:
        logger.error(f"Failed to load merged blogs file: {e}")
        sys.exit(1)
    
    # Verify required columns
    if args.constraint_column not in constraints_df.columns:
        logger.error(f"Column '{args.constraint_column}' not found in constraints file")
        logger.info(f"Available columns: {list(constraints_df.columns)}")
        sys.exit(1)
    
    if args.merged_content_column not in merged_df.columns:
        logger.error(f"Column '{args.merged_content_column}' not found in merged blogs file")
        logger.info(f"Available columns: {list(merged_df.columns)}")
        sys.exit(1)
    
    # Initialize LLM client
    if args.provider == "openai":
        client = OpenAIClient(log_usage=True)
    else:
        client = AnthropicClient(log_usage=True)
    
    # Initialize constraint fitter
    try:
        fitter = ConstraintFitter(
            llm_client=client,
            model=args.model,
            content_type=args.domain,           
            retry_attempts=args.retry_attempts
        )
        
        # Prepare data for fitting
        # We need to merge constraints with merged blogs
        # Assuming both have 'instruction_number' for matching
        
        if "instruction_number" in constraints_df.columns and "instruction_number" in merged_df.columns:
            # Merge on instruction_number
            merged_data = pd.merge(
                constraints_df,
                merged_df[[args.merged_content_column, "instruction_number"]],
                on="instruction_number",
                how="inner"
            )
            logger.info(f"Matched {len(merged_data)} samples by instruction_number")
        else:
            # Assume same order and length
            if len(constraints_df) != len(merged_df):
                logger.warning(
                    f"Constraints ({len(constraints_df)}) and merged blogs ({len(merged_df)}) "
                    "have different lengths. Using first min(len) samples."
                )
            
            # Create merged dataframe
            min_len = min(len(constraints_df), len(merged_df))
            merged_data = constraints_df.iloc[:min_len].copy()
            merged_data[args.merged_content_column] = merged_df[args.merged_content_column].iloc[:min_len].values
            logger.info(f"Processing {len(merged_data)} samples")
        
        # Fit constraints directly to merged content
        logger.info(f"Fitting constraints to merged blogs for {len(merged_data)} samples")
        
        results = []
        for idx, row in merged_data.iterrows():
            instruction_num = row.get("instruction_number", idx + 1)
            
            # Extract fields
            task = row.get("main_task", "Revise the blog to satisfy constraints")
            constraints = row[args.constraint_column]
            merged_content = row[args.merged_content_column]
            
            logger.info(f"Processing sample #{instruction_num}")
            
            try:
                # Use merged blog as "base content" for fitting
                fitted_content, tokens = fitter.fit_content(
                    task=task,
                    base_content=merged_content,  # Using merged blog directly
                    constraints=constraints,
                    log=True
                )
                
                result = row.copy()
                result["merged_blog_original"] = merged_content
                result["fitted_content"] = fitted_content
                result["fitted_length"] = len(fitted_content)
                result["original_length"] = len(merged_content)
                
                # Count constraints
                constraint_list = [c.strip() for c in constraints.split("\n") if c.strip() and not c.strip().startswith("Constraint")]
                num_constraints = len([c for c in constraint_list if c and c[0].isdigit()])
                result["num_constraints"] = num_constraints
                
                result["model_used"] = args.model
                result["tokens_used"] = tokens
                result["timestamp"] = pd.Timestamp.now().isoformat()
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to fit constraints for sample #{instruction_num}: {e}")
                # Keep original with error marker
                result = row.copy()
                result["fitted_content"] = merged_content
                result["error"] = str(e)
                results.append(result)
        
        result_df = pd.DataFrame(results)
        
        # Save results
        result_df.to_csv(args.output_path, index=False, encoding="utf-8")
        logger.info(f"Fitted content saved to {args.output_path}")
        
        logger.info(f"Successfully fitted content for {len(result_df)} samples")
        
        # Print usage summary
        usage = get_total_usage()
        logger.info(f"Total tokens used: {usage['total_tokens']}")
        
    except Exception as e:
        logger.error(f"Direct constraint fitting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Direct constraint fitting complete!")


if __name__ == "__main__":
    main()
