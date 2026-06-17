"""
Constraint satisfaction evaluator module.
"""

import pandas as pd
import logging
import re
import threading
from time import sleep
from typing import Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cs4.core.prompts import get_evaluation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class FatalEvalError(Exception):
    """Raised when an eval API error is non-retryable (credit/auth/bad-request).

    Distinct from transient errors (rate-limit, timeout, server overload) which
    are worth retrying. A FatalEvalError should halt the whole batch immediately
    rather than burning attempts that cannot succeed.
    """


# Substrings that mark a definitely-fatal condition regardless of status code.
_FATAL_SUBSTRINGS = (
    "credit balance is too low",
    "billing",
    "invalid_api_key",
    "invalid x-api-key",
    "authentication",
    "permission",
)


def is_fatal_eval_error(e: Exception) -> bool:
    """Classify an exception as fatal (no point retrying) vs transient.

    Fatal: HTTP 400/401/403 (bad request, auth, permission) or a message that
    names a credit/billing/auth problem. Transient: 429 (rate limit), 408/5xx
    (timeout, server, overloaded), connection errors -> worth a backoff retry.
    """
    msg = str(getattr(e, "message", "") or str(e)).lower()
    if any(s in msg for s in _FATAL_SUBSTRINGS):
        return True
    status = getattr(e, "status_code", None)
    if status is None:
        # anthropic/openai sometimes nest the code on a .response
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        if status in (400, 401, 403):
            return True
        if status in (408, 409, 425, 429) or 500 <= status < 600:
            return False  # transient
    # Unknown error shape: treat as transient so a one-off blip can recover,
    # but the bounded retry count still prevents infinite loops.
    return False


class ConstraintEvaluator:
    """Evaluate constraint satisfaction in generated content."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0,
        terse: bool = False
    ):
        """
        Initialize constraint evaluator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_EVALUATION_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.terse = terse

        self.logger = logging.getLogger("CS4Evaluator")
    
    def evaluate_content(
        self,
        content: str,
        constraints: str,
        log: bool = True
    ) -> Tuple[str, int, int]:
        """
        Evaluate content against constraints.
        
        Args:
            content: Generated content to evaluate
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (satisfaction_results, num_satisfied, tokens_used)
        """
        prompt = get_evaluation_prompt(
            content_type=self.content_type,
            content=content,
            constraints=constraints,
            terse=self.terse
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model
                    )
                    results = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model    
                    )
                    results = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Extract number of satisfied constraints
                num_satisfied = self._extract_satisfaction_count(results)
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                    self.logger.info(f"Constraints satisfied: {num_satisfied}")
                
                return results, num_satisfied, tokens

            except Exception as e:
                # Fatal errors (credit/auth/bad-request) never recover on retry:
                # surface immediately so the batch can halt instead of burning calls.
                if is_fatal_eval_error(e):
                    self.logger.error(f"FATAL eval error (not retrying): {e}")
                    raise FatalEvalError(str(e)) from e
                # Transient (rate-limit/timeout/server): exponential backoff.
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} transient failure: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay * (2 ** (attempt - 1)))
                else:
                    raise

        raise RuntimeError("Failed to evaluate content")
    
    def _extract_satisfaction_count(self, results: str) -> int:
        """Extract the number of satisfied constraints from evaluation results."""
        # Count actual "Yes" lines - more reliable than LLM's stated count
        yes_count = len(re.findall(r'^\d+\.\s+Yes', results, re.MULTILINE))
        if yes_count > 0:
            return yes_count
        
        # Fallback: use LLM's stated count
        match = re.search(r'Number of constraints satisfied:\s*(\d+)', results)
        if match:
            return int(match.group(1))
        
        return 0
    
    def evaluate_batch(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        constraints_column: str = "constraints",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate constraint satisfaction for a batch of samples.
        Preserves all original columns and adds new evaluation columns.
        
        Args:
            df: Input DataFrame (can be any CSV with content and constraints)
            content_column: Name of column with content to evaluate
            constraints_column: Name of column with constraints
            output_path: Optional path to save results (saved incrementally)
            
        Returns:
            DataFrame with all original columns plus evaluation results
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        if constraints_column not in df.columns:
            raise ValueError(f"Column '{constraints_column}' not found in DataFrame")
        
        # Check for instruction_number
        has_instruction_num = "instruction_number" in df.columns
        if not has_instruction_num:
            self.logger.warning("No 'instruction_number' column found, using index")
        
        # Check for subset_size column (from bucketed constraints)
        has_subset_size = "subset_size" in df.columns
        if has_subset_size:
            self.logger.info("Using 'subset_size' column for constraint count")
        else:
            self.logger.info("No 'subset_size' column found, will parse constraints with regex")
        
        self.logger.info(f"Evaluating {len(df)} samples")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize new columns
        result_df["satisfaction_results"] = ""
        result_df["num_satisfied"] = 0
        result_df["total_constraints"] = 0
        result_df["satisfaction_rate"] = 0.0
        result_df["eval_model"] = ""
        result_df["eval_tokens"] = 0
        result_df["eval_timestamp"] = ""
        
        for idx, row in df.iterrows():
            content = row[content_column]
            constraints = row[constraints_column]
            instruction_num = row["instruction_number"] if has_instruction_num else idx + 1
            
            self.logger.info(f"Evaluating sample #{instruction_num} (row {idx + 1}/{len(df)})")
            
            try:
                satisfaction_results, num_satisfied, tokens = self.evaluate_content(
                    content=content,
                    constraints=constraints,
                    log=True
                )
                
                # Count total constraints - use subset_size if available, otherwise parse
                if has_subset_size:
                    total_constraints = int(row["subset_size"])
                else:
                    total_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                satisfaction_rate = num_satisfied / total_constraints if total_constraints > 0 else 0.0
                
                result_df.at[idx, "satisfaction_results"] = satisfaction_results
                result_df.at[idx, "num_satisfied"] = num_satisfied
                result_df.at[idx, "total_constraints"] = total_constraints
                result_df.at[idx, "satisfaction_rate"] = satisfaction_rate
                result_df.at[idx, "eval_model"] = self.model
                result_df.at[idx, "eval_tokens"] = tokens
                result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate sample {instruction_num}: {e}"
                )
                # Get total_constraints even in error case
                if has_subset_size:
                    error_total_constraints = int(row["subset_size"])
                else:
                    error_total_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                result_df.at[idx, "satisfaction_results"] = ""
                result_df.at[idx, "num_satisfied"] = 0
                result_df.at[idx, "total_constraints"] = error_total_constraints
                result_df.at[idx, "satisfaction_rate"] = 0.0
                result_df.at[idx, "eval_model"] = self.model
                result_df.at[idx, "eval_tokens"] = 0
                result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
            
            # Save incrementally after each row to prevent data loss
            if output_path:
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                self.logger.debug(f"Progress saved (row {idx + 1}/{len(df)})")
        
        if output_path:
            self.logger.info(f"All evaluation results saved to {output_path}")

        return result_df

    def evaluate_batch_parallel(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        constraints_column: str = "constraints",
        output_path: Optional[str] = None,
        max_workers: int = 5,
    ) -> pd.DataFrame:
        """Parallel version of evaluate_batch.

        Same output schema and column preservation as evaluate_batch, but runs
        up to `max_workers` evaluation calls concurrently. A FatalEvalError in
        any worker aborts the whole batch immediately (remaining work is skipped
        and the error re-raised) so a credit/auth failure cannot silently fill
        the output with zeros.
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        if constraints_column not in df.columns:
            raise ValueError(f"Column '{constraints_column}' not found in DataFrame")

        has_instruction_num = "instruction_number" in df.columns
        has_subset_size = "subset_size" in df.columns

        self.logger.info(
            f"Evaluating {len(df)} samples with {max_workers} parallel workers"
        )

        result_df = df.copy()
        result_df["satisfaction_results"] = ""
        result_df["num_satisfied"] = 0
        result_df["total_constraints"] = 0
        result_df["satisfaction_rate"] = 0.0
        result_df["eval_model"] = ""
        result_df["eval_tokens"] = 0
        result_df["eval_timestamp"] = ""

        def total_constraints_for(row):
            if has_subset_size:
                return int(row["subset_size"])
            return len(re.findall(r'^\d+\.', row[constraints_column], re.MULTILINE))

        abort = threading.Event()

        def _work(idx, row):
            if abort.is_set():
                return idx, None  # short-circuit once a fatal error is seen
            instruction_num = row["instruction_number"] if has_instruction_num else idx + 1
            results, num_satisfied, tokens = self.evaluate_content(
                content=row[content_column],
                constraints=row[constraints_column],
                log=False,
            )
            return idx, (results, num_satisfied, tokens, instruction_num)

        completed = 0
        fatal_error = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_idx = {
                executor.submit(_work, idx, row): idx
                for idx, row in df.iterrows()
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    _, payload = fut.result()
                except FatalEvalError as e:
                    fatal_error = e
                    abort.set()  # stop the rest from starting real work
                    continue
                except Exception as e:
                    # transient retries already exhausted -> record empty row
                    row = df.loc[idx]
                    result_df.at[idx, "satisfaction_results"] = ""
                    result_df.at[idx, "num_satisfied"] = 0
                    result_df.at[idx, "total_constraints"] = total_constraints_for(row)
                    result_df.at[idx, "satisfaction_rate"] = 0.0
                    result_df.at[idx, "eval_model"] = self.model
                    result_df.at[idx, "eval_tokens"] = 0
                    result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
                    self.logger.error(f"Failed to evaluate row {idx}: {e}")
                    continue

                if payload is None:
                    continue  # aborted short-circuit

                results, num_satisfied, tokens, instruction_num = payload
                row = df.loc[idx]
                total = total_constraints_for(row)
                rate = num_satisfied / total if total > 0 else 0.0
                result_df.at[idx, "satisfaction_results"] = results
                result_df.at[idx, "num_satisfied"] = num_satisfied
                result_df.at[idx, "total_constraints"] = total
                result_df.at[idx, "satisfaction_rate"] = rate
                result_df.at[idx, "eval_model"] = self.model
                result_df.at[idx, "eval_tokens"] = tokens
                result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()

                completed += 1
                self.logger.info(
                    f"Evaluated #{instruction_num} ({completed}/{len(df)}): "
                    f"{num_satisfied}/{total} satisfied"
                )
                if output_path and completed % 5 == 0:
                    result_df.to_csv(output_path, index=False, encoding="utf-8")

        if fatal_error is not None:
            # Do not write a half-zeroed file on a fatal failure.
            raise FatalEvalError(
                f"Aborted parallel eval after fatal error: {fatal_error}"
            )

        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"All evaluation results saved to {output_path}")

        return result_df

    def evaluate_batch_anthropic(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        constraints_column: str = "constraints",
        output_path: Optional[str] = None,
        poll_seconds: int = 30,
    ) -> pd.DataFrame:
        """Evaluate via the Anthropic Message Batches API (async, ~50% cheaper).

        Submits one request per row (custom_id = instruction_number), polls until
        the batch ends, then maps results back by custom_id. Same output schema as
        evaluate_batch. Anthropic judge only. Results arrive at the end (no
        incremental save); the batch_id is logged so a crashed poll is recoverable.
        """
        # Lazy imports: keep module import safe regardless of SDK internals.
        import time
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        if not isinstance(self.llm_client, AnthropicClient):
            raise ValueError("evaluate_batch_anthropic requires an AnthropicClient")
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        if constraints_column not in df.columns:
            raise ValueError(f"Column '{constraints_column}' not found in DataFrame")

        has_instruction_num = "instruction_number" in df.columns
        has_subset_size = "subset_size" in df.columns

        def total_constraints_for(row):
            if has_subset_size:
                return int(row["subset_size"])
            return len(re.findall(r'^\d+\.', row[constraints_column], re.MULTILINE))

        # custom_id -> df index label; guard against collisions.
        cid_to_idx = {}
        requests = []
        for idx, row in df.iterrows():
            cid = str(row["instruction_number"]) if has_instruction_num else str(idx)
            if cid in cid_to_idx:
                raise ValueError(f"Duplicate custom_id '{cid}' — instruction_number must be unique")
            cid_to_idx[cid] = idx
            prompt = get_evaluation_prompt(
                content_type=self.content_type,
                content=row[content_column],
                constraints=row[constraints_column],
                terse=self.terse,
            )
            requests.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                ),
            ))

        batch = self.llm_client.create_batch(requests)
        self.logger.info(f"Submitted batch {batch.id} with {len(requests)} requests "
                         f"(recover with this id if interrupted)")

        # Poll until ended.
        while True:
            b = self.llm_client.retrieve_batch(batch.id)
            if b.processing_status == "ended":
                break
            rc = b.request_counts
            self.logger.info(f"batch {batch.id}: processing={rc.processing} "
                             f"succeeded={rc.succeeded} errored={rc.errored}")
            time.sleep(poll_seconds)

        # Prepare output frame (same schema as evaluate_batch).
        result_df = df.copy()
        result_df["satisfaction_results"] = ""
        result_df["num_satisfied"] = 0
        result_df["total_constraints"] = 0
        result_df["satisfaction_rate"] = 0.0
        result_df["eval_model"] = ""
        result_df["eval_tokens"] = 0
        result_df["eval_timestamp"] = ""

        n_ok = 0
        for res in self.llm_client.batch_results(batch.id):
            idx = cid_to_idx.get(res.custom_id)
            if idx is None:
                self.logger.error(f"Result custom_id {res.custom_id} not in input — skipped")
                continue
            row = df.loc[idx]
            total = total_constraints_for(row)
            result_df.at[idx, "total_constraints"] = total
            result_df.at[idx, "eval_model"] = self.model
            result_df.at[idx, "eval_timestamp"] = datetime.now().isoformat()
            if res.result.type == "succeeded":
                msg = res.result.message
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                num_sat = self._extract_satisfaction_count(text)
                tokens = msg.usage.input_tokens + msg.usage.output_tokens
                result_df.at[idx, "satisfaction_results"] = text
                result_df.at[idx, "num_satisfied"] = num_sat
                result_df.at[idx, "satisfaction_rate"] = num_sat / total if total > 0 else 0.0
                result_df.at[idx, "eval_tokens"] = tokens
                n_ok += 1
            else:
                self.logger.error(f"custom_id {res.custom_id}: result.type={res.result.type}")

        self.logger.info(f"Batch complete: {n_ok}/{len(df)} succeeded")
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"All evaluation results saved to {output_path}")
        return result_df


# Legacy function for backward compatibility
def evaluate_constraints(
    input_path: str,
    output_path: str,
    model: str = None,
    content_type: str = "blog"
):
    """
    Legacy interface for constraint evaluation.
    
    Args:
        input_path: Path to CSV with generated content
        output_path: Path to save evaluation results
        model: LLM model identifier
        content_type: Type of content (blog, story, news)
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    evaluator = ConstraintEvaluator(model=model, content_type=content_type)
    df = pd.read_csv(input_path, encoding="utf-8")
    
    result_df = evaluator.evaluate_batch(df, output_path=output_path)
    
    return result_df
