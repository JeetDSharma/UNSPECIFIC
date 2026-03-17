#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from tqdm import tqdm
from cs4.utils.config_loader import load_yaml, stamp, fill_vars
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.utils.io_utils import ensure_dir

RETRY_LIMIT_DEFAULT = 5
CHUNK_SIZE_MB = 1


def safe_filename(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def load_urls(path_or_csv: str):
    """Load URLs from text file or comma-separated list."""
    if os.path.isfile(path_or_csv):
        with open(path_or_csv, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [u.strip() for u in path_or_csv.split(",") if u.strip()]


def download_one(url: str, out_dir: Path, retry: int, chunk_bytes: int, logger):
    name = safe_filename(url)
    dst = out_dir / name

    if dst.exists() and dst.stat().st_size > 0:
        logger.info(f"skip existing: {name}")
        return

    for attempt in range(1, retry + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                desc = f"{name} (try {attempt}/{retry})"
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                    leave=False,
                ) as pbar:
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_bytes):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            logger.info(f"done: {name}")
            return
        except Exception as e:
            logger.warning(f"error {name} (try {attempt}): {e}")
            time.sleep(2 * attempt)
    logger.error(f"failed: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to download_config.yaml")
    ap.add_argument("--logging", default="configs/logging_config.yaml")
    args = ap.parse_args()

    # Load configuration first
    cfg = load_yaml(args.config)
    ts = stamp()

    # Expand timestamp placeholders
    logs_dir = Path(fill_vars(cfg["logs_dir"], timestamp=ts))
    out_dir = Path(fill_vars(cfg["output_dir"], timestamp=ts))
    ensure_dir(logs_dir)
    ensure_dir(out_dir)

    # Set up per-job log file
    job_log = logs_dir / "cs4_download.log"
    setup_logging(args.logging, job_log_file=job_log)
    logger = get_logger("CS4Downloader")

    # Resolve parameters
    urls = load_urls(cfg["urls_file"])
    if not urls:
        logger.error("No URLs found — check your urls.txt or config path.")
        sys.exit(1)

    workers = int(cfg.get("workers", 4))
    retry = int(cfg.get("retry_limit", RETRY_LIMIT_DEFAULT))
    chunk_bytes = int(cfg.get("chunk_mb", CHUNK_SIZE_MB)) * 1024 * 1024

    logger.info(f"download start: {len(urls)} files -> {out_dir}")

    # Start parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(
            tqdm(
                ex.map(lambda u: download_one(u, out_dir, retry, chunk_bytes, logger), urls),
                total=len(urls),
                desc="overall",
            )
        )

    logger.info("download complete.")
    logger.info(f"files saved in: {out_dir}")


if __name__ == "__main__":
    main()
