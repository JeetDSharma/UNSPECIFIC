#!/usr/bin/env python3
"""
parse_blogs.py
---------------
Scans one or more Dolma JSONL files, extracts genuine blog posts,
and merges them into a single CSV file.
"""

import argparse, csv, json, re, sys
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs4.utils.config_loader import load_yaml, stamp, fill_vars
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.utils.io_utils import ensure_dir


# ---------- Blog URL Detection ----------
BLOG_DOMAINS = [
    "wordpress.com", "blogspot.com", "medium.com", "substack.com",
    "tumblr.com", "ghost.io", "weebly.com", "wixsite.com", "squarespace.com",
    "livejournal.com", "typepad.com", "hubpages.com", "dev.to", "hashnode.dev",
    "github.io", "gitlab.io", "netlify.app", "vercel.app",
    "notion.site", "over-blog.com", "canalblog.com",
    "hatena.ne.jp", "ameblo.jp", "blog.sina.com.cn"
]

BLOG_DOMAINS_RE = re.compile(
    r"(?:^|\.)(%s)$" % "|".join(re.escape(d) for d in BLOG_DOMAINS)
)
BLOG_SUBDOMAIN_RE = re.compile(r"(^|\.)blog\d*\.")
# BLOG_PATH_RE = re.compile(r"/(blog|blogs|post|posts|single-post|entry)(/|$)", re.I)
BLOG_PATH_RE = re.compile(r"/(blog|blogs)(/|$)", re.I)

def is_blog_url(url: str) -> bool:
    """Return True if URL likely represents a blog article."""
    try:
        u = url.lower()
        parsed = urlparse(u)
        host = parsed.hostname or ""
        path = parsed.path or ""

        # direct domain or subdomain indicators
        if BLOG_DOMAINS_RE.search(host):
            return True
        if BLOG_SUBDOMAIN_RE.search(host):
            return True
        # blog/post in path but not category/tag indexes
        if BLOG_PATH_RE.search(path):
            if any(x in path for x in ("/category/", "/tag/", "/author/")):
                return False
            return True
        return False
    except Exception:
        return False


# ---------- Core Parsing ----------
def parse_jsonl_to_csv(input_path: Path, writer: csv.DictWriter, logger, min_len: int, max_len: int):
    total, kept = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
                url = obj.get("metadata", {}).get("url", "")
                text = obj.get("text", "")
                text_length = len(text.strip().replace("\n", " "))
                if not url or not text:
                    continue
                if text_length < min_len or text_length > max_len:
                    continue
                if not is_blog_url(url):
                    continue

                writer.writerow({
                    "id": obj.get("id", ""),
                    "url": url,
                    "created": obj.get("created", ""),
                    "added": obj.get("added", ""),
                    "source": obj.get("source", ""),
                    "text": text.strip().replace("\n", " "),
                    "text_length": text_length,
                })
                kept += 1
            except Exception as e:
                logger.warning(f"{input_path.name}: failed line {total}: {e}")
    logger.info(f"{input_path.name}: processed {total:,} lines, kept {kept:,} blogs.")
    return total, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to parse_config.yaml")
    ap.add_argument("--logging", default="configs/logging_config.yaml")
    ap.add_argument("--vars", nargs="*", default=[], help="override vars like unzip_timestamp=20251007_141230_unzip")
    args = ap.parse_args()

    # ---- Load config ----
    cfg = load_yaml(args.config)
    kv = dict(v.split("=", 1) for v in args.vars) if args.vars else {}
    ts = stamp()

    input_jsonl = Path(fill_vars(cfg["input_jsonl"], timestamp=ts, **kv))
    output_csv = Path(fill_vars(cfg["output_csv"], timestamp=ts, **kv))
    logs_dir = Path(fill_vars(cfg["logs_dir"], timestamp=ts, **kv))
    ensure_dir(logs_dir)
    ensure_dir(output_csv.parent)

    # ---- Setup logging ----
    job_log = logs_dir / "cs4_parse.log"
    setup_logging(args.logging, job_log_file=job_log)
    logger = get_logger("CS4Parser")

    # ---- Gather files ----
    if input_jsonl.is_dir():
        jsonl_files = sorted(
            [p for p in input_jsonl.rglob("*.jsonl") if p.is_file()]
        )
    elif input_jsonl.is_file():
        jsonl_files = [input_jsonl]
    else:
        logger.error(f"Input path not found: {input_jsonl}")
        return

    if not jsonl_files:
        logger.error(f"No JSONL files found in {input_jsonl}")
        return

    logger.info(f"Found {len(jsonl_files)} JSONL files to parse.")
    min_len = int(cfg.get("min_text_length", 0))
    max_len = int(cfg.get("max_text_length", 1000000))
    # ---- Write merged CSV ----
    total_lines = total_kept = 0
    with open(output_csv, "w", newline="", encoding=cfg.get("encoding", "utf-8")) as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["id", "url", "created", "added", "source", "text", "text_length"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        for jf in tqdm(jsonl_files, desc="Parsing JSONL files"):
            t, k = parse_jsonl_to_csv(jf, writer, logger, min_len, max_len)
            total_lines += t
            total_kept += k

    logger.info(f"Blog parsing complete.")
    logger.info(f"Total lines: {total_lines:,}, kept blogs: {total_kept:,}")
    logger.info(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
