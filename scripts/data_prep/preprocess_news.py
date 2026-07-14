import pandas as pd
import re
import argparse


def clean_article(text):
    if not isinstance(text, str):
        return text

    text = re.sub(r'\n?\s*ADVERTISEMENT\s*\n?', '\n', text)

    text = re.sub(r'\n{2,}', '\n\n', text)

    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)

    return text.strip()


def is_good_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False

    if re.search(r'kAm|jC2|E96|H:==|@C', text):
        return False

    bad_tokens = re.findall(r'[A-Za-z0-9@=:;%\^\[\]\\/<>\?\$]{12,}', text)
    if len(bad_tokens) > 20:
        return False

    total_chars = len(text)
    readable_chars = sum(c.isalpha() or c.isspace() or c in ".,!?;:'\"()-" for c in text)
    ratio = readable_chars / total_chars if total_chars > 0 else 0

    if ratio < 0.8:
        return False

    noisy_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r'[@=]{2,}|[A-Z0-9]{6,}|[\^\\]{1,}', line):
            noisy_lines += 1

    if noisy_lines > 5:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Clean extracted news CSV")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to output cleaned CSV")
    parser.add_argument("--min-words", type=int, default=200, help="Minimum number of words in article content")
    parser.add_argument("--max-words", type=int, default=3000, help="Maximum number of words in article content")
    args = parser.parse_args()

    print(f"[INFO] Reading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    print(f"[INFO] Total rows before cleaning: {len(df)}")

    df = df[df["content"].notna()]

    if "title" in df.columns:
        df = df[df["title"].notna()]
        df = df[df["title"].astype(str).str.len() > 10]

    df["content"] = df["content"].apply(clean_article)

    df["word_count"] = df["content"].astype(str).str.split().str.len()
    df = df[(df["word_count"] > args.min_words) & (df["word_count"] < args.max_words)]

    print(f"[INFO] Rows after length filtering: {len(df)}")

    df = df[df["content"].apply(is_good_text)]

    print(f"[INFO] Rows after text quality filtering: {len(df)}")

    df = df.drop_duplicates(subset=["content"])

    print(f"[INFO] Rows after deduplication: {len(df)}")

    wanted_cols = ["title", "content", "link", "publication_date"]
    existing_cols = [col for col in wanted_cols if col in df.columns]
    df_final = df[existing_cols].copy()

    if len(df_final) > 0:
        row = df_final.iloc[0]
        print("\n[INFO] Example clean article:")
        if "title" in row:
            print("Title:", row["title"])
        if "content" in row:
            print("\nContent preview:\n", row["content"][:500])

    print(f"\n[INFO] Remaining clean articles: {len(df_final)}")
    df_final.to_csv(args.output_csv, index=False)
    print(f"[INFO] Cleaned CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()