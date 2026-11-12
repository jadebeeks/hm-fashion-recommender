#!/usr/bin/env bash
# Download the H&M CSV files from Kaggle into data/raw/.
# Prerequisites: see data/README.md (accept competition rules + kaggle.json token).
set -euo pipefail

COMPETITION="h-and-m-personalized-fashion-recommendations"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw"
mkdir -p "$OUT"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "The kaggle command was not found. Run: pip install -r requirements.txt" >&2
  exit 1
fi

for f in articles.csv customers.csv transactions_train.csv; do
  if [ -f "$OUT/$f" ]; then
    echo "Already have $f, skipping."
    continue
  fi
  echo "Downloading $f ..."
  kaggle competitions download -c "$COMPETITION" -f "$f" -p "$OUT"
  # Kaggle may deliver a single file zipped
  if [ -f "$OUT/$f.zip" ]; then
    unzip -o -q "$OUT/$f.zip" -d "$OUT" && rm "$OUT/$f.zip"
  fi
done

echo "Done. Files in $OUT:"
ls -lh "$OUT"
