#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: ./run.sh <input_pdf_dir> <output_path>}"
output_path="${2:?usage: ./run.sh <input_pdf_dir> <output_path>}"

cd /app
exec python3 -m scripts.predict "$input_dir" "$output_path"
