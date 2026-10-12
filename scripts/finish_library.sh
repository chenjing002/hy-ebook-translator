#!/bin/bash
# Resume the model download, then translate and publish the supplied documents.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p .state
status=.state/finish.status
printf 'downloading\n' > "$status"

until .venv/bin/python scripts/download_parallel.py --workers 16; do
  printf 'download retry at %s\n' "$(date)"
  sleep 60
done

until ./hy setup --endpoint https://huggingface.co; do
  printf 'metadata retry at %s\n' "$(date)"
  sleep 60
done

printf 'translating sample\n' > "$status"
./hy run mlx-primer.md --strict || { printf 'failed: sample\n' > "$status"; exit 1; }
printf 'translating report\n' > "$status"
./hy run '浑水做空辉山乳业报告.md' --strict || { printf 'failed: report\n' > "$status"; exit 1; }
printf 'translating book\n' > "$status"
./hy run "Financial Statement Analysis_ A Practitioner's Guide.pdf.md" --strict || {
  printf 'failed: book\n' > "$status"; exit 1;
}

./hy publish --pages || { printf 'failed: publish\n' > "$status"; exit 1; }
printf 'complete\n' > "$status"
