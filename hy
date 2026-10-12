#!/bin/sh
# hy — entry point for the translation pipeline.
# Uses the project virtualenv when present so no activation step is needed.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$here/.venv/bin/python" ]; then
  exec "$here/.venv/bin/python" -m hymt "$@"
fi
exec python3 -m hymt "$@"
