#!/usr/bin/env sh
# Format + lint staged Python files, then restage the results
files=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
[ -z "$files" ] && exit 0

uv run ruff format $files
uv run ruff check --fix $files
check=$?
git add $files  # restage formatting + safe autofixes
[ $check -eq 0 ] || { echo "pre-commit: ruff found unfixable issues above: fix and re-commit"; exit 1; }
