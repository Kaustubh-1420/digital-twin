#!/usr/bin/env bash
# Deploy the backend to the HF Space as a single orphan commit.
#
# Why: `git push --force space main` is rejected by HF because the repo holds
# frontend/public/sample-avatar.glb (13 MB, over the 10 MiB non-LFS limit).
# The Space only needs the backend, so we push just that.
#
# Deploys committed HEAD (not the working tree), so every Space deploy maps to
# a GitHub commit. README_hf.md is published as README.md (the Space reads its
# config from the YAML header; the repo README has none).
#
# Usage: tools/deploy_space.sh [--dry-run]

set -euo pipefail

FILES=(app.py backend setup.sh requirements.txt packages.txt LICENSE README_hf.md)
REMOTE=space

cd "$(git rev-parse --show-toplevel)"

dry_run=false
[[ "${1:-}" == "--dry-run" ]] && dry_run=true

if ! git diff --quiet HEAD -- "${FILES[@]}"; then
  echo "Uncommitted changes in deployed paths; commit them first:" >&2
  git status --short -- "${FILES[@]}" >&2
  exit 1
fi

url=$(git remote get-url "$REMOTE")
sha=$(git rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git archive HEAD "${FILES[@]}" | tar -x -C "$tmp"
mv "$tmp/README_hf.md" "$tmp/README.md"

echo "Deploying $sha to $REMOTE:"
(cd "$tmp" && find . -type f | sed 's#^\./#  #' | sort)

if $dry_run; then
  echo "Dry run: nothing pushed."
  exit 0
fi

git -C "$tmp" init -q -b main
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -m "Deploy $sha from GitHub main"
git -C "$tmp" push -q --force "$url" main
echo "Pushed. Space rebuilds now; check its logs."
