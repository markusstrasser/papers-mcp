#!/usr/bin/env bash
# install-hooks.sh — wire up the shared git pre-commit hooks for this repo.
#
# Currently installs:
#   pre-commit → ~/Projects/skills/hooks/pre-commit-no-large-binaries.sh
#                rejects PDFs >1KB and large binary archives. Bypass with
#                GIT_ALLOW_BINARIES=1.
#
# Re-run after a fresh clone. Idempotent.

set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
SHARED="$HOME/Projects/skills/hooks/pre-commit-no-large-binaries.sh"

if [[ ! -x "$SHARED" ]]; then
  echo "  ✗ missing shared hook: $SHARED" >&2
  exit 1
fi

ln -sf "$SHARED" "$REPO_ROOT/.git/hooks/pre-commit"
echo "  ✓ .git/hooks/pre-commit → $SHARED"
echo "  (bypass: GIT_ALLOW_BINARIES=1)"
