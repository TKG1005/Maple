#!/usr/bin/env bash
set -euo pipefail
PATCH_FILE=${1:-/tmp/rqid.patch}
TARGET=pokemon-showdown/lib/process-manager.ts
BRANCH=feature/node-ipc-server-development

echo "Resume script starting..."
echo "Branch: $(git rev-parse --abbrev-ref HEAD || echo unknown)"
echo "Target: $TARGET"
if [ ! -f "$PATCH_FILE" ]; then
  echo "ERROR: patch file not found: $PATCH_FILE"; exit 2
fi

# Ensure on correct branch
git fetch origin || true
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
else
  echo "Branch $BRANCH does not exist locally; creating from HEAD"
  git checkout -b "$BRANCH"
fi

# Optional safety: show current diff for the target
echo "Current diff for $TARGET (if any):"
git --no-pager diff -- "$TARGET" || true

# Check patch applicability
echo "Checking patch..."
if ! git apply --check "$PATCH_FILE"; then
  echo "Patch does not apply cleanly. Aborting. Inspect $PATCH_FILE and $TARGET"
  exit 3
fi

# Apply patch
echo "Applying patch..."
git apply "$PATCH_FILE"

# Show result
echo "Patch applied. Showing modified lines:"
git --no-pager diff -- "$TARGET" | sed -n '1,200p'

echo "Build/type-check suggestion: run 'npm run build' or 'npx tsc --noEmit'"
echo "If OK, commit: git add $TARGET && git commit -m 'Add rqid injection to Raw
ProcessManager.pipeStream (IPC)'"

