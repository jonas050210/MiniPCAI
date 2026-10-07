#!/usr/bin/env bash
#
# Run the test suite in CI and make the failures readable without the raw log.
#
# GitHub's log download is not always reachable, so every failing test is also
# emitted as a workflow annotation (readable through the checks API:
# `gh api repos/{owner}/{repo}/check-runs/{id}/annotations`) and the tail of the
# output is written to the job summary.
#
# Usage: scripts/ci_pytest.sh [command prefix ...]
#        scripts/ci_pytest.sh xvfb-run -a      # headless Linux

set -uo pipefail

prefix=("$@")
output=$(mktemp)

if [ ${#prefix[@]} -gt 0 ]; then
    "${prefix[@]}" python -m pytest -q --tb=line -rf >"$output" 2>&1
else
    python -m pytest -q --tb=line -rf >"$output" 2>&1
fi
status=$?

cat "$output"

if [ "$status" -ne 0 ]; then
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            echo "### pytest failed (exit $status)"
            echo ""
            echo '```'
            tail -n 80 "$output"
            echo '```'
        } >>"$GITHUB_STEP_SUMMARY"
    fi
    grep -E "^(FAILED|ERROR)" "$output" | head -40 | while IFS= read -r line; do
        echo "::error::$line"
    done
fi

exit "$status"
