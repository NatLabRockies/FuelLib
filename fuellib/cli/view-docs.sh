#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCS_INDEX="$REPO_ROOT/docs/_build/html/index.html"

if [[ ! -f "$DOCS_INDEX" ]]; then
    echo "Error: Documentation has not been built yet."
    echo "Expected to find: $DOCS_INDEX"
    echo "Build the docs first (e.g. 'uv run task build-docs')."
    exit 1
fi

if [[ -n "${BROWSER:-}" ]]; then
    exec "$BROWSER" "$DOCS_INDEX"
fi

# Fall back to platform-specific openers when BROWSER is not set.
case "$(uname -s)" in
    Darwin)
        exec open "$DOCS_INDEX"
        ;;
    Linux)
        if grep -qi microsoft /proc/version 2>/dev/null && command -v explorer.exe >/dev/null 2>&1; then
            exec explorer.exe "$(wslpath -w "$DOCS_INDEX")"
        elif command -v xdg-open >/dev/null 2>&1; then
            exec xdg-open "$DOCS_INDEX"
        fi
        ;;
esac

echo "Error: The BROWSER environment variable is not set and no"
echo "platform-specific fallback (open/xdg-open/explorer.exe) was found."
echo "Set BROWSER to a web browser executable, for example:"
echo "  export BROWSER='/mnt/c/Program Files/Mozilla Firefox/firefox.exe'"
exit 1