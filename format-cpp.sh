#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
JSON_FILE="$SCRIPT_DIR/build/compile_commands.json"
jq -r '.[].file' "$JSON_FILE" | xargs -I{} clang-format-16 -i {}
