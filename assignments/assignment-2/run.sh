#!/bin/bash

# Find Python command
get_python() {
    if command -v uv >/dev/null 2>&1; then
        echo "uv run python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        echo "ERROR: No Python found. Install python3, python, or uv" >&2
        exit 1
    fi
}

# Show help
show_help() {
    echo "Assignment 2 Calculator"
    echo ""
    echo "Supports Interpreters:"
    echo "  - uv"
    echo "  - python3"
    echo "  - python"
    echo ""
    echo "USAGE:"
    echo "  ./run.sh           # Run calculator"
    echo "  ./run.sh --help    # Show this help"
}

# Main execution
cd "$(dirname "$0")"

case "${1:-}" in
    --help|-h)
        show_help
        ;;
    *)
        # Run calculator
        $(get_python) src/calculator.py
        ;;
esac
