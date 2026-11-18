#!/usr/bin/env bash
# Shared by the experiment entry points. Source this file; do not execute it.
if [[ -n "${BASH_VERSION:-}" ]]; then
    _RUNTIME_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _RUNTIME_SOURCE="${(%):-%x}"
else
    printf '%s\n' 'runtime.sh must be sourced by bash or zsh' >&2
    return 1
fi
_RUNTIME_REPO_DIR="$(cd "$(dirname "$_RUNTIME_SOURCE")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$_RUNTIME_REPO_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$_RUNTIME_REPO_DIR/.venv/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
export PYTHON_BIN
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$_RUNTIME_REPO_DIR/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$_RUNTIME_REPO_DIR/.cache/matplotlib}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
unset _RUNTIME_REPO_DIR _RUNTIME_SOURCE
