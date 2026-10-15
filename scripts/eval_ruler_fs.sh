#!/usr/bin/env bash
# Finite RULER grid; see eval_ruler_common.sh for environment options.
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_ruler_common.sh" FS "$@"
