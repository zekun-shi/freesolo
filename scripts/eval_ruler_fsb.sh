#!/usr/bin/env bash
# Two-forest FS-B; see eval_ruler_common.sh for checkpoint and calibration options.
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_ruler_common.sh" FS-B "$@"
