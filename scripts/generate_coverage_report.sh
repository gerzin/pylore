#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH/.."

# shellcheck source=/dev/null
source venv/bin/activate
coverage run -m pytest
coverage html
deactivate
