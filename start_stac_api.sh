#!/usr/bin/env bash
# Start the FESOM STAC API server.
# Usage: ./start_stac_api.sh [PORT]
# Default port: 9092

PORT=${1:-9092}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../.conda/envs/dummy/bin/python"

# Fall back to whatever python is on PATH if the explicit path doesn't exist
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(which python)"
fi

echo "Starting FESOM STAC API on port ${PORT} ..."
exec "$PYTHON" -m uvicorn stac_api:app --host 0.0.0.0 --port "${PORT}" --app-dir "${SCRIPT_DIR}"
