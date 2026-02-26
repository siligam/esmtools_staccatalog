#!/usr/bin/env bash
# Start the STAC Browser locally at http://localhost:8080
# This avoids the mixed-content block that occurs when using
# the hosted browser at https://radiantearth.github.io with a local HTTP catalog.
#
# Usage: ./start_stac_browser.sh [PORT]
# Default port: 8080

set -e

PORT=${1:-8080}
BROWSER_DIR="$HOME/stac-browser"

# Clone repo if not present
if [ ! -d "$BROWSER_DIR" ]; then
    echo "Cloning STAC Browser repository..."
    git clone https://github.com/radiantearth/stac-browser.git "$BROWSER_DIR"
fi

cd "$BROWSER_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies (this may take a minute)..."
    npm install
fi

CATALOG_URL="http://localhost:9091/catalog.json"

echo ""
echo "Starting STAC Browser at http://localhost:${PORT}"
echo "Open this URL in your browser:"
echo "  http://localhost:${PORT}/#/external/${CATALOG_URL}"
echo ""

# stac-browser v2 uses vue-cli-service, v3 uses vite
if npm run | grep -q "^  serve$"; then
    npm run serve -- --port "${PORT}"
elif npm run | grep -q "^  dev$"; then
    npx vite --port "${PORT}"
else
    npm start -- --port "${PORT}"
fi
