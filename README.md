# ESM Tools STAC Catalog

A STAC (SpatioTemporal Asset Catalog) implementation for ESM Tools experiment outputs, providing standardized metadata and web-browsable access to climate model results.

---

## Overview

This project converts ESM Tools experiment outputs into a STAC catalog, making climate model data discoverable and accessible via web browsers and APIs. The catalog includes rich metadata, deterministic UUID-based IDs, and datacube extensions for multidimensional data.

---

## Project Structure

```
.
├── build_catalog.py          # Main script: generates STAC catalog from experiments.json
├── experiments.py            # Extracts experiment metadata from finished_config.yaml
├── experiments.json          # Minimal experiment metadata (input to build_catalog.py)
├── catalog/                  # Generated STAC catalog files
│   ├── catalog.json          # Root catalog
│   ├── {exp}/catalog.json    # Experiment-level catalogs
│   ├── {exp}/{model}/collection.json  # Model collections
│   └── {exp}/{model}/{uuid}-{item}.json  # Individual data items
├── start_stac_browser.sh     # Launch local STAC browser UI
├── start_stac_api.sh         # Launch STAC API service
├── stac_browser.py           # STAC browser implementation
├── stac_api.py               # STAC API implementation
└── validate_catalog.py       # Catalog validation utilities
```

---

## Workflow

### 1. Extract Experiment Metadata

The `experiments.py` script processes ESM Tools experiment directories:

- Scans `/albedo/work/user/pgierz/SciComp/Tutorials/AWIESM_Basics/experiments/`
- Reads `{exp}_finished_config.yaml` files from each experiment's `config/` directory
- Extracts essential metadata (model components, output directories, experiment settings)
- Generates `experiments.json` with minimal, clean metadata for catalog building

> **Note**: Currently uses `experiments.json` for simplicity. Future development may use YAML files directly.

### 2. Build STAC Catalog

Run the catalog generator:

```bash
~/.conda/envs/dummy/bin/python build_catalog.py
```

The `build_catalog.py` script creates a complete STAC catalog with:

- **Title fields** at all levels (catalog, experiment, collection, item)
- **Deterministic UUID5-based IDs** for uniqueness while maintaining readability
- **Datacube extension** with dimensions and variable information
- **CF metadata** and scientific extensions for climate data standards

### 3. Serve and Browse Catalog

#### Option A: Local STAC Browser

```bash
./start_stac_browser.sh
```

This starts:
- STAC API on port 9090
- STAC Browser UI on port 8080

Access the catalog at:
```
http://localhost:8080/external/http:/localhost:9090/catalog.json
```

#### Option B: External STAC Browser

If you have a separate STAC Browser instance (e.g., https://radiantearth.github.io/stac-browser/), simply point it to your local API:
```
http://localhost:8080/external/http:/localhost:9090/catalog.json
```

#### Option C: STAC API Only

```bash
python start_stac_api.py
```

Provides a RESTful STAC API at `http://localhost:9090` for programmatic access.

---

## Features

### Enhanced Metadata

- **Titles**: Human-readable titles at every catalog level
- **UUID5 IDs**: Short, deterministic IDs (`{8hex}-{name}`) for uniqueness and endpoint usability
- **Datacube Extension**: Full dimensional metadata (`time`, `latitude`, `longitude`) and variable listings
- **CF Compliance**: Climate and Forecast conventions integration

### Catalog Levels

1. **Root Catalog**: Links to all experiments
2. **Experiment Catalogs**: Per-experiment metadata and component links
3. **Collections**: Model-level data with all variables
4. **Items**: Individual NetCDF files with variable-specific metadata

---

## Quick Start

```bash
# 1. Generate experiment metadata
python experiments.py

# 2. Build STAC catalog
~/.conda/envs/dummy/bin/python build_catalog.py

# 3. Start browser and API
./start_stac_browser.sh

# 4. Open browser
# Navigate to: http://localhost:8080/external/http:/localhost:9090/catalog.json
```

---

## Dependencies

- Python environment: `~/.conda/envs/dummy/bin/python`
- Key packages: `pystac`, `xarray`, `pyyaml`, `numpy`

---

## Future Development

- Direct YAML file processing (bypass experiments.json)
- Enhanced validation with `stac-validator`
- Additional STAC extensions (e.g., projection, EO)
- Performance optimizations for large experiment sets
