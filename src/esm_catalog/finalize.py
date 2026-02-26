"""Finalize catalog structure after Snakemake generates items."""

import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

STAC_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"


def make_short_id(name: str) -> str:
    """Generate deterministic short ID."""
    uid = uuid.uuid5(STAC_UUID_NAMESPACE, name)
    return f"{uid.hex[:8]}-{name}"


def write_json(path: Path, data: dict) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_exp_base_path(exp_data: dict) -> Path | None:
    """Get experiment base directory from file paths."""
    for model_files in exp_data.get("files", {}).values():
        if model_files:
            return Path(model_files[0]).parent.parent.parent
    return None


def finalize_catalog(
    experiments_json_path: str | Path,
    output_dir: str | Path,
    distributed: bool = True,
) -> None:
    """Generate catalog.json and collection.json files after items exist.

    Args:
        experiments_json_path: Path to experiments.json
        output_dir: Output directory (or parent for distributed mode)
        distributed: Whether items are in experiment directories
    """
    with open(experiments_json_path) as f:
        experiments = json.load(f)

    output_dir = Path(output_dir)

    if distributed:
        _finalize_distributed(experiments, output_dir)
    else:
        _finalize_unified(experiments, output_dir)


def _finalize_distributed(experiments: dict, root_catalog_dir: Path) -> None:
    """Finalize distributed catalog structure."""
    root_links = [
        {"rel": "root", "href": "./catalog.json", "type": "application/json"},
        {"rel": "self", "href": "./catalog.json", "type": "application/json"},
    ]

    for exp_name, exp_data in experiments.items():
        exp_base = get_exp_base_path(exp_data)
        if not exp_base:
            logger.warning(f"Skipping {exp_name}: no files found")
            continue

        catalog_dir = exp_base / "catalog"
        exp_meta = exp_data.get("experiment_meta", {})

        # Relative path from root catalog.json to experiment catalog.json
        try:
            rel_path = (catalog_dir / "catalog.json").relative_to(root_catalog_dir)
            root_links.append({
                "rel": "child",
                "href": f"./{rel_path}",
                "type": "application/json",
                "title": exp_name,
            })
        except ValueError:
            # Absolute path if not relative
            root_links.append({
                "rel": "child",
                "href": str(catalog_dir / "catalog.json"),
                "type": "application/json",
                "title": exp_name,
            })

        # Build experiment catalog
        exp_child_links = []
        for model_name, model_files in exp_data.get("files", {}).items():
            if not model_files:
                continue

            model_dir = catalog_dir / model_name

            # Find all item JSONs in this model directory
            item_files = sorted(model_dir.glob("*.json")) if model_dir.exists() else []
            if not item_files:
                logger.warning(f"No items found for {exp_name}/{model_name}")
                continue

            # Load items to build collection metadata
            items = []
            for item_file in item_files:
                if item_file.name == "collection.json":
                    continue
                try:
                    with open(item_file) as f:
                        items.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Could not read {item_file}: {e}")

            if not items:
                continue

            # Update item links to point to collection
            for item_file in item_files:
                if item_file.name == "collection.json":
                    continue
                try:
                    with open(item_file) as f:
                        item_data = json.load(f)
                    item_data["links"] = [
                        {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                        {"rel": "self", "href": f"./{item_file.name}", "type": "application/geo+json"},
                        {"rel": "parent", "href": "./collection.json", "type": "application/json"},
                        {"rel": "collection", "href": "./collection.json", "type": "application/json"},
                    ]
                    write_json(item_file, item_data)
                except Exception as e:
                    logger.warning(f"Could not update {item_file}: {e}")

            # Build collection
            variables = sorted({
                item.get("properties", {}).get("variable", "")
                for item in items
                if item.get("properties", {}).get("variable")
            })

            datetimes = [
                item.get("properties", {}).get("datetime")
                for item in items
                if item.get("properties", {}).get("datetime")
            ]
            temporal_extent = [min(datetimes), max(datetimes)] if datetimes else [None, None]

            collection = {
                "type": "Collection",
                "id": make_short_id(f"{exp_name}-{model_name}"),
                "title": model_name,
                "stac_version": "1.0.0",
                "license": "proprietary",
                "description": f"{model_name} output for experiment {exp_name}",
                "stac_extensions": [DATACUBE_EXTENSION],
                "extent": {
                    "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
                    "temporal": {"interval": [temporal_extent]},
                },
                "cube:dimensions": {
                    "time": {"type": "temporal", "extent": temporal_extent},
                    "longitude": {"type": "spatial", "axis": "x", "extent": [-180.0, 180.0], "reference_system": "EPSG:4326"},
                    "latitude": {"type": "spatial", "axis": "y", "extent": [-90.0, 90.0], "reference_system": "EPSG:4326"},
                },
                "cube:variables": {
                    var: {"type": "data", "dimensions": ["time", "latitude", "longitude"]}
                    for var in variables
                },
                "links": [
                    {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                    {"rel": "self", "href": "./collection.json", "type": "application/json"},
                    {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
                ] + [
                    {"rel": "item", "href": f"./{Path(item['id']).name}.json", "type": "application/geo+json"}
                    for item in items
                ],
            }
            write_json(model_dir / "collection.json", collection)

            exp_child_links.append({
                "rel": "child",
                "href": f"./{model_name}/collection.json",
                "type": "application/json",
                "title": model_name,
            })

        # Write experiment catalog
        exp_catalog = {
            "type": "Catalog",
            "id": make_short_id(exp_name),
            "title": exp_name,
            "stac_version": "1.0.0",
            "description": (
                f"Experiment {exp_name} | "
                f"Setup: {exp_meta.get('setup_name', 'unknown')} | "
                f"Scenario: {exp_meta.get('scenario', 'unknown')}"
            ),
            "links": [
                {"rel": "root", "href": "../catalog.json", "type": "application/json"},
                {"rel": "self", "href": "./catalog.json", "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
            ] + exp_child_links,
        }
        write_json(catalog_dir / "catalog.json", exp_catalog)
        logger.info(f"Finalized: {exp_name}")

    # Write root catalog
    root_catalog = {
        "type": "Catalog",
        "id": make_short_id("esm-stac-catalog"),
        "title": "ESM Tools Experiments",
        "stac_version": "1.0.0",
        "description": "STAC catalog of ESM Tools experiment output",
        "links": root_links,
    }
    write_json(root_catalog_dir / "catalog.json", root_catalog)
    logger.info(f"Root catalog: {root_catalog_dir / 'catalog.json'}")


def _finalize_unified(experiments: dict, output_dir: Path) -> None:
    """Finalize unified catalog structure."""
    root_links = [
        {"rel": "root", "href": "./catalog.json", "type": "application/json"},
        {"rel": "self", "href": "./catalog.json", "type": "application/json"},
    ]

    for exp_name, exp_data in experiments.items():
        exp_dir = output_dir / exp_name
        exp_meta = exp_data.get("experiment_meta", {})

        root_links.append({
            "rel": "child",
            "href": f"./{exp_name}/catalog.json",
            "type": "application/json",
            "title": exp_name,
        })

        exp_child_links = []
        for model_name, model_files in exp_data.get("files", {}).items():
            if not model_files:
                continue

            model_dir = exp_dir / model_name
            item_files = sorted(model_dir.glob("*.json")) if model_dir.exists() else []

            items = []
            for item_file in item_files:
                if item_file.name == "collection.json":
                    continue
                try:
                    with open(item_file) as f:
                        items.append(json.load(f))
                except Exception:
                    pass

            if not items:
                continue

            # Update item links
            for item_file in item_files:
                if item_file.name == "collection.json":
                    continue
                try:
                    with open(item_file) as f:
                        item_data = json.load(f)
                    item_data["links"] = [
                        {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                        {"rel": "self", "href": f"./{item_file.name}", "type": "application/geo+json"},
                        {"rel": "parent", "href": "./collection.json", "type": "application/json"},
                        {"rel": "collection", "href": "./collection.json", "type": "application/json"},
                    ]
                    write_json(item_file, item_data)
                except Exception:
                    pass

            variables = sorted({
                item.get("properties", {}).get("variable", "")
                for item in items
                if item.get("properties", {}).get("variable")
            })

            datetimes = [
                item.get("properties", {}).get("datetime")
                for item in items
                if item.get("properties", {}).get("datetime")
            ]
            temporal_extent = [min(datetimes), max(datetimes)] if datetimes else [None, None]

            collection = {
                "type": "Collection",
                "id": make_short_id(f"{exp_name}-{model_name}"),
                "title": model_name,
                "stac_version": "1.0.0",
                "license": "proprietary",
                "description": f"{model_name} output for experiment {exp_name}",
                "stac_extensions": [DATACUBE_EXTENSION],
                "extent": {
                    "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
                    "temporal": {"interval": [temporal_extent]},
                },
                "cube:dimensions": {
                    "time": {"type": "temporal", "extent": temporal_extent},
                    "longitude": {"type": "spatial", "axis": "x", "extent": [-180.0, 180.0], "reference_system": "EPSG:4326"},
                    "latitude": {"type": "spatial", "axis": "y", "extent": [-90.0, 90.0], "reference_system": "EPSG:4326"},
                },
                "cube:variables": {
                    var: {"type": "data", "dimensions": ["time", "latitude", "longitude"]}
                    for var in variables
                },
                "links": [
                    {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                    {"rel": "self", "href": "./collection.json", "type": "application/json"},
                    {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
                ] + [
                    {"rel": "item", "href": f"./{Path(item['id']).name}.json", "type": "application/geo+json"}
                    for item in items
                ],
            }
            write_json(model_dir / "collection.json", collection)

            exp_child_links.append({
                "rel": "child",
                "href": f"./{model_name}/collection.json",
                "type": "application/json",
                "title": model_name,
            })

        exp_catalog = {
            "type": "Catalog",
            "id": make_short_id(exp_name),
            "title": exp_name,
            "stac_version": "1.0.0",
            "description": (
                f"Experiment {exp_name} | "
                f"Setup: {exp_meta.get('setup_name', 'unknown')} | "
                f"Scenario: {exp_meta.get('scenario', 'unknown')}"
            ),
            "links": [
                {"rel": "root", "href": "../catalog.json", "type": "application/json"},
                {"rel": "self", "href": "./catalog.json", "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
            ] + exp_child_links,
        }
        write_json(exp_dir / "catalog.json", exp_catalog)
        logger.info(f"Finalized: {exp_name}")

    root_catalog = {
        "type": "Catalog",
        "id": make_short_id("esm-stac-catalog"),
        "title": "ESM Tools Experiments",
        "stac_version": "1.0.0",
        "description": "STAC catalog of ESM Tools experiment output",
        "links": root_links,
    }
    write_json(output_dir / "catalog.json", root_catalog)
    logger.info(f"Root catalog: {output_dir / 'catalog.json'}")
