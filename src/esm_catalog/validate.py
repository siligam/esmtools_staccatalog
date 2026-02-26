"""Validate STAC catalog structure and required fields."""

import json
from pathlib import Path
from typing import Any

from loguru import logger

CATALOG_REQUIRED = {"type", "id", "stac_version", "description", "links"}
COLLECTION_REQUIRED = CATALOG_REQUIRED | {"extent", "license"}
ITEM_REQUIRED = {"type", "stac_version", "id", "geometry", "bbox", "properties", "links", "assets"}


def check_fields(
    obj_id: str,
    data: dict[str, Any],
    required: set[str],
    errors: list[tuple[str, str]],
) -> None:
    """Check that all required fields are present."""
    missing = required - data.keys()
    if missing:
        errors.append((obj_id, f"Missing required fields: {sorted(missing)}"))


def validate_item(data: dict[str, Any], errors: list[tuple[str, str]]) -> None:
    """Validate a STAC Item."""
    obj_id = data.get("id", "<unknown>")
    check_fields(obj_id, data, ITEM_REQUIRED, errors)

    props = data.get("properties", {})
    if "datetime" not in props and not (
        "start_datetime" in props and "end_datetime" in props
    ):
        errors.append((
            obj_id,
            "Missing 'datetime' (or 'start_datetime'+'end_datetime') in properties",
        ))

    if data.get("assets") == {}:
        errors.append((obj_id, "assets is empty"))


def validate_catalog(catalog_dir: str | Path) -> list[tuple[str, str]]:
    """Validate all JSON files in a STAC catalog.

    Checks required fields per STAC 1.0 spec:
        Catalog:    type, id, stac_version, description, links
        Collection: above + extent (spatial + temporal), license
        Item:       type, stac_version, id, geometry, bbox, properties.datetime, links, assets

    Args:
        catalog_dir: Path to catalog directory

    Returns:
        List of (object_id, error_message) tuples
    """
    catalog_dir = Path(catalog_dir)
    errors: list[tuple[str, str]] = []
    counts = {"Catalog": 0, "Collection": 0, "Item": 0}

    for json_file in sorted(catalog_dir.rglob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        stac_type = data.get("type")
        rel_path = json_file.relative_to(catalog_dir)

        if stac_type == "FeatureCollection":
            for feature in data.get("features", []):
                validate_item(feature, errors)
                counts["Item"] += 1

        elif stac_type == "Catalog":
            check_fields(data.get("id"), data, CATALOG_REQUIRED, errors)
            counts["Catalog"] += 1

        elif stac_type == "Collection":
            check_fields(data.get("id"), data, COLLECTION_REQUIRED, errors)
            extent = data.get("extent", {})
            if "spatial" not in extent:
                errors.append((data.get("id"), "extent missing 'spatial'"))
            if "temporal" not in extent:
                errors.append((data.get("id"), "extent missing 'temporal'"))
            counts["Collection"] += 1

        elif stac_type == "Feature":
            validate_item(data, errors)
            counts["Item"] += 1

        else:
            errors.append((str(rel_path), f"Unknown or missing 'type': {stac_type!r}"))

    logger.info(
        f"Validated: {counts['Catalog']} catalogs, "
        f"{counts['Collection']} collections, "
        f"{counts['Item']} items"
    )

    if errors:
        logger.error(f"{len(errors)} validation error(s):")
        for obj_id, msg in errors:
            logger.error(f"  [{obj_id}] {msg}")
    else:
        logger.info("All objects valid.")

    return errors
