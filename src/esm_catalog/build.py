"""Build STAC catalog from experiment metadata."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import xarray as xr
from loguru import logger

GLOBAL_BBOX = [-180.0, -90.0, 180.0, 90.0]
GLOBAL_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[
        [-180.0, -90.0],
        [180.0, -90.0],
        [180.0, 90.0],
        [-180.0, 90.0],
        [-180.0, -90.0],
    ]],
}

STAC_EXTENSIONS = [
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    "https://stac-extensions.github.io/cf/v0.2.0/schema.json",
    "https://stac-extensions.github.io/file/v1.0.0/schema.json",
    "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
]

DATACUBE_EXTENSION = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"

# UUID namespace for deterministic IDs
STAC_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_short_id(name: str) -> str:
    """Generate deterministic short ID: first 8 hex chars of UUID5 + name."""
    uid = uuid.uuid5(STAC_UUID_NAMESPACE, name)
    return f"{uid.hex[:8]}-{name}"


def extract_variable_name(filename: str) -> str:
    """Extract variable name from filename (e.g., ssh.fesom.185001.01.nc -> ssh)."""
    stem = Path(filename).stem
    parts = stem.split(".")
    return parts[0] if parts else "unknown"


def extract_cf_parameters(ds: xr.Dataset) -> list[dict[str, str]]:
    """Extract CF parameters from xarray Dataset variables."""
    cf_parameters = []
    for var_name, da in ds.data_vars.items():
        standard_name = (
            da.attrs.get("standard_name")
            or da.attrs.get("description")
            or da.attrs.get("long_name")
        )
        if not standard_name:
            continue
        param = {"name": standard_name, "variable": var_name}
        unit = da.attrs.get("units")
        if unit:
            param["unit"] = unit
        cf_parameters.append(param)
    return cf_parameters


def parse_datetime_from_filename(filename: str) -> datetime | None:
    """Extract datetime from filename with YYYYMM pattern."""
    match = re.search(r"\.(\d{6})\.\d{2}\.nc$", filename)
    if match:
        yyyymm = match.group(1)
        try:
            return datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1, tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def parse_iso_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not dt_str:
        return None
    return datetime.fromisoformat(str(dt_str)).replace(tzinfo=timezone.utc)


def build_item(
    filepath: str | Path,
    exp_meta: dict[str, Any],
    model_name: str,
    meta: dict[str, Any],
    exp_name: str | None = None,
) -> pystac.Item:
    """Build a STAC Item from a NetCDF file.

    Args:
        filepath: Path to NetCDF file
        exp_meta: Experiment-level metadata
        model_name: Name of model component (e.g., "fesom")
        meta: Model-level metadata
        exp_name: Experiment name (defaults to exp_meta['expid'])

    Returns:
        pystac.Item with datacube extension and CF parameters
    """
    p = Path(filepath)
    md = meta.get("metadata", {}) or {}

    if exp_name is None:
        exp_name = exp_meta.get("expid", p.parts[-1])

    initial_dt = parse_iso_datetime(exp_meta.get("initial_date"))
    item_dt = parse_datetime_from_filename(p.name) or initial_dt or datetime.now(timezone.utc)

    # Extract CF parameters from NetCDF
    cf_parameters = []
    nc_conventions = "CF-UGRID"
    fesom_attrs = {}

    try:
        ds = xr.open_dataset(filepath, decode_times=True)
        cf_parameters = extract_cf_parameters(ds)
        nc_conventions = ds.attrs.get("Conventions", "CF-UGRID")
        fesom_attrs = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in ds.attrs.items()
            if k.startswith("FESOM_")
        }
        ds.close()
    except Exception as e:
        logger.debug(f"Could not read NetCDF {filepath}: {e}")

    try:
        file_size = p.stat().st_size
    except Exception:
        file_size = None

    variable_name = extract_variable_name(p.name)

    # Extract YYYYMM for title
    m = re.search(r"\.(\d{6})\.\d{2}\.nc$", p.name)
    yyyymm_str = m.group(1) if m else ""
    item_title = f"{variable_name}_{model_name}_{yyyymm_str}"

    properties = {
        "model": model_name,
        "model_type": meta.get("type"),
        "grid": "unstructured-mesh",
        "conventions": nc_conventions,
        "institution": md.get("Institute", "Alfred Wegener Institute"),
        "source_type": "OGCM",
        "frequency": "monthly",
        "variable": variable_name,
        "experiment": exp_name,
        "scenario": exp_meta.get("scenario"),
        "resolution": exp_meta.get("resolution"),
        "setup": exp_meta.get("setup_name"),
    }
    properties.update(fesom_attrs)

    if cf_parameters:
        properties["cf:parameter"] = cf_parameters
    if md.get("Publications"):
        properties["sci:citation"] = md["Publications"]

    item = pystac.Item(
        id=make_short_id(p.stem),
        geometry=GLOBAL_GEOMETRY,
        bbox=GLOBAL_BBOX,
        datetime=item_dt,
        properties=properties,
        stac_extensions=STAC_EXTENSIONS,
    )
    item.common_metadata.title = item_title

    # Datacube extension
    dt_iso = item_dt.isoformat()
    item.properties["cube:dimensions"] = {
        "time": {"type": "temporal", "extent": [dt_iso, dt_iso]},
        "longitude": {
            "type": "spatial",
            "axis": "x",
            "extent": [-180.0, 180.0],
            "reference_system": "EPSG:4326",
        },
        "latitude": {
            "type": "spatial",
            "axis": "y",
            "extent": [-90.0, 90.0],
            "reference_system": "EPSG:4326",
        },
    }
    item.properties["cube:variables"] = {
        variable_name: {
            "type": "data",
            "dimensions": ["time", "latitude", "longitude"],
        }
    }

    # Asset
    asset_extra = {
        "xarray:open_kwargs": {"engine": "netcdf4", "decode_times": True},
        "xarray:storage_options": {},
        "netcdf:format": "netcdf4",
        "netcdf:convention": nc_conventions,
    }
    if file_size is not None:
        asset_extra["file:size"] = file_size
    if cf_parameters:
        asset_extra["cf:parameter"] = cf_parameters

    item.add_asset(
        "data",
        pystac.Asset(
            href=str(filepath),
            media_type="application/x-netcdf",
            roles=["data"],
            title=p.name,
            extra_fields=asset_extra,
        ),
    )

    return item


def write_json(path: Path, data: dict) -> None:
    """Write JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_catalog(
    root_catalog: pystac.Catalog,
    collections_map: dict[str, list[pystac.Item]],
    output_dir: str | Path,
) -> None:
    """Save catalog with relative hrefs for portability.

    Structure:
        {output_dir}/catalog.json
        {output_dir}/{exp}/catalog.json
        {output_dir}/{exp}/{model}/collection.json
        {output_dir}/{exp}/{model}/{item_id}.json
    """
    output_dir = Path(output_dir)

    # Root catalog
    root_dict = {
        "type": "Catalog",
        "id": root_catalog.id,
        "title": root_catalog.title or root_catalog.id,
        "stac_version": "1.0.0",
        "description": root_catalog.description,
        "links": [
            {"rel": "root", "href": "./catalog.json", "type": "application/json"},
            {"rel": "self", "href": "./catalog.json", "type": "application/json"},
        ] + [
            {
                "rel": "child",
                "href": f"./{exp_cat.title}/catalog.json",
                "type": "application/json",
                "title": exp_cat.title or exp_cat.id,
            }
            for exp_cat in root_catalog.get_children()
        ],
    }
    write_json(output_dir / "catalog.json", root_dict)

    for exp_cat in root_catalog.get_children():
        exp_dir = output_dir / exp_cat.title
        child_collections = list(exp_cat.get_children())

        # Experiment catalog
        exp_dict = {
            "type": "Catalog",
            "id": exp_cat.id,
            "title": exp_cat.title or exp_cat.id,
            "stac_version": "1.0.0",
            "description": exp_cat.description,
            "links": [
                {"rel": "root", "href": "../catalog.json", "type": "application/json"},
                {"rel": "self", "href": "./catalog.json", "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
            ] + [
                {
                    "rel": "child",
                    "href": f"./{col.extra_fields.get('model', col.id)}/collection.json",
                    "type": "application/json",
                    "title": col.title or col.id,
                }
                for col in child_collections
            ],
        }
        write_json(exp_dir / "catalog.json", exp_dict)

        for col in child_collections:
            model_slug = col.extra_fields.get("model", col.id)
            col_dir = exp_dir / model_slug
            items = collections_map[col.id]

            # Items
            item_links = []
            for item in items:
                item_filename = f"{item.id}.json"
                item_dict = item.to_dict()
                item_dict["links"] = [
                    {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                    {"rel": "self", "href": f"./{item_filename}", "type": "application/geo+json"},
                    {"rel": "parent", "href": "./collection.json", "type": "application/json"},
                    {"rel": "collection", "href": "./collection.json", "type": "application/json"},
                ]

                # Ensure datacube extension
                item_exts = item_dict.get("stac_extensions", [])
                if DATACUBE_EXTENSION not in item_exts:
                    item_dict["stac_extensions"] = item_exts + [DATACUBE_EXTENSION]

                write_json(col_dir / item_filename, item_dict)
                item_links.append({
                    "rel": "item",
                    "href": f"./{item_filename}",
                    "type": "application/geo+json",
                })

            # Collection
            col_dict = col.to_dict()
            if "title" not in col_dict:
                col_dict["title"] = col.title or model_slug

            col_exts = col_dict.get("stac_extensions", [])
            if DATACUBE_EXTENSION not in col_exts:
                col_dict["stac_extensions"] = col_exts + [DATACUBE_EXTENSION]

            temporal_interval = (
                col_dict.get("extent", {})
                .get("temporal", {})
                .get("interval", [[None, None]])[0]
            )
            col_dict["cube:dimensions"] = {
                "time": {
                    "type": "temporal",
                    "extent": [temporal_interval[0], temporal_interval[1]],
                },
                "longitude": {
                    "type": "spatial",
                    "axis": "x",
                    "extent": [-180.0, 180.0],
                    "reference_system": "EPSG:4326",
                },
                "latitude": {
                    "type": "spatial",
                    "axis": "y",
                    "extent": [-90.0, 90.0],
                    "reference_system": "EPSG:4326",
                },
            }

            col_variable_names = sorted({
                item.properties.get("variable", "")
                for item in items
                if item.properties.get("variable")
            })
            col_dict["cube:variables"] = {
                var: {"type": "data", "dimensions": ["time", "latitude", "longitude"]}
                for var in col_variable_names
            }

            col_dict["links"] = [
                {"rel": "root", "href": "../../catalog.json", "type": "application/json"},
                {"rel": "self", "href": "./collection.json", "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json", "type": "application/json"},
            ] + item_links

            write_json(col_dir / "collection.json", col_dict)


def build_catalog(experiments_json_path: str | Path, output_dir: str | Path) -> pystac.Catalog:
    """Build complete STAC catalog from experiments.json.

    Args:
        experiments_json_path: Path to experiments.json
        output_dir: Output directory for catalog

    Returns:
        Root pystac.Catalog
    """
    with open(experiments_json_path) as f:
        experiments = json.load(f)

    root_catalog = pystac.Catalog(
        id=make_short_id("esm-stac-catalog"),
        title="ESM Tools Experiments",
        description="STAC catalog of ESM Tools experiment output",
    )

    collections_map: dict[str, list[pystac.Item]] = {}

    for exp_name, exp_data in experiments.items():
        exp_meta = exp_data.get("experiment_meta", {})
        model_meta = exp_data.get("model_meta", {})
        files = exp_data.get("files", {})

        initial_dt = parse_iso_datetime(exp_meta.get("initial_date"))
        final_dt = parse_iso_datetime(exp_meta.get("final_date"))

        exp_catalog = pystac.Catalog(
            id=make_short_id(exp_name),
            title=exp_name,
            description=(
                f"Experiment {exp_name} | "
                f"Setup: {exp_meta.get('setup_name', 'unknown')} | "
                f"Scenario: {exp_meta.get('scenario', 'unknown')} | "
                f"Resolution: {exp_meta.get('resolution', 'unknown')}"
            ),
        )
        root_catalog.add_child(exp_catalog)

        for model_name, model_files in files.items():
            meta = model_meta.get(model_name, {})
            md = meta.get("metadata", {}) or {}

            extent = pystac.Extent(
                spatial=pystac.SpatialExtent(bboxes=[GLOBAL_BBOX]),
                temporal=pystac.TemporalExtent(intervals=[[initial_dt, final_dt]]),
            )

            variable_names = sorted({
                Path(fp).stem.split(".")[0] for fp in model_files
            })
            keywords = [exp_name, model_name] + variable_names

            collection = pystac.Collection(
                id=make_short_id(f"{exp_name}-{model_name}"),
                title=model_name,
                description=(
                    md.get("Description")
                    or f"{model_name} output for experiment {exp_name}"
                ),
                extent=extent,
                extra_fields={
                    "model": model_name,
                    "model_type": meta.get("type"),
                    "version": str(meta.get("version")) if meta.get("version") else None,
                    "resolution": meta.get("resolution"),
                    "scenario": meta.get("scenario"),
                    "institute": md.get("Institute"),
                    "authors": md.get("Authors"),
                    "license": md.get("License"),
                    "publications": md.get("Publications"),
                    "keywords": keywords,
                },
            )
            exp_catalog.add_child(collection)

            items = [
                build_item(fp, exp_meta, model_name, meta, exp_name=exp_name)
                for fp in model_files
            ]
            collections_map[collection.id] = items

            logger.debug(f"  {model_name}: {len(items)} items")

        logger.info(f"Built experiment: {exp_name}")

    save_catalog(root_catalog, collections_map, output_dir)
    return root_catalog
