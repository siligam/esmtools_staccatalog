import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pystac
import xarray as xr


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

STAC_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_short_id(name):
    """Return a short, deterministic ID: first 8 hex chars of UUID5 + '-' + name.

    This gives a unique yet human-readable ID that is safe to use in URL
    endpoints without excessive typing.
    """
    uid = uuid.uuid5(STAC_UUID_NAMESPACE, name)
    prefix = uid.hex[:8]
    return f"{prefix}-{name}"


def make_item_title(variable_name, model_name, yyyymm):
    """Construct a meaningful human-readable title for an item."""
    return f"{variable_name}_{model_name}_{yyyymm}"


def extract_variable_name(filename):
    """Extract variable name from FESOM filename (e.g. ssh.fesom.185001.01.nc -> ssh)."""
    stem = Path(filename).stem
    parts = stem.split(".")
    return parts[0] if parts else "unknown"


def extract_cf_parameters(ds):
    """Return list of CF parameter dicts from xarray Dataset variables."""
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


def parse_datetime_from_filename(filename, fallback):
    """Extract datetime from filenames like MLD1.fesom.185001.01.nc (YYYYMM pattern)."""
    match = re.search(r'\.(\d{6})\.\d{2}\.nc$', filename)
    if match:
        yyyymm = match.group(1)
        try:
            return datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1, tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback


def parse_iso_datetime(dt_str):
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)


def build_item(filepath, exp_meta, model_name, meta, exp_name=None):
    """Build a single STAC Item from one NetCDF file.

    This is the primary entry point for Snakemake rules that process one file
    at a time. The returned item is not yet attached to any collection.

    Args:
        filepath: Path-like or str pointing to the NetCDF file.
        exp_meta: Dict of experiment-level metadata (from experiments.json
                  ``experiment_meta`` block, or equivalent).
        model_name: Name of the model component (e.g. ``"fesom"``).
        meta: Dict of model-level metadata (from experiments.json
              ``model_meta.<model_name>`` block, or equivalent).
        exp_name: Experiment name (top-level key in experiments.json). Falls
                  back to ``exp_meta['expid']`` if not provided.

    Returns:
        pystac.Item with all extensions and enriched properties.
    """
    p = Path(filepath)
    md = meta.get("metadata", {}) or {}
    if exp_name is None:
        exp_name = exp_meta.get("expid", p.parts[-1])
    initial_dt = parse_iso_datetime(exp_meta.get("initial_date"))
    item_dt = parse_datetime_from_filename(p.name, initial_dt)

    # Open NetCDF to extract CF parameters and dataset attributes
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
    except Exception:
        cf_parameters = []
        nc_conventions = "CF-UGRID"
        fesom_attrs = {}

    try:
        file_size = p.stat().st_size
    except Exception:
        file_size = None

    variable_name = extract_variable_name(p.name)

    # Derive YYYYMM string for title
    m = re.search(r'\.(\d{6})\.\d{2}\.nc$', p.name)
    yyyymm_str = m.group(1) if m else ""
    item_title = make_item_title(variable_name, model_name, yyyymm_str)

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


def build_items(filepaths, exp_meta, model_name, meta, exp_name=None):
    """Build STAC Items for a list of NetCDF files.

    Convenience wrapper around :func:`build_item` for batch processing.
    Use this when all files for a model component are available at once.

    Args:
        filepaths: Iterable of Path-like or str values pointing to NetCDF files.
        exp_meta: Dict of experiment-level metadata.
        model_name: Name of the model component.
        meta: Dict of model-level metadata.
        exp_name: Experiment name (top-level key in experiments.json).

    Returns:
        List of pystac.Item objects, one per file.
    """
    return [build_item(fp, exp_meta, model_name, meta, exp_name=exp_name) for fp in filepaths]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_catalog(root_catalog, collections_map, output_dir):
    """
    Custom save that writes:
      {output_dir}/catalog.json
      {output_dir}/{exp}/catalog.json
      {output_dir}/{exp}/{exp}-{model}/collection.json
      {output_dir}/{exp}/{exp}-{model}/{item_id}.json  <- flat, no per-item subfolders

    All hrefs are relative so the catalog is portable regardless of where it
    is served from.
    """
    output_dir = Path(output_dir)

    # --- root catalog.json ---
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
            {"rel": "child", "href": f"./{exp_cat.title}/catalog.json",
             "type": "application/json", "title": exp_cat.title or exp_cat.id}
            for exp_cat in root_catalog.get_children()
        ],
    }
    write_json(output_dir / "catalog.json", root_dict)

    for exp_cat in root_catalog.get_children():
        exp_dir = output_dir / exp_cat.title
        child_collections = list(exp_cat.get_children())

        # --- experiment catalog.json ---
        exp_dict = {
            "type": "Catalog",
            "id": exp_cat.id,
            "title": exp_cat.title or exp_cat.id,
            "stac_version": "1.0.0",
            "description": exp_cat.description,
            "links": [
                {"rel": "root",   "href": "../catalog.json",  "type": "application/json"},
                {"rel": "self",   "href": "./catalog.json",   "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json",  "type": "application/json"},
            ] + [
                {"rel": "child",
                 "href": f"./{col.extra_fields.get('model', col.id)}/collection.json",
                 "type": "application/json", "title": col.title or col.id}
                for col in child_collections
            ],
        }
        write_json(exp_dir / "catalog.json", exp_dict)

        for col in child_collections:
            model_slug = col.extra_fields.get("model", col.id)
            col_dir = exp_dir / model_slug
            items = collections_map[col.id]

            # --- individual item .json files (flat, no subfolders) ---
            item_links = []
            for item in items:
                item_filename = f"{item.id}.json"
                item_dict = item.to_dict()
                item_dict["links"] = [
                    {"rel": "root",       "href": "../../catalog.json", "type": "application/json"},
                    {"rel": "self",       "href": f"./{item_filename}", "type": "application/geo+json"},
                    {"rel": "parent",     "href": "./collection.json",  "type": "application/json"},
                    {"rel": "collection", "href": "./collection.json",  "type": "application/json"},
                ]
                # Inject datacube extension at item level
                item_exts = item_dict.get("stac_extensions", [])
                if DATACUBE_EXTENSION not in item_exts:
                    item_dict["stac_extensions"] = item_exts + [DATACUBE_EXTENSION]
                item_var = item.properties.get("variable", "")
                item_dt = item_dict["properties"]["datetime"]
                item_dict["properties"]["cube:dimensions"] = {
                    "time": {"type": "temporal", "extent": [item_dt, item_dt]},
                    "longitude": {"type": "spatial", "axis": "x", "extent": [-180.0, 180.0], "reference_system": "EPSG:4326"},
                    "latitude": {"type": "spatial", "axis": "y", "extent": [-90.0, 90.0], "reference_system": "EPSG:4326"},
                }
                if item_var:
                    item_dict["properties"]["cube:variables"] = {
                        item_var: {
                            "type": "data",
                            "dimensions": ["time", "latitude", "longitude"],
                        }
                    }
                write_json(col_dir / item_filename, item_dict)
                item_links.append(
                    {"rel": "item", "href": f"./{item_filename}", "type": "application/geo+json"}
                )

            # --- collection.json with rel:item links to each flat item file ---
            col_dict = col.to_dict()
            if "title" not in col_dict:
                col_dict["title"] = col.title or model_slug
            # Inject datacube extension into each collection
            col_exts = col_dict.get("stac_extensions", [])
            if DATACUBE_EXTENSION not in col_exts:
                col_dict["stac_extensions"] = col_exts + [DATACUBE_EXTENSION]
            temporal_interval = col_dict.get("extent", {}).get("temporal", {}).get("interval", [[None, None]])[0]
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
            # Collect unique variable names across all items in this collection
            col_variable_names = sorted({
                item.properties.get("variable", "")
                for item in items
                if item.properties.get("variable")
            })
            col_dict["cube:variables"] = {
                var: {
                    "type": "data",
                    "dimensions": ["time", "latitude", "longitude"],
                }
                for var in col_variable_names
            }
            col_dict["links"] = [
                {"rel": "root",   "href": "../../catalog.json", "type": "application/json"},
                {"rel": "self",   "href": "./collection.json",  "type": "application/json"},
                {"rel": "parent", "href": "../catalog.json",    "type": "application/json"},
            ] + item_links
            write_json(col_dir / "collection.json", col_dict)


def build_catalog(experiments_json_path, output_dir):
    with open(experiments_json_path) as f:
        experiments = json.load(f)

    root_catalog = pystac.Catalog(
        id=make_short_id("fesom-stac-catalog"),
        title="ESM Tools Experiments",
        description="STAC catalog of AWIESM experiment output",
    )

    # collections_map keeps items per collection id for the custom save
    collections_map = {}

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

            # Collect variable names from the model files for keyword population
            variable_names = sorted({
                Path(fp).stem.split(".")[0]
                for fp in model_files
            })
            keywords = [exp_name, model_name] + variable_names

            collection = pystac.Collection(
                id=make_short_id(f"{exp_name}-{model_name}"),
                title=model_name,
                description=(
                    md.get("Description") or
                    f"{model_name} output for experiment {exp_name}"
                ),
                extent=extent,
                extra_fields={
                    "model":        model_name,
                    "model_type":   meta.get("type"),
                    "version":      str(meta.get("version")) if meta.get("version") else None,
                    "resolution":   meta.get("resolution"),
                    "scenario":     meta.get("scenario"),
                    "institute":    md.get("Institute"),
                    "authors":      md.get("Authors"),
                    "license":      md.get("License"),
                    "publications": md.get("Publications"),
                    "keywords":     keywords,
                },
            )
            exp_catalog.add_child(collection)

            items = build_items(model_files, exp_meta, model_name, meta, exp_name=exp_name)
            collections_map[collection.id] = items

    save_catalog(root_catalog, collections_map, output_dir)
    print(f"Catalog saved to {output_dir}")
    return root_catalog


if __name__ == "__main__":
    build_catalog("experiments.json", "catalog")
