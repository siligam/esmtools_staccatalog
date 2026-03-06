"""
FESOM STAC API
==============
STAC API server built on stac-fastapi, using pystac to load the catalog.

Loads the local ``catalog/`` directory at startup via pystac and exposes
standard STAC API endpoints. Items can be filtered by the custom properties
``experiment``, ``model``, and ``variable`` via dedicated query parameters on
GET /search or body fields on POST /search.

Each item in search results gets an injected ``xarray_snippet`` asset showing
a ready-to-run Python snippet for opening the matched files with xarray.

Run:
    ./start_stac_api.sh [PORT]          # default port 9092
    uvicorn stac_api:app --port 9092    # or directly

Then browse:
    http://localhost:9092          – landing page (load in STAC Browser)
    http://localhost:9092/docs     – Swagger UI
"""

import json as _json
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import attr
import pystac
import uvicorn
from upath import UPath
from fastapi import Query
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from stac_fastapi.api.app import StacApi
from stac_fastapi.api.models import create_request_model
from stac_fastapi.types.config import ApiSettings
from stac_fastapi.types.core import AsyncBaseCoreClient
from stac_fastapi.types.errors import NotFoundError
from stac_fastapi.types.rfc3339 import rfc3339_str_to_datetime, str_to_interval
from stac_fastapi.types.search import BaseSearchGetRequest, BaseSearchPostRequest
from stac_fastapi.extensions.core.aggregation import AggregationExtension
from stac_fastapi.extensions.core.aggregation.client import AsyncBaseAggregationClient
from stac_fastapi.extensions.core.aggregation.types import Aggregation, AggregationCollection
from stac_fastapi.extensions.core.filter import FilterExtension
from stac_fastapi.extensions.core.filter.client import AsyncBaseFiltersClient


# ---------------------------------------------------------------------------
# Catalog directory (relative to this file)
# ---------------------------------------------------------------------------
CATALOG_DIR = Path(__file__).parent / "catalog"


# ---------------------------------------------------------------------------
# xarray / intake snippet helpers
# ---------------------------------------------------------------------------

def _xarray_snippet(
    hrefs: List[str],
    variable: str = "",
    experiment: str = "",
    model: str = "",
    item_url: str = "",
) -> str:
    """Return a fenced Python snippet that opens a single item with xarray directly."""
    code: List[str] = ["import xarray as xr", "from upath import UPath"]
    if experiment:
        code.append(f"# experiment : {experiment!r}")
    if model:
        code.append(f"# model      : {model!r}")
    if variable:
        code.append(f"# variable   : {variable!r}")
    code.append("")

    if item_url and len(hrefs) == 1:
        # Use the direct file path approach which works with netcdf4
        path = hrefs[0]
        code += [
            f'# Load the dataset directly with netcdf4 engine',
            f'ds = xr.open_dataset(UPath(r"{path}"), engine="netcdf4", decode_times=True)',
        ]
        code += [
            "",
            "# --- alternatively with intake-stac ---",
            "# import intake",
            f'# item = intake.open_stac_item("{item_url}")',
            '# ds = item["data"].to_dask()',
        ]
    elif len(hrefs) == 1:
        path = hrefs[0]
        code += [
            f'# Load the dataset directly with netcdf4 engine',
            f'ds = xr.open_dataset(UPath(r"{path}"), engine="netcdf4", decode_times=True)',
        ]
    else:
        code += [
            "# load multiple files by href",
            "paths = [",
        ]
        for h in hrefs:
            code.append(f'    UPath(r"{h}"),')
        code.append("]")
        code.append('ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", decode_times=True)')

    inner = "\n".join(code)
    return f"```python\n{inner}\n```"


def _collection_xarray_snippet(
    items: List[dict],
    api_url: str = "http://localhost:9092",
    collection_id: str = "",
) -> str:
    """Return a fenced Python snippet that opens a collection via pystac-client."""
    variables: List[str] = []
    experiment = ""
    model = ""

    for item in items:
        props = item.get("properties", {})
        if not experiment:
            experiment = props.get("experiment", "")
        if not model:
            model = props.get("model", "")
        var = props.get("variable", "")
        if var and var not in variables:
            variables.append(var)

    if not items:
        return "```python\n# No data files found in this selection.\n```"

    variable_str = variables[0] if len(variables) == 1 else ", ".join(sorted(variables))
    col_id = collection_id or "<collection-id>"

    # Build the filter expression string.
    filter_parts: List[str] = []
    if variables:
        filter_parts.append(f"variable = '{variable_str}'")
    if experiment:
        filter_parts.append(f"experiment = '{experiment}'")
    filter_expr = " AND ".join(filter_parts) if filter_parts else ""

    code: List[str] = [
        "import pystac_client",
        "import xarray as xr", 
        "from upath import UPath",
        "",
        f'# Connect to the STAC API',
        f'catalog = pystac_client.Client.open("{api_url}")',
        "",
        "# Search for items in this collection",
        f"results = catalog.search(collections=[\"{col_id}\"]",
    ]
    
    if filter_expr:
        code.extend([
            f'                           filter="{filter_expr}",',
            "                           filter_lang=\"cql2-text\",",
        ])
    
    code.extend([
        "                           max_items=None,",
        ")",
        "",
        "# Load all matching datasets",
        "paths = [UPath(item.assets[\"data\"].href) for item in results.items()]",
        'ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", decode_times=True)',
    ])

    inner = "\n".join(code)
    return f"```python\n{inner}\n```"


def _inject_snippet(item: dict, base_url: str = "") -> dict:
    """Return a copy of item dict with an xarray_snippet asset added."""
    item = dict(item)
    assets = dict(item.get("assets", {}))
    props = item.get("properties", {})
    hrefs = [a["href"] for a in assets.values() if "href" in a and a["href"] != "inline"]

    # Derive the item's API URL from its self link or from base_url + collection + id.
    item_url = ""
    for lnk in item.get("links", []):
        if lnk.get("rel") == "self":
            item_url = lnk.get("href", "")
            break
    if not item_url and base_url:
        col_id = item.get("collection", "")
        item_id = item.get("id", "")
        if col_id and item_id:
            item_url = f"{base_url}/collections/{col_id}/items/{item_id}"

    assets["xarray_snippet"] = {
        "href": "inline",
        "type": "text/x-python",
        "title": "Open with intake-stac",
        "roles": ["metadata"],
        "description": _xarray_snippet(
            hrefs,
            variable=props.get("variable", ""),
            experiment=props.get("experiment", ""),
            model=props.get("model", ""),
            item_url=item_url,
        ),
    }
    item["assets"] = assets
    return item


# ---------------------------------------------------------------------------
# CQL2 filter parser (equality-only, enough for STAC Browser filter UI)
# ---------------------------------------------------------------------------

def _parse_cql2_text(expr: str) -> Dict[str, Any]:
    """Parse CQL2-TEXT equality and comparison expressions.

    Handles:  variable = 'ssh'
              FESOM_evp_rheol_steps < 120
              experiment = 'basic-001' AND FESOM_evp_rheol_steps >= 100

    Returns {prop: (op, value_str)} for comparisons,
            {prop: ('=', value_str)} for equality.
    """
    result: Dict[str, Any] = {}
    # Match: identifier  op  'quoted' or unquoted-number
    pattern = r"\b(\w+)\s*(<=|>=|<|>|=)\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+?)(?:\s|$|AND|OR))"
    for m in re.finditer(pattern, expr, re.IGNORECASE):
        prop = m.group(1)
        op   = m.group(2)
        value = m.group(3) if m.group(3) is not None else (
                m.group(4) if m.group(4) is not None else
                (m.group(5) or "").strip())
        if prop.upper() in ("AND", "OR", "NOT"):
            continue
        # Lowercase only well-known fields; preserve case for FESOM_ and other props.
        key = prop.lower() if prop.lower() in ("variable", "experiment", "model", "component") else prop
        result[key] = (op, value)
    return result


def _parse_cql2_json(expr: Any) -> Dict[str, Any]:
    """Parse CQL2-JSON equality and comparison expressions → {prop: (op, value_str)}."""
    result: Dict[str, Any] = {}
    try:
        data = _json.loads(expr) if isinstance(expr, str) else expr
    except Exception:
        return result

    _cql2_op_map = {
        "eq": "=", "=": "=",
        "neq": "!=", "!=": "!=", "<>": "!=",
        "lt": "<", "<": "<",
        "lte": "<=", "<=": "<=",
        "gt": ">", ">": ">",
        "gte": ">=", ">=": ">="
    }

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        op = str(node.get("op", "")).lower()
        args = node.get("args", [])
        if op in _cql2_op_map and len(args) == 2:
            prop_node, val_node = args
            if isinstance(prop_node, dict) and "property" in prop_node:
                prop = prop_node["property"]
                key = prop.lower() if prop.lower() in ("variable", "experiment", "model", "component") else prop
                result[key] = (_cql2_op_map[op], str(val_node))
        elif op in ("and", "or") and isinstance(args, list):
            for arg in args:
                _walk(arg)

    _walk(data)
    return result


def _cql_val(cql: Dict[str, Any], key: str) -> Optional[str]:
    """Extract a plain string value from a CQL2 parsed dict for a given key.

    The parsers now return (op, value) tuples. For well-known equality fields
    (variable, experiment, model) we only want the value part.
    """
    v = cql.get(key)
    if v is None:
        return None
    if isinstance(v, tuple):
        return v[1]  # (op, value) → value
    return str(v)


def _parse_cql2(expr: Optional[str], lang: str = "cql2-text") -> Dict[str, str]:
    """Dispatch to the correct CQL2 parser."""
    if not expr:
        return {}
    if "json" in lang.lower():
        return _parse_cql2_json(expr)
    return _parse_cql2_text(expr)


# ---------------------------------------------------------------------------
# Queryables JSON Schema (OGC API - Features Part 3 / STAC Filter Extension)
# ---------------------------------------------------------------------------

_OGC_QUERYABLES_REL = "http://www.opengis.net/def/rel/ogc/1.0/queryables"

def _prop_json_type(value: Any) -> str:
    """Map a Python value to a JSON Schema type string."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _make_queryables(
    base_url: str,
    collection_id: str = "",
    variables: Optional[List[str]] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
    unique_values: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Build the OGC queryables JSON Schema document.

    extra_properties maps property name → a representative sample value used
    to infer the JSON Schema type.  Every key becomes a queryable field.
    unique_values maps property name → list of unique values for enum dropdowns.
    """
    q_id = (
        f"{base_url}/collections/{collection_id}/queryables"
        if collection_id
        else f"{base_url}/queryables"
    )
    title = f"Queryables – {collection_id}" if collection_id else "FESOM STAC Queryables"

    unique_vals = unique_values or {}

    variable_schema: Dict[str, Any] = {
        "title": "Variable",
        "description": "FESOM output variable (e.g. ssh, sst, MLD1)",
        "type": "string",
    }
    if variables:
        variable_schema["enum"] = sorted(variables)
    elif "variable" in unique_vals:
        variable_schema["enum"] = unique_vals["variable"]

    experiment_schema: Dict[str, Any] = {
        "title": "Experiment",
        "description": "Experiment name (e.g. basic-001, garfield-001)",
        "type": "string",
    }
    if "experiment" in unique_vals:
        experiment_schema["enum"] = unique_vals["experiment"]

    model_schema: Dict[str, Any] = {
        "title": "Model",
        "description": "Model component (e.g. fesom)",
        "type": "string",
    }
    if "model" in unique_vals:
        model_schema["enum"] = unique_vals["model"]

    component_schema: Dict[str, Any] = {
        "title": "Component",
        "description": "Model component name (e.g. fesom)",
        "type": "string",
    }
    if "component" in unique_vals:
        component_schema["enum"] = unique_vals["component"]

    # Fixed well-known queryables.
    props: Dict[str, Any] = {
        "id": {"title": "Item ID", "type": "string"},
        "collection": {"title": "Collection", "type": "string"},
        "datetime": {
            "title": "Date / Time",
            "type": "string",
            "format": "date-time",
        },
        "variable": variable_schema,
        "experiment": experiment_schema,
        "model": model_schema,
        "component": component_schema,
    }

    # Dynamically add every other item property.
    _skip = {"datetime", "start_datetime", "end_datetime", "created", "updated",
             "cf:parameter", "cube:dimensions", "cube:variables"}
    for key, sample_val in sorted((extra_properties or {}).items()):
        if key in props or key in _skip:
            continue
        prop_schema: Dict[str, Any] = {
            "title": key,
            "type": _prop_json_type(sample_val),
        }
        # Add enum if we have unique values for this property
        if key in unique_vals and _prop_json_type(sample_val) == "string":
            prop_schema["enum"] = unique_vals[key]
        props[key] = prop_schema

    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$id": q_id,
        "type": "object",
        "title": title,
        "properties": props,
    }


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

class CatalogLoader:
    """Load and index the fesom_stac2 catalog using pystac.

    Structure: Root Catalog → Experiment Catalogs → Model Collections → Items
    """

    def __init__(self, catalog_dir: Path):
        self.catalog_dir = catalog_dir
        self.root: Optional[pystac.Catalog] = None
        # {exp_id: {model_id: pystac.Collection}}
        self.model_collections: Dict[str, Dict[str, pystac.Collection]] = {}

    def load(self) -> None:
        root_path = self.catalog_dir / "catalog.json"
        self.root = pystac.Catalog.from_file(str(root_path))
        for exp_cat in self.root.get_children():
            if not isinstance(exp_cat, pystac.Catalog):
                continue
            # Key by the human-readable title (e.g. "basic-001"), not the
            # UUID-prefixed id, so experiment filter matches item properties.
            exp_key = exp_cat.title or exp_cat.id
            self.model_collections[exp_key] = {}
            for child in exp_cat.get_children():
                if isinstance(child, pystac.Collection):
                    self.model_collections[exp_key][child.id] = child
        n_cols = sum(len(v) for v in self.model_collections.values())
        print(f"Loaded catalog from {self.catalog_dir}: "
              f"{len(self.model_collections)} experiments, {n_cols} collections")

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def _collection_keywords(self, exp_id: str, col: pystac.Collection) -> List[str]:
        """Build high-level keyword list (not individual variables)."""
        model_name = col.extra_fields.get("model", "")
        model_type = col.extra_fields.get("model_type", "")
        scenario = col.extra_fields.get("scenario", "")
        resolution = col.extra_fields.get("resolution", "")
        
        keywords = [exp_id]
        for kw in [model_name, model_type, scenario, resolution]:
            if kw and kw not in keywords:
                keywords.append(kw)
        return keywords

    def get_all_collections(self) -> List[Dict[str, Any]]:
        result = []
        for exp_id, models in self.model_collections.items():
            for model_id, col in models.items():
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                if not d.get("keywords"):
                    d["keywords"] = self._collection_keywords(exp_id, col)
                result.append(d)
        return result

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        for exp_id, models in self.model_collections.items():
            if collection_id in models:
                col = models[collection_id]
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                if not d.get("keywords"):
                    d["keywords"] = self._collection_keywords(exp_id, col)
                return d
        return None

    def _find_collection(self, collection_id: str) -> Optional[Tuple[str, pystac.Collection]]:
        for exp_id, models in self.model_collections.items():
            if collection_id in models:
                return exp_id, models[collection_id]
        return None

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def get_all_property_keys(self, collection_id: str = "") -> Dict[str, Any]:
        """Return a dict of property_name -> sample_value across all items.

        Used to build the dynamic queryables schema.  Only the first non-None
        value seen for each key is kept as the representative sample.
        """
        sample: Dict[str, Any] = {}
        for exp_id, models in self.model_collections.items():
            for model_id, col in models.items():
                if collection_id and model_id != collection_id:
                    continue
                for item in col.get_all_items():
                    for k, v in (item.properties or {}).items():
                        if k not in sample and v is not None and not isinstance(v, (list, dict)):
                            sample[k] = v
        return sample

    def get_unique_property_values(self, collection_id: str = "") -> Dict[str, List[str]]:
        """Return a dict of property_name -> list of unique values across all items.

        Used to populate enum arrays in queryables for dropdown menus.
        Only collects values for string-type properties.
        """
        values: Dict[str, set] = {}
        for exp_id, models in self.model_collections.items():
            for model_id, col in models.items():
                if collection_id and model_id != collection_id:
                    continue
                # Add experiment and model from collection metadata
                if "experiment" not in values:
                    values["experiment"] = set()
                values["experiment"].add(exp_id)
                
                component = col.extra_fields.get("model", "")
                if component:
                    if "model" not in values:
                        values["model"] = set()
                    if "component" not in values:
                        values["component"] = set()
                    values["model"].add(component)
                    values["component"].add(component)
                
                for item in col.get_all_items():
                    for k, v in (item.properties or {}).items():
                        if v is None or isinstance(v, (list, dict, bool)):
                            continue
                        # Only collect string values for enum
                        if isinstance(v, str):
                            if k not in values:
                                values[k] = set()
                            values[k].add(v)
        
        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in values.items()}

    def _inject_exp_component(
        self, d: Dict[str, Any], exp_id: str, component: str
    ) -> Dict[str, Any]:
        """Ensure experiment and component are present in item properties.

        Also injects them into ``properties.keywords`` so STAC Browser renders
        them as keyword badges on item cards (requires showKeywordsInItemCards).
        """
        props = d.setdefault("properties", {})
        if not props.get("experiment"):
            props["experiment"] = exp_id
        if not props.get("component"):
            props["component"] = component
        # Inject into keywords so STAC Browser shows them as visible badges.
        keywords: List[str] = list(props.get("keywords") or [])
        for tag in [exp_id, component]:
            if tag and tag not in keywords:
                keywords.append(tag)
        props["keywords"] = keywords
        return d

    def get_items_for_collection(self, collection_id: str) -> List[Dict[str, Any]]:
        found = self._find_collection(collection_id)
        if not found:
            return []
        exp_id, col = found
        component = col.extra_fields.get("model", "")
        return [
            self._inject_exp_component(
                item.to_dict(include_self_link=False), exp_id, component
            )
            for item in col.get_all_items()
        ]

    def get_item(self, collection_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        found = self._find_collection(collection_id)
        if not found:
            return None
        exp_id, col = found
        component = col.extra_fields.get("model", "")
        for item in col.get_all_items():
            if item.id == item_id:
                d = item.to_dict(include_self_link=False)
                return self._inject_exp_component(d, exp_id, component)
        return None

    def search(
        self,
        experiment: Optional[str] = None,
        model: Optional[str] = None,
        variable: Optional[str] = None,
        collections: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        datetime_filter: Optional[str] = None,
        limit: int = 10,
        properties: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for exp_id, models in self.model_collections.items():
            if experiment and exp_id != experiment:
                continue
            for model_id, col in models.items():
                if collections and model_id not in collections:
                    continue
                for item in col.get_all_items():
                    props = item.properties or {}
                    if model and props.get("model") != model:
                        continue
                    if variable and props.get("variable") != variable:
                        continue
                    if ids and item.id not in ids:
                        continue
                    if properties:
                        match = True
                        for k, filter_val in properties.items():
                            pv = props.get(k)
                            if pv is None:
                                match = False
                                break
                            # filter_val is either a (op, value_str) tuple or a plain string.
                            op, v = filter_val if isinstance(filter_val, tuple) else ("=", filter_val)
                            # Try numeric comparison first, fall back to string.
                            try:
                                pv_cmp: Any = float(pv)
                                v_cmp: Any  = float(v)
                            except (TypeError, ValueError):
                                pv_cmp = str(pv)
                                v_cmp  = str(v)
                            if op == "=":
                                match = pv_cmp == v_cmp
                            elif op in ("!=", "<>"):
                                match = pv_cmp != v_cmp
                            elif op == "<":
                                match = pv_cmp < v_cmp
                            elif op == "<=":
                                match = pv_cmp <= v_cmp
                            elif op == ">":
                                match = pv_cmp > v_cmp
                            elif op == ">=":
                                match = pv_cmp >= v_cmp
                            else:
                                match = pv_cmp == v_cmp
                            if not match:
                                break
                        if not match:
                            continue
                    item_dict = item.to_dict(include_self_link=False)
                    if datetime_filter and not self._matches_datetime(item_dict, datetime_filter):
                        continue
                    self._inject_exp_component(item_dict, exp_id, col.extra_fields.get("model", ""))
                    item_dict["collection"] = model_id
                    results.append(item_dict)
                    if limit is not None and len(results) >= limit:
                        return results
        return results

    # ------------------------------------------------------------------
    # Datetime filtering
    # ------------------------------------------------------------------

    def _parse_item_interval(
        self, item: Dict[str, Any]
    ) -> Optional[Tuple[Any, Any]]:
        props = item.get("properties", {})
        if props.get("datetime"):
            try:
                dt = rfc3339_str_to_datetime(props["datetime"])
                return (dt, dt)
            except ValueError:
                return None
        start = props.get("start_datetime")
        end = props.get("end_datetime")
        if start or end:
            try:
                s = rfc3339_str_to_datetime(start) if start else None
                e = rfc3339_str_to_datetime(end) if end else None
                return (s, e)
            except ValueError:
                return None
        return None

    def _matches_datetime(self, item: Dict[str, Any], datetime_filter: str) -> bool:
        interval = str_to_interval(datetime_filter)
        if interval is None:
            return False
        item_interval = self._parse_item_interval(item)
        if item_interval is None:
            return False
        q_start, q_end = interval if isinstance(interval, tuple) else (interval, interval)
        i_start, i_end = item_interval
        if q_end and i_start and q_end < i_start:
            return False
        if i_end and q_start and i_end < q_start:
            return False
        return True


# ---------------------------------------------------------------------------
# STAC API client
# ---------------------------------------------------------------------------

@attr.s
class FesomsClient(AsyncBaseCoreClient):
    """Async STAC client backed by the local fesom_stac2 catalog."""

    catalog_dir: Path = attr.ib(default=CATALOG_DIR)
    loader: CatalogLoader = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.loader = CatalogLoader(self.catalog_dir)
        self.loader.load()

    def _base_url(self, request: Optional[Request]) -> str:
        if request is None:
            return ""
        return str(request.base_url).rstrip("/")

    def _collection_links(self, base_url: str, collection_id: str) -> List[Dict]:
        if not base_url:
            return []
        return [
            {"rel": "self",  "href": f"{base_url}/collections/{collection_id}",        "type": "application/json"},
            {"rel": "root",  "href": f"{base_url}/",                                   "type": "application/json"},
            {"rel": "items", "href": f"{base_url}/collections/{collection_id}/items",  "type": "application/geo+json"},
            # Filter extension: tell the STAC Browser where to find queryable fields.
            {"rel": _OGC_QUERYABLES_REL, "href": f"{base_url}/collections/{collection_id}/queryables", "type": "application/schema+json", "title": "Queryables"},
        ]

    def _item_links(self, base_url: str, collection_id: str, item_id: str) -> List[Dict]:
        if not base_url:
            return []
        return [
            {"rel": "self",       "href": f"{base_url}/collections/{collection_id}/items/{item_id}", "type": "application/geo+json"},
            {"rel": "collection", "href": f"{base_url}/collections/{collection_id}",                 "type": "application/json"},
            {"rel": "root",       "href": f"{base_url}/",                                            "type": "application/json"},
        ]

    def _item_collection_links(
        self,
        base_url: str,
        collection_id: str,
        offset: int = 0,
        limit: int = 10,
        total: int = 0,
        extra_params: str = "",
    ) -> List[Dict]:
        if not base_url:
            return []
        base_items = f"{base_url}/collections/{collection_id}/items"
        qs = f"limit={limit}{extra_params}"
        links = [
            {"rel": "self",       "href": f"{base_items}?{qs}&token={offset}", "type": "application/geo+json"},
            {"rel": "collection", "href": f"{base_url}/collections/{collection_id}",  "type": "application/json"},
            {"rel": "root",       "href": f"{base_url}/",                             "type": "application/json"},
        ]
        if offset + limit < total:
            links.append({"rel": "next", "href": f"{base_items}?{qs}&token={offset + limit}", "type": "application/geo+json"})
        if offset > 0:
            prev = max(0, offset - limit)
            links.append({"rel": "prev", "href": f"{base_items}?{qs}&token={prev}", "type": "application/geo+json"})
        return links

    def _search_links(self, base_url: str) -> List[Dict]:
        if not base_url:
            return []
        return [
            {"rel": "self", "href": f"{base_url}/search", "type": "application/geo+json"},
            {"rel": "root", "href": f"{base_url}/",       "type": "application/json"},
        ]

    def _search_pagination_links(
        self,
        base_url: str,
        offset: int = 0,
        limit: int = 10,
        total: int = 0,
        extra_params: str = "",
    ) -> List[Dict]:
        """Generate pagination links for search endpoint."""
        if not base_url:
            return []
        base_search = f"{base_url}/search"
        qs = f"limit={limit}{extra_params}"
        links = [
            {"rel": "self",       "href": f"{base_search}?{qs}&token={offset}", "type": "application/geo+json"},
            {"rel": "root",       "href": f"{base_url}/",                             "type": "application/json"},
        ]
        if offset + limit < total:
            links.append({"rel": "next", "href": f"{base_search}?{qs}&token={offset + limit}", "type": "application/geo+json"})
        if offset > 0:
            prev = max(0, offset - limit)
            links.append({"rel": "prev", "href": f"{base_search}?{qs}&token={prev}", "type": "application/geo+json"})
        return links

    # ------------------------------------------------------------------
    # Required endpoints
    # ------------------------------------------------------------------

    async def all_collections(self, **kwargs) -> Dict:
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)
        cols = self.loader.get_all_collections()
        for col in cols:
            col["links"] = self._collection_links(base_url, col["id"])
        return {
            "collections": cols,
            "links": [
                {"rel": "self", "href": f"{base_url}/collections", "type": "application/json"},
                {"rel": "root", "href": f"{base_url}/",            "type": "application/json"},
            ],
            "numberMatched": len(cols),
            "numberReturned": len(cols),
        }

    async def get_collection(self, collection_id: str, **kwargs) -> Dict:
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)
        col = self.loader.get_collection(collection_id)
        if col is None:
            raise NotFoundError(f"Collection {collection_id!r} not found")
        col["links"] = self._collection_links(base_url, collection_id)

        # Read any active filter params so datacube and snippet reflect the selection.
        variable: Optional[str] = None
        experiment: Optional[str] = None
        model: Optional[str] = None
        if request:
            variable   = request.query_params.get("variable")   or None
            experiment = request.query_params.get("experiment") or None
            model      = request.query_params.get("model")      or None
            cql_expr   = request.query_params.get("filter")
            cql_lang   = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql        = _parse_cql2(cql_expr, cql_lang)
                variable   = variable   or _cql_val(cql, "variable")
                experiment = experiment or _cql_val(cql, "experiment")
                model      = model      or _cql_val(cql, "model")

        if variable or experiment or model:
            # Narrow cube:variables to only the variables present in filtered items.
            all_items = self.loader.get_items_for_collection(collection_id)
            if variable:
                all_items = [i for i in all_items if i.get("properties", {}).get("variable") == variable]
            if experiment:
                all_items = [i for i in all_items if i.get("properties", {}).get("experiment") == experiment]
            if model:
                all_items = [i for i in all_items if i.get("properties", {}).get("model") == model]

            filtered_vars = sorted({
                i.get("properties", {}).get("variable", "")
                for i in all_items
                if i.get("properties", {}).get("variable")
            })
            if filtered_vars:
                col["cube:variables"] = {
                    v: {"type": "data", "dimensions": ["time", "latitude", "longitude"]}
                    for v in filtered_vars
                }

            # Inject xarray/intake snippet as a collection asset.
            snippet = _collection_xarray_snippet(all_items, api_url=base_url, collection_id=collection_id)
            col.setdefault("assets", {})["xarray_snippet"] = {
                "href": "inline",
                "type": "text/x-python",
                "title": "Open filtered selection with xarray / intake-xarray",
                "roles": ["metadata"],
                "description": snippet,
            }

        return col

    async def get_item(self, item_id: str, collection_id: str, **kwargs) -> Dict:
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)
        item = self.loader.get_item(collection_id, item_id)
        if item is None:
            raise NotFoundError(f"Item {item_id!r} not found in collection {collection_id!r}")
        item["collection"] = collection_id
        item["links"] = self._item_links(base_url, collection_id, item_id)

        # Single-file snippet for this item.
        item = _inject_snippet(item, base_url=base_url)

        # Second asset: load ALL timesteps for the same variable from this collection.
        variable = item.get("properties", {}).get("variable", "")
        experiment = item.get("properties", {}).get("experiment", "")
        model = item.get("properties", {}).get("model", "")
        if variable:
            all_items = self.loader.get_items_for_collection(collection_id)
            same_var = [i for i in all_items if i.get("properties", {}).get("variable") == variable]
            all_snippet = _collection_xarray_snippet(same_var, api_url=base_url, collection_id=collection_id)
            item["assets"]["load_all_timesteps"] = {
                "href": "inline",
                "type": "text/x-python",
                "title": f"Load all '{variable}' timesteps from this collection",
                "roles": ["metadata"],
                "description": all_snippet,
            }

        return item

    async def item_collection(
        self,
        collection_id: str,
        bbox: Optional[List[float]] = None,
        datetime: Optional[str] = None,
        limit: int = 10,
        token: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)

        # Pull custom filter params – either our simple ?variable= shortcut
        # or a CQL2 expression submitted by the STAC Browser filter UI.
        variable: Optional[str] = None
        experiment: Optional[str] = None
        model: Optional[str] = None
        component: Optional[str] = None
        extra_props: Dict[str, str] = {}
        _well_known = {"variable", "experiment", "model", "component"}
        if request:
            variable   = request.query_params.get("variable")   or None
            experiment = request.query_params.get("experiment")  or None
            model      = request.query_params.get("model")       or None
            component  = request.query_params.get("component")   or None
            cql_expr  = request.query_params.get("filter")
            cql_lang  = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql = _parse_cql2(cql_expr, cql_lang)
                variable   = variable   or _cql_val(cql, "variable")
                experiment = experiment or _cql_val(cql, "experiment")
                model      = model      or _cql_val(cql, "model")
                component  = component  or _cql_val(cql, "component")
                extra_props = {k: v for k, v in cql.items() if k not in _well_known}

        items = self.loader.get_items_for_collection(collection_id)

        # Apply filters before pagination so numberMatched is accurate.
        if variable:
            items = [i for i in items if i.get("properties", {}).get("variable") == variable]
        if experiment:
            items = [i for i in items if i.get("properties", {}).get("experiment") == experiment]
        if model:
            items = [i for i in items if i.get("properties", {}).get("model") == model]
        if component:
            items = [i for i in items if i.get("properties", {}).get("component") == component]
        if extra_props:
            items = [
                i for i in items
                if all(str(i.get("properties", {}).get(k)) == v for k, v in extra_props.items())
            ]
        if datetime:
            items = [i for i in items if self.loader._matches_datetime(i, datetime)]

        total_matched = len(items)

        # Build the collection-level snippet from ALL matched items (before page slice).
        collection_snippet = _collection_xarray_snippet(items, api_url=base_url, collection_id=collection_id)

        # Parse token from request query parameters for pagination
        token = None
        if request:
            token = request.query_params.get("token") or None
        
        offset = 0
        if token:
            try:
                offset = int(token)
            except ValueError:
                pass
        page = items[offset: offset + limit]

        for item in page:
            item["collection"] = collection_id
            item["links"] = self._item_links(base_url, collection_id, item["id"])

        # Build extra query-string params to preserve filters in next/prev links.
        from urllib.parse import urlencode, quote_plus
        
        extra_params = {}
        if variable:
            extra_params["variable"] = variable
        if experiment:
            extra_params["experiment"] = experiment
        if model:
            extra_params["model"] = model
        if component:
            extra_params["component"] = component
        if datetime:
            extra_params["datetime"] = datetime

        # Add CQL2 filter parameters if present
        if request:
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                extra_params["filter"] = cql_expr
                extra_params["filter-lang"] = cql_lang

        # Convert to properly encoded query string
        extra_qs = ""
        if extra_params:
            encoded_params = urlencode(extra_params, doseq=True)
            extra_qs = f"&{encoded_params}"

        return {
            "type": "FeatureCollection",
            "features": [_inject_snippet(i, base_url=base_url) for i in page],
            "links": self._item_collection_links(
                base_url, collection_id,
                offset=offset, limit=limit, total=total_matched,
                extra_params=extra_qs,
            ),
            "numberMatched": total_matched,
            "numberReturned": len(page),
            # Non-standard extension field: ready-to-run snippet for the full selection.
            "fesom:code_snippet": collection_snippet,
            # Asset entry so STAC Browser surfaces the snippet in the Assets panel.
            "assets": {
                "xarray_snippet": {
                    "href": "inline",
                    "type": "text/x-python",
                    "title": "Open filtered selection with xarray / intake-xarray",
                    "roles": ["metadata"],
                    "description": collection_snippet,
                }
            },
        }

    async def post_search(self, search_request: BaseSearchPostRequest, **kwargs) -> Dict:
        experiment = getattr(search_request, "experiment", None)
        model      = getattr(search_request, "model", None)
        variable   = getattr(search_request, "variable", None)
        component  = getattr(search_request, "component", None)
        extra_props: Dict[str, str] = {}
        _well_known = {"variable", "experiment", "model", "component"}

        # stac-fastapi populates filter_expr from the POST body 'filter' key.
        # It may arrive as a raw string (cql2-text) or a dict (cql2-json).
        raw_filter = getattr(search_request, "filter_expr", None)
        filter_lang = getattr(search_request, "filter_lang", "cql2-text") or "cql2-text"
        if raw_filter is None:
            # Also try the request body directly for cql2-text strings that
            # the typed model may have dropped.
            request = kwargs.get("request")
            if request:
                try:
                    body = await request.json()
                    raw_filter = body.get("filter")
                    filter_lang = body.get("filter-lang", "cql2-text")
                except Exception:
                    pass
        if raw_filter is not None:
            if isinstance(raw_filter, str):
                cql = _parse_cql2_text(raw_filter)
            else:
                cql = _parse_cql2_json(raw_filter)
            variable   = variable   or _cql_val(cql, "variable")
            experiment = experiment or _cql_val(cql, "experiment")
            model      = model      or _cql_val(cql, "model")
            component  = component  or _cql_val(cql, "component")
            extra_props = {k: v for k, v in cql.items() if k not in _well_known}

        return await self.get_search(
            collections=search_request.collections,
            ids=search_request.ids,
            bbox=search_request.bbox,
            datetime=search_request.datetime,
            limit=search_request.limit,
            experiment=experiment,
            model=model,
            variable=variable,
            component=component,
            extra_props=extra_props or None,
            **kwargs,
        )

    async def get_search(
        self,
        collections: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        intersects: Optional[Any] = None,
        datetime: Optional[str] = None,
        limit: Optional[int] = 10,
        token: Optional[str] = None,
        experiment: Optional[str] = None,
        model: Optional[str] = None,
        variable: Optional[str] = None,
        component: Optional[str] = None,
        extra_props: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict:
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)

        # Merge CQL2 filter from GET query params (STAC Browser filter UI).
        _well_known = {"variable", "experiment", "model", "component"}
        if request:
            component  = component  or request.query_params.get("component") or None
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql = _parse_cql2(cql_expr, cql_lang)
                variable   = variable   or _cql_val(cql, "variable")
                experiment = experiment or _cql_val(cql, "experiment")
                model      = model      or _cql_val(cql, "model")
                component  = component  or _cql_val(cql, "component")
                extra_from_get = {k: v for k, v in cql.items() if k not in _well_known}
                if extra_from_get:
                    extra_props = {**(extra_props or {}), **extra_from_get}

        # If component is set but not in extra_props yet, route it through properties filter.
        if component:
            extra_props = {**(extra_props or {}), "component": ("=", component)}

        # Parse token from request query parameters for pagination
        if request:
            token = request.query_params.get("token") or None

        offset = 0
        if token:
            try:
                offset = int(token)
            except ValueError:
                pass

        # Get ALL matching items to get total count, then apply pagination
        all_items = self.loader.search(
            experiment=experiment,
            model=model,
            variable=variable,
            collections=collections,
            ids=ids,
            datetime_filter=datetime,
            limit=None,  # Get all items for total count
            properties=extra_props or None,
        )

        total_matched = len(all_items)
        page = all_items[offset: offset + (limit or 10)]

        for item in page:
            col_id = item.get("collection", "")
            item["links"] = self._item_links(base_url, col_id, item["id"])

        # Build extra query-string params to preserve filters in next/prev links.
        from urllib.parse import urlencode

        extra_params = {}
        if variable:
            extra_params["variable"] = variable
        if experiment:
            extra_params["experiment"] = experiment
        if model:
            extra_params["model"] = model
        if component:
            extra_params["component"] = component
        if datetime:
            extra_params["datetime"] = datetime
        if collections:
            extra_params["collections"] = collections
        if ids:
            extra_params["ids"] = ids
            
        # Add CQL2 filter parameters if present
        if request:
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                extra_params["filter"] = cql_expr
                extra_params["filter-lang"] = cql_lang
        
        # Convert to properly encoded query string
        extra_qs = ""
        if extra_params:
            encoded_params = urlencode(extra_params, doseq=True)
            extra_qs = f"&{encoded_params}"

        collection_snippet = _collection_xarray_snippet(page, api_url=base_url, collection_id="")

        return {
            "type": "FeatureCollection",
            "features": [_inject_snippet(i, base_url=base_url) for i in page],
            "links": self._search_pagination_links(
                base_url, offset=offset, limit=limit or 10, total=total_matched,
                extra_params=extra_qs,
            ),
            "numberMatched": total_matched,
            "numberReturned": len(page),
            # Non-standard extension field: ready-to-run snippet for the current page.
            "fesom:code_snippet": collection_snippet,
        }


# ---------------------------------------------------------------------------
# Custom search request models (GET + POST) with FESOM-specific filters
# ---------------------------------------------------------------------------

@attr.s
class FesomsearchGetRequest(BaseSearchGetRequest):
    experiment: Annotated[
        Optional[str],
        Query(description="Filter by experiment name (e.g. garfield-001)"),
    ] = attr.ib(default=None)
    model: Annotated[
        Optional[str],
        Query(description="Filter by model component (e.g. fesom)"),
    ] = attr.ib(default=None)
    component: Annotated[
        Optional[str],
        Query(description="Filter by component name (e.g. fesom)"),
    ] = attr.ib(default=None)
    variable: Annotated[
        Optional[str],
        Query(description="Filter by variable name (e.g. ssh)"),
    ] = attr.ib(default=None)
    token: Annotated[
        Optional[str],
        Query(description="Pagination token (offset)"),
    ] = attr.ib(default=None)


class FesomsearchPostRequest(BaseSearchPostRequest):
    experiment: Optional[str] = None
    model: Optional[str] = None
    component: Optional[str] = None
    variable: Optional[str] = None


search_get_model = create_request_model(
    "FesomsearchGetRequest",
    base_model=FesomsearchGetRequest,
    mixins=[],
    request_type="GET",
)

search_post_model = create_request_model(
    "FesomsearchPostRequest",
    base_model=FesomsearchPostRequest,
    mixins=[],
    request_type="POST",
)


# ---------------------------------------------------------------------------
# Filter Extension client  (queryables backed by our catalog)
# ---------------------------------------------------------------------------

@attr.s
class FesomsFiltersClient(AsyncBaseFiltersClient):
    """Queryables client: returns JSON Schema for filterable item properties."""

    loader: CatalogLoader = attr.ib()

    async def get_queryables(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        request: Optional[Request] = kwargs.get("request")
        base_url = str(request.base_url).rstrip("/") if request else ""

        variables: Optional[List[str]] = None
        if collection_id:
            col = self.loader.get_collection(collection_id)
            if col:
                # Get variables from cube:variables instead of keywords
                cube_vars = col.get("cube:variables", {})
                if cube_vars:
                    variables = sorted(list(cube_vars.keys()))
        extra_properties = self.loader.get_all_property_keys(collection_id or "")
        unique_values = self.loader.get_unique_property_values(collection_id or "")
        return _make_queryables(base_url, collection_id or "", variables or None, extra_properties, unique_values)


# ---------------------------------------------------------------------------
# Aggregation Extension client  (counts & frequency buckets)
# ---------------------------------------------------------------------------

@attr.s
class FesomsAggregationClient(AsyncBaseAggregationClient):
    """Aggregation client: computes summary stats over catalog items."""

    loader: CatalogLoader = attr.ib()

    def _base_url(self, request: Optional[Request]) -> str:
        return str(request.base_url).rstrip("/") if request else ""

    async def get_aggregations(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> AggregationCollection:
        """Return the list of available aggregation fields."""
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)
        self_path = (
            f"/collections/{collection_id}/aggregations" if collection_id else "/aggregations"
        )
        return AggregationCollection(
            type="AggregationCollection",
            aggregations=[
                Aggregation(name="total_count",  data_type="integer"),
                Aggregation(name="variable",     data_type="string"),
                Aggregation(name="experiment",   data_type="string"),
                Aggregation(name="model",        data_type="string"),
                Aggregation(name="datetime",     data_type="datetime"),
            ],
            links=[
                {"rel": "root", "type": "application/json", "href": f"{base_url}/"},
                {"rel": "self", "type": "application/json", "href": f"{base_url}{self_path}"},
            ],
        )

    async def aggregate(
        self,
        collection_id: Optional[str] = None,
        aggregations: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        bbox: Optional[Any] = None,
        intersects: Optional[Any] = None,
        datetime: Optional[Any] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> AggregationCollection:
        """Compute and return aggregation buckets."""
        request: Optional[Request] = kwargs.get("request")
        base_url = self._base_url(request)
        self_path = (
            f"/collections/{collection_id}/aggregate" if collection_id else "/aggregate"
        )

        # Collect all relevant items.
        items: List[Dict[str, Any]] = []
        if collection_id:
            items = self.loader.get_items_for_collection(collection_id)
        else:
            for exp_id, models in self.loader.model_collections.items():
                for model_id, col in models.items():
                    if collections and model_id not in collections:
                        continue
                    items.extend(
                        item.to_dict(include_self_link=False)
                        for item in col.get_all_items()
                    )

        agg_names = set(aggregations) if aggregations else {
            "total_count", "variable", "experiment", "model", "datetime"
        }

        result: List[Aggregation] = []

        if "total_count" in agg_names:
            result.append(Aggregation(name="total_count", data_type="integer", value=len(items)))

        for field in ("variable", "experiment", "model"):
            if field not in agg_names:
                continue
            freq: Dict[str, int] = {}
            for item in items:
                val = item.get("properties", {}).get(field, "")
                if val:
                    freq[val] = freq.get(val, 0) + 1
            buckets = [
                {"key": k, "data_type": "string", "frequency": {"count": v}}
                for k, v in sorted(freq.items(), key=lambda x: -x[1])
            ]
            result.append(Aggregation(name=field, data_type="string", buckets=buckets))

        if "datetime" in agg_names:
            freq = {}
            for item in items:
                dt = item.get("properties", {}).get("datetime", "")
                if dt:
                    ym = dt[:7]  # YYYY-MM
                    freq[ym] = freq.get(ym, 0) + 1
            buckets = [
                {"key": k, "data_type": "datetime", "frequency": {"count": v}}
                for k, v in sorted(freq.items())
            ]
            result.append(Aggregation(name="datetime", data_type="datetime", buckets=buckets))

        return AggregationCollection(
            type="AggregationCollection",
            aggregations=result,
            links=[
                {"rel": "root", "type": "application/json", "href": f"{base_url}/"},
                {"rel": "self", "type": "application/json", "href": f"{base_url}{self_path}"},
            ],
        )


# ---------------------------------------------------------------------------
# Build the API
# ---------------------------------------------------------------------------

settings = ApiSettings(
    stac_fastapi_title="FESOM STAC API",
    stac_fastapi_description=(
        "STAC API for FESOM/AWIESM experiment output. "
        "Filter items by experiment, model, and variable via GET /search query params. "
        "Results include an xarray_snippet asset with ready-to-run Python code."
    ),
    stac_fastapi_version="1.0.0",
    stac_fastapi_landing_id="fesom-stac-api",
    enable_response_models=True,
)

stac_client = FesomsClient()
_filter_client = FesomsFiltersClient(loader=stac_client.loader)
_agg_client    = FesomsAggregationClient(loader=stac_client.loader)

api = StacApi(
    settings=settings,
    client=stac_client,
    extensions=[
        FilterExtension(client=_filter_client),
        AggregationExtension(client=_agg_client),
    ],
    search_get_request_model=search_get_model,
    search_post_request_model=search_post_model,
    middlewares=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

app = api.app


# ---------------------------------------------------------------------------
# Landing-page patch: inject queryables + children links
# ---------------------------------------------------------------------------
# The extensions add conformance classes automatically, but the landing page
# links must be augmented manually for the STAC Browser to discover them.

_CHILDREN_CONFORMANCE = "https://api.stacspec.org/v1.0.0/children"


@app.middleware("http")
async def _patch_root_links(request: Request, call_next):
    """Inject one 'child' link per experiment into the landing page response.

    STAC Browser navigates the hierarchy by following rel='child' links on the
    landing page.  Each experiment gets its own child link pointing to
    /catalogs/{exp_name}, which in turn lists model collections as children.
    """
    response = await call_next(request)
    if request.url.path not in ("/", ""):
        return response

    body_bytes = b""
    async for chunk in response.body_iterator:
        body_bytes += chunk

    try:
        data = _json.loads(body_bytes)
    except Exception:
        from starlette.responses import Response
        return Response(content=body_bytes, status_code=response.status_code,
                        headers=dict(response.headers), media_type=response.media_type)

    base_url = str(request.base_url).rstrip("/")
    links: List[Dict] = data.get("links", [])

    # Remove stale 'child' links and the 'data' link (/collections).
    # The 'data' rel causes STAC Browser to render all model collections flat
    # alongside experiment catalogs, breaking the hierarchy view.
    links = [lnk for lnk in links if lnk.get("rel") not in ("child", "data")]

    # One child link per experiment so the browser can walk the hierarchy.
    for exp_name in sorted(stac_client.loader.model_collections):
        links.append({
            "rel": "child",
            "href": f"{base_url}/catalogs/{exp_name}",
            "type": "application/json",
            "title": exp_name,
        })

    # Queryables link for the Filter extension.
    if not any(lnk.get("rel") == _OGC_QUERYABLES_REL for lnk in links):
        links.append({
            "rel": _OGC_QUERYABLES_REL,
            "href": f"{base_url}/queryables",
            "type": "application/schema+json",
            "title": "Queryables",
        })

    data["links"] = links
    new_body = _json.dumps(data).encode()
    from starlette.responses import Response
    headers = dict(response.headers)
    headers["content-length"] = str(len(new_body))
    return Response(content=new_body, status_code=response.status_code,
                    headers=headers, media_type=response.media_type)


# ---------------------------------------------------------------------------
# Children Extension  (GET /children, GET /catalogs/{id}, GET /catalogs/{id}/children)
# ---------------------------------------------------------------------------
# No built-in stac-fastapi support exists; implemented as plain FastAPI routes.
# Exposes the two-level hierarchy: Root → Experiment catalogs → Collections.

def _experiment_catalog_dict(exp_name: str, base_url: str) -> Dict[str, Any]:
    """Build a lightweight STAC Catalog dict for an experiment."""
    return {
        "type": "Catalog",
        "id": exp_name,
        "title": exp_name,
        "stac_version": "1.0.0",
        "description": f"Experiment {exp_name}",
        "links": [
            {"rel": "root",     "href": f"{base_url}/",                           "type": "application/json"},
            {"rel": "self",     "href": f"{base_url}/catalogs/{exp_name}",         "type": "application/json"},
            {"rel": "parent",   "href": f"{base_url}/",                            "type": "application/json"},
            {"rel": "children", "href": f"{base_url}/catalogs/{exp_name}/children", "type": "application/json"},
        ],
    }


@app.get("/children", tags=["Children Extension"])
async def root_children(request: Request):
    """Children Extension – list experiment sub-catalogs of the root catalog."""
    base_url = str(request.base_url).rstrip("/")
    children = [
        _experiment_catalog_dict(exp_id, base_url)
        for exp_id in sorted(stac_client.loader.model_collections)
    ]
    return {
        "children": children,
        "links": [
            {"rel": "root", "href": f"{base_url}/",        "type": "application/json"},
            {"rel": "self", "href": f"{base_url}/children", "type": "application/json"},
        ],
    }


@app.get("/catalogs/{catalog_id}", tags=["Children Extension"])
async def get_catalog(catalog_id: str, request: Request):
    """Children Extension – return an experiment sub-catalog with child links per model."""
    base_url = str(request.base_url).rstrip("/")
    models = stac_client.loader.model_collections.get(catalog_id)
    if models is None:
        from starlette.responses import JSONResponse as _JSONResponse
        return _JSONResponse({"detail": f"Catalog {catalog_id!r} not found"}, status_code=404)

    cat = _experiment_catalog_dict(catalog_id, base_url)
    # Inject one 'child' link per model collection so STAC Browser can drill down.
    for col_id, col in models.items():
        model_name = col.extra_fields.get("model", col.title or col_id)
        cat["links"].append({
            "rel": "child",
            "href": f"{base_url}/collections/{col_id}",
            "type": "application/json",
            "title": model_name,
        })
    return cat


@app.get("/catalogs/{catalog_id}/children", tags=["Children Extension"])
async def catalog_children(catalog_id: str, request: Request):
    """Children Extension – list model collections within an experiment."""
    base_url = str(request.base_url).rstrip("/")
    models = stac_client.loader.model_collections.get(catalog_id)
    if models is None:
        from starlette.responses import JSONResponse as _JSONResponse
        return _JSONResponse({"detail": f"Catalog {catalog_id!r} not found"}, status_code=404)

    children = []
    for col_id, col in models.items():
        col_dict = col.to_dict(include_self_link=False)
        col_dict["links"] = [
            {"rel": "root",   "href": f"{base_url}/",                              "type": "application/json"},
            {"rel": "self",   "href": f"{base_url}/collections/{col_id}",           "type": "application/json"},
            {"rel": "parent", "href": f"{base_url}/catalogs/{catalog_id}",           "type": "application/json"},
            {"rel": "items",  "href": f"{base_url}/collections/{col_id}/items",      "type": "application/geo+json"},
            {"rel": _OGC_QUERYABLES_REL, "href": f"{base_url}/collections/{col_id}/queryables", "type": "application/schema+json"},
        ]
        children.append(col_dict)

    return {
        "children": children,
        "links": [
            {"rel": "root",   "href": f"{base_url}/",                              "type": "application/json"},
            {"rel": "self",   "href": f"{base_url}/catalogs/{catalog_id}/children", "type": "application/json"},
            {"rel": "parent", "href": f"{base_url}/catalogs/{catalog_id}",          "type": "application/json"},
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FESOM STAC API Server")
    parser.add_argument("-p", "--port", type=int, default=9092)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("stac_api:app", host=args.host, port=args.port, reload=True)
