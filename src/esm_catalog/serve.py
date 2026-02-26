"""STAC API server for serving ESM catalogs.

Features:
- Full STAC API with search, filter, aggregation
- Children Extension for hierarchy navigation
- xarray snippet injection for easy data access
- Queryables with variable enum for filter UI
"""

import json as _json
import re
from pathlib import Path
from typing import Any, Optional

import attr
import pystac
import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from stac_fastapi.api.app import StacApi
from stac_fastapi.types.config import ApiSettings
from stac_fastapi.types.core import AsyncBaseCoreClient
from stac_fastapi.types.errors import NotFoundError
from stac_fastapi.types.search import BaseSearchPostRequest
from stac_fastapi.extensions.core.filter import FilterExtension
from stac_fastapi.extensions.core.filter.client import AsyncBaseFiltersClient
from stac_fastapi.extensions.core.aggregation import AggregationExtension
from stac_fastapi.extensions.core.aggregation.client import AsyncBaseAggregationClient
from stac_fastapi.extensions.core.aggregation.types import Aggregation, AggregationCollection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OGC_QUERYABLES_REL = "http://www.opengis.net/def/rel/ogc/1.0/queryables"


# ---------------------------------------------------------------------------
# xarray / intake snippet helpers
# ---------------------------------------------------------------------------

def _xarray_snippet(
    hrefs: list[str],
    variable: str = "",
    experiment: str = "",
    model: str = "",
) -> str:
    """Return a fenced Python snippet that opens files with xarray."""
    code: list[str] = ["import xarray as xr"]
    if experiment:
        code.append(f"# experiment : {experiment!r}")
    if model:
        code.append(f"# model      : {model!r}")
    if variable:
        code.append(f"# variable   : {variable!r}")
    code.append("")

    if len(hrefs) == 1:
        path = hrefs[0]
        code += [
            f'ds = xr.open_dataset(r"{path}", engine="netcdf4", decode_times=True)',
            "",
            "# --- alternatively with intake-xarray ---",
            "# import intake",
            f'# source = intake.open_netcdf(r"{path}", xarray_kwargs={{"engine": "netcdf4", "decode_times": True}})',
            "# ds = source.read()",
        ]
    else:
        code.append("paths = [")
        for h in hrefs:
            code.append(f'    r"{h}",')
        code += [
            "]",
            "",
            'ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", decode_times=True)',
            "",
            "# --- alternatively with intake-xarray ---",
            "# import intake",
            "# source = intake.open_netcdf(",
            "#     paths,",
            '#     xarray_kwargs={"engine": "netcdf4", "combine": "by_coords", "decode_times": True},',
            "# )",
            "# ds = source.read()",
        ]

    inner = "\n".join(code)
    return f"```python\n{inner}\n```"


def _collection_xarray_snippet(
    items: list[dict],
    api_url: str = "",
    collection_id: str = "",
) -> str:
    """Return a Python snippet that queries STAC API and opens results with xarray."""
    variables: list[str] = []
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

    filter_parts: list[str] = []
    if variables:
        filter_parts.append(f"variable = '{variable_str}'")
    if experiment:
        filter_parts.append(f"experiment = '{experiment}'")
    filter_expr = " AND ".join(filter_parts) if filter_parts else ""

    code: list[str] = [
        "import xarray as xr",
        "import pystac_client",
        "",
    ]
    if experiment:
        code.append(f"# experiment : {experiment!r}")
    if model:
        code.append(f"# model      : {model!r}")
    if variables:
        code.append(f"# variable(s): {variable_str!r}")
    code += [
        "",
        f'catalog = pystac_client.Client.open("{api_url}")',
        "results = catalog.search(",
        f'    collections=["{col_id}"],',
    ]
    if filter_expr:
        code += [
            f'    filter="{filter_expr}",',
            '    filter_lang="cql2-text",',
        ]
    code += [
        "    max_items=None,",
        ")",
        "",
        'paths = [item.assets["data"].href for item in results.items()]',
        "",
        'ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", decode_times=True)',
    ]

    inner = "\n".join(code)
    return f"```python\n{inner}\n```"


def _inject_snippet(item: dict) -> dict:
    """Return a copy of item dict with an xarray_snippet asset added."""
    item = dict(item)
    assets = dict(item.get("assets", {}))
    props = item.get("properties", {})
    hrefs = [a["href"] for a in assets.values() if "href" in a and a["href"] != "inline"]
    assets["xarray_snippet"] = {
        "href": "inline",
        "type": "text/x-python",
        "title": "Open with xarray",
        "roles": ["metadata"],
        "description": _xarray_snippet(
            hrefs,
            variable=props.get("variable", ""),
            experiment=props.get("experiment", ""),
            model=props.get("model", ""),
        ),
    }
    item["assets"] = assets
    return item


# ---------------------------------------------------------------------------
# CQL2 filter parser
# ---------------------------------------------------------------------------

def _parse_cql2_text(expr: str) -> dict[str, str]:
    """Parse simple CQL2-TEXT equality expressions."""
    result: dict[str, str] = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")", expr, re.IGNORECASE):
        prop = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        result[prop] = value
    return result


def _parse_cql2_json(expr: str) -> dict[str, str]:
    """Parse simple CQL2-JSON equality expressions."""
    result: dict[str, str] = {}
    try:
        data = _json.loads(expr)
    except Exception:
        return result

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        op = str(node.get("op", "")).lower()
        args = node.get("args", [])
        if op == "=" and len(args) == 2:
            prop_node, val_node = args
            if isinstance(prop_node, dict) and "property" in prop_node:
                result[prop_node["property"]] = str(val_node)
        elif op in ("and", "or") and isinstance(args, list):
            for arg in args:
                _walk(arg)

    _walk(data)
    return result


def _parse_cql2(expr: Optional[str], lang: str = "cql2-text") -> dict[str, str]:
    """Dispatch to the correct CQL2 parser."""
    if not expr:
        return {}
    if "json" in lang.lower():
        return _parse_cql2_json(expr)
    return _parse_cql2_text(expr)


# ---------------------------------------------------------------------------
# Queryables
# ---------------------------------------------------------------------------

def _make_queryables(
    base_url: str,
    collection_id: str = "",
    variables: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the OGC queryables JSON Schema document."""
    q_id = (
        f"{base_url}/collections/{collection_id}/queryables"
        if collection_id
        else f"{base_url}/queryables"
    )
    title = f"Queryables - {collection_id}" if collection_id else "ESM STAC Queryables"

    variable_schema: dict[str, Any] = {
        "title": "Variable",
        "description": "Model output variable (e.g. ssh, sst, MLD1)",
        "type": "string",
    }
    if variables:
        variable_schema["enum"] = sorted(variables)

    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$id": q_id,
        "type": "object",
        "title": title,
        "properties": {
            "id": {"title": "Item ID", "type": "string"},
            "collection": {"title": "Collection", "type": "string"},
            "datetime": {
                "title": "Date / Time",
                "type": "string",
                "format": "date-time",
            },
            "variable": variable_schema,
            "experiment": {
                "title": "Experiment",
                "description": "Experiment name (e.g. basic-001, garfield-001)",
                "type": "string",
            },
            "model": {
                "title": "Model",
                "description": "Model component (e.g. fesom, oasis3mct)",
                "type": "string",
            },
        },
    }


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

class CatalogLoader:
    """Load and index STAC catalog using pystac.

    Structure: Root Catalog -> Experiment Catalogs -> Model Collections -> Items
    """

    def __init__(self, catalog_dir: Path):
        self.catalog_dir = catalog_dir
        self.root: Optional[pystac.Catalog] = None
        self.model_collections: dict[str, dict[str, pystac.Collection]] = {}

    def load(self) -> None:
        root_path = self.catalog_dir / "catalog.json"
        self.root = pystac.Catalog.from_file(str(root_path))

        for exp_cat in self.root.get_children():
            if not isinstance(exp_cat, pystac.Catalog):
                continue
            # Key by title (e.g. "basic-001"), not UUID-prefixed id
            exp_key = exp_cat.title or exp_cat.id
            self.model_collections[exp_key] = {}
            for child in exp_cat.get_children():
                if isinstance(child, pystac.Collection):
                    self.model_collections[exp_key][child.id] = child

        n_cols = sum(len(v) for v in self.model_collections.values())
        print(f"Loaded catalog: {len(self.model_collections)} experiments, {n_cols} collections")

    def _collection_keywords(self, exp_id: str, col: pystac.Collection) -> list[str]:
        """Build keyword list from experiment, model, and variables."""
        model_name = col.extra_fields.get("model", "")
        keywords = [exp_id]
        if model_name and model_name not in keywords:
            keywords.append(model_name)
        for item in col.get_all_items():
            var = (item.properties or {}).get("variable")
            if var and var not in keywords:
                keywords.append(var)
        return keywords

    def get_all_collections(self) -> list[dict[str, Any]]:
        result = []
        for exp_id, models in self.model_collections.items():
            for model_id, col in models.items():
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                if not d.get("keywords"):
                    d["keywords"] = self._collection_keywords(exp_id, col)
                result.append(d)
        return result

    def get_collection(self, collection_id: str) -> Optional[dict[str, Any]]:
        for exp_id, models in self.model_collections.items():
            if collection_id in models:
                col = models[collection_id]
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                if not d.get("keywords"):
                    d["keywords"] = self._collection_keywords(exp_id, col)
                return d
        return None

    def _find_collection(self, collection_id: str) -> Optional[tuple[str, pystac.Collection]]:
        for exp_id, models in self.model_collections.items():
            if collection_id in models:
                return exp_id, models[collection_id]
        return None

    def get_items_for_collection(self, collection_id: str) -> list[dict[str, Any]]:
        found = self._find_collection(collection_id)
        if not found:
            return []
        _, col = found
        return [item.to_dict(include_self_link=False) for item in col.get_all_items()]

    def get_item(self, collection_id: str, item_id: str) -> Optional[dict[str, Any]]:
        found = self._find_collection(collection_id)
        if not found:
            return None
        _, col = found
        for item in col.get_all_items():
            if item.id == item_id:
                return item.to_dict(include_self_link=False)
        return None

    def search(
        self,
        experiment: Optional[str] = None,
        model: Optional[str] = None,
        variable: Optional[str] = None,
        collections: Optional[list[str]] = None,
        ids: Optional[list[str]] = None,
        datetime_filter: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
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
                    item_dict = item.to_dict(include_self_link=False)
                    item_dict["collection"] = model_id
                    results.append(item_dict)
                    if len(results) >= limit:
                        return results
        return results


# ---------------------------------------------------------------------------
# STAC API Client
# ---------------------------------------------------------------------------

@attr.s
class CatalogClient(AsyncBaseCoreClient):
    """STAC API client backed by local catalog."""

    loader: CatalogLoader = attr.ib(kw_only=True)

    def _base_url(self, request: Optional[Request]) -> str:
        if request is None:
            return ""
        return str(request.base_url).rstrip("/")

    def _collection_links(self, base_url: str, collection_id: str) -> list[dict]:
        if not base_url:
            return []
        return [
            {"rel": "self", "href": f"{base_url}/collections/{collection_id}", "type": "application/json"},
            {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            {"rel": "items", "href": f"{base_url}/collections/{collection_id}/items", "type": "application/geo+json"},
            {"rel": _OGC_QUERYABLES_REL, "href": f"{base_url}/collections/{collection_id}/queryables", "type": "application/schema+json", "title": "Queryables"},
        ]

    def _item_links(self, base_url: str, collection_id: str, item_id: str) -> list[dict]:
        if not base_url:
            return []
        return [
            {"rel": "self", "href": f"{base_url}/collections/{collection_id}/items/{item_id}", "type": "application/geo+json"},
            {"rel": "collection", "href": f"{base_url}/collections/{collection_id}", "type": "application/json"},
            {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
        ]

    async def all_collections(self, **kwargs) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        cols = self.loader.get_all_collections()
        for col in cols:
            col["links"] = self._collection_links(base_url, col["id"])
        return {
            "collections": cols,
            "links": [
                {"rel": "self", "href": f"{base_url}/collections", "type": "application/json"},
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            ],
            "numberMatched": len(cols),
            "numberReturned": len(cols),
        }

    async def get_collection(self, collection_id: str, **kwargs) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        col = self.loader.get_collection(collection_id)
        if col is None:
            raise NotFoundError(f"Collection {collection_id!r} not found")
        col["links"] = self._collection_links(base_url, collection_id)

        # Read filter params for datacube and snippet
        variable: Optional[str] = None
        experiment: Optional[str] = None
        model: Optional[str] = None
        if request:
            variable = request.query_params.get("variable") or None
            experiment = request.query_params.get("experiment") or None
            model = request.query_params.get("model") or None
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql = _parse_cql2(cql_expr, cql_lang)
                variable = variable or cql.get("variable")
                experiment = experiment or cql.get("experiment")
                model = model or cql.get("model")

        if variable or experiment or model:
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

            snippet = _collection_xarray_snippet(all_items, api_url=base_url, collection_id=collection_id)
            col.setdefault("assets", {})["xarray_snippet"] = {
                "href": "inline",
                "type": "text/x-python",
                "title": "Open filtered selection with xarray",
                "roles": ["metadata"],
                "description": snippet,
            }

        return col

    async def get_item(self, item_id: str, collection_id: str, **kwargs) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        item = self.loader.get_item(collection_id, item_id)
        if item is None:
            raise NotFoundError(f"Item {item_id!r} not found")
        item["collection"] = collection_id
        item["links"] = self._item_links(base_url, collection_id, item_id)

        # Single-file snippet
        item = _inject_snippet(item)

        # All timesteps snippet
        variable = item.get("properties", {}).get("variable", "")
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
        bbox: Optional[list[float]] = None,
        datetime: Optional[str] = None,
        limit: int = 10,
        token: Optional[str] = None,
        **kwargs,
    ) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)

        variable: Optional[str] = None
        experiment: Optional[str] = None
        model: Optional[str] = None
        if request:
            variable = request.query_params.get("variable") or None
            experiment = request.query_params.get("experiment") or None
            model = request.query_params.get("model") or None
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql = _parse_cql2(cql_expr, cql_lang)
                variable = variable or cql.get("variable")
                experiment = experiment or cql.get("experiment")
                model = model or cql.get("model")

        items = self.loader.get_items_for_collection(collection_id)

        if variable:
            items = [i for i in items if i.get("properties", {}).get("variable") == variable]
        if experiment:
            items = [i for i in items if i.get("properties", {}).get("experiment") == experiment]
        if model:
            items = [i for i in items if i.get("properties", {}).get("model") == model]

        total_matched = len(items)
        collection_snippet = _collection_xarray_snippet(items, api_url=base_url, collection_id=collection_id)

        offset = int(token) if token else 0
        page = items[offset:offset + limit]

        for item in page:
            item["collection"] = collection_id
            item["links"] = self._item_links(base_url, collection_id, item["id"])

        return {
            "type": "FeatureCollection",
            "features": [_inject_snippet(i) for i in page],
            "links": [
                {"rel": "self", "href": f"{base_url}/collections/{collection_id}/items", "type": "application/geo+json"},
                {"rel": "collection", "href": f"{base_url}/collections/{collection_id}", "type": "application/json"},
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            ],
            "numberMatched": total_matched,
            "numberReturned": len(page),
            "assets": {
                "xarray_snippet": {
                    "href": "inline",
                    "type": "text/x-python",
                    "title": "Open filtered selection with xarray",
                    "roles": ["metadata"],
                    "description": collection_snippet,
                }
            },
        }

    async def post_search(self, search_request: BaseSearchPostRequest, **kwargs) -> dict:
        return await self.get_search(
            collections=search_request.collections,
            ids=search_request.ids,
            bbox=search_request.bbox,
            datetime=search_request.datetime,
            limit=search_request.limit,
            **kwargs,
        )

    async def get_search(
        self,
        collections: Optional[list[str]] = None,
        ids: Optional[list[str]] = None,
        bbox: Optional[list[float]] = None,
        intersects: Optional[Any] = None,
        datetime: Optional[str] = None,
        limit: Optional[int] = 10,
        **kwargs,
    ) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)

        variable = None
        experiment = None
        model = None
        if request:
            variable = request.query_params.get("variable")
            experiment = request.query_params.get("experiment")
            model = request.query_params.get("model")
            cql_expr = request.query_params.get("filter")
            cql_lang = request.query_params.get("filter-lang", "cql2-text")
            if cql_expr:
                cql = _parse_cql2(cql_expr, cql_lang)
                variable = variable or cql.get("variable")
                experiment = experiment or cql.get("experiment")
                model = model or cql.get("model")

        items = self.loader.search(
            experiment=experiment,
            model=model,
            variable=variable,
            collections=collections,
            ids=ids,
            datetime_filter=datetime,
            limit=limit or 10,
        )

        for item in items:
            col_id = item.get("collection", "")
            item["links"] = self._item_links(base_url, col_id, item["id"])

        collection_snippet = _collection_xarray_snippet(items, api_url=base_url, collection_id="")

        return {
            "type": "FeatureCollection",
            "features": [_inject_snippet(i) for i in items],
            "links": [
                {"rel": "self", "href": f"{base_url}/search", "type": "application/geo+json"},
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            ],
            "numberMatched": len(items),
            "numberReturned": len(items),
            "assets": {
                "xarray_snippet": {
                    "href": "inline",
                    "type": "text/x-python",
                    "title": "Open search results with xarray",
                    "roles": ["metadata"],
                    "description": collection_snippet,
                }
            },
        }


# ---------------------------------------------------------------------------
# Filter Extension Client
# ---------------------------------------------------------------------------

@attr.s
class FiltersClient(AsyncBaseFiltersClient):
    """Queryables client with variable enum."""

    loader: CatalogLoader = attr.ib(kw_only=True)

    async def get_queryables(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        request = kwargs.get("request")
        base_url = str(request.base_url).rstrip("/") if request else ""

        variables: Optional[list[str]] = None
        if collection_id:
            col = self.loader.get_collection(collection_id)
            if col:
                exp_id = col.get("experiment_id", "")
                model = col.get("model", "")
                variables = [
                    kw for kw in col.get("keywords", [])
                    if kw and kw not in (exp_id, model)
                ]
        return _make_queryables(base_url, collection_id or "", variables or None)


# ---------------------------------------------------------------------------
# Aggregation Extension Client
# ---------------------------------------------------------------------------

@attr.s
class AggregationClient(AsyncBaseAggregationClient):
    """Aggregation client for summary stats."""

    loader: CatalogLoader = attr.ib(kw_only=True)

    def _base_url(self, request: Optional[Request]) -> str:
        return str(request.base_url).rstrip("/") if request else ""

    async def get_aggregations(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> AggregationCollection:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        self_path = f"/collections/{collection_id}/aggregations" if collection_id else "/aggregations"
        return AggregationCollection(
            type="AggregationCollection",
            aggregations=[
                Aggregation(name="total_count", data_type="integer"),
                Aggregation(name="variable", data_type="string"),
                Aggregation(name="experiment", data_type="string"),
                Aggregation(name="model", data_type="string"),
                Aggregation(name="datetime", data_type="datetime"),
            ],
            links=[
                {"rel": "root", "type": "application/json", "href": f"{base_url}/"},
                {"rel": "self", "type": "application/json", "href": f"{base_url}{self_path}"},
            ],
        )

    async def aggregate(
        self,
        collection_id: Optional[str] = None,
        aggregations: Optional[list[str]] = None,
        collections: Optional[list[str]] = None,
        ids: Optional[list[str]] = None,
        bbox: Optional[Any] = None,
        intersects: Optional[Any] = None,
        datetime: Optional[Any] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> AggregationCollection:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        self_path = f"/collections/{collection_id}/aggregate" if collection_id else "/aggregate"

        items: list[dict[str, Any]] = []
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

        result: list[Aggregation] = []

        if "total_count" in agg_names:
            result.append(Aggregation(name="total_count", data_type="integer", value=len(items)))

        for field in ("variable", "experiment", "model"):
            if field not in agg_names:
                continue
            freq: dict[str, int] = {}
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
                    ym = dt[:7]
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
# App factory
# ---------------------------------------------------------------------------

def create_app(catalog_dir: Path):
    """Create FastAPI STAC application with all features."""
    loader = CatalogLoader(catalog_dir)
    loader.load()

    client = CatalogClient(loader=loader)
    filter_client = FiltersClient(loader=loader)
    agg_client = AggregationClient(loader=loader)

    settings = ApiSettings(
        stac_fastapi_title="ESM STAC API",
        stac_fastapi_description="STAC API for ESM Tools experiment output",
        stac_fastapi_version="1.0.0",
        stac_fastapi_landing_id="esm-stac-api",
        enable_response_models=True,
    )

    api = StacApi(
        settings=settings,
        client=client,
        extensions=[
            FilterExtension(client=filter_client),
            AggregationExtension(client=agg_client),
        ],
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

    # -----------------------------------------------------------------------
    # Landing page middleware: inject child links for hierarchy navigation
    # -----------------------------------------------------------------------
    @app.middleware("http")
    async def patch_root_links(request: Request, call_next):
        response = await call_next(request)
        if request.url.path not in ("/", ""):
            return response

        body_bytes = b""
        async for chunk in response.body_iterator:
            body_bytes += chunk

        try:
            data = _json.loads(body_bytes)
        except Exception:
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        base_url = str(request.base_url).rstrip("/")
        links: list[dict] = data.get("links", [])

        # Remove 'data' link (causes flat hierarchy in STAC Browser)
        links = [lnk for lnk in links if lnk.get("rel") not in ("child", "data")]

        # Add child link per experiment
        for exp_name in sorted(loader.model_collections):
            links.append({
                "rel": "child",
                "href": f"{base_url}/catalogs/{exp_name}",
                "type": "application/json",
                "title": exp_name,
            })

        # Add queryables link
        if not any(lnk.get("rel") == _OGC_QUERYABLES_REL for lnk in links):
            links.append({
                "rel": _OGC_QUERYABLES_REL,
                "href": f"{base_url}/queryables",
                "type": "application/schema+json",
                "title": "Queryables",
            })

        data["links"] = links
        new_body = _json.dumps(data).encode()
        headers = dict(response.headers)
        headers["content-length"] = str(len(new_body))
        return Response(
            content=new_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

    # -----------------------------------------------------------------------
    # Children Extension routes
    # -----------------------------------------------------------------------
    def _experiment_catalog_dict(exp_name: str, base_url: str) -> dict[str, Any]:
        return {
            "type": "Catalog",
            "id": exp_name,
            "title": exp_name,
            "stac_version": "1.0.0",
            "description": f"Experiment {exp_name}",
            "links": [
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
                {"rel": "self", "href": f"{base_url}/catalogs/{exp_name}", "type": "application/json"},
                {"rel": "parent", "href": f"{base_url}/", "type": "application/json"},
                {"rel": "children", "href": f"{base_url}/catalogs/{exp_name}/children", "type": "application/json"},
            ],
        }

    @app.get("/children", tags=["Children Extension"])
    async def root_children(request: Request):
        base_url = str(request.base_url).rstrip("/")
        children = [
            _experiment_catalog_dict(exp_id, base_url)
            for exp_id in sorted(loader.model_collections)
        ]
        return {
            "children": children,
            "links": [
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
                {"rel": "self", "href": f"{base_url}/children", "type": "application/json"},
            ],
        }

    @app.get("/catalogs/{catalog_id}", tags=["Children Extension"])
    async def get_catalog(catalog_id: str, request: Request):
        base_url = str(request.base_url).rstrip("/")
        models = loader.model_collections.get(catalog_id)
        if models is None:
            return JSONResponse({"detail": f"Catalog {catalog_id!r} not found"}, status_code=404)

        cat = _experiment_catalog_dict(catalog_id, base_url)
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
        base_url = str(request.base_url).rstrip("/")
        models = loader.model_collections.get(catalog_id)
        if models is None:
            return JSONResponse({"detail": f"Catalog {catalog_id!r} not found"}, status_code=404)

        children = []
        for col_id, col in models.items():
            col_dict = col.to_dict(include_self_link=False)
            col_dict["links"] = [
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
                {"rel": "self", "href": f"{base_url}/collections/{col_id}", "type": "application/json"},
                {"rel": "parent", "href": f"{base_url}/catalogs/{catalog_id}", "type": "application/json"},
                {"rel": "items", "href": f"{base_url}/collections/{col_id}/items", "type": "application/geo+json"},
                {"rel": _OGC_QUERYABLES_REL, "href": f"{base_url}/collections/{col_id}/queryables", "type": "application/schema+json"},
            ]
            children.append(col_dict)

        return {
            "children": children,
            "links": [
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
                {"rel": "self", "href": f"{base_url}/catalogs/{catalog_id}/children", "type": "application/json"},
                {"rel": "parent", "href": f"{base_url}/catalogs/{catalog_id}", "type": "application/json"},
            ],
        }

    return app


def run_server(catalog_dir: str | Path, host: str = "0.0.0.0", port: int = 9092) -> None:
    """Run the STAC API server."""
    app = create_app(Path(catalog_dir))
    uvicorn.run(app, host=host, port=port)
