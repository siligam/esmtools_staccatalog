"""STAC API server for serving catalogs."""

import json as _json
import re
from pathlib import Path
from typing import Annotated, Any, Optional

import attr
import pystac
import uvicorn
from fastapi import Query
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
from stac_fastapi.extensions.core.filter import FilterExtension
from stac_fastapi.extensions.core.filter.client import AsyncBaseFiltersClient


def _parse_cql2_text(expr: str) -> dict[str, str]:
    """Parse simple CQL2-TEXT equality expressions."""
    result: dict[str, str] = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")", expr, re.IGNORECASE):
        prop = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        result[prop] = value
    return result


class CatalogLoader:
    """Load and index STAC catalog using pystac."""

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
            exp_key = exp_cat.title or exp_cat.id
            self.model_collections[exp_key] = {}
            for child in exp_cat.get_children():
                if isinstance(child, pystac.Collection):
                    self.model_collections[exp_key][child.id] = child

        n_cols = sum(len(v) for v in self.model_collections.values())
        print(f"Loaded catalog: {len(self.model_collections)} experiments, {n_cols} collections")

    def get_all_collections(self) -> list[dict[str, Any]]:
        result = []
        for exp_id, models in self.model_collections.items():
            for model_id, col in models.items():
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                result.append(d)
        return result

    def get_collection(self, collection_id: str) -> Optional[dict[str, Any]]:
        for exp_id, models in self.model_collections.items():
            if collection_id in models:
                col = models[collection_id]
                d = col.to_dict(include_self_link=False)
                d["experiment_id"] = exp_id
                return d
        return None

    def get_items_for_collection(self, collection_id: str) -> list[dict[str, Any]]:
        for models in self.model_collections.values():
            if collection_id in models:
                col = models[collection_id]
                return [item.to_dict(include_self_link=False) for item in col.get_all_items()]
        return []

    def get_item(self, collection_id: str, item_id: str) -> Optional[dict[str, Any]]:
        for models in self.model_collections.values():
            if collection_id in models:
                for item in models[collection_id].get_all_items():
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
        return col

    async def get_item(self, item_id: str, collection_id: str, **kwargs) -> dict:
        request = kwargs.get("request")
        base_url = self._base_url(request)
        item = self.loader.get_item(collection_id, item_id)
        if item is None:
            raise NotFoundError(f"Item {item_id!r} not found")
        item["collection"] = collection_id
        item["links"] = self._item_links(base_url, collection_id, item_id)
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

        variable = None
        if request:
            variable = request.query_params.get("variable")
            cql_expr = request.query_params.get("filter")
            if cql_expr:
                cql = _parse_cql2_text(cql_expr)
                variable = variable or cql.get("variable")

        items = self.loader.get_items_for_collection(collection_id)
        if variable:
            items = [i for i in items if i.get("properties", {}).get("variable") == variable]

        offset = int(token) if token else 0
        page = items[offset:offset + limit]

        for item in page:
            item["collection"] = collection_id
            item["links"] = self._item_links(base_url, collection_id, item["id"])

        return {
            "type": "FeatureCollection",
            "features": page,
            "links": [
                {"rel": "self", "href": f"{base_url}/collections/{collection_id}/items", "type": "application/geo+json"},
                {"rel": "collection", "href": f"{base_url}/collections/{collection_id}", "type": "application/json"},
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            ],
            "numberMatched": len(items),
            "numberReturned": len(page),
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
            if cql_expr:
                cql = _parse_cql2_text(cql_expr)
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

        return {
            "type": "FeatureCollection",
            "features": items,
            "links": [
                {"rel": "self", "href": f"{base_url}/search", "type": "application/geo+json"},
                {"rel": "root", "href": f"{base_url}/", "type": "application/json"},
            ],
            "numberMatched": len(items),
            "numberReturned": len(items),
        }


@attr.s
class FiltersClient(AsyncBaseFiltersClient):
    """Queryables client for filter extension."""

    loader: CatalogLoader = attr.ib(kw_only=True)

    async def get_queryables(
        self, collection_id: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        request = kwargs.get("request")
        base_url = str(request.base_url).rstrip("/") if request else ""

        return {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$id": f"{base_url}/queryables",
            "type": "object",
            "title": "ESM Catalog Queryables",
            "properties": {
                "id": {"title": "Item ID", "type": "string"},
                "collection": {"title": "Collection", "type": "string"},
                "datetime": {"title": "Date/Time", "type": "string", "format": "date-time"},
                "variable": {"title": "Variable", "type": "string"},
                "experiment": {"title": "Experiment", "type": "string"},
                "model": {"title": "Model", "type": "string"},
            },
        }


def create_app(catalog_dir: Path):
    """Create FastAPI STAC application."""
    loader = CatalogLoader(catalog_dir)
    loader.load()

    client = CatalogClient(loader=loader)
    filter_client = FiltersClient(loader=loader)

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
        extensions=[FilterExtension(client=filter_client)],
        middlewares=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
    )

    return api.app


def run_server(catalog_dir: str | Path, host: str = "0.0.0.0", port: int = 9092) -> None:
    """Run the STAC API server.

    Args:
        catalog_dir: Path to catalog directory
        host: Host to bind to
        port: Port to serve on
    """
    app = create_app(Path(catalog_dir))
    uvicorn.run(app, host=host, port=port)
