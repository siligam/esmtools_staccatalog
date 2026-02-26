"""Extract experiment metadata from ESM Tools experiment directories."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def find_experiments(experiments_dir: str | Path) -> dict[str, Any]:
    """Scan experiment directories and extract metadata.

    Args:
        experiments_dir: Path to directory containing experiment subdirectories.
            Each experiment should have config/{name}_finished_config.yaml.

    Returns:
        Dictionary mapping experiment names to their metadata:
        {
            "exp_name": {
                "files": {"model": ["/path/to/file.nc", ...]},
                "model_meta": {"model": {...}},
                "experiment_meta": {...}
            }
        }
    """
    experiments_dir = Path(experiments_dir)
    results = {}

    for entry in experiments_dir.iterdir():
        if not entry.is_dir():
            continue

        name = entry.name
        yaml_file = entry / "config" / f"{name}_finished_config.yaml"

        if not yaml_file.exists():
            logger.debug(f"Skipping {name} - no finished_config.yaml")
            continue

        try:
            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Skipping {name} - could not load config: {e}")
            continue

        component_files: dict[str, list[str]] = defaultdict(list)
        model_meta: dict[str, Any] = {}

        for component, config in yaml_data.items():
            if not isinstance(config, dict):
                continue

            outdata_dir = config.get("experiment_outdata_dir")
            if not outdata_dir:
                continue

            outdata_path = Path(outdata_dir)
            files = sorted(p for p in outdata_path.glob("*.nc") if p.exists())

            if files:
                logger.debug(f"  {component}: {len(files)} files")
                component_files[component] = [str(f) for f in files]
                model_meta[component] = {
                    "type": config.get("type"),
                    "version": config.get("version"),
                    "resolution": config.get("resolution"),
                    "scenario": config.get("scenario"),
                    "repository": config.get("repository"),
                    "metadata": config.get("metadata", {}),
                }

        general = yaml_data.get("general", {})
        all_models = general.get("models", [])

        experiment_meta = {
            "expid": general.get("expid"),
            "setup_name": general.get("setup_name"),
            "initial_date": general.get("initial_date"),
            "final_date": general.get("final_date"),
            "scenario": general.get("scenario"),
            "resolution": general.get("resolution"),
            "models": all_models,
            "models_without_nc_output": [m for m in all_models if m not in model_meta],
        }

        results[name] = {
            "files": dict(component_files),
            "model_meta": model_meta,
            "experiment_meta": experiment_meta,
        }

        logger.info(f"Found experiment: {name}")

    return results


def write_experiments_json(results: dict[str, Any], output_path: str | Path) -> None:
    """Write experiment metadata to JSON file.

    Args:
        results: Dictionary from find_experiments()
        output_path: Path to output JSON file
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
