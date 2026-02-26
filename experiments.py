import shutil
from pathlib import Path
import yaml
from collections import defaultdict
from itertools import filterfalse
import json


EXP_DIR = Path("/albedo/work/user/pgierz/SciComp/Tutorials/AWIESM_Basics/experiments")

def find_experiments(exp_dir):
    results = {}
    experiments = defaultdict(list)
    for entry in exp_dir.iterdir():
        name = entry.name
        print(name)
        if entry.is_dir():
            yaml_file = entry / "config" / f"{name}_finished_config.yaml"
            if not yaml_file.exists():
                print(f"Skipping {name} - no finished_config.yaml")
                continue
            try:
                with open(yaml_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)
            except Exception as e:
                print(f"Skipping {name} - could not load finished_config.yaml: {e}")
                continue
            component_files = defaultdict(list)
            model_meta = {}
            for component, config in yaml_data.items():
                if isinstance(config, dict):
                    outdata_dir = config.get("experiment_outdata_dir")
                    if outdata_dir:
                        outdata_dir = Path(outdata_dir)
                        files = list(outdata_dir.glob("*.nc"))
                        files = filter(lambda p: Path(p).exists(), files)
                        files = sorted(files)
                        if files:
                            print("  ", component)
                            print("    ",len(files))
                            files = list(map(str, files))
                            component_files[component].extend(files)
                            model_meta[component] = {
                                "type":       config.get("type"),
                                "version":    config.get("version"),
                                "resolution": config.get("resolution"),
                                "scenario":   config.get("scenario"),
                                "repository": config.get("repository"),
                                "metadata":   config.get("metadata", {}),
                            }
            general = yaml_data.get("general", {})
            all_models = general.get("models", [])
            experiment_meta = {
                "expid":                   general.get("expid"),
                "setup_name":              general.get("setup_name"),
                "initial_date":            general.get("initial_date"),
                "final_date":              general.get("final_date"),
                "scenario":                general.get("scenario"),
                "resolution":              general.get("resolution"),
                "models":                  all_models,
                "models_without_nc_output": [m for m in all_models if m not in model_meta],
            }
            results[name] = {
                "files":           dict(component_files),
                "model_meta":      model_meta,
                "experiment_meta": experiment_meta,
            }
    return results


def write_experiments_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    from pprint import pprint
    results = find_experiments(EXP_DIR)
    pprint(results)
    write_experiments_json(results, "experiments.json")
    
    
