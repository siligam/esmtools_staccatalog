"""
yaml representation of the fesom data
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_experiment_yaml(experiment_dir: Path) -> Optional[Dict]:
    """Load the finished configuration YAML file for an experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Parsed YAML content or None if file not found
    """
    yaml_file = experiment_dir / "config" / f"{experiment_dir.name}_finished_config.yaml"
    
    if not yaml_file.exists():
        return None
        
    try:
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load YAML file {yaml_file}: {e}")
        return None


def extract_model_metadata(yaml_data: Dict) -> Dict[str, Any]:
    """Extract model information from YAML data.
    
    Args:
        yaml_data: Parsed YAML configuration
        
    Returns:
        Dictionary with model metadata
    """
    models = []
    model_versions = []
    model_components = []
    
    # Extract information for each model component
    for model_name, model_config in yaml_data.items():
        if isinstance(model_config, dict) and 'model' in model_config:
            model_info = {
                'name': model_config.get('model', model_name),
                'version': model_config.get('version'),
                'type': model_config.get('type'),
                'description': model_config.get('metadata', {}).get('Description'),
                'institute': model_config.get('metadata', {}).get('Institute'),
                'authors': model_config.get('metadata', {}).get('Authors'),
                'publications': model_config.get('metadata', {}).get('Publications'),
                'license': model_config.get('metadata', {}).get('License'),
                'contact': model_config.get('contact'),
                'repository': model_config.get('repository'),
                'resolution': model_config.get('resolution'),
                'scenario': model_config.get('scenario')
            }
            
            models.append(model_info)
            
            if model_config.get('version'):
                model_versions.append(model_config['version'])
            if model_config.get('model'):
                model_components.append(model_config['model'])
    
    return {
        'models': models,
        'versions': model_versions,
        'components': model_components
    }


def extract_experiment_metadata(yaml_data: Dict) -> Dict[str, Any]:
    """Extract experiment-level metadata from YAML data.
    
    Args:
        yaml_data: Parsed YAML configuration
        
    Returns:
        Dictionary with experiment metadata
    """
    # Look for general experiment information
    experiment_info = {}
    
    # Extract from ECHAM section (usually has the most complete metadata)
    echam_config = yaml_data.get('echam', {})
    if echam_config:
        experiment_info.update({
            'institute': echam_config.get('metadata', {}).get('Institute'),
            'description': echam_config.get('metadata', {}).get('Description'),
            'authors': echam_config.get('metadata', {}).get('Authors'),
            'publications': echam_config.get('metadata', {}).get('Publications'),
            'license': echam_config.get('metadata', {}).get('License'),
            'contact': echam_config.get('contact')
        })
    
    # Extract time information
    for component in ['echam', 'fesom']:
        if component in yaml_data:
            comp_config = yaml_data[component]
            if 'pseudo_start_date' in comp_config:
                experiment_info['start_date'] = comp_config['pseudo_start_date']
            if 'pseudo_end_date' in comp_config:
                experiment_info['end_date'] = comp_config['pseudo_end_date']
            if 'scenario' in comp_config:
                experiment_info['scenario'] = comp_config['scenario']
            if 'resolution' in comp_config:
                experiment_info['resolution'] = comp_config['resolution']
            break
    
    return experiment_info


def extract_providers(yaml_data: Dict) -> List[Dict[str, Any]]:
    """Extract provider information from YAML data.
    
    Args:
        yaml_data: Parsed YAML configuration
        
    Returns:
        List of provider dictionaries
    """
    providers = []
    
    # Extract institute information
    echam_config = yaml_data.get('echam', {})
    institute = echam_config.get('metadata', {}).get('Institute')
    
    if institute:
        # Map institute names to URLs
        institute_urls = {
            'MPI-Met': 'https://www.mpimet.mpg.de/',
            'AWI': 'https://www.awi.de/',
            'Alfred Wegener Institute': 'https://www.awi.de/'
        }
        
        providers.append({
            'name': institute,
            'roles': ['producer', 'licensor'],
            'url': institute_urls.get(institute)
        })
    
    # Add AWI as processor/host (common for FESOM)
    if 'fesom' in yaml_data:
        providers.append({
            'name': 'Alfred Wegener Institute',
            'roles': ['processor', 'host'],
            'url': 'https://www.awi.de/'
        })
    
    # Add GitLab repositories as additional providers
    repositories = set()
    for component, config in yaml_data.items():
        if isinstance(config, dict) and 'repository' in config:
            repositories.add(config['repository'])
    
    for repo in repositories:
        if 'gitlab.dkrz.de' in repo:
            providers.append({
                'name': 'DKRZ GitLab',
                'roles': ['repository'],
                'url': 'https://gitlab.dkrz.de/'
            })
        elif 'gitlab.awi.de' in repo:
            providers.append({
                'name': 'AWI GitLab',
                'roles': ['repository'],
                'url': 'https://gitlab.awi.de/'
            })
    
    return providers


def extract_summaries(yaml_data: Dict) -> Dict[str, List[str]]:
    """Extract summary information for STAC summaries field.
    
    Args:
        yaml_data: Parsed YAML configuration
        
    Returns:
        Dictionary with summary lists
    """
    summaries = {}
    
    # Model components
    components = []
    versions = []
    resolutions = []
    scenarios = []
    institutes = []
    forcing = []
    
    for component, config in yaml_data.items():
        if isinstance(config, dict) and 'model' in config:
            components.append(config['model'])
            if config.get('version'):
                versions.append(config['version'])
            if config.get('resolution'):
                resolutions.append(config['resolution'])
            if config.get('scenario'):
                scenarios.append(config['scenario'])
            if config.get('metadata', {}).get('Institute'):
                institutes.append(config['metadata']['Institute'])
            if component == 'echam' and config.get('ocean_resolution'):
                forcing.append(config['ocean_resolution'])
    
    if components:
        summaries['model:component'] = list(set(components))
    if versions:
        summaries['model:version'] = list(set(versions))
    if resolutions:
        summaries['model:resolution'] = list(set(resolutions))
    if scenarios:
        summaries['model:scenario'] = list(set(scenarios))
    if institutes:
        summaries['model:institute'] = list(set(institutes))
    if forcing:
        summaries['model:forcing'] = list(set(forcing))
    
    return summaries


def extract_scientific_metadata(yaml_data: Dict) -> Dict[str, Any]:
    """Extract scientific metadata (DOIs, citations) from YAML data.
    
    Args:
        yaml_data: Parsed YAML configuration
        
    Returns:
        Dictionary with scientific metadata
    """
    scientific = {}
    
    echam_config = yaml_data.get('echam', {})
    metadata = echam_config.get('metadata', {})
    
    # Extract DOI from publications
    publications = metadata.get('Publications', '')
    if '<https://doi.org/' in publications:
        doi_start = publications.find('<https://doi.org/') + len('<https://doi.org/')
        doi_end = publications.find('>', doi_start)
        if doi_end > doi_start:
            scientific['sci:doi'] = publications[doi_start:doi_end]
    
    # Extract citation
    if publications:
        scientific['sci:citation'] = publications.replace('<', '').replace('>', '')
    
    return scientific


def create_enhanced_collection_metadata(experiment_dir: Path) -> Dict[str, Any]:
    """Create enhanced metadata for a STAC collection from YAML configuration.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary with enhanced metadata for STAC collection
    """
    yaml_data = load_experiment_yaml(experiment_dir)
    
    if not yaml_data:
        return {}
    
    # Extract all metadata
    model_info = extract_model_metadata(yaml_data)
    experiment_info = extract_experiment_metadata(yaml_data)
    providers = extract_providers(yaml_data)
    summaries = extract_summaries(yaml_data)
    scientific = extract_scientific_metadata(yaml_data)
    
    # Create enhanced metadata
    enhanced_metadata = {
        'title': f"AWIESM {experiment_dir.name}: {experiment_info.get('resolution', 'Unknown')} {experiment_info.get('scenario', 'Simulation')}",
        'description': experiment_info.get('description', f"AWIESM coupled climate model simulation: {experiment_dir.name}"),
        'keywords': [
            'AWIESM',
            'climate-model',
            'coupled-simulation'
        ] + model_info.get('components', []),
        'providers': providers,
        'summaries': summaries
    }
    
    # Add scientific metadata if available
    enhanced_metadata.update(scientific)
    
    # Add experiment-specific metadata
    if experiment_info.get('institute'):
        enhanced_metadata['institute'] = experiment_info['institute']
    if experiment_info.get('start_date'):
        enhanced_metadata['start_date'] = experiment_info['start_date']
    if experiment_info.get('end_date'):
        enhanced_metadata['end_date'] = experiment_info['end_date']
    
    return enhanced_metadata


def get_stac_extensions() -> List[str]:
    """Get list of STAC extensions for enhanced metadata.
    
    Returns:
        List of STAC extension URLs
    """
    return [
        "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
        "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
        "https://stac-extensions.github.io/provider/v1.0.0/schema.json"
    ]


def merge_metadata_lists(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple metadata dictionaries into a single comprehensive metadata.
    
    Args:
        metadata_list: List of metadata dictionaries from individual experiments
        
    Returns:
        Merged metadata dictionary
    """
    if not metadata_list:
        return {}
    
    merged = {
        'title': "AWIESM Experiments Collection",
        'description': "Collection of AWIESM coupled climate model simulations",
        'keywords': ['AWIESM', 'climate-model', 'coupled-simulation', 'ensemble'],
        'experiments': []
    }
    
    # Collect all unique values across experiments
    all_models = []
    all_components = []
    all_versions = []
    all_resolutions = []
    all_scenarios = []
    all_institutes = []
    all_providers = []
    all_summaries = {}
    scientific_metadata = {}
    experiment_dates = []
    
    for metadata in metadata_list:
        if not metadata:
            continue
            
        # Add experiment info
        exp_info = {
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'start_date': metadata.get('start_date'),
            'end_date': metadata.get('end_date'),
            'scenario': metadata.get('scenario'),
            'resolution': metadata.get('resolution')
        }
        merged['experiments'].append(exp_info)
        
        # Collect model information
        if 'models' in metadata:
            all_models.extend(metadata['models'])
        if 'keywords' in metadata:
            all_components.extend([k for k in metadata['keywords'] if k not in ['AWIESM', 'climate-model', 'coupled-simulation']])
        
        # Collect dates for temporal extent
        if metadata.get('start_date'):
            experiment_dates.append(metadata['start_date'])
        if metadata.get('end_date'):
            experiment_dates.append(metadata['end_date'])
        
        # Collect providers (avoid duplicates)
        if 'providers' in metadata:
            for provider in metadata['providers']:
                provider_key = f"{provider.get('name', '')}_{provider.get('url', '')}"
                if not any(p.get('name', '') == provider.get('name', '') for p in all_providers):
                    all_providers.append(provider)
        
        # Merge summaries
        if 'summaries' in metadata:
            for key, values in metadata['summaries'].items():
                if key not in all_summaries:
                    all_summaries[key] = []
                all_summaries[key].extend(values)
        
        # Collect scientific metadata
        for key, value in metadata.items():
            if key.startswith('sci:'):
                scientific_metadata[key] = value
    
    # Create unique lists
    if all_components:
        merged['keywords'] = list(set(merged['keywords'] + all_components))
    
    # Create merged summaries
    merged_summaries = {}
    for key, values in all_summaries.items():
        unique_values = list(set(values))
        if unique_values:
            merged_summaries[key] = unique_values
    
    if merged_summaries:
        merged['summaries'] = merged_summaries
    
    # Add providers
    if all_providers:
        merged['providers'] = all_providers
    
    # Add scientific metadata
    merged.update(scientific_metadata)
    
    # Add temporal extent if dates available
    if experiment_dates:
        merged['temporal_extent'] = {
            'start_date': min(experiment_dates),
            'end_date': max(experiment_dates)
        }
    
    # Add experiment count
    merged['experiment_count'] = len(metadata_list)
    
    return merged


def process_multiple_experiments(experiments_dir: Path) -> Dict[str, Any]:
    """Process all experiments in a directory and merge their metadata.
    
    Args:
        experiments_dir: Path to the directory containing experiment subdirectories
        
    Returns:
        Merged metadata dictionary for all experiments
    """
    if not experiments_dir.exists():
        print(f"Error: Experiments directory {experiments_dir} does not exist")
        return {}
    
    print(f"Scanning experiments directory: {experiments_dir}")
    
    # Find all experiment subdirectories
    experiment_dirs = []
    for item in experiments_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has a config directory with finished config
            config_file = item / "config" / f"{item.name}_finished_config.yaml"
            if config_file.exists():
                experiment_dirs.append(item)
            else:
                print(f"Warning: No finished config found for {item.name}")
    
    if not experiment_dirs:
        print("No valid experiments found")
        return {}
    
    print(f"Found {len(experiment_dirs)} experiments to process")
    
    # Process each experiment
    metadata_list = []
    successful_experiments = []
    
    for exp_dir in experiment_dirs:
        print(f"Processing {exp_dir.name}...")
        metadata = create_enhanced_collection_metadata(exp_dir)
        if metadata:
            metadata_list.append(metadata)
            successful_experiments.append(exp_dir.name)
        else:
            print(f"Warning: Could not extract metadata from {exp_dir.name}")
    
    if not metadata_list:
        print("No metadata could be extracted from any experiment")
        return {}
    
    print(f"Successfully processed {len(metadata_list)} experiments")
    print(f"Experiments: {', '.join(successful_experiments)}")
    
    # Merge all metadata
    merged_metadata = merge_metadata_lists(metadata_list)
    
    return merged_metadata


if __name__ == "__main__":
    # Test the extractor with a sample experiment
    test_dir = Path("/albedo/work/user/pgierz/SciComp/Tutorials/AWIESM_Basics/experiments/basic-001")
    
    print("Testing YAML Metadata Extractor")
    print("=" * 50)
    
    metadata = create_enhanced_collection_metadata(test_dir)
    
    print("\nEnhanced Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    print(f"\nSTAC Extensions: {get_stac_extensions()}")
    
    # Test multiple experiments processing
    print("\n" + "=" * 50)
    print("Testing Multiple Experiments Processing")
    print("=" * 50)
    
    experiments_dir = Path("/albedo/work/user/pgierz/SciComp/Tutorials/AWIESM_Basics/experiments")
    merged_metadata = process_multiple_experiments(experiments_dir)
    
    if merged_metadata:
        print("\nMerged Metadata:")
        for key, value in merged_metadata.items():
            if key == 'experiments':
                print(f"{key}: {len(value)} experiments")
                for exp in value[:3]:  # Show first 3
                    print(f"  - {exp.get('title', 'Unknown')}")
                if len(value) > 3:
                    print(f"  ... and {len(value) - 3} more")
            elif key == 'summaries':
                print(f"{key}:")
                for sum_key, sum_values in value.items():
                    print(f"  {sum_key}: {sum_values}")
            else:
                print(f"{key}: {value}")
