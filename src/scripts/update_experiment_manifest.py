#!/usr/bin/env python3
"""
Script to update experiment_manifest.yaml with enhanced information from experiment_mapping.json
"""
import yaml
import json
from pathlib import Path
import os

def load_yaml(file_path):
    """Load YAML file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_yaml(data, file_path):
    """Save data to YAML file with proper formatting"""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                  allow_unicode=True, width=120)

def detect_is_static(name, type_field):
    """Detect if an experiment is static with special handling for static turns"""
    name_lower = name.lower()
    type_lower = type_field.lower() if type_field else ""
    
    # Special case: if it's a turn (in name or type), it's not static even if "static" is in the name
    if "turn" in name_lower or "turn" in type_lower:
        return False
    
    # Check for static indicators
    if any(keyword in name_lower for keyword in ["static", "setup", "waiting", "sync"]):
        return True
    
    # Check type field (already handled turn case above)
    if type_field and "static" in type_lower:
        return True
    
    return False

def get_data_types(exp_path):
    """Get available data types for an experiment by checking the directory"""
    data_types = []
    
    # Check for GPS
    if (exp_path / "GPS").exists():
        data_types.append("GPS")
    
    # Check for IMU sensors
    if (exp_path / "IMU").exists():
        imu_path = exp_path / "IMU"
    else:
        imu_path = exp_path  # Some experiments have sensors directly in the root
    
    # Check for each sensor type
    for sensor in ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]:
        if (imu_path / sensor).exists():
            data_types.append(sensor)
    
    return data_types

def create_experiment_category_map(mapping_data):
    """Create a map from experiment name to category"""
    exp_to_category = {}
    
    for category, info in mapping_data["evaluation_experiments"].items():
        for exp in info.get("morning", []):
            if exp not in exp_to_category:
                exp_to_category[exp] = []
            exp_to_category[exp].append(category)
            
        for exp in info.get("afternoon", []):
            if exp not in exp_to_category:
                exp_to_category[exp] = []
            exp_to_category[exp].append(category)
    
    return exp_to_category

def update_experiment_entry(exp, session, exp_to_category, data_root):
    """Update a single experiment entry with enhanced information"""
    name = exp["name"]
    
    # Get categories
    categories = exp_to_category.get(name, [])
    if categories:
        exp["category"] = categories[0] if len(categories) == 1 else categories
    
    # Detect if static - pass the type field to check for "static_turn"
    exp["is_static"] = detect_is_static(name, exp.get("type"))
    
    # Add paths
    relative_path = f"{session}/Experiments/{name}"
    exp["paths"] = {
        "relative": relative_path,
        "full_path": f"/data/raw/{relative_path}"
    }
    
    # Get data types if directory exists
    full_path = data_root / relative_path
    if full_path.exists():
        exp["data_types"] = get_data_types(full_path)
    
    return exp

def scan_all_experiments(data_root):
    """Scan data directory to find all experiments"""
    all_experiments = {"morning": [], "afternoon": []}
    
    for session in ["morning", "afternoon"]:
        exp_dir = data_root / session / "Experiments"
        if exp_dir.exists():
            for exp_path in sorted(exp_dir.iterdir()):
                if exp_path.is_dir():
                    all_experiments[session].append(exp_path.name)
    
    return all_experiments

def get_experiments_in_manifest(manifest):
    """Get set of all experiments already in manifest"""
    existing = set()
    
    # From evaluation experiments
    for session in ["morning", "afternoon"]:
        if session in manifest.get("evaluation_experiments", {}):
            for exp in manifest["evaluation_experiments"][session]:
                existing.add((exp["name"], session))
    
    # From static experiments
    for session in ["morning", "afternoon"]:
        if session in manifest.get("static_experiments", {}):
            for exp in manifest["static_experiments"][session]:
                existing.add((exp["name"], session))
    
    return existing

def create_new_experiment_entry(name, session, exp_to_category, data_root):
    """Create a new experiment entry for experiments not in the manifest"""
    exp = {
        "name": name,
        "path": f"data/raw/{session}/Experiments/{name}",
        "type": "unknown",
        "description": f"{name.replace('_', ' ')}"
    }
    
    # Get categories if mapped
    categories = exp_to_category.get(name, [])
    if categories:
        exp["category"] = categories[0] if len(categories) == 1 else categories
    
    # Detect if static
    exp["is_static"] = detect_is_static(name, exp.get("type"))
    
    # Add paths
    relative_path = f"{session}/Experiments/{name}"
    exp["paths"] = {
        "relative": relative_path,
        "full_path": f"/data/raw/{relative_path}"
    }
    
    # Get data types
    full_path = data_root / relative_path
    if full_path.exists():
        exp["data_types"] = get_data_types(full_path)
    
    return exp

def main():
    # Paths
    config_dir = Path("/mnt/c/Users/ben/Documents/EngD/09 Data collection/01_analysis_pipeline/analysis-pipeline/config/experiments")
    manifest_path = config_dir / "experiment_manifest.yaml"
    mapping_path = config_dir / "experiment_mapping.json"
    data_root = Path("/mnt/c/Users/ben/Documents/EngD/09 Data collection/01_analysis_pipeline/analysis-pipeline/data/raw")
    
    # Load files
    print("Loading experiment manifest...")
    manifest = load_yaml(manifest_path)
    
    print("Loading experiment mapping...")
    mapping = load_json(mapping_path)
    
    # Create experiment to category map
    exp_to_category = create_experiment_category_map(mapping)
    
    # Scan all experiments in data directory
    print("\nScanning data directory for all experiments...")
    all_experiments = scan_all_experiments(data_root)
    
    # Get existing experiments
    existing_experiments = get_experiments_in_manifest(manifest)
    
    # Update evaluation experiments
    print("\nUpdating evaluation experiments...")
    for session in ["morning", "afternoon"]:
        if session in manifest["evaluation_experiments"]:
            for i, exp in enumerate(manifest["evaluation_experiments"][session]):
                manifest["evaluation_experiments"][session][i] = update_experiment_entry(
                    exp, session, exp_to_category, data_root
                )
                print(f"  Updated: {exp['name']} ({session})")
    
    # Update static experiments
    print("\nUpdating static experiments...")
    for session in ["morning", "afternoon"]:
        if session in manifest["static_experiments"]:
            for i, exp in enumerate(manifest["static_experiments"][session]):
                # Static experiments have different path structure
                name = exp["name"]
                exp["is_static"] = True  # These are definitely static
                
                # Check both possible paths
                if "all_expts" in exp.get("path", ""):
                    relative_path = f"{session}/Experiments/{name}"
                else:
                    relative_path = f"{session}/Experiments/{name}"
                    
                exp["paths"] = {
                    "relative": relative_path,
                    "full_path": f"/data/raw/{relative_path}"
                }
                
                # Get data types
                full_path = data_root / relative_path
                if full_path.exists():
                    exp["data_types"] = get_data_types(full_path)
                
                manifest["static_experiments"][session][i] = exp
                print(f"  Updated: {exp['name']} ({session})")
    
    # Add missing experiments
    print("\nAdding missing experiments...")
    if "all_experiments" not in manifest:
        manifest["all_experiments"] = {"morning": [], "afternoon": []}
    
    for session in ["morning", "afternoon"]:
        for exp_name in all_experiments[session]:
            if (exp_name, session) not in existing_experiments:
                new_exp = create_new_experiment_entry(exp_name, session, exp_to_category, data_root)
                
                # Decide where to put it based on static detection
                if new_exp["is_static"]:
                    if session not in manifest["static_experiments"]:
                        manifest["static_experiments"][session] = []
                    manifest["static_experiments"][session].append(new_exp)
                    print(f"  Added to static: {exp_name} ({session})")
                else:
                    # Check if it's in a category
                    if "category" in new_exp:
                        manifest["evaluation_experiments"][session].append(new_exp)
                        print(f"  Added to evaluation: {exp_name} ({session})")
                    else:
                        # Put in all_experiments section
                        manifest["all_experiments"][session].append(new_exp)
                        print(f"  Added to uncategorized: {exp_name} ({session})")
    
    # Clean up empty sections
    if not any(manifest["all_experiments"].values()):
        del manifest["all_experiments"]
    
    # Save updated manifest
    print("\nSaving updated manifest...")
    save_yaml(manifest, manifest_path)
    print(f"Updated manifest saved to: {manifest_path}")
    
    # Print summary
    print("\nSummary:")
    total_eval = sum(len(manifest["evaluation_experiments"].get(s, [])) for s in ["morning", "afternoon"])
    total_static = sum(len(manifest["static_experiments"].get(s, [])) for s in ["morning", "afternoon"])
    total_other = sum(len(manifest.get("all_experiments", {}).get(s, [])) for s in ["morning", "afternoon"])
    print(f"  Total evaluation experiments: {total_eval}")
    print(f"  Total static experiments: {total_static}")
    print(f"  Total uncategorized experiments: {total_other}")
    print(f"  Total experiments: {total_eval + total_static + total_other}")

if __name__ == "__main__":
    main()