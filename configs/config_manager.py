import os
import json

CURRENT_CONFIG_NAME = None
CURRENT_CONFIG = None

# Setting Configuration

def set_config(config_name):
    global CURRENT_CONFIG
    CURRENT_CONFIG = load_config(config_name)

def load_config(config_name, config_directory='configs'):
    global CURRENT_CONFIG_NAME

    if CURRENT_CONFIG_NAME is not None:
        raise ValueError("Configuration already set. Please call `set_config` only once.")

    CURRENT_CONFIG_NAME = config_name

    base_config_path = os.path.join(config_directory, 'base.json')
    specific_config_path = os.path.join(config_directory, f'{config_name}.json')
    
    # Load the base configuration
    with open(base_config_path, 'r') as base_file:
        base_config = json.load(base_file)
    
    # Try to load the specific configuration
    try:
        with open(specific_config_path, 'r') as specific_file:
            specific_config = json.load(specific_file)
    except FileNotFoundError:
        raise ValueError(f"Config '{config_name}.json' not found in {config_directory}")
    
    # Merge the specific configuration into the base configuration
    merged_config = {**base_config, **specific_config}
    
    return merged_config


# Getting Values from Configuration

def get_config_value(key):
    return CURRENT_CONFIG[key]

def print_config():
    if CURRENT_CONFIG is None:
        raise ValueError("No configuration set. Please call `set_config` first.")

    print(f"Current configuration: {CURRENT_CONFIG_NAME}")
    print(json.dumps(CURRENT_CONFIG, indent=4))