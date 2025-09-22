#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Custom checkpoint loader that handles module path changes
"""
import torch
import io
import pickle
from typing import Any, Dict


class ModuleRenamingUnpickler(pickle.Unpickler):
    """Custom unpickler that handles module renaming"""
    
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        
    def find_class(self, module, name):
        # Handle module renaming
        renamed_module = module
        
        # Map old module paths to new ones
        module_mappings = {
            'models.dyMEAN': 'models.SE3nn',
            'models.dyMEAN.modules': 'models.SE3nn.modules',
            'models.dyMEAN.modules.am_egnn': 'models.SE3nn.modules.am_egnn',
            'models.dyMEAN.nn_utils': 'models.SE3nn.nn_utils',
        }
        
        for old_path, new_path in module_mappings.items():
            if module.startswith(old_path):
                renamed_module = module.replace(old_path, new_path)
                print(f"Remapping module: {module} -> {renamed_module}")
                break
        
        return super().find_class(renamed_module, name)


def load_checkpoint_with_module_mapping(checkpoint_path, map_location='cpu'):
    """
    Load a checkpoint with module path remapping
    
    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device mapping for loading
        
    Returns:
        Loaded checkpoint object
    """
    print(f"Loading checkpoint with module remapping: {checkpoint_path}")
    
    # First, try regular loading
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        print(f"Module not found during regular loading: {e}")
        print("Attempting to load with module remapping...")
        
        # Create a custom pickle module with our unpickler
        class CustomPickle:
            Unpickler = ModuleRenamingUnpickler
            
            @staticmethod
            def load(file):
                return ModuleRenamingUnpickler(file).load()
            
            # Pass through other pickle functions
            Pickler = pickle.Pickler
            dump = pickle.dump
            dumps = pickle.dumps
            loads = pickle.loads
            HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
            DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
        
        # Load with the custom pickle module
        with open(checkpoint_path, 'rb') as f:
            return torch.load(f, map_location=map_location, pickle_module=CustomPickle)


def update_state_dict_keys(state_dict):
    """
    Update state dict keys to match new module names
    """
    new_state_dict = {}
    
    key_mappings = {
        'dyMEAN': 'SE3nn',
    }
    
    for key, value in state_dict.items():
        new_key = key
        for old_str, new_str in key_mappings.items():
            if old_str in key:
                new_key = key.replace(old_str, new_str)
                print(f"Renaming state dict key: {key} -> {new_key}")
                break
        new_state_dict[new_key] = value
    
    return new_state_dict
