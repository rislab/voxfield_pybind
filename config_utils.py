'''
Created Date: Nov 17 2023
Author: Jonathan Lee
-----
Last Modified: Nov 17 2023
Modified By: Jonathan Lee
-----
Description:

A simple commandline utility to compare config files 
and inspect matching/inconsistent key-value pairs
'''

import yaml
import argparse
from pathlib import Path
from pprint import pprint
from easydict import EasyDict

cmdline_funcs = []

# decorator
def register_cmdline_func(func):
  cmdline_funcs.append(func)
  return func

@register_cmdline_func
def missing(A: dict, B: dict) -> dict:
    '''Returns (A-B) the dict of keys in a A but not in B'''
    flat_A = flatten(A)
    flat_B = flatten(B)
    A_keys = set(flat_A.keys())
    B_keys = set(flat_B.keys())
    diff = A_keys.difference(B_keys)
    missing = {k: flat_A[k] for k in list(diff)}
    return missing

@register_cmdline_func
def shared(A: dict, B: dict) -> dict:
    '''Returns (A U B) the dict of keys in a A and B'''
    flat_A = flatten(A)
    flat_B = flatten(B)
    A_keys = set(flat_A.keys())
    B_keys = set(flat_B.keys())
    intersect = A_keys.intersection(B_keys)
    shared = {k: flat_A[k] for k in list(intersect)}
    return shared

@register_cmdline_func
def changed(A: dict, B: dict) -> dict: 
    '''Returns a dict of shared key-(A.value, B.value) pairs where the values are inconsistent'''
    flat_A = flatten(A)
    flat_B = flatten(B)
    A_keys = set(flat_A.keys())
    B_keys = set(flat_B.keys())
    intersect = list(A_keys.intersection(B_keys))
    changed = {}
    for k in intersect:
        if flat_A[k] != flat_B[k]:
            changed[k] = (flat_A[k], flat_B[k])

    return changed

@register_cmdline_func
def combined(A: dict, B: dict) -> dict:
    '''Returns (A | B) with values of A taking precedent over B'''
    flat_A = flatten(A)
    flat_B = flatten(B)
    A_keys = set(flat_A.keys())
    B_keys = set(flat_B.keys())
    union = A_keys.union(B_keys)
    shared = {k: flat_A[k] if flat_A.get(k) != None else flat_B[k] for k in list(union)}
    return shared



### Other utilities

def flatten(d: dict) -> dict:
    '''Flatten a dictionary so all keys are at root level'''
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten(v).items())
        else:
            if (isinstance(k, str)):
                k = k.strip()
            if (isinstance(v, str)):
                v = v.strip()
            items.append((k, v))
    return dict(items)

    
def load_config(config_file: str) -> dict:
    config_dict = dict()

    ext = config_file.split(".")[1] 
    if (ext == 'yaml' or ext == 'yml'):
        try:
            config_dict = yaml.safe_load(open(config_file))
        except:
            print("Error trying to load the config file in YAML format")

    if (ext == 'json'):
        try:
            config_dict = json.loads(open(config_file))
        except:
            print("Error trying to load the config file in JSON format")

    return config_dict

# Run as cmdline tool
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_A', type=str) 
    parser.add_argument('config_file_B', type=str) 

    # Register the functions
    for func in cmdline_funcs:
        option = "--" + func.__name__
        parser.add_argument(option, action='store_true', help=func.__doc__)
    args = parser.parse_args()

    # Run the requested function
    for func in cmdline_funcs:
        if args.__dict__[func.__name__] == True:
            abspath_A = Path(args.config_file_A).resolve()
            abspath_B = Path(args.config_file_B).resolve()
            A = load_config(str(abspath_A))
            B = load_config(str(abspath_B))
            out = func(A, B)
            print(yaml.dump(out))
            # pprint(out)
