import os
import re
from pathlib import Path
def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def dict_camel_to_snake(in_dict: dict) -> dict:
    out_dict = {}
    for k,v in in_dict.items():
        out_dict[camel_to_snake(k)] = v
    return out_dict

def get_vars_from_path(path: Path) -> dict:
    out_dict = {}
    if os.path.isdir(path):
        name = os.path.split(os.path.abspath(path))[1]
    elif os.path.isfile(path):
        filename = os.path.split(os.path.abspath(path))[1]
        name = os.path.splitext(os.path.splitext(filename)[0])[0] #Do twice for, for example .nii.gz
    
    kv_pairs = name.split('_')
    for kv_pair in kv_pairs:
        kv_pair_split = kv_pair.split('-')
        assert len(kv_pair_split) in [1,2], f'Something went wrong splitting the folder name into key-value pairs for folder {path} in this part: {kv_pair_split}'
        if len(kv_pair_split) == 1:
            assert not('suffix' in out_dict), f"Found multiple suffixes (or incorrect key-value separation using '-') for folder: {path} "
            out_dict['suffix'] = v
            
        elif len(kv_pair_split) == 2:
            k,v = kv_pair.split('-')

            if k=='sub':
                out_dict['participant_id'] = kv_pair
            elif k=='ses':
                out_dict['session_id'] = kv_pair
            else:
                out_dict[k] = v

            
    return out_dict

def get_basename_extension(file):
    fname, ext = os.path.splitext(file)
    if ext == ".gz":
        fname, ext = os.path.splitext(fname)
    basename = os.path.basename(fname)
    return basename, ext