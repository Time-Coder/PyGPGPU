import hashlib
import re
import os
import pickle
from charset_normalizer import from_path
from typing import Union, Any

import numpy as np


def modify_time(file_path:str)->float:
    if not os.path.isfile(file_path):
        return 0
    
    return os.path.getmtime(file_path)

def md5sum(data_or_path:Union[str, np.ndarray, bytes, bytearray])->str:
    md5_hash = hashlib.md5()

    if isinstance(data_or_path, str):
        chunk_size:int = 4096
        
        with open(data_or_path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                md5_hash.update(chunk)
    elif isinstance(data_or_path, (bytes, bytearray)):
        md5_hash.update(data_or_path)

    elif isinstance(data_or_path, np.ndarray):
        md5_hash.update(data_or_path.tobytes())
    
    return md5_hash.hexdigest()

def md5sums(content:str)->str:
    return md5sum(content.encode("utf-8"))

def cat(file_name: str) -> str:
    return str(from_path(file_name).best())
    
def echo(content:str, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "w", encoding='utf-8') as file:
        file.write(content)

def save_var(var:Any, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "wb") as file:
        pickle.dump(var, file)

def load_var(file_name:str)->Any:
    with open(file_name, "rb") as file:
        return pickle.load(file)
    
def save_bin(bin:Union[bytes | bytearray], file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "wb") as file:
        file.write(bin)

def load_bin(file_name:str)->bytes:
    with open(file_name, "rb") as file:
        return file.read()

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    illegal_chars = r'[\\/:*?"<>|]'
    safe_name = re.sub(illegal_chars, '_', filename)
    
    safe_name = safe_name.strip().strip('.')
    
    if not safe_name:
        safe_name = "unnamed"
    
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if safe_name.upper() in reserved_names:
        safe_name = f"_{safe_name}_"
    
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name

def complete_basename(file_name:str):
    return '.'.join(os.path.basename(file_name).split('.')[:-1])