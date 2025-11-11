import hashlib
import re
import os
import pickle
from charset_normalizer import from_path
from typing import Union, Any, Tuple

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

def detect_work_size(total_size: int, shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    local_size = detect_local_size(total_size, shape)
    global_size = detect_global_size(local_size, shape)
    return global_size, local_size

def detect_global_size(local_size:Tuple[int, ...], shape: Tuple[int, ...]):
    global_ws = []
    for s, l in zip(shape, local_size):
        if l == 0:
            raise ValueError("Local size cannot be zero.")
        g = ((s + l - 1) // l) * l
        global_ws.append(g)

    return tuple(global_ws)

def detect_local_size(total_size: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    ndim = len(shape)
    
    if ndim == 1:
        l = min(total_size, shape[0])
        l = 1 << (l.bit_length() - 1)
        return (l,)
    
    if ndim == 2:
        w, h = shape
        exp_total = total_size.bit_length() - 1
        if (1 << exp_total) != total_size:
            raise ValueError("total_size must be a power of two")
        
        candidates = []
        for ex in range(exp_total + 1):
            ey = exp_total - ex
            lx, ly = 1 << ex, 1 << ey
            if lx <= w and ly <= h:
                candidates.append((lx, ly))
        
        if not candidates:
            best_prod = 0
            best_pair = None
            for ex in range(10):
                for ey in range(10):
                    lx, ly = 1 << ex, 1 << ey
                    if lx <= w and ly <= h:
                        prod = lx * ly
                        if prod <= total_size and prod > best_prod:
                            best_prod = prod
                            best_pair = (lx, ly)
            if best_pair:
                return best_pair
            else:
                raise ValueError(f"No valid local size for {shape}, {total_size}")
        
        def score(pair):
            lx, ly = pair
            ratio_x = lx / w
            ratio_y = ly / h
            imbalance = abs(ratio_x - ratio_y)
            penalty = 0
            if lx < 8: penalty += 10
            if ly < 8: penalty += 10
            return imbalance + penalty
        
        best = min(candidates, key=score)
        return best
    
    if ndim == 3:
        w, h, d = shape
        exp_total = total_size.bit_length() - 1
        if (1 << exp_total) != total_size:
            raise ValueError("total_size must be a power of two")
        
        candidates = []
        for ex in range(exp_total + 1):
            for ey in range(exp_total - ex + 1):
                ez = exp_total - ex - ey
                lx, ly, lz = 1 << ex, 1 << ey, 1 << ez
                if lx <= w and ly <= h and lz <= d:
                    candidates.append((lx, ly, lz))
        
        if not candidates:
            raise ValueError(f"No valid 3D local size for {shape}, {total_size}")
        
        def score(triple):
            lx, ly, lz = triple
            rx, ry, rz = lx / w, ly / h, lz / d
            avg = (rx + ry + rz) / 3
            var = (rx - avg)**2 + (ry - avg)**2 + (rz - avg)**2
            penalty = sum(10 for x in triple if x < 8)
            return var + penalty
        
        return min(candidates, key=score)
    
    raise NotImplementedError("Only 1D/2D/3D supported")