from ctypes import Structure
import copy

import numpy as np


def __has_structure(arr)->bool:
    if isinstance(arr, Structure):
        return True

    try:
        for ele in arr:
            if __has_structure(ele):
                return True
    except:
        pass

    return False

def __change_element(arr):
    try:
        for i, ele in enumerate(arr):
            if isinstance(ele, Structure):
                arr[i] = np_array(ele)
                continue

            if isinstance(ele, tuple):
                arr[i] = list(ele)

            __change_element(ele)
    except:
        pass

np_array = np.array
        
def array(*args, **kwargs):
    obj = args[0]

    if isinstance(obj, Structure) or not __has_structure(obj):
        return np_array(*args, **kwargs)
    
    obj = copy.deepcopy(obj)
    if isinstance(obj, tuple):
        obj = list(obj)
    
    __change_element(obj)
    return np_array(obj, *args[1:], **kwargs)

if np.array != array:
    np.array = array