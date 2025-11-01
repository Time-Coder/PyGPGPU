from __future__ import annotations
from typing import Dict, Union, Any
from ctypes import c_uint, c_ulong, sizeof
import enum


class Constante:

    __instances:Dict[Union[str, int], Constante] = {}

    def __init__(self, name:str, value:Any):
        self.__name:str = name
        Constante.__instances[name] = self

    @property
    def name(self)->str:
        return self.__name
    
    @staticmethod
    def get(key:int)->Constante:
        return Constante.__instances[key]


class IntConstante(int, Constante):

    def __new__(cls, name:str, value:int):
        instance = int.__new__(cls, value)
        return instance
    
    @property
    def value(self)->int:
        return int(self)
    
    def __str__(self)->str:
        return f"{self.name}({int.__str__(self)})"
    
    def __repr__(self)->str:
        return str(self)


class FloatConstante(float, Constante):

    def __new__(cls, name:str, value:float):
        instance = float.__new__(cls, value)
        return instance
    
    @property
    def value(self)->float:
        return float(self)
    
    def __str__(self)->str:
        return f"{self.name}({float.__str__(self)})"
    
    def __repr__(self)->str:
        return str(self)
    

class IntEnum(enum.IntEnum):

    def __str__(self)->str:
        return f"{self.name}({self.value})"
    
    def __repr__(self)->str:
        return str(self)
    
    @classmethod
    def dtype(cls):
        return c_uint
    
    @classmethod
    def size(cls):
        return sizeof(cls.dtype())
    

class IntFlag(enum.IntFlag):

    def __str__(self)->str:
        if len(list(self)) != 1:
            return "0"
        else:
            return f"{self.name}({self.value})"
    
    def __repr__(self)->str:
        return str(self)
    
    @classmethod
    def dtype(cls):
        return c_uint
    
    @classmethod
    def size(cls):
        return sizeof(cls.dtype())
    

CL_NAME_VERSION_MAX_NAME_SIZE = IntConstante("CL_NAME_VERSION_MAX_NAME_SIZE", 64)
CL_NAME_VERSION_MAX_NAME_SIZE_KHR = IntConstante("CL_NAME_VERSION_MAX_NAME_SIZE_KHR", 64)