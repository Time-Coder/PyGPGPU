from __future__ import annotations
from typing import Dict, Union


class CLConstante:

    __instances:Dict[Union[str, int], CLConstante] = {}

    def __init__(self, name:str, value:int):
        self.__name:str = name
        self.__value:int = value
        CLConstante.__instances[name] = self
        CLConstante.__instances[value] = self

    @property
    def name(self)->str:
        return self.__name
    
    @property
    def value(self)->int:
        return self.__value
    
    @staticmethod
    def get(key:int)->CLConstante:
        return CLConstante.__instances[key]
    
    def __int__(self):
        return self.__value
    
    def __str__(self):
        return f"{self.__name}({self.__value})"
    
    def __eq__(self, other:int)->bool:
        return (self.__value == other)
    