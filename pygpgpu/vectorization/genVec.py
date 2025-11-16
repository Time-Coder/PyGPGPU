from __future__ import annotations

from typing import Set, List, Dict, Union, Any, Optional, Tuple, TYPE_CHECKING
from .helper import generate_getter_swizzles, generate_setter_swizzles, is_number
from .genType import genType, MathForm, Flavor
from abc import abstractmethod

if TYPE_CHECKING:
    from .genMat import genMat


class genVec(genType):

    _attr_index_map:Dict[str, int] = {
        'x': 0,
        'y': 1,
        'z': 2,
        'w': 3,
        'r': 0,
        'g': 1,
        'b': 2,
        'a': 3,
        's': 0,
        't': 1,
        'p': 2,
        'q': 3,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13, 
        'E': 14,
        'F': 15,
        'a': 10,
        'b': 11,
        'c': 12,
        'd': 13, 
        'e': 14,
        'f': 15,
    }

    _all_attrs:Set[str] = {
        '_data', '_related_mat', '_mat_start_index', '_on_changed'
    }

    __namespaces:Dict[Flavor, List[str]] = {
        Flavor.GL: ['xyzw', 'rgba', 'stpq'],
        Flavor.CL:  ['xyzw', 'rgba', '0123456789ABCDEF']
    }

    def __init__(self, *args):
        genType.__init__(self)
        self._related_mat:Optional[genMat] = None
        self._mat_start_index:int = -1

        i: int = 0
        n_elements: int = self.n_elements
        n_args: int = len(args)
        error_message:str = f"invalid arguments for {self.__class__.__name__}()"

        if n_args == 0:
            return
        
        if n_args == 1:
            arg = args[0]
            if is_number(arg):
                for i in range(n_elements):
                    self._data[i] = self.cast(arg)
                return
        
        for i_arg, arg in enumerate(args):
            if is_number(arg):
                self._data[i] = self.cast(arg)

                i += 1
                if i == n_elements:
                    if n_args != 1 and i_arg != n_args - 1:
                        raise ValueError(error_message)
                    
                    return

            elif isinstance(arg, (genVec,tuple,list,str,bytes,bytearray)):
                sub_n_arg: int = len(arg)
                for sub_i_arg, value in enumerate(arg):
                    self._data[i] = self.cast(value)

                    i += 1
                    if i == n_elements:
                        if n_args != 1 and (i_arg != n_args - 1 or sub_i_arg != sub_n_arg - 1):
                            raise ValueError(error_message)
                        
                        return
            else:
                raise TypeError(f"invalid argument type(s) for {self.__class__.__name__}()")
            
        raise ValueError(error_message)
    
    @property
    def math_form(self)->MathForm:
        return MathForm.Vec

    @abstractmethod
    def __len__(self)->int:
        pass

    @property
    def shape(self)->Tuple[int]:
        return (len(self),)
    
    @staticmethod
    def vec_type(flavor:Flavor, dtype:type, size:int)->type:
        return genType.gen_type(flavor, MathForm.Vec, dtype, (size,))

    def _update_data(self, indices:Optional[List[int]] = None):
        genType._update_data(self, indices)

        if self._related_mat is not None:
            if indices is None:
                indices = range(len(self))

            for index in indices:
                self._related_mat._data[self._mat_start_index + index] = self._data[index]

            self._related_mat._call_on_changed()

    def __getattr__(self, name:str)->Union[float,bool,int,genVec]:
        if not self.__in_getter_swizzles(name):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if self.flavor == Flavor.CL and name[0] == 's':
            name = name[1:]

        vec_type = self.vec_type(self.flavor, self.dtype, len(name))
        return vec_type(*(self._data[self._attr_index_map[ch]] for ch in name))

    def __setattr__(self, name:str, value:Union[float,bool,int,genVec]):
        if name in self._all_attrs:
            super().__setattr__(name, value)
            return
        
        if not self.__in_setter_swizzles(name):
            if self.__in_getter_swizzles(name):
                raise AttributeError(f"property '{name}' of '{self.__class__.__name__}' object has no setter")
            
            if self.__in_total_swizzles(name):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            super().__setattr__(name, value)
            return

        update_indices:List[int] = []

        used_name:str = name
        if self.flavor == Flavor.CL and name[0] == 's':
            used_name:str = name[1:]

        value_is_vec:bool = (isinstance(value, genVec) and len(value) == len(used_name))
        if not value_is_vec and not is_number(value):
            raise TypeError(f"can not set '{value.__class__.__name__}' object to property '{name}' of '{self.__class__.__name__}' object")
        
        for i, ch in enumerate(used_name):
            index:int = self._attr_index_map[ch]
            self._data[index] = value[i] if value_is_vec else value
            update_indices.append(index)
        self._update_data(update_indices)
    
    def __getitem__(self, index:Union[int,slice])->Union[float,int,bool,genVec]:
        result = self._data[index]
        if isinstance(result, list):
            n_result = len(result)
            if n_result == 1:
                return result[0]
            elif n_result in [1, 2, 3, 4, 8, 16]:
                result_type = self.vec_type(self.flavor, self.dtype, n_result)
                return result_type(*result)
            else:
                return result
        else:
            return result
    
    def __setitem__(self, index:Union[int,slice], value:Union[float,int,bool,genVec])->None:
        if isinstance(index, int):
            self._data[index] = value
            self._update_data([index])
            return

        start, stop, step = index.indices(len(self))
        value_is_vec:bool = (not is_number(value))
        update_indices:List[int] = []
        for i in range(start, stop, step):
            self._data[i] = self.cast(value[i] if value_is_vec else value)
            update_indices.append(i)

        self._update_data(update_indices)
        
    def value_ptr(self):
        return self._data
        
    def __in_getter_swizzles(self, name:str):
        if self.flavor == Flavor.CL:
            if name[0] == 's':
                name = name[1:]
            if len(name) not in [1, 2, 3, 4, 8, 16]:
                return False
        elif self.flavor == Flavor.GL:
            if len(name) not in [1, 2, 3, 4]:
                return False

        len_self:int = len(self)
        for namespace in genVec.__namespaces[self.flavor]:
            namespace = namespace[:len_self]
            all_in_namespace:bool = True
            for ch in name:
                if ch not in namespace:
                    all_in_namespace:bool = False
                    break
            if all_in_namespace:
                return True

        return False
    
    def __in_setter_swizzles(self, name:str):
        len_self:int = len(self)
        if self.flavor == Flavor.CL:
            if name[0] == 's':
                name = name[1:]
            len_name = len(name)
            if len_name not in [1, 2, 3, 4, 8, 16] or len_name > len_self:
                return False
        elif self.flavor == Flavor.GL:
            if len(name) not in [1, 2, 3, 4] or len_name > len_self:
                return False

        for namespace in genVec.__namespaces[self.flavor]:
            namespace = namespace[:len_self]
            all_in_namespace:bool = True
            for ch in name:
                if ch not in namespace:
                    all_in_namespace:bool = False
                    break
            if all_in_namespace:
                return True

        return False
    
    def __in_total_swizzles(self, name:str):
        if self.flavor == Flavor.CL:
            if name[0] == 's':
                name = name[1:]
            if len(name) not in [1, 2, 3, 4, 8, 16]:
                return False
        elif self.flavor == Flavor.GL:
            if len(name) not in [1, 2, 3, 4]:
                return False

        for namespace in genVec.__namespaces[self.flavor]:
            all_in_namespace:bool = True
            for ch in name:
                if ch not in namespace:
                    all_in_namespace:bool = False
                    break
            if all_in_namespace:
                return True

        return False

    def __iter__(self):
        return iter(self._data)
    
    def __contains__(self, value:Any)->bool:
        return (value in self._data)
