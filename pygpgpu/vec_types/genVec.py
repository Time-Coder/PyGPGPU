from __future__ import annotations
from ctypes import Structure

from .helper import from_import


class genVec(Structure):

    _operator_funcs = {
        "+": (lambda a, b: a + b),
        "-": (lambda a, b: a - b),
        "*": (lambda a, b: a * b),
        "/": (lambda a, b: a / b),
        "//": (lambda a, b: a // b),
        "**": (lambda a, b: a ** b),
        ">": (lambda a, b: a > b),
        ">=": (lambda a, b: a >= b),
        "<": (lambda a, b: a < b),
        "<=": (lambda a, b: a <= b),
        "==": (lambda a, b: a == b),
        "!=": (lambda a, b: a != b),
    }

    def __init__(self, *args)->None:
        values = []
        for arg in args:
            if hasattr(arg, '__len__'):
                values.extend(arg)
            else:
                values.append(arg)

        if len(values) == 0:
            values = [0] * len(self)

        elif len(values) == 1:
            values = values * len(self)

        if len(values) != len(self):
            raise ValueError(f"{self.__class__.__name__}.__init__ accepts {len(self)} values, got {len(values)}")

        Structure.__init__(self, *values)

    def __getattr__(self, name:str):
        processed_name = name
        if name.startswith('s') and len(name) > 1:
            processed_name = name[1:]

        if len(processed_name) not in [1, 2, 3, 4, 8, 16]:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
        values = []
        for char in processed_name:
            if char not in self.swizzle_map:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            values.append(self[self.swizzle_map[char]])
        
        base_name = self.__class__.__name__[:-1]
        if self.__class__.__name__.endswith('16'):
            base_name = self.__class__.__name__[:-2]
        
        if len(values) == 1:
            return values[0]
        
        type_name = base_name + str(len(values))
        vec_type = from_import(f".{type_name}", type_name)
        return vec_type(*values)
            
    def __setattr__(self, name:str, value):
        processed_name = name
        if name.startswith('s') and len(name) > 1:
            processed_name = name[1:]

        if len(processed_name) not in [1, 2, 3, 4, 8, 16]:
            super().__setattr__(name, value)
            return
            
        indices = []
        for char in processed_name:
            if char not in self.swizzle_map:
                super().__setattr__(name, value)
                return
            
            if self.swizzle_map[char] in indices:
                raise AttributeError(f"property '{name}' of '{self.__class__.__name__}' object has no setter")
            
            indices.append(self.swizzle_map[char])
                
        if hasattr(value, '__len__'):
            if len(value) != len(indices):
                raise ValueError(f"property '{name}' of '{self.__class__.__name__}' object only accepts value with length {len(indices)}, got {len(value)}")
            
            for i, v in zip(indices, value):
                converted_value = self._convert_value_for_field(i, v)
                super().__setattr__(self._fields_[i][0], converted_value)
        else:
            if len(indices) == 1:
                converted_value = self._convert_value_for_field(indices[0], value)
                super().__setattr__(self._fields_[indices[0]][0], converted_value)
            else:
                raise ValueError(f"property '{name}' of '{self.__class__.__name__}' object only accepts value with length {len(indices)}, got scalar value")

    def __getitem__(self, index:int):
        return getattr(self, self._fields_[index][0])
        
    def __setitem__(self, index:int, value):
        converted_value = self._convert_value_for_field(index, value)
        super().__setattr__(self._fields_[index][0], converted_value)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(str(val) for val in self)})"
        
    def __str__(self):
        return self.__repr__()
        
    def _op(self, operator:str, other)->genVec:
        func = self._operator_funcs[operator]
        if isinstance(other, genVec):
            result_type = self._get_larger_type(other)
            result_values = [func(a, b) for a, b in zip(self, other)]
        else:
            result_type = self.__class__
            result_values = [func(a, other) for a in self]
            
        return result_type(*result_values)
    
    def _rop(self, operator:str, other)->genVec:
        func = self._operator_funcs[operator]
        if isinstance(other, genVec):
            result_type = self._get_larger_type(other)
            result_values = [func(b, a) for a, b in zip(self, other)]
        else:
            result_type = self.__class__
            result_values = [func(other, a) for a in self]
            
        return result_type(*result_values)
    
    def _iop(self, operator:str, other)->genVec:
        func = self._operator_funcs[operator]
        if isinstance(other, genVec):
            for i, b in enumerate(other):
                self[i] = func(self[i], b)
        else:
            for i in range(len(self)):
                self[i] = func(self[i], other)
        return self
    
    def _compare_op(self, operator:str, other)->genVec:
        func = self._operator_funcs[operator]
        if isinstance(other, genVec):
            result_values = [int(func(a, b)) for a, b in zip(self, other)]
        else:
            result_values = [int(func(a, other)) for a in self]
            
        int_vec_type = self._get_int_vector_type()
        return int_vec_type(*result_values)

    def __add__(self, other)->genVec:
        return self._op("+", other)
        
    def __radd__(self, other):
        return self._rop("+", other)
        
    def __sub__(self, other):
        return self._op("-", other)
        
    def __rsub__(self, other):
        return self._rop("-", other)
        
    def __mul__(self, other):
        return self._op("*", other)
        
    def __rmul__(self, other):
        return self._rop("*", other)
        
    def __truediv__(self, other):
        return self._op("/", other)
        
    def __rtruediv__(self, other):
        return self._rop("/", other)
        
    def __floordiv__(self, other):
        return self._op("//", other)
        
    def __rfloordiv__(self, other):
        return self._rop("//", other)
        
    def __pow__(self, other):
        return self._op("**", other)
        
    def __rpow__(self, other):
        return self._rop("**", other)
        
    def __neg__(self):
        values = [-a for a in self]
        return self.__class__(*values)
        
    def __pos__(self):
        return self.__class__(*self)
        
    def __invert__(self):
        values = [~a for a in self]
        return self.__class__(*values)
        
    def __eq__(self, other):
        return self._compare_op("==", other)
        
    def __ne__(self, other):
        return self._compare_op("!=", other)
        
    def __lt__(self, other):
        return self._compare_op("<", other)
        
    def __le__(self, other):
        return self._compare_op("<=", other)
        
    def __gt__(self, other):
        return self._compare_op(">", other)
        
    def __ge__(self, other):
        return self._compare_op(">=", other)
        
    def __iadd__(self, other):
        return self._iop("+", other)
        
    def __isub__(self, other):
        return self._iop("-", other)
        
    def __imul__(self, other):
        return self._iop("*", other)
        
    def __itruediv__(self, other):
        return self._iop("/", other)
        
    def __ifloordiv__(self, other):
        return self._iop("//", other)
        
    def __ipow__(self, other):
        return self._iop("**", other)
        
    def __iter__(self):
        return (getattr(self, field[0]) for field in self._fields_ if field[0] != '_')
        
    def _convert_value_for_field(self, index, value):
        field_type = self._fields_[index][1]
        try:
            if isinstance(value, (int, float)):
                if field_type.__name__ in ['c_int32', 'c_int16', 'c_int8', 'c_uint32', 'c_uint16', 'c_uint8', 'c_long', 'c_ulong']:
                    converted = field_type(int(value))
                elif field_type.__name__ in ['c_float', 'c_double']:
                    converted = field_type(float(value))
                else:
                    converted = field_type(value)
                    
                if hasattr(converted, 'value'):
                    return converted.value
                return converted
            else:
                converted = field_type(value)
                if hasattr(converted, 'value'):
                    return converted.value
                return converted
        except (ValueError, TypeError):
            return value
        
    def _get_larger_type(self, other):
        type_priority = {
            'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2,
            'int': 3, 'uint': 3, 'long': 4, 'ulong': 4,
            'float': 5, 'double': 6
        }
        
        self_class_name = self.__class__.__name__
        other_class_name = other.__class__.__name__
        
        self_base_name = ''.join([c for c in self_class_name if not c.isdigit()])
        other_base_name = ''.join([c for c in other_class_name if not c.isdigit()])
        
        self_dim = len(self)
        other_dim = len(other)
        
        if self_dim != other_dim:
            raise ValueError("Cannot operate on vectors with different dimensions")
            
        self_priority = type_priority.get(self_base_name, 0)
        other_priority = type_priority.get(other_base_name, 0)
        
        result_type_name = self_base_name if self_priority >= other_priority else other_base_name
        full_result_type_name = result_type_name + str(self_dim)
        
        return from_import(f".{full_result_type_name}", full_result_type_name)
        
    def _get_division_type(self, other):
        type_priority = {
            'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2,
            'int': 3, 'uint': 3, 'long': 4, 'ulong': 4,
            'float': 5, 'double': 6
        }
        
        self_class_name = self.__class__.__name__
        other_class_name = other.__class__.__name__
        
        self_base_name = ''.join([c for c in self_class_name if not c.isdigit()])
        other_base_name = ''.join([c for c in other_class_name if not c.isdigit()])
        
        self_dim = len(self)
        other_dim = len(other)
        
        if self_dim != other_dim:
            raise ValueError("Cannot operate on vectors with different dimensions")
            
        self_priority = type_priority.get(self_base_name, 0)
        other_priority = type_priority.get(other_base_name, 0)
        
        min_float_type = 'float'
        if self_priority >= 6 or other_priority >= 6:
            min_float_type = 'double'
            
        max_priority = max(self_priority, other_priority, type_priority[min_float_type])
        
        for type_name, priority in type_priority.items():
            if priority == max_priority:
                result_type_name = type_name
                break
        else:
            result_type_name = min_float_type
            
        full_result_type_name = result_type_name + str(self_dim)
        return from_import(f".{full_result_type_name}", full_result_type_name)
        
    def _get_scalar_division_type(self, scalar):
        type_priority = {
            'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2,
            'int': 3, 'uint': 3, 'long': 4, 'ulong': 4,
            'float': 5, 'double': 6
        }
        
        self_class_name = self.__class__.__name__
        self_base_name = ''.join([c for c in self_class_name if not c.isdigit()])
        
        if isinstance(scalar, int):
            scalar_type_name = 'int'
        elif isinstance(scalar, float):
            scalar_type_name = 'float'
        else:
            scalar_class_name = scalar.__class__.__name__
            scalar_type_name = ''.join([c for c in scalar_class_name if not c.isdigit()])
            
        self_priority = type_priority.get(self_base_name, 0)
        scalar_priority = type_priority.get(scalar_type_name, 0)
        
        min_float_type = 'float'
        if self_priority >= 6 or scalar_priority >= 6:
            min_float_type = 'double'
            
        max_priority = max(self_priority, scalar_priority, type_priority[min_float_type])
        
        for type_name, priority in type_priority.items():
            if priority == max_priority:
                result_type_name = type_name
                break
        else:
            result_type_name = min_float_type
            
        full_result_type_name = result_type_name + str(len(self))
        return from_import(f".{full_result_type_name}", full_result_type_name)
        
    def _get_int_vector_type(self):
        class_name = self.__class__.__name__
        dim = ''.join([c for c in class_name if c.isdigit()])
        
        int_type_name = 'int' + dim
        return from_import(f".{int_type_name}", int_type_name)