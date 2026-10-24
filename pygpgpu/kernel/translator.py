from __future__ import annotations

import ast
import hashlib
import inspect
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .builtins import (
    ATOMIC_BUILTINS,
    GENERIC_BUILTINS,
    INDEX_BUILTINS,
    emit_atomic,
    emit_barrier,
    emit_index,
)
from .builtins import gpu as GPU_BUILTINS


class PyKernelSyntaxError(SyntaxError):
    """Python kernel source contains syntax the backend cannot represent."""

    def __init__(self, message:str, node:Optional[ast.AST]=None, filename:str="<kernel>"):
        lineno:int = (getattr(node, "lineno", 0) if node is not None else 0)
        super().__init__(f"{message} (line {lineno})" if lineno else message)
        self.node = node
        self.filename = filename


INT_TYPES:Set[str] = {"bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"}
FLOAT_TYPES:Set[str] = {"float", "double"}
SCALAR_TYPES:Set[str] = INT_TYPES | FLOAT_TYPES

DTYPE_MAP:Dict[str, str] = {
    "bool": "bool",
    "int8": "char", "int16": "short", "int32": "int", "int64": "long",
    "uint8": "uchar", "uint16": "ushort", "uint32": "uint", "uint64": "ulong",
    "float16": "half", "float32": "float", "float64": "double",
}

PTR = "*"  # 类型记法: ("*", base) 表示数组参数,字符串表示标量

_C_FIELD_TYPES:Dict[Any, str] = {}

def _vec_registry()->Dict[str, Any]:
    """vec 类型名 -> ctypes 类(如 float4),惰性构建。"""
    global _VEC_REGISTRY
    try:
        return _VEC_REGISTRY
    except NameError:
        pass
    from ..opencl.driver import CLInfo
    _VEC_REGISTRY = dict(CLInfo.vec_types)
    for name, cls in _VEC_REGISTRY.items():
        for field_name, field_type in cls._fields_:
            _C_FIELD_TYPES[(name, field_name)] = _c_scalar_name(field_type)
    return _VEC_REGISTRY

def _c_scalar_name(ctypes_type:Any)->str:
    import ctypes as _ctypes
    sizes = {_ctypes.c_float: "float", _ctypes.c_double: "double",
             _ctypes.c_int8: "char", _ctypes.c_int16: "short", _ctypes.c_int32: "int",
             _ctypes.c_int64: "long", _ctypes.c_uint8: "uchar", _ctypes.c_uint16: "ushort",
             _ctypes.c_uint32: "uint", _ctypes.c_uint64: "ulong"}
    return sizes.get(ctypes_type, "float")

def is_vec(t:Any)->bool:
    return isinstance(t, str) and t in _vec_registry()

_MATH_FUNCS:Dict[str, Optional[str]] = {
    "sqrt": None, "exp": None, "log": None, "log2": None, "log10": None,
    "sin": None, "cos": None, "tan": None, "asin": None, "acos": None,
    "atan": None, "atan2": None, "sinh": None, "cosh": None, "tanh": None,
    "fabs": None, "pow": None, "hypot": None, "erf": None,
    "floor": "int", "ceil": "int", "trunc": "int",
}

_MATH_CONSTS:Dict[str, str] = {
    "pi": "3.14159265358979323846f",
    "e": "2.71828182845904523536f",
    "tau": "6.28318530717958647692f",
    "inf": "INFINITY",
}


def is_int(t:Any)->bool:
    return isinstance(t, str) and t in INT_TYPES

def is_float(t:Any)->bool:
    return isinstance(t, str) and t in FLOAT_TYPES

def is_ptr(t:Any)->bool:
    return isinstance(t, tuple) and len(t) == 2 and t[0] == PTR

def promote(t1:Any, t2:Any)->Any:
    if is_ptr(t1) or is_ptr(t2):
        raise PyKernelSyntaxError("cannot mix arrays with scalars in arithmetic")
    if is_vec(t1) or is_vec(t2):
        raise PyKernelSyntaxError("vector arithmetic is handled separately")
    if "double" in (t1, t2):
        return "double"
    if is_float(t1) or is_float(t2):
        return "float"
    rank:List[str] = ["bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"]
    if t1 in rank and t2 in rank:
        return t1 if rank.index(t1) >= rank.index(t2) else t2
    return "int"

def infer_ctype(value:Any)->Any:
    """调用时实参 -> C 类型。"""
    registry:Dict[str, Any] = _vec_registry()
    dtype_to_vec:Dict[Any, str] = {cls.dtype: name for name, cls in registry.items()}

    if isinstance(value, np.void):
        vec_name:str = dtype_to_vec.get(value.dtype)
        if vec_name is None:
            raise PyKernelSyntaxError(f"unsupported numpy structured scalar: {value.dtype}")
        return vec_name

    for vec_name, cls in registry.items():
        if isinstance(value, cls):
            return vec_name

    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None:
            vec_name = dtype_to_vec.get(value.dtype)
            if vec_name is not None:
                return (PTR, vec_name)
            raise PyKernelSyntaxError(f"unsupported numpy structured dtype: {value.dtype}")
        if value.dtype.name not in DTYPE_MAP:
            raise PyKernelSyntaxError(f"unsupported numpy dtype: {value.dtype.name}")
        return (PTR, DTYPE_MAP[value.dtype.name])
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    raise PyKernelSyntaxError(f"unsupported argument type: {type(value).__name__} (only numpy arrays and int/float/bool scalars are supported)")

def type_sig(types:List[Any])->str:
    parts:List[str] = []
    for t in types:
        parts.append((t[1] + "[]") if is_ptr(t) else str(t))
    return "_".join(parts) or "void"

def short_hash(text:str)->str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


class FunctionTranslator:
    """将单个 Python 函数翻译为 C 函数(kernel 或设备函数)。

    is_kernel=True 时函数体最外层的
    `for i in range(<array>.shape[0])` 循环会被翻译为 work-item。
    """

    def __init__(self, func:Callable, param_types:List[Any], backend:str,
                 resolve_global:Callable[[str], Any], device_cache:Dict[Any, Any],
                 is_kernel:bool=False):
        self.func:Callable = func
        self.filename:str = getattr(getattr(func, "__code__", None), "co_filename", "<kernel>")
        self.backend:str = backend
        self.resolve_global:Callable[[str], Any] = resolve_global
        self.device_cache:Dict[Any, Any] = device_cache
        self.is_kernel:bool = is_kernel

        try:
            tree:ast.Module = ast.parse(self._trim_source(), mode="exec")
            if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
                raise PyKernelSyntaxError("decorated object must be a plain function")
            self.tree:ast.FunctionDef = tree.body[0]
        except SyntaxError as e:
            raise PyKernelSyntaxError(f"cannot parse function source: {e}") from e

        self.params:List[Tuple[str, Any]] = []
        self.shapes_used:List[Tuple[str, int]] = []
        self.gid_param:Optional[str] = None
        self.work_size_dims:List[Tuple[str, int]] = []
        self.atomic_targets:Set[str] = set()
        self.return_type:Optional[Any] = None
        self.lines:List[str] = []
        self._indent:int = 1
        self.scopes:List[Dict[str, Any]] = []

        args:List[ast.arg] = self.tree.args.args
        if self.tree.args.vararg or self.tree.args.kwarg or self.tree.args.kwonlyargs or self.tree.args.defaults:
            raise PyKernelSyntaxError("*args/**kwargs/keyword-only args/defaults are not supported", self.tree.args)
        if len(args) != len(param_types):
            raise PyKernelSyntaxError(f"{func.__name__}() takes {len(args)} arguments but {len(param_types)} were given")

        scope:Dict[str, Any] = {}
        for arg, t in zip(args, param_types):
            if arg.annotation is not None:
                raise PyKernelSyntaxError("type annotations are not needed, remove them", arg)
            scope[arg.arg] = t
            self.params.append((arg.arg, t))
        self.scopes.append(scope)

    # ---------- 基础工具 ----------

    def _error(self, message:str, node:Optional[ast.AST]=None)->Any:
        raise PyKernelSyntaxError(message, node, self.filename)

    def _emit(self, line:str)->None:
        self.lines.append("    " * self._indent + line)

    def _lookup(self, name:str)->Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _define(self, name:str, t:Any, node:ast.AST)->None:
        existing:Any = self._lookup(name)
        if existing is None:
            self.scopes[-1][name] = t
        elif existing != t:
            if is_int(existing) and is_float(t):
                self.scopes[-1][name] = t
            else:
                self._error(f"variable '{name}' changes type from {existing} to {t}", node)

    def _shape_var(self, array_name:str, dim:int)->str:
        key = (array_name, dim)
        if key not in self.shapes_used:
            self.shapes_used.append(key)
        return f"_{array_name}_shape{dim}"

    def _trim_source(self)->str:
        try:
            source:str = inspect.getsource(self.func)
        except OSError as e:
            raise PyKernelSyntaxError(f"cannot read source of {self.func.__name__}: {e}") from e
        lines:List[str] = source.split("\n")
        start:int = next((i for i, line in enumerate(lines) if line.lstrip().startswith("def ")), 0)
        return "\n".join(lines[start:])

    # ---------- 翻译入口 ----------

    def translate(self)->None:
        body:List[ast.stmt] = self.tree.body
        if self.is_kernel:
            self._translate_magic_loop(body)
        else:
            for stmt in body:
                self._visit_statement(stmt)
            if self.return_type is None:
                self.return_type = "void"

    @staticmethod
    def _is_shape0_range(iter_node:ast.expr)->bool:
        return (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "range" and len(iter_node.args) == 1 and not iter_node.keywords
                and isinstance(iter_node.args[0], ast.Subscript)
                and isinstance(iter_node.args[0].value, ast.Attribute)
                and iter_node.args[0].value.attr == "shape"
                and isinstance(iter_node.args[0].value.value, ast.Name)
                and isinstance(iter_node.args[0].slice, ast.Constant)
                and iter_node.args[0].slice.value == 0)

    def _translate_magic_loop(self, body:List[ast.stmt])->None:
        magic_index:Optional[int] = None
        for idx, stmt in enumerate(body):
            if isinstance(stmt, ast.For) and self._is_shape0_range(stmt.iter):
                magic_index = idx
                break

        if magic_index is None:
            # 无魔法循环:用户通过 gpu.get_global_id 等内建自行索引,
            # work size 由调用方 global_size 或输出数组形状自动推断
            for stmt in body:
                self._visit_statement(stmt)
            return

        # 魔法循环之前的语句照常翻译(如局部变量初始化)
        for stmt in body[:magic_index]:
            self._visit_statement(stmt)

        loop:ast.For = body[magic_index]
        range_call:ast.Call = loop.iter
        bound:ast.expr = range_call.args[0]

        array_name:str = bound.value.value.id
        if array_name not in {n for n, _ in self.params} or not is_ptr(self._lookup(array_name)):
            self._error(f"'{array_name}' is not an array parameter of this kernel", bound)

        self.gid_param = array_name

        # 2D 模式: 外层 range(arr.shape[0]) 内嵌 range(arr.shape[1]) -> 二维 NDRange
        inner_loop:Optional[ast.For] = None
        if (len(loop.body) == 1 and isinstance(loop.body[0], ast.For)
                and isinstance(loop.body[0].iter, ast.Call)
                and isinstance(loop.body[0].iter.func, ast.Name) and loop.body[0].iter.func.id == "range"
                and len(loop.body[0].iter.args) == 1
                and isinstance(loop.body[0].iter.args[0], ast.Subscript)
                and isinstance(loop.body[0].iter.args[0].value, ast.Attribute)
                and loop.body[0].iter.args[0].value.attr == "shape"
                and isinstance(loop.body[0].iter.args[0].value.value, ast.Name)
                and loop.body[0].iter.args[0].value.value.id == array_name
                and isinstance(loop.body[0].iter.args[0].slice, ast.Constant)
                and loop.body[0].iter.args[0].slice.value == 1):
            inner_loop = loop.body[0]

        def _emit_gid(var:str, dim:int, component:int)->None:
            self._shape_var(array_name, dim)
            if self.backend == "opencl":
                self._emit(f"const int {var} = get_global_id({component});")
            else:
                component_expr:str = (".x" if component == 0 else ".y" if component == 1 else ".z")
                self._emit(f"const int {var} = (int)(blockIdx{component_expr} * blockDim{component_expr} + threadIdx{component_expr});")

        if inner_loop is None:
            gid_bound:str = self._shape_var(array_name, 0)
            target:str = loop.target.id
            self._define(target, "int", loop.target)
            _emit_gid(target, 0, 0)
            self._emit(f"if ({target} >= {gid_bound}) return;")

            # 循环体即 work-item 主体
            for stmt in loop.body:
                self._visit_statement(stmt)
        else:
            outer_target:str = loop.target.id
            inner_target:str = inner_loop.target.id
            outer_bound:str = self._shape_var(array_name, 0)
            inner_bound:str = self._shape_var(array_name, 1)
            self._define(outer_target, "int", loop.target)
            self._define(inner_target, "int", inner_loop.target)
            # gid0 对应最内层(行主序的 x), gid1 对应外层(y)
            _emit_gid(inner_target, 1, 0)
            _emit_gid(outer_target, 0, 1)
            self._emit(f"if ({outer_target} >= {outer_bound} || {inner_target} >= {inner_bound}) return;")

            for stmt in inner_loop.body:
                self._visit_statement(stmt)

        # work size 来源: 1D -> (shape0); 2D -> (shape1, shape0)
        self.work_size_dims:List[Tuple[str, int]] = (
            [(array_name, 1), (array_name, 0)] if inner_loop is not None else [(array_name, 0)])

    # ---------- 语句 ----------

    def _visit_statement(self, node:ast.stmt)->None:
        if isinstance(node, ast.Assign):
            self._visit_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_augassign(node)
        elif isinstance(node, ast.If):
            cond, _ = self._visit_expression(node.test)
            self._emit(f"if ({cond}) {{")
            self._push_scope_block(node.body)
            if node.orelse:
                self._emit("} else {")
                self._push_scope_block(node.orelse)
            self._emit("}")
        elif isinstance(node, ast.While):
            cond, _ = self._visit_expression(node.test)
            self._emit(f"while ({cond}) {{")
            self._push_scope_block(node.body)
            self._emit("}")
        elif isinstance(node, ast.For):
            self._visit_for(node)
        elif isinstance(node, ast.Return):
            self._visit_return(node)
        elif isinstance(node, ast.Expr):
            # 表达式语句(如 gpu.atomic_add(...)/gpu.barrier())必须 emit,否则副作用丢失
            code, _ = self._visit_expression(node.value)
            if code:
                self._emit(f"{code};")
        elif isinstance(node, ast.Break):
            self._emit("break;")
        elif isinstance(node, ast.Continue):
            self._emit("continue;")
        elif isinstance(node, ast.Pass):
            pass
        else:
            self._error(f"unsupported statement: {node.__class__.__name__}", node)

    def _push_scope_block(self, statements:List[ast.stmt])->None:
        self.scopes.append({})
        self._indent += 1
        for stmt in statements:
            self._visit_statement(stmt)
        self._indent -= 1
        self.scopes.pop()

    def _visit_for(self, node:ast.For)->None:
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            self._error("only `for ... in range(...)` loops are supported", node.iter)
        args:List[ast.expr] = node.iter.args
        if len(args) not in (1, 2, 3) or node.iter.keywords:
            self._error("range() takes 1-3 positional arguments", node.iter)
        if not isinstance(node.target, ast.Name):
            self._error("loop target must be a plain variable", node.target)

        values:List[Tuple[str, Any]] = [self._visit_expression(a) for a in args]
        for a, (_, t) in zip(args, values):
            if not is_int(t):
                self._error("range() arguments must be integers", a)
        target:str = node.target.id
        self._define(target, "int", node.target)

        if len(values) == 1:
            start, stop, step = "0", values[0][0], None
        elif len(values) == 2:
            start, stop, step = values[0][0], values[1][0], None
        else:
            start, stop, step = values[0][0], values[1][0], values[2][0]

        if step is None:
            self._emit(f"for (int {target} = {start}; {target} < {stop}; {target}++) {{")
        elif re.fullmatch(r"-?\d+", step):
            cond_op:str = (">" if step.startswith("-") else "<")
            self._emit(f"for (int {target} = {start}; {target} {cond_op} {stop}; {target} += {step}) {{")
        else:
            self._error("range() step must be a constant integer", node.iter.args[-1])
        self._push_scope_block(node.body)
        self._emit("}")

    def _visit_return(self, node:ast.Return)->None:
        if node.value is None:
            self.return_type = self.return_type or "void"
            self._emit("return;")
            return

        if self.is_kernel:
            self._error("a @kernel function must not return values; write results into an array argument instead", node)

        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            self._error("cannot return an array from a function", node)

        if self.return_type is None:
            self.return_type = t
        elif self.return_type != t:
            unified:Any = promote(self.return_type, t)
            if unified != self.return_type:
                code = f"(({self.return_type})({code}))"
            self.return_type = unified
        self._emit(f"return {code};")

    def _visit_assign(self, node:ast.Assign)->None:
        if len(node.targets) != 1:
            self._error("chained assignment is not supported", node)
        target:ast.expr = node.targets[0]

        # 元组解包/交换: a, b = b, a
        if isinstance(target, ast.Tuple):
            if not isinstance(node.value, ast.Tuple) or len(target.elts) != len(node.value.elts):
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not all(isinstance(t, ast.Name) for t in target.elts):
                self._error("tuple assignment targets must be plain variables", node)

            values:List[Tuple[str, Any]] = [self._visit_expression(v) for v in node.value.elts]
            temps:List[str] = []
            for i, (code, t) in enumerate(values):
                if is_ptr(t):
                    self._error("cannot swap array parameters", node)
                tmp:str = f"_swap{i}"
                self._define(tmp, t, node)
                self._emit(f"{t} {tmp} = {code};")
                temps.append((tmp, t))

            for (tmp, t), target_item in zip(temps, target.elts):
                name:str = target_item.id
                existing:Any = self._lookup(name)
                if existing is not None and is_ptr(existing):
                    self._error(f"cannot assign to array parameter '{name}'", node)
                self._define(name, t, node)
                self._emit(f"{name} = {tmp};")
            return
        code, t = self._visit_expression(node.value)
        if is_ptr(t):
            self._error("cannot assign an array to a variable", node)

        if isinstance(target, ast.Name):
            existing:Any = self._lookup(target.id)
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{target.id}'", node)
            is_new:bool = (existing is None)
            self._define(target.id, t, node)
            decl:str = ""
            if is_new:
                decl = {"bool": "bool ", "float": "float ", "double": "double "}.get(t, f"{t} ")
            self._emit(f"{decl}{target.id} = {code};")
        elif isinstance(target, ast.Subscript):
            base_code, base_t = self._visit_array_target(target.value)
            index_code, _ = self._visit_subscript_index(target)
            self._emit(f"{base_code}[{index_code}] = {code};")
        else:
            self._error(f"unsupported assignment target: {target.__class__.__name__}", node)

    def _visit_augassign(self, node:ast.AugAssign)->None:
        target_code, target_t = self._visit_target(node.target)
        value_code, value_t = self._visit_expression(node.value)
        if isinstance(node.target, ast.Subscript) and not is_int(self._lookup_type_of_index(node)):
            pass

        name:str = node.op.__class__.__name__
        if name == "Div":
            if is_int(target_t) and is_int(value_t):
                self._error("integer /= integer changes type (use // or float operands)", node)
            self._emit(f"{target_code} /= {value_code};")
        elif name == "FloorDiv":
            if is_int(target_t) and is_int(value_t):
                self._emit(f"{target_code} /= {value_code};")
            else:
                self._emit(f"{target_code} = floor(({target_t})({target_code}) / ({target_t})({value_code}));")
        elif name == "Pow":
            self._error("**= is not supported, use pow()", node)
        elif name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor", "Add", "Sub", "Mult", "Mod"):
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^",
                          "Add": "+", "Sub": "-", "Mult": "*", "Mod": "%"}[name]
            self._emit(f"{target_code} {symbol}= {value_code};")
        else:
            self._error(f"unsupported operator: {name}", node)

    def _lookup_type_of_index(self, node:ast.AugAssign)->Any:
        return "int"

    def _visit_target(self, target:ast.expr)->Tuple[str, Any]:
        if isinstance(target, ast.Name):
            t:Any = self._lookup(target.id)
            if t is None:
                self._error(f"variable '{target.id}' used before assignment", target)
            if is_ptr(t):
                self._error(f"cannot assign to array parameter '{target.id}'", target)
            return target.id, t
        if isinstance(target, ast.Subscript):
            base_code, base_t = self._visit_array_target(target.value)
            index_code, _ = self._visit_subscript_index(target)
            return f"{base_code}[{index_code}]", base_t
        self._error(f"unsupported assignment target: {target.__class__.__name__}", target)

    def _visit_array_target(self, node:ast.expr)->Tuple[str, Any]:
        code, t = self._visit_expression(node)
        if not is_ptr(t):
            self._error("only array (numpy) arguments can be indexed", node)
        return code, t[1]

    def _visit_subscript_index(self, node:ast.Subscript)->Tuple[str, Any]:
        index:ast.expr = node.slice
        if isinstance(index, ast.Tuple):
            array_name:str = self._array_name(node.value)
            dims:List[Tuple[str, Any]] = [self._visit_expression(e) for e in index.elts]
            terms:List[str] = []
            for d, (code, t) in enumerate(dims):
                if not is_int(t):
                    self._error("array indices must be integers", node)
                term:str = code
                for inner in range(d + 1, len(dims)):
                    term = f"{term} * {self._shape_var(array_name, inner)}"
                terms.append(term)
            return " + ".join(terms), "int"
        code, t = self._visit_expression(index)
        if not is_int(t):
            self._error("array index must be an integer", node)
        return code, t

    def _array_name(self, node:ast.expr)->str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self._array_name(node.value)
        self._error("unsupported array expression", node)

    # ---------- 表达式 ----------

    def _visit_expression(self, node:ast.expr)->Tuple[str, Any]:
        if isinstance(node, ast.Constant):
            return self._visit_constant(node)
        if isinstance(node, ast.Name):
            return self._visit_name(node)
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._visit_compare(node)
        if isinstance(node, ast.BoolOp):
            parts:List[str] = [self._visit_expression(v)[0] for v in node.values]
            symbol:str = ("&&" if isinstance(node.op, ast.And) else "||")
            return f"({' ' + symbol + ' '.join(parts)})", "bool"
        if isinstance(node, ast.IfExp):
            cond, _ = self._visit_expression(node.test)
            then_code, tt = self._visit_expression(node.body)
            else_code, et = self._visit_expression(node.orelse)
            if tt != et:
                unified:Any = promote(tt, et)
                return f"({cond} ? ({unified})({then_code}) : ({unified})({else_code}))", unified
            return f"({cond} ? {then_code} : {else_code})", tt
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        self._error(f"unsupported expression: {node.__class__.__name__}", node)

    def _visit_constant(self, node:ast.Constant)->Tuple[str, Any]:
        value:Any = node.value
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            text:str = repr(value)
            if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
                text += ".0"
            if not text.endswith(("f", "F", "y")) and "inf" not in text:
                text += "f"
            return text, "float"
        self._error(f"unsupported constant: {value!r}", node)

    def _visit_name(self, node:ast.Name)->Tuple[str, Any]:
        t:Any = self._lookup(node.id)
        if t is not None:
            return node.id, t
        value:Any = self.resolve_global(node.id)
        if isinstance(value, bool):
            return self._visit_constant(ast.Constant(value=value))
        if isinstance(value, (int, float)) and not isinstance(value, np.ndarray):
            return self._visit_constant(ast.Constant(value=value))
        self._error(f"unknown name '{node.id}' (only parameters, local variables and int/float/bool global constants are supported)", node)

    def _vec_componentwise(self, op:str, left:str, vec:str, right:str="", scalar:str="")->str:
        """CUDA 没有向量算术/数学重载,展开为逐分量。

        op 为中缀符号(+ - * /)时生成 `l.c op r.c`;为函数名(sqrt 等)时生成 `op(l.c, r.c)`。
        right 为向量操作数,scalar 为广播标量操作数。
        """
        cls:Any = _vec_registry()[vec]
        components:List[str] = [field_name for field_name, _ in cls._fields_]
        maker:str = f"make_{vec}"
        infix:bool = (op in ("+", "-", "*", "/"))
        terms:List[str] = []
        for c in components:
            if scalar:
                terms.append(f"({left}.{c} {op} {scalar})" if infix else f"{op}({left}.{c}, {scalar})")
            elif right:
                terms.append(f"({left}.{c} {op} {right}.{c})" if infix else f"{op}({left}.{c}, {right}.{c})")
            else:
                terms.append(f"({op}{left}.{c})" if infix else f"{op}({left}.{c})")
        return f"{maker}({', '.join(terms)})"

    def _visit_binop(self, node:ast.BinOp)->Tuple[str, Any]:
        left, lt = self._visit_expression(node.left)
        right, rt = self._visit_expression(node.right)
        name:str = node.op.__class__.__name__

        if is_vec(lt) or is_vec(rt):
            symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}.get(name)
            if symbol is None:
                self._error(f"operator {name} is not supported on vector types; operate per-component (v.x + v.y)", node)

            if is_vec(lt) and is_vec(rt):
                if lt != rt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                if self.backend == "opencl":
                    return f"({left} {symbol} {right})", lt
                return self._vec_componentwise(symbol, left, lt, right), lt

            # 向量与标量: 广播逐分量
            vec, vec_side_left = (lt, True) if is_vec(lt) else (rt, False)
            if self.backend == "opencl":
                code:str = f"({left} {symbol} {right})" if vec_side_left else f"({right} {symbol} {left})"
                return code, vec
            if vec_side_left:
                return self._vec_componentwise(symbol, left, vec, scalar=right), vec
            return self._vec_componentwise(symbol, right, vec, scalar=left), vec

        if name == "Div":
            t:Any = promote(lt, rt) if not (is_int(lt) and is_int(rt)) else "float"
            if is_int(lt) and is_int(rt):
                return f"((({t})({left})) / (({t})({right})))", t
            return f"({left} / {right})", t

        if name == "FloorDiv":
            if is_int(lt) and is_int(rt):
                return f"({left} / {right})", promote(lt, rt)
            t:Any = promote(lt, rt)
            return f"(({t})floor(({t})({left}) / ({t})({right})))", t

        if name == "Pow":
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value == 2:
                return f"({left} * {left})", lt
            t:Any = promote(lt, rt)
            if is_int(t):
                t = "float"
            return f"(({t})pow(({t})({left}), ({t})({right})))", t

        if name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor"):
            if not (is_int(lt) and is_int(rt)):
                self._error(f"operator {name} requires integer operands", node)
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^"}[name]
            return f"({left} {symbol} {right})", promote(lt, rt)

        if name == "Mod":
            t:Any = promote(lt, rt)
            if is_int(t):
                return f"({left} % {right})", t
            return f"fmod(({t})({left}), ({t})({right}))", t

        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*"}.get(name)
        if symbol is None:
            self._error(f"unsupported operator: {name}", node)
        return f"({left} {symbol} {right})", promote(lt, rt)

    def _visit_unaryop(self, node:ast.UnaryOp)->Tuple[str, Any]:
        code, t = self._visit_expression(node.operand)
        name:str = node.op.__class__.__name__
        if name == "USub":
            return f"(-{code})", t
        if name == "UAdd":
            return f"(+{code})", t
        if name == "Invert":
            if not is_int(t):
                self._error("~ requires an integer operand", node)
            return f"(~{code})", t
        if name == "Not":
            return f"(!{code})", "bool"
        self._error(f"unsupported unary operator: {name}", node)

    def _visit_compare(self, node:ast.Compare)->Tuple[str, Any]:
        symbols:Dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=", "Eq": "==", "NotEq": "!="}
        parts:List[str] = []
        left, _ = self._visit_expression(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            name:str = op.__class__.__name__
            if name not in symbols:
                self._error(f"unsupported comparison: {name}", op)
            right, _ = self._visit_expression(comparator)
            parts.append(f"({left} {symbols[name]} {right})")
            left = right
        code = (" && ".join(parts)) if len(parts) > 1 else parts[0]
        return f"({code})", "bool"

    def _visit_subscript(self, node:ast.Subscript)->Tuple[str, Any]:
        value:ast.expr = node.value
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            array_name:str = self._array_name(value.value)
            t:Any = self._lookup(array_name)
            if not is_ptr(t):
                self._error(f"'.shape' is only available on array parameters", node)
            index:ast.expr = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                return self._shape_var(array_name, index.value), "int"
            self._error("only `<array>.shape[<constant>]` is supported", node)

        base_code, base_t = self._visit_array_target(value)
        index_code, _ = self._visit_subscript_index(node)
        return f"{base_code}[{index_code}]", base_t

    def _visit_attribute(self, node:ast.Attribute)->Tuple[str, Any]:
        if isinstance(node.value, ast.Name):
            module:Any = self.resolve_global(node.value.id)
            if module is math or module is np:
                if node.attr in _MATH_CONSTS:
                    return _MATH_CONSTS[node.attr], "float"
                self._error(f"{node.value.id}.{node.attr} is not supported", node)

        base_code, base_t = self._visit_expression(node.value)
        if is_vec(base_t):
            field_key:(str, str) = (base_t, node.attr)
            if field_key not in _C_FIELD_TYPES:
                self._error(
                    f"vector type {base_t} has no component '.{node.attr}' "
                    f"(only single components like .x/.y/.z/.w; swizzles are not supported)", node)
            return f"({base_code}.{node.attr})", _C_FIELD_TYPES[field_key]
        self._error(f"unsupported attribute access: .{node.attr}", node)

    # ---------- 调用 ----------

    def _visit_call(self, node:ast.Call)->Tuple[str, Any]:
        func:ast.expr = node.func
        if node.keywords:
            self._error("keyword arguments are not supported in kernel code", node)

        if isinstance(func, ast.Name):
            name:str = func.id
            args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
            arg_codes:List[str] = [c for c, _ in args]

            if name in ("int", "float", "bool"):
                if len(args) != 1:
                    self._error(f"{name}() takes exactly 1 argument", node)
                t:Any = {"int": "int", "float": "float", "bool": "bool"}[name]
                return f"(({t})({arg_codes[0]}))", t

            if name in ("abs", "min", "max", "round"):
                return self._emit_builtin(name, args, node)

            if name in _vec_registry():
                return self._emit_vec_constructor(name, args, node)

            target:Any = self.resolve_global(name)
            if callable(target):
                return self._emit_device_call(name, target, args, node)
            self._error(f"unknown function '{name}'", node)

        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module:Any = self.resolve_global(func.value.id)
            if module is GPU_BUILTINS:
                args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
                return self._emit_gpu_builtin(func.attr, args, node)
            if module is math or module is np:
                args:List[Tuple[str, Any]] = [self._visit_expression(a) for a in node.args]
                return self._emit_math_call(func.attr, args, node, module_name=func.value.id)
            self._error(f"unsupported method call: .{func.attr}", node)

        self._error("unsupported call expression", node)

    def _emit_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        if name == "abs":
            if len(args) != 1:
                self._error("abs() takes exactly 1 argument", node)
            code, t = args[0]
            if is_float(t):
                return f"fabs({code})", t
            if is_int(t):
                return f"abs({code})", t
            self._error("abs() requires a numeric argument", node)

        if name in ("min", "max"):
            if len(args) != 2:
                self._error(f"{name}() takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            t:Any = promote(lt, rt)
            if is_float(t):
                return f"{'fmin' if name == 'min' else 'fmax'}(({t})({lcode}), ({t})({rcode}))", t
            return f"{'min' if name == 'min' else 'max'}(({t})({lcode}), ({t})({rcode}))", t

        if name == "round":
            if len(args) != 1:
                self._error("round() takes exactly 1 argument", node)
            code, t = args[0]
            if is_float(t):
                return f"((int)rint(({t})({code})))", "int"
            return f"({code})", t

        self._error(f"unsupported builtin: {name}", node)

    def _emit_math_call(self, attr:str, args:List[Tuple[str, Any]], node:ast.Call, module_name:str="math")->Tuple[str, Any]:
        if attr not in _MATH_FUNCS:
            self._error(f"{module_name}.{attr} is not supported", node)
        ret_override:Optional[str] = _MATH_FUNCS[attr]
        arg_codes:List[str] = []
        ret:Any = (ret_override or "float")
        vec_arg:Optional[str] = None
        for i, (code, t) in enumerate(args):
            if is_ptr(t):
                # python 语义: math.* 只接受标量; np.*(arr) 是宿主端逐元素运算,
                # 返回新数组, 无法在 work-item 内核中表示 -> 要求索引到元素
                arr_desc:str = (node.args[i].id if isinstance(node.args[i], ast.Name) else "the argument")
                self._error(
                    f"{module_name}.{attr}() cannot be applied to the array '{arr_desc}': "
                    f"in python, {module_name}.{attr}(arr) would be an element-wise host operation "
                    f"returning a new array, which a per-work-item kernel cannot represent. "
                    f"Index the element first, e.g. {module_name}.{attr}(arr[i]).", node)
            if is_vec(t):
                vec_arg = (vec_arg or t)
            if ret_override is None and t == "double":
                ret = "double"

        if vec_arg is not None:
            # python 语义: math/np 函数不接受向量结构体(np.sqrt(float4_value) 会 TypeError)。
            # OpenCL 原生支持逐分量向量数学; CUDA 通过逐分量展开实现。
            if ret_override is not None:
                self._error(
                    f"{module_name}.{attr}() on vector type {vec_arg} is not supported "
                    f"(it returns {ret_override} for scalars; apply it per-component, e.g. "
                    f"{attr}(v.x))", node)
            if len(args) != 1:
                self._error(f"{module_name}.{attr}() on vectors takes exactly 1 vector argument", node)
            if self.backend == "opencl":
                return f"{attr}({args[0][0]})", vec_arg
            return self._vec_componentwise(attr, args[0][0], vec_arg), vec_arg

        for (code, t) in args:
            arg_codes.append(code if (t == ret or (is_float(t) and is_float(ret))) else f"(({ret})({code}))")
        c_func:str = attr
        return f"{c_func}({', '.join(arg_codes)})", ret

    def _emit_vec_constructor(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        cls:Any = _vec_registry()[name]
        expected:int = len(cls._fields_)
        if len(args) != expected:
            self._error(f"{name}() takes exactly {expected} component arguments ({expected} were required, {len(args)} given)", node)
        codes:List[str] = []
        for i, (code, t) in enumerate(args):
            if is_ptr(t) or is_vec(t):
                self._error(f"{name}() components must be scalars", node)
            field_type:str = _C_FIELD_TYPES[(name, cls._fields_[i][0])]
            codes.append(code if t == field_type else f"(({field_type})({code}))")
        if self.backend == "opencl":
            return f"({name})({', '.join(codes)})", name
        return f"make_{name}({', '.join(codes)})", name

    # ---------- gpu.* 设备内建 ----------

    def _emit_gpu_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        if name in INDEX_BUILTINS:
            if len(args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, int):
                self._error(f"gpu.{name}(dim) requires a constant integer dimension (0, 1 or 2)", node)
            return emit_index(name, node.args[0].value, self.backend), "int"

        if name == "barrier":
            if args:
                self._error("gpu.barrier() takes no arguments", node)
            return emit_barrier(self.backend), "void"

        if name in ATOMIC_BUILTINS:
            if len(args) != 3:
                self._error(f"gpu.{name}(array, index, value) takes exactly 3 arguments", node)
            (arr_code, arr_t), (idx_code, idx_t), (val_code, val_t) = args
            if not is_ptr(arr_t):
                self._error(f"gpu.{name}() first argument must be an array", node)
            if isinstance(node.args[0], ast.Name):
                # 原子操作通过指针强转写内存,静态分析无法识别该模式,
                # 记录目标参数,由 PyKernel 在编译后强制标记为可写
                self.atomic_targets.add(node.args[0].id)
            if arr_t[1] not in ("int", "uint", "long", "ulong"):
                self._error(f"gpu.{name}() only supports integer arrays, got {arr_t[1]}", node)
            if not is_int(idx_t) or not is_int(val_t):
                self._error(f"gpu.{name}() index and value must be integers", node)
            return emit_atomic(name, arr_code, idx_code, val_code, arr_t[1], self.backend), arr_t[1]

        if name in GENERIC_BUILTINS:
            return self._emit_generic_builtin(name, args, node)

        self._error(f"unknown gpu builtin: gpu.{name}", node)

    def _emit_generic_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        arg_codes:List[str] = [c for c, _ in args]
        arg_types:List[Any] = [t for _, t in args]
        vec_t:Any = next((t for t in arg_types if is_vec(t)), None)

        if name in ("rsqrt", "exp2"):
            if len(args) != 1:
                self._error(f"gpu.{name}() takes exactly 1 argument", node)
            code, t = args[0]
            if is_vec(t):
                if self.backend == "opencl":
                    return f"{name}({code})", t
                return self._vec_componentwise(name, code, t), t
            code = f"{GENERIC_BUILTINS[name][1]}({code})" if self.backend == "cuda" else f"{name}({code})"
            return code, t

        if name == "clamp":
            if len(args) != 3:
                self._error("gpu.clamp(a, lo, hi) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self.backend == "opencl":
                    return f"clamp({', '.join(arg_codes)})", vec_t
                return self._emit_vec_clamp(arg_codes, vec_t), vec_t
            if self.backend == "opencl":
                return f"clamp({', '.join(arg_codes)})", args[0][1]
            return f"fminf(fmaxf({arg_codes[0]}, {arg_codes[1]}), {arg_codes[2]})", args[0][1]

        if name == "mix":
            if len(args) != 3:
                self._error("gpu.mix(a, b, t) takes exactly 3 arguments", node)
            if vec_t is not None:
                if self.backend == "opencl":
                    return f"mix({', '.join(arg_codes)})", vec_t
                return self._emit_vec_mix(arg_codes, vec_t), vec_t
            a, b, t = arg_codes
            return f"(({a}) * (1.0f - ({t})) + ({b}) * ({t}))", args[0][1]

        if name == "mad":
            if len(args) != 3:
                self._error("gpu.mad(a, b, c) takes exactly 3 arguments", node)
            if vec_t is not None and self.backend == "opencl":
                return f"mad({', '.join(arg_codes)})", vec_t
            a, b, c = arg_codes
            return f"((({a}) * ({b})) + ({c}))", args[0][1]

        if name == "dot":
            if len(args) != 2:
                self._error("gpu.dot(a, b) takes exactly 2 arguments", node)
            (lcode, lt), (rcode, rt) = args
            if is_vec(lt) and is_vec(rt):
                if lt != rt:
                    self._error(f"gpu.dot() operands must have the same type, got {lt} and {rt}", node)
                if self.backend == "opencl":
                    return f"dot({lcode}, {rcode})", "float"
                return self._emit_vec_dot(lcode, rcode, lt), "float"
            if not (self._is_scalar(lt) and self._is_scalar(rt)):
                self._error("gpu.dot() requires vectors or scalars", node)
            return f"(({lcode}) * ({rcode}))", promote(lt, rt)

        if name in ("cross", "normalize", "length", "distance"):
            return self._emit_vec_builtin(name, args, node)

        self._error(f"unknown gpu builtin: gpu.{name}", node)

    def _is_scalar(self, t:Any)->bool:
        return isinstance(t, str) and not is_vec(t)

    def _emit_vec_clamp(self, codes:List[str], vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = [f_name for f_name, _ in cls._fields_]
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"fminf(fmaxf({codes[0]}.{c}, {codes[1]}.{c}), {codes[2]}.{c})" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_mix(self, codes:List[str], vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = [f_name for f_name, _ in cls._fields_]
        maker:str = f"make_{vec}"
        inner:str = ", ".join(
            f"(({codes[0]}.{c}) * (1.0f - ({codes[2]}.{c})) + ({codes[1]}.{c}) * ({codes[2]}.{c}))"
            for c in components)
        return f"{maker}({inner})"

    def _emit_vec_dot(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = [f_name for f_name, _ in cls._fields_]
        terms:str = " + ".join(f"(({lcode}.{c}) * ({rcode}.{c}))" for c in components)
        return f"({terms})"

    def _emit_vec_builtin(self, name:str, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        opencl_name, _ = GENERIC_BUILTINS[name]
        expected_args:int = (1 if name in ("normalize", "length") else 2)
        if len(args) != expected_args:
            self._error(f"gpu.{name}() takes exactly {expected_args} arguments", node)

        vec_types:List[Any] = [t for _, t in args if is_vec(t)]
        if not vec_types:
            self._error(f"gpu.{name}() requires vector arguments "
                        f"(python has no such functions; use fabs/abs for scalars)", node)
        vec:str = vec_types[0]
        if any(t != vec for t in vec_types):
            self._error(f"gpu.{name}() operands must have the same vector type", node)

        if self.backend == "opencl":
            return f"{opencl_name}({', '.join(c for c, _ in args)})", vec

        codes:List[str] = [c for c, _ in args]
        if name == "normalize":
            return self._emit_vec_normalize(codes[0], vec), vec
        if name == "length":
            return f"sqrtf({self._emit_vec_dot(codes[0], codes[0], vec)})", "float"
        if name == "distance":
            diff:str = self._vec_componentwise("-", codes[0], vec, right=codes[1])
            return f"sqrtf({self._emit_vec_dot(diff, diff, vec)})", "float"
        if name == "cross":
            return self._emit_vec_cross(codes[0], codes[1], vec), vec
        self._error(f"gpu.{name} is not supported", node)

    def _emit_vec_normalize(self, code:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = [f_name for f_name, _ in cls._fields_]
        maker:str = f"make_{vec}"
        dot_expr:str = self._emit_vec_dot(code, code, vec)
        inner:str = ", ".join(f"(({code}.{c}) * rsqrtf({dot_expr}))" for c in components)
        return f"{maker}({inner})"

    def _emit_vec_cross(self, lcode:str, rcode:str, vec:str)->str:
        cls:Any = _vec_registry()[vec]
        components:List[str] = [f_name for f_name, _ in cls._fields_]
        maker:str = f"make_{vec}"
        x, y, z = components[0], components[1], components[2]
        inner:str = (f"(({lcode}.{y}) * ({rcode}.{z}) - ({lcode}.{z}) * ({rcode}.{y})), "
                     f"(({lcode}.{z}) * ({rcode}.{x}) - ({lcode}.{x}) * ({rcode}.{z})), "
                     f"(({lcode}.{x}) * ({rcode}.{y}) - ({lcode}.{y}) * ({rcode}.{x}))")
        if len(components) == 4:
            inner += ", 0.0f"
        elif len(components) != 3:
            self._error(f"gpu.cross() requires a 3 or 4 component vector, got {vec}")
        return f"{maker}({inner})"

    def _emit_device_call(self, name:str, target:Any, args:List[Tuple[str, Any]], node:ast.Call)->Tuple[str, Any]:
        is_kernel_ref:bool = getattr(target, "is_pykernel", False)
        raw_func:Callable = (target.raw_func if is_kernel_ref else target)
        if is_kernel_ref:
            self._error(f"'{name}' is a @kernel function and cannot be called from device code; use a plain python function instead", node)

        sig_key:Tuple[Callable, Tuple[Any, ...]] = (raw_func, tuple(t for _, t in args))
        entry:Optional[Dict[str, Any]] = self.device_cache.get(sig_key)
        if entry is not None:
            if entry["status"] == "in_progress":
                self._error(f"recursive call detected: '{name}'", node)
            return f"{entry['c_name']}({', '.join(c for c, _ in args)})", entry["return_type"]

        self.device_cache[sig_key] = {"status": "in_progress"}
        try:
            callee:FunctionTranslator = FunctionTranslator(
                raw_func, [t for _, t in args], self.backend, self.resolve_global, self.device_cache, is_kernel=False)
            callee.translate()
            c_name:str = self._device_func_name(raw_func, [t for _, t in args])
            entry = {
                "status": "done",
                "c_name": c_name,
                "return_type": callee.return_type,
                "code": callee.render_device(c_name),
            }
            self.device_cache[sig_key] = entry
        except Exception:
            self.device_cache.pop(sig_key, None)
            raise

        return f"{entry['c_name']}({', '.join(c for c, _ in args)})", entry["return_type"]

    def _device_func_name(self, func:Callable, types:List[Any])->str:
        module_file:str = getattr(getattr(func, "__code__", None), "co_filename", "")
        base:str = (os.path.basename(module_file).replace(".", "_") + "_" + func.__name__) if module_file else func.__name__
        return f"pydev_{base}_{short_hash(type_sig(types))}"

    def render_device(self, c_name:str)->str:
        params:List[str] = []
        for name, t in self.params:
            if is_ptr(t):
                params.append(f"{t[1]}* {name}")
            else:
                params.append(f"{t} {name}")
        ret:str = (self.return_type or "void")
        prefix:str = ("__device__ " if self.backend == "cuda" else "")
        header:str = f"{prefix}{ret} {c_name}({', '.join(params)})"
        body:str = "\n".join(self.lines)
        return f"{header} {{\n{body}\n}}"

    def render_kernel(self, c_name:str)->str:
        params:List[str] = []
        for name, t in self.params:
            if is_ptr(t):
                qualifier:str = ("__global " if self.backend == "opencl" else "")
                params.append(f"{qualifier}{t[1]}* {name}")
            else:
                params.append(f"{t} {name}")
        for array_name, dim in self.shapes_used:
            params.append(f"const int {self._shape_var(array_name, dim)}")

        prefix:str = ("__kernel " if self.backend == "opencl" else "__global__ ")
        header:str = f"{prefix}void {c_name}({', '.join(params)})"
        body:str = "\n".join(self.lines)
        return f"{header} {{\n{body}\n}}"
