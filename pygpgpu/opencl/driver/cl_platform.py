# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 4
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                type_ = type_._type_
                if hasattr(type_, 'as_dict'):
                    value = [type_.as_dict(v) for v in value]
                else:
                    value = [i for i in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass





cl_char = ctypes.c_byte
cl_uchar = ctypes.c_ubyte
cl_short = ctypes.c_int16
cl_ushort = ctypes.c_uint16
cl_int = ctypes.c_int32
cl_uint = ctypes.c_uint32
cl_long = ctypes.c_int64
cl_ulong = ctypes.c_uint64
cl_half = ctypes.c_uint16
cl_float = ctypes.c_float
cl_double = ctypes.c_double
cl_GLuint = ctypes.c_uint32
cl_GLint = ctypes.c_int32
cl_GLenum = ctypes.c_uint32
class union___m128(Union):
    pass

union___m128._pack_ = 1 # source:False
union___m128._fields_ = [
    ('m128_f32', ctypes.c_float * 4),
    ('m128_u64', ctypes.c_uint64 * 2),
    ('m128_i8', ctypes.c_char * 16),
    ('m128_i16', ctypes.c_int16 * 8),
    ('m128_i32', ctypes.c_int32 * 4),
    ('m128_i64', ctypes.c_int64 * 2),
    ('m128_u8', ctypes.c_ubyte * 16),
    ('m128_u16', ctypes.c_uint16 * 8),
    ('m128_u32', ctypes.c_uint32 * 4),
]

__cl_float4 = union___m128
class union___m128i(Union):
    pass

union___m128i._pack_ = 1 # source:False
union___m128i._fields_ = [
    ('m128i_i8', ctypes.c_char * 16),
    ('m128i_i16', ctypes.c_int16 * 8),
    ('m128i_i32', ctypes.c_int32 * 4),
    ('m128i_i64', ctypes.c_int64 * 2),
    ('m128i_u8', ctypes.c_ubyte * 16),
    ('m128i_u16', ctypes.c_uint16 * 8),
    ('m128i_u32', ctypes.c_uint32 * 4),
    ('m128i_u64', ctypes.c_uint64 * 2),
]

__cl_uchar16 = union___m128i
__cl_char16 = union___m128i
__cl_ushort8 = union___m128i
__cl_short8 = union___m128i
__cl_uint4 = union___m128i
__cl_int4 = union___m128i
__cl_ulong2 = union___m128i
__cl_long2 = union___m128i
class struct___m128d(Structure):
    pass

struct___m128d._pack_ = 1 # source:False
struct___m128d._fields_ = [
    ('m128d_f64', ctypes.c_double * 2),
]

__cl_double2 = struct___m128d
class union___m64(Union):
    pass

union___m64._pack_ = 1 # source:False
union___m64._fields_ = [
    ('m64_u64', ctypes.c_uint64),
    ('m64_f32', ctypes.c_float * 2),
    ('m64_i8', ctypes.c_char * 8),
    ('m64_i16', ctypes.c_int16 * 4),
    ('m64_i32', ctypes.c_int32 * 2),
    ('m64_i64', ctypes.c_int64),
    ('m64_u8', ctypes.c_ubyte * 8),
    ('m64_u16', ctypes.c_uint16 * 4),
    ('m64_u32', ctypes.c_uint32 * 2),
]

__cl_uchar8 = union___m64
__cl_char8 = union___m64
__cl_ushort4 = union___m64
__cl_short4 = union___m64
__cl_uint2 = union___m64
__cl_int2 = union___m64
__cl_ulong1 = union___m64
__cl_long1 = union___m64
__cl_float2 = union___m64
class union_cl_char2(Union):
    pass

class struct_cl_char2_0(Structure):
    pass

struct_cl_char2_0._pack_ = 1 # source:False
struct_cl_char2_0._fields_ = [
    ('x', ctypes.c_byte),
    ('y', ctypes.c_byte),
]

class struct_cl_char2_1(Structure):
    pass

struct_cl_char2_1._pack_ = 1 # source:False
struct_cl_char2_1._fields_ = [
    ('s0', ctypes.c_byte),
    ('s1', ctypes.c_byte),
]

class struct_cl_char2_2(Structure):
    pass

struct_cl_char2_2._pack_ = 1 # source:False
struct_cl_char2_2._fields_ = [
    ('lo', ctypes.c_byte),
    ('hi', ctypes.c_byte),
]

union_cl_char2._pack_ = 1 # source:False
union_cl_char2._anonymous_ = ('_0', '_1', '_2',)
union_cl_char2._fields_ = [
    ('s', ctypes.c_byte * 2),
    ('_0', struct_cl_char2_0),
    ('_1', struct_cl_char2_1),
    ('_2', struct_cl_char2_2),
]

cl_char2 = union_cl_char2
class union_cl_char4(Union):
    pass

class struct_cl_char4_0(Structure):
    pass

struct_cl_char4_0._pack_ = 1 # source:False
struct_cl_char4_0._fields_ = [
    ('x', ctypes.c_byte),
    ('y', ctypes.c_byte),
    ('z', ctypes.c_byte),
    ('w', ctypes.c_byte),
]

class struct_cl_char4_1(Structure):
    pass

struct_cl_char4_1._pack_ = 1 # source:False
struct_cl_char4_1._fields_ = [
    ('s0', ctypes.c_byte),
    ('s1', ctypes.c_byte),
    ('s2', ctypes.c_byte),
    ('s3', ctypes.c_byte),
]

class struct_cl_char4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_char2),
    ('hi', cl_char2),
     ]

union_cl_char4._pack_ = 1 # source:False
union_cl_char4._anonymous_ = ('_0', '_1', '_2',)
union_cl_char4._fields_ = [
    ('s', ctypes.c_byte * 4),
    ('_0', struct_cl_char4_0),
    ('_1', struct_cl_char4_1),
    ('_2', struct_cl_char4_2),
]

cl_char4 = union_cl_char4
cl_char3 = union_cl_char4
class union_cl_char8(Union):
    pass

class struct_cl_char8_0(Structure):
    pass

struct_cl_char8_0._pack_ = 1 # source:False
struct_cl_char8_0._fields_ = [
    ('x', ctypes.c_byte),
    ('y', ctypes.c_byte),
    ('z', ctypes.c_byte),
    ('w', ctypes.c_byte),
]

class struct_cl_char8_1(Structure):
    pass

struct_cl_char8_1._pack_ = 1 # source:False
struct_cl_char8_1._fields_ = [
    ('s0', ctypes.c_byte),
    ('s1', ctypes.c_byte),
    ('s2', ctypes.c_byte),
    ('s3', ctypes.c_byte),
    ('s4', ctypes.c_byte),
    ('s5', ctypes.c_byte),
    ('s6', ctypes.c_byte),
    ('s7', ctypes.c_byte),
]

class struct_cl_char8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_char4),
    ('hi', cl_char4),
     ]

union_cl_char8._pack_ = 1 # source:False
union_cl_char8._anonymous_ = ('_0', '_1', '_2',)
union_cl_char8._fields_ = [
    ('s', ctypes.c_byte * 8),
    ('_0', struct_cl_char8_0),
    ('_1', struct_cl_char8_1),
    ('_2', struct_cl_char8_2),
    ('v8', globals()['__cl_char8']),
]

cl_char8 = union_cl_char8
class union_cl_char16(Union):
    pass

class struct_cl_char16_0(Structure):
    pass

struct_cl_char16_0._pack_ = 1 # source:False
struct_cl_char16_0._fields_ = [
    ('x', ctypes.c_byte),
    ('y', ctypes.c_byte),
    ('z', ctypes.c_byte),
    ('w', ctypes.c_byte),
    ('__spacer4', ctypes.c_byte),
    ('__spacer5', ctypes.c_byte),
    ('__spacer6', ctypes.c_byte),
    ('__spacer7', ctypes.c_byte),
    ('__spacer8', ctypes.c_byte),
    ('__spacer9', ctypes.c_byte),
    ('sa', ctypes.c_byte),
    ('sb', ctypes.c_byte),
    ('sc', ctypes.c_byte),
    ('sd', ctypes.c_byte),
    ('se', ctypes.c_byte),
    ('sf', ctypes.c_byte),
]

class struct_cl_char16_1(Structure):
    pass

struct_cl_char16_1._pack_ = 1 # source:False
struct_cl_char16_1._fields_ = [
    ('s0', ctypes.c_byte),
    ('s1', ctypes.c_byte),
    ('s2', ctypes.c_byte),
    ('s3', ctypes.c_byte),
    ('s4', ctypes.c_byte),
    ('s5', ctypes.c_byte),
    ('s6', ctypes.c_byte),
    ('s7', ctypes.c_byte),
    ('s8', ctypes.c_byte),
    ('s9', ctypes.c_byte),
    ('sA', ctypes.c_byte),
    ('sB', ctypes.c_byte),
    ('sC', ctypes.c_byte),
    ('sD', ctypes.c_byte),
    ('sE', ctypes.c_byte),
    ('sF', ctypes.c_byte),
]

class struct_cl_char16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_char8),
    ('hi', cl_char8),
     ]

union_cl_char16._pack_ = 1 # source:False
union_cl_char16._anonymous_ = ('_0', '_1', '_2',)
union_cl_char16._fields_ = [
    ('s', ctypes.c_byte * 16),
    ('_0', struct_cl_char16_0),
    ('_1', struct_cl_char16_1),
    ('_2', struct_cl_char16_2),
    ('v8', union___m64 * 2),
    ('v16', globals()['__cl_char16']),
]

cl_char16 = union_cl_char16
class union_cl_uchar2(Union):
    pass

class struct_cl_uchar2_0(Structure):
    pass

struct_cl_uchar2_0._pack_ = 1 # source:False
struct_cl_uchar2_0._fields_ = [
    ('x', ctypes.c_ubyte),
    ('y', ctypes.c_ubyte),
]

class struct_cl_uchar2_1(Structure):
    pass

struct_cl_uchar2_1._pack_ = 1 # source:False
struct_cl_uchar2_1._fields_ = [
    ('s0', ctypes.c_ubyte),
    ('s1', ctypes.c_ubyte),
]

class struct_cl_uchar2_2(Structure):
    pass

struct_cl_uchar2_2._pack_ = 1 # source:False
struct_cl_uchar2_2._fields_ = [
    ('lo', ctypes.c_ubyte),
    ('hi', ctypes.c_ubyte),
]

union_cl_uchar2._pack_ = 1 # source:False
union_cl_uchar2._anonymous_ = ('_0', '_1', '_2',)
union_cl_uchar2._fields_ = [
    ('s', ctypes.c_ubyte * 2),
    ('_0', struct_cl_uchar2_0),
    ('_1', struct_cl_uchar2_1),
    ('_2', struct_cl_uchar2_2),
]

cl_uchar2 = union_cl_uchar2
class union_cl_uchar4(Union):
    pass

class struct_cl_uchar4_0(Structure):
    pass

struct_cl_uchar4_0._pack_ = 1 # source:False
struct_cl_uchar4_0._fields_ = [
    ('x', ctypes.c_ubyte),
    ('y', ctypes.c_ubyte),
    ('z', ctypes.c_ubyte),
    ('w', ctypes.c_ubyte),
]

class struct_cl_uchar4_1(Structure):
    pass

struct_cl_uchar4_1._pack_ = 1 # source:False
struct_cl_uchar4_1._fields_ = [
    ('s0', ctypes.c_ubyte),
    ('s1', ctypes.c_ubyte),
    ('s2', ctypes.c_ubyte),
    ('s3', ctypes.c_ubyte),
]

class struct_cl_uchar4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uchar2),
    ('hi', cl_uchar2),
     ]

union_cl_uchar4._pack_ = 1 # source:False
union_cl_uchar4._anonymous_ = ('_0', '_1', '_2',)
union_cl_uchar4._fields_ = [
    ('s', ctypes.c_ubyte * 4),
    ('_0', struct_cl_uchar4_0),
    ('_1', struct_cl_uchar4_1),
    ('_2', struct_cl_uchar4_2),
]

cl_uchar4 = union_cl_uchar4
cl_uchar3 = union_cl_uchar4
class union_cl_uchar8(Union):
    pass

class struct_cl_uchar8_0(Structure):
    pass

struct_cl_uchar8_0._pack_ = 1 # source:False
struct_cl_uchar8_0._fields_ = [
    ('x', ctypes.c_ubyte),
    ('y', ctypes.c_ubyte),
    ('z', ctypes.c_ubyte),
    ('w', ctypes.c_ubyte),
]

class struct_cl_uchar8_1(Structure):
    pass

struct_cl_uchar8_1._pack_ = 1 # source:False
struct_cl_uchar8_1._fields_ = [
    ('s0', ctypes.c_ubyte),
    ('s1', ctypes.c_ubyte),
    ('s2', ctypes.c_ubyte),
    ('s3', ctypes.c_ubyte),
    ('s4', ctypes.c_ubyte),
    ('s5', ctypes.c_ubyte),
    ('s6', ctypes.c_ubyte),
    ('s7', ctypes.c_ubyte),
]

class struct_cl_uchar8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uchar4),
    ('hi', cl_uchar4),
     ]

union_cl_uchar8._pack_ = 1 # source:False
union_cl_uchar8._anonymous_ = ('_0', '_1', '_2',)
union_cl_uchar8._fields_ = [
    ('s', ctypes.c_ubyte * 8),
    ('_0', struct_cl_uchar8_0),
    ('_1', struct_cl_uchar8_1),
    ('_2', struct_cl_uchar8_2),
    ('v8', globals()['__cl_uchar8']),
]

cl_uchar8 = union_cl_uchar8
class union_cl_uchar16(Union):
    pass

class struct_cl_uchar16_0(Structure):
    pass

struct_cl_uchar16_0._pack_ = 1 # source:False
struct_cl_uchar16_0._fields_ = [
    ('x', ctypes.c_ubyte),
    ('y', ctypes.c_ubyte),
    ('z', ctypes.c_ubyte),
    ('w', ctypes.c_ubyte),
    ('__spacer4', ctypes.c_ubyte),
    ('__spacer5', ctypes.c_ubyte),
    ('__spacer6', ctypes.c_ubyte),
    ('__spacer7', ctypes.c_ubyte),
    ('__spacer8', ctypes.c_ubyte),
    ('__spacer9', ctypes.c_ubyte),
    ('sa', ctypes.c_ubyte),
    ('sb', ctypes.c_ubyte),
    ('sc', ctypes.c_ubyte),
    ('sd', ctypes.c_ubyte),
    ('se', ctypes.c_ubyte),
    ('sf', ctypes.c_ubyte),
]

class struct_cl_uchar16_1(Structure):
    pass

struct_cl_uchar16_1._pack_ = 1 # source:False
struct_cl_uchar16_1._fields_ = [
    ('s0', ctypes.c_ubyte),
    ('s1', ctypes.c_ubyte),
    ('s2', ctypes.c_ubyte),
    ('s3', ctypes.c_ubyte),
    ('s4', ctypes.c_ubyte),
    ('s5', ctypes.c_ubyte),
    ('s6', ctypes.c_ubyte),
    ('s7', ctypes.c_ubyte),
    ('s8', ctypes.c_ubyte),
    ('s9', ctypes.c_ubyte),
    ('sA', ctypes.c_ubyte),
    ('sB', ctypes.c_ubyte),
    ('sC', ctypes.c_ubyte),
    ('sD', ctypes.c_ubyte),
    ('sE', ctypes.c_ubyte),
    ('sF', ctypes.c_ubyte),
]

class struct_cl_uchar16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uchar8),
    ('hi', cl_uchar8),
     ]

union_cl_uchar16._pack_ = 1 # source:False
union_cl_uchar16._anonymous_ = ('_0', '_1', '_2',)
union_cl_uchar16._fields_ = [
    ('s', ctypes.c_ubyte * 16),
    ('_0', struct_cl_uchar16_0),
    ('_1', struct_cl_uchar16_1),
    ('_2', struct_cl_uchar16_2),
    ('v8', union___m64 * 2),
    ('v16', globals()['__cl_uchar16']),
]

cl_uchar16 = union_cl_uchar16
class union_cl_short2(Union):
    pass

class struct_cl_short2_0(Structure):
    pass

struct_cl_short2_0._pack_ = 1 # source:False
struct_cl_short2_0._fields_ = [
    ('x', ctypes.c_int16),
    ('y', ctypes.c_int16),
]

class struct_cl_short2_1(Structure):
    pass

struct_cl_short2_1._pack_ = 1 # source:False
struct_cl_short2_1._fields_ = [
    ('s0', ctypes.c_int16),
    ('s1', ctypes.c_int16),
]

class struct_cl_short2_2(Structure):
    pass

struct_cl_short2_2._pack_ = 1 # source:False
struct_cl_short2_2._fields_ = [
    ('lo', ctypes.c_int16),
    ('hi', ctypes.c_int16),
]

union_cl_short2._pack_ = 1 # source:False
union_cl_short2._anonymous_ = ('_0', '_1', '_2',)
union_cl_short2._fields_ = [
    ('s', ctypes.c_int16 * 2),
    ('_0', struct_cl_short2_0),
    ('_1', struct_cl_short2_1),
    ('_2', struct_cl_short2_2),
]

cl_short2 = union_cl_short2
class union_cl_short4(Union):
    pass

class struct_cl_short4_0(Structure):
    pass

struct_cl_short4_0._pack_ = 1 # source:False
struct_cl_short4_0._fields_ = [
    ('x', ctypes.c_int16),
    ('y', ctypes.c_int16),
    ('z', ctypes.c_int16),
    ('w', ctypes.c_int16),
]

class struct_cl_short4_1(Structure):
    pass

struct_cl_short4_1._pack_ = 1 # source:False
struct_cl_short4_1._fields_ = [
    ('s0', ctypes.c_int16),
    ('s1', ctypes.c_int16),
    ('s2', ctypes.c_int16),
    ('s3', ctypes.c_int16),
]

class struct_cl_short4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_short2),
    ('hi', cl_short2),
     ]

union_cl_short4._pack_ = 1 # source:False
union_cl_short4._anonymous_ = ('_0', '_1', '_2',)
union_cl_short4._fields_ = [
    ('s', ctypes.c_int16 * 4),
    ('_0', struct_cl_short4_0),
    ('_1', struct_cl_short4_1),
    ('_2', struct_cl_short4_2),
    ('v4', globals()['__cl_short4']),
]

cl_short4 = union_cl_short4
cl_short3 = union_cl_short4
class union_cl_short8(Union):
    pass

class struct_cl_short8_0(Structure):
    pass

struct_cl_short8_0._pack_ = 1 # source:False
struct_cl_short8_0._fields_ = [
    ('x', ctypes.c_int16),
    ('y', ctypes.c_int16),
    ('z', ctypes.c_int16),
    ('w', ctypes.c_int16),
]

class struct_cl_short8_1(Structure):
    pass

struct_cl_short8_1._pack_ = 1 # source:False
struct_cl_short8_1._fields_ = [
    ('s0', ctypes.c_int16),
    ('s1', ctypes.c_int16),
    ('s2', ctypes.c_int16),
    ('s3', ctypes.c_int16),
    ('s4', ctypes.c_int16),
    ('s5', ctypes.c_int16),
    ('s6', ctypes.c_int16),
    ('s7', ctypes.c_int16),
]

class struct_cl_short8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_short4),
    ('hi', cl_short4),
     ]

union_cl_short8._pack_ = 1 # source:False
union_cl_short8._anonymous_ = ('_0', '_1', '_2',)
union_cl_short8._fields_ = [
    ('s', ctypes.c_int16 * 8),
    ('_0', struct_cl_short8_0),
    ('_1', struct_cl_short8_1),
    ('_2', struct_cl_short8_2),
    ('v4', union___m64 * 2),
    ('v8', globals()['__cl_short8']),
]

cl_short8 = union_cl_short8
class union_cl_short16(Union):
    pass

class struct_cl_short16_0(Structure):
    pass

struct_cl_short16_0._pack_ = 1 # source:False
struct_cl_short16_0._fields_ = [
    ('x', ctypes.c_int16),
    ('y', ctypes.c_int16),
    ('z', ctypes.c_int16),
    ('w', ctypes.c_int16),
    ('__spacer4', ctypes.c_int16),
    ('__spacer5', ctypes.c_int16),
    ('__spacer6', ctypes.c_int16),
    ('__spacer7', ctypes.c_int16),
    ('__spacer8', ctypes.c_int16),
    ('__spacer9', ctypes.c_int16),
    ('sa', ctypes.c_int16),
    ('sb', ctypes.c_int16),
    ('sc', ctypes.c_int16),
    ('sd', ctypes.c_int16),
    ('se', ctypes.c_int16),
    ('sf', ctypes.c_int16),
]

class struct_cl_short16_1(Structure):
    pass

struct_cl_short16_1._pack_ = 1 # source:False
struct_cl_short16_1._fields_ = [
    ('s0', ctypes.c_int16),
    ('s1', ctypes.c_int16),
    ('s2', ctypes.c_int16),
    ('s3', ctypes.c_int16),
    ('s4', ctypes.c_int16),
    ('s5', ctypes.c_int16),
    ('s6', ctypes.c_int16),
    ('s7', ctypes.c_int16),
    ('s8', ctypes.c_int16),
    ('s9', ctypes.c_int16),
    ('sA', ctypes.c_int16),
    ('sB', ctypes.c_int16),
    ('sC', ctypes.c_int16),
    ('sD', ctypes.c_int16),
    ('sE', ctypes.c_int16),
    ('sF', ctypes.c_int16),
]

class struct_cl_short16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_short8),
    ('hi', cl_short8),
     ]

union_cl_short16._pack_ = 1 # source:False
union_cl_short16._anonymous_ = ('_0', '_1', '_2',)
union_cl_short16._fields_ = [
    ('s', ctypes.c_int16 * 16),
    ('_0', struct_cl_short16_0),
    ('_1', struct_cl_short16_1),
    ('_2', struct_cl_short16_2),
    ('v4', union___m64 * 4),
    ('v8', union___m128i * 2),
]

cl_short16 = union_cl_short16
class union_cl_ushort2(Union):
    pass

class struct_cl_ushort2_0(Structure):
    pass

struct_cl_ushort2_0._pack_ = 1 # source:False
struct_cl_ushort2_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
]

class struct_cl_ushort2_1(Structure):
    pass

struct_cl_ushort2_1._pack_ = 1 # source:False
struct_cl_ushort2_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
]

class struct_cl_ushort2_2(Structure):
    pass

struct_cl_ushort2_2._pack_ = 1 # source:False
struct_cl_ushort2_2._fields_ = [
    ('lo', ctypes.c_uint16),
    ('hi', ctypes.c_uint16),
]

union_cl_ushort2._pack_ = 1 # source:False
union_cl_ushort2._anonymous_ = ('_0', '_1', '_2',)
union_cl_ushort2._fields_ = [
    ('s', ctypes.c_uint16 * 2),
    ('_0', struct_cl_ushort2_0),
    ('_1', struct_cl_ushort2_1),
    ('_2', struct_cl_ushort2_2),
]

cl_ushort2 = union_cl_ushort2
class union_cl_ushort4(Union):
    pass

class struct_cl_ushort4_0(Structure):
    pass

struct_cl_ushort4_0._pack_ = 1 # source:False
struct_cl_ushort4_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
]

class struct_cl_ushort4_1(Structure):
    pass

struct_cl_ushort4_1._pack_ = 1 # source:False
struct_cl_ushort4_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
]

class struct_cl_ushort4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ushort2),
    ('hi', cl_ushort2),
     ]

union_cl_ushort4._pack_ = 1 # source:False
union_cl_ushort4._anonymous_ = ('_0', '_1', '_2',)
union_cl_ushort4._fields_ = [
    ('s', ctypes.c_uint16 * 4),
    ('_0', struct_cl_ushort4_0),
    ('_1', struct_cl_ushort4_1),
    ('_2', struct_cl_ushort4_2),
    ('v4', globals()['__cl_ushort4']),
]

cl_ushort4 = union_cl_ushort4
cl_ushort3 = union_cl_ushort4
class union_cl_ushort8(Union):
    pass

class struct_cl_ushort8_0(Structure):
    pass

struct_cl_ushort8_0._pack_ = 1 # source:False
struct_cl_ushort8_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
]

class struct_cl_ushort8_1(Structure):
    pass

struct_cl_ushort8_1._pack_ = 1 # source:False
struct_cl_ushort8_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
    ('s4', ctypes.c_uint16),
    ('s5', ctypes.c_uint16),
    ('s6', ctypes.c_uint16),
    ('s7', ctypes.c_uint16),
]

class struct_cl_ushort8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ushort4),
    ('hi', cl_ushort4),
     ]

union_cl_ushort8._pack_ = 1 # source:False
union_cl_ushort8._anonymous_ = ('_0', '_1', '_2',)
union_cl_ushort8._fields_ = [
    ('s', ctypes.c_uint16 * 8),
    ('_0', struct_cl_ushort8_0),
    ('_1', struct_cl_ushort8_1),
    ('_2', struct_cl_ushort8_2),
    ('v4', union___m64 * 2),
    ('v8', globals()['__cl_ushort8']),
]

cl_ushort8 = union_cl_ushort8
class union_cl_ushort16(Union):
    pass

class struct_cl_ushort16_0(Structure):
    pass

struct_cl_ushort16_0._pack_ = 1 # source:False
struct_cl_ushort16_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
    ('__spacer4', ctypes.c_uint16),
    ('__spacer5', ctypes.c_uint16),
    ('__spacer6', ctypes.c_uint16),
    ('__spacer7', ctypes.c_uint16),
    ('__spacer8', ctypes.c_uint16),
    ('__spacer9', ctypes.c_uint16),
    ('sa', ctypes.c_uint16),
    ('sb', ctypes.c_uint16),
    ('sc', ctypes.c_uint16),
    ('sd', ctypes.c_uint16),
    ('se', ctypes.c_uint16),
    ('sf', ctypes.c_uint16),
]

class struct_cl_ushort16_1(Structure):
    pass

struct_cl_ushort16_1._pack_ = 1 # source:False
struct_cl_ushort16_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
    ('s4', ctypes.c_uint16),
    ('s5', ctypes.c_uint16),
    ('s6', ctypes.c_uint16),
    ('s7', ctypes.c_uint16),
    ('s8', ctypes.c_uint16),
    ('s9', ctypes.c_uint16),
    ('sA', ctypes.c_uint16),
    ('sB', ctypes.c_uint16),
    ('sC', ctypes.c_uint16),
    ('sD', ctypes.c_uint16),
    ('sE', ctypes.c_uint16),
    ('sF', ctypes.c_uint16),
]

class struct_cl_ushort16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ushort8),
    ('hi', cl_ushort8),
     ]

union_cl_ushort16._pack_ = 1 # source:False
union_cl_ushort16._anonymous_ = ('_0', '_1', '_2',)
union_cl_ushort16._fields_ = [
    ('s', ctypes.c_uint16 * 16),
    ('_0', struct_cl_ushort16_0),
    ('_1', struct_cl_ushort16_1),
    ('_2', struct_cl_ushort16_2),
    ('v4', union___m64 * 4),
    ('v8', union___m128i * 2),
]

cl_ushort16 = union_cl_ushort16
class union_cl_half2(Union):
    pass

class struct_cl_half2_0(Structure):
    pass

struct_cl_half2_0._pack_ = 1 # source:False
struct_cl_half2_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
]

class struct_cl_half2_1(Structure):
    pass

struct_cl_half2_1._pack_ = 1 # source:False
struct_cl_half2_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
]

class struct_cl_half2_2(Structure):
    pass

struct_cl_half2_2._pack_ = 1 # source:False
struct_cl_half2_2._fields_ = [
    ('lo', ctypes.c_uint16),
    ('hi', ctypes.c_uint16),
]

union_cl_half2._pack_ = 1 # source:False
union_cl_half2._anonymous_ = ('_0', '_1', '_2',)
union_cl_half2._fields_ = [
    ('s', ctypes.c_uint16 * 2),
    ('_0', struct_cl_half2_0),
    ('_1', struct_cl_half2_1),
    ('_2', struct_cl_half2_2),
]

cl_half2 = union_cl_half2
class union_cl_half4(Union):
    pass

class struct_cl_half4_0(Structure):
    pass

struct_cl_half4_0._pack_ = 1 # source:False
struct_cl_half4_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
]

class struct_cl_half4_1(Structure):
    pass

struct_cl_half4_1._pack_ = 1 # source:False
struct_cl_half4_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
]

class struct_cl_half4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_half2),
    ('hi', cl_half2),
     ]

union_cl_half4._pack_ = 1 # source:False
union_cl_half4._anonymous_ = ('_0', '_1', '_2',)
union_cl_half4._fields_ = [
    ('s', ctypes.c_uint16 * 4),
    ('_0', struct_cl_half4_0),
    ('_1', struct_cl_half4_1),
    ('_2', struct_cl_half4_2),
]

cl_half4 = union_cl_half4
cl_half3 = union_cl_half4
class union_cl_half8(Union):
    pass

class struct_cl_half8_0(Structure):
    pass

struct_cl_half8_0._pack_ = 1 # source:False
struct_cl_half8_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
]

class struct_cl_half8_1(Structure):
    pass

struct_cl_half8_1._pack_ = 1 # source:False
struct_cl_half8_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
    ('s4', ctypes.c_uint16),
    ('s5', ctypes.c_uint16),
    ('s6', ctypes.c_uint16),
    ('s7', ctypes.c_uint16),
]

class struct_cl_half8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_half4),
    ('hi', cl_half4),
     ]

union_cl_half8._pack_ = 1 # source:False
union_cl_half8._anonymous_ = ('_0', '_1', '_2',)
union_cl_half8._fields_ = [
    ('s', ctypes.c_uint16 * 8),
    ('_0', struct_cl_half8_0),
    ('_1', struct_cl_half8_1),
    ('_2', struct_cl_half8_2),
]

cl_half8 = union_cl_half8
class union_cl_half16(Union):
    pass

class struct_cl_half16_0(Structure):
    pass

struct_cl_half16_0._pack_ = 1 # source:False
struct_cl_half16_0._fields_ = [
    ('x', ctypes.c_uint16),
    ('y', ctypes.c_uint16),
    ('z', ctypes.c_uint16),
    ('w', ctypes.c_uint16),
    ('__spacer4', ctypes.c_uint16),
    ('__spacer5', ctypes.c_uint16),
    ('__spacer6', ctypes.c_uint16),
    ('__spacer7', ctypes.c_uint16),
    ('__spacer8', ctypes.c_uint16),
    ('__spacer9', ctypes.c_uint16),
    ('sa', ctypes.c_uint16),
    ('sb', ctypes.c_uint16),
    ('sc', ctypes.c_uint16),
    ('sd', ctypes.c_uint16),
    ('se', ctypes.c_uint16),
    ('sf', ctypes.c_uint16),
]

class struct_cl_half16_1(Structure):
    pass

struct_cl_half16_1._pack_ = 1 # source:False
struct_cl_half16_1._fields_ = [
    ('s0', ctypes.c_uint16),
    ('s1', ctypes.c_uint16),
    ('s2', ctypes.c_uint16),
    ('s3', ctypes.c_uint16),
    ('s4', ctypes.c_uint16),
    ('s5', ctypes.c_uint16),
    ('s6', ctypes.c_uint16),
    ('s7', ctypes.c_uint16),
    ('s8', ctypes.c_uint16),
    ('s9', ctypes.c_uint16),
    ('sA', ctypes.c_uint16),
    ('sB', ctypes.c_uint16),
    ('sC', ctypes.c_uint16),
    ('sD', ctypes.c_uint16),
    ('sE', ctypes.c_uint16),
    ('sF', ctypes.c_uint16),
]

class struct_cl_half16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_half8),
    ('hi', cl_half8),
     ]

union_cl_half16._pack_ = 1 # source:False
union_cl_half16._anonymous_ = ('_0', '_1', '_2',)
union_cl_half16._fields_ = [
    ('s', ctypes.c_uint16 * 16),
    ('_0', struct_cl_half16_0),
    ('_1', struct_cl_half16_1),
    ('_2', struct_cl_half16_2),
]

cl_half16 = union_cl_half16
class union_cl_int2(Union):
    pass

class struct_cl_int2_0(Structure):
    pass

struct_cl_int2_0._pack_ = 1 # source:False
struct_cl_int2_0._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
]

class struct_cl_int2_1(Structure):
    pass

struct_cl_int2_1._pack_ = 1 # source:False
struct_cl_int2_1._fields_ = [
    ('s0', ctypes.c_int32),
    ('s1', ctypes.c_int32),
]

class struct_cl_int2_2(Structure):
    pass

struct_cl_int2_2._pack_ = 1 # source:False
struct_cl_int2_2._fields_ = [
    ('lo', ctypes.c_int32),
    ('hi', ctypes.c_int32),
]

union_cl_int2._pack_ = 1 # source:False
union_cl_int2._anonymous_ = ('_0', '_1', '_2',)
union_cl_int2._fields_ = [
    ('s', ctypes.c_int32 * 2),
    ('_0', struct_cl_int2_0),
    ('_1', struct_cl_int2_1),
    ('_2', struct_cl_int2_2),
    ('v2', globals()['__cl_int2']),
]

cl_int2 = union_cl_int2
class union_cl_int4(Union):
    pass

class struct_cl_int4_0(Structure):
    pass

struct_cl_int4_0._pack_ = 1 # source:False
struct_cl_int4_0._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('z', ctypes.c_int32),
    ('w', ctypes.c_int32),
]

class struct_cl_int4_1(Structure):
    pass

struct_cl_int4_1._pack_ = 1 # source:False
struct_cl_int4_1._fields_ = [
    ('s0', ctypes.c_int32),
    ('s1', ctypes.c_int32),
    ('s2', ctypes.c_int32),
    ('s3', ctypes.c_int32),
]

class struct_cl_int4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_int2),
    ('hi', cl_int2),
     ]

union_cl_int4._pack_ = 1 # source:False
union_cl_int4._anonymous_ = ('_0', '_1', '_2',)
union_cl_int4._fields_ = [
    ('s', ctypes.c_int32 * 4),
    ('_0', struct_cl_int4_0),
    ('_1', struct_cl_int4_1),
    ('_2', struct_cl_int4_2),
    ('v2', union___m64 * 2),
    ('v4', globals()['__cl_int4']),
]

cl_int4 = union_cl_int4
cl_int3 = union_cl_int4
class union_cl_int8(Union):
    pass

class struct_cl_int8_0(Structure):
    pass

struct_cl_int8_0._pack_ = 1 # source:False
struct_cl_int8_0._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('z', ctypes.c_int32),
    ('w', ctypes.c_int32),
]

class struct_cl_int8_1(Structure):
    pass

struct_cl_int8_1._pack_ = 1 # source:False
struct_cl_int8_1._fields_ = [
    ('s0', ctypes.c_int32),
    ('s1', ctypes.c_int32),
    ('s2', ctypes.c_int32),
    ('s3', ctypes.c_int32),
    ('s4', ctypes.c_int32),
    ('s5', ctypes.c_int32),
    ('s6', ctypes.c_int32),
    ('s7', ctypes.c_int32),
]

class struct_cl_int8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_int4),
    ('hi', cl_int4),
     ]

union_cl_int8._pack_ = 1 # source:False
union_cl_int8._anonymous_ = ('_0', '_1', '_2',)
union_cl_int8._fields_ = [
    ('s', ctypes.c_int32 * 8),
    ('_0', struct_cl_int8_0),
    ('_1', struct_cl_int8_1),
    ('_2', struct_cl_int8_2),
    ('v2', union___m64 * 4),
    ('v4', union___m128i * 2),
]

cl_int8 = union_cl_int8
class union_cl_int16(Union):
    pass

class struct_cl_int16_0(Structure):
    pass

struct_cl_int16_0._pack_ = 1 # source:False
struct_cl_int16_0._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
    ('z', ctypes.c_int32),
    ('w', ctypes.c_int32),
    ('__spacer4', ctypes.c_int32),
    ('__spacer5', ctypes.c_int32),
    ('__spacer6', ctypes.c_int32),
    ('__spacer7', ctypes.c_int32),
    ('__spacer8', ctypes.c_int32),
    ('__spacer9', ctypes.c_int32),
    ('sa', ctypes.c_int32),
    ('sb', ctypes.c_int32),
    ('sc', ctypes.c_int32),
    ('sd', ctypes.c_int32),
    ('se', ctypes.c_int32),
    ('sf', ctypes.c_int32),
]

class struct_cl_int16_1(Structure):
    pass

struct_cl_int16_1._pack_ = 1 # source:False
struct_cl_int16_1._fields_ = [
    ('s0', ctypes.c_int32),
    ('s1', ctypes.c_int32),
    ('s2', ctypes.c_int32),
    ('s3', ctypes.c_int32),
    ('s4', ctypes.c_int32),
    ('s5', ctypes.c_int32),
    ('s6', ctypes.c_int32),
    ('s7', ctypes.c_int32),
    ('s8', ctypes.c_int32),
    ('s9', ctypes.c_int32),
    ('sA', ctypes.c_int32),
    ('sB', ctypes.c_int32),
    ('sC', ctypes.c_int32),
    ('sD', ctypes.c_int32),
    ('sE', ctypes.c_int32),
    ('sF', ctypes.c_int32),
]

class struct_cl_int16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_int8),
    ('hi', cl_int8),
     ]

union_cl_int16._pack_ = 1 # source:False
union_cl_int16._anonymous_ = ('_0', '_1', '_2',)
union_cl_int16._fields_ = [
    ('s', ctypes.c_int32 * 16),
    ('_0', struct_cl_int16_0),
    ('_1', struct_cl_int16_1),
    ('_2', struct_cl_int16_2),
    ('v2', union___m64 * 8),
    ('v4', union___m128i * 4),
]

cl_int16 = union_cl_int16
class union_cl_uint2(Union):
    pass

class struct_cl_uint2_0(Structure):
    pass

struct_cl_uint2_0._pack_ = 1 # source:False
struct_cl_uint2_0._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
]

class struct_cl_uint2_1(Structure):
    pass

struct_cl_uint2_1._pack_ = 1 # source:False
struct_cl_uint2_1._fields_ = [
    ('s0', ctypes.c_uint32),
    ('s1', ctypes.c_uint32),
]

class struct_cl_uint2_2(Structure):
    pass

struct_cl_uint2_2._pack_ = 1 # source:False
struct_cl_uint2_2._fields_ = [
    ('lo', ctypes.c_uint32),
    ('hi', ctypes.c_uint32),
]

union_cl_uint2._pack_ = 1 # source:False
union_cl_uint2._anonymous_ = ('_0', '_1', '_2',)
union_cl_uint2._fields_ = [
    ('s', ctypes.c_uint32 * 2),
    ('_0', struct_cl_uint2_0),
    ('_1', struct_cl_uint2_1),
    ('_2', struct_cl_uint2_2),
    ('v2', globals()['__cl_uint2']),
]

cl_uint2 = union_cl_uint2
class union_cl_uint4(Union):
    pass

class struct_cl_uint4_0(Structure):
    pass

struct_cl_uint4_0._pack_ = 1 # source:False
struct_cl_uint4_0._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
    ('w', ctypes.c_uint32),
]

class struct_cl_uint4_1(Structure):
    pass

struct_cl_uint4_1._pack_ = 1 # source:False
struct_cl_uint4_1._fields_ = [
    ('s0', ctypes.c_uint32),
    ('s1', ctypes.c_uint32),
    ('s2', ctypes.c_uint32),
    ('s3', ctypes.c_uint32),
]

class struct_cl_uint4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uint2),
    ('hi', cl_uint2),
     ]

union_cl_uint4._pack_ = 1 # source:False
union_cl_uint4._anonymous_ = ('_0', '_1', '_2',)
union_cl_uint4._fields_ = [
    ('s', ctypes.c_uint32 * 4),
    ('_0', struct_cl_uint4_0),
    ('_1', struct_cl_uint4_1),
    ('_2', struct_cl_uint4_2),
    ('v2', union___m64 * 2),
    ('v4', globals()['__cl_uint4']),
]

cl_uint4 = union_cl_uint4
cl_uint3 = union_cl_uint4
class union_cl_uint8(Union):
    pass

class struct_cl_uint8_0(Structure):
    pass

struct_cl_uint8_0._pack_ = 1 # source:False
struct_cl_uint8_0._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
    ('w', ctypes.c_uint32),
]

class struct_cl_uint8_1(Structure):
    pass

struct_cl_uint8_1._pack_ = 1 # source:False
struct_cl_uint8_1._fields_ = [
    ('s0', ctypes.c_uint32),
    ('s1', ctypes.c_uint32),
    ('s2', ctypes.c_uint32),
    ('s3', ctypes.c_uint32),
    ('s4', ctypes.c_uint32),
    ('s5', ctypes.c_uint32),
    ('s6', ctypes.c_uint32),
    ('s7', ctypes.c_uint32),
]

class struct_cl_uint8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uint4),
    ('hi', cl_uint4),
     ]

union_cl_uint8._pack_ = 1 # source:False
union_cl_uint8._anonymous_ = ('_0', '_1', '_2',)
union_cl_uint8._fields_ = [
    ('s', ctypes.c_uint32 * 8),
    ('_0', struct_cl_uint8_0),
    ('_1', struct_cl_uint8_1),
    ('_2', struct_cl_uint8_2),
    ('v2', union___m64 * 4),
    ('v4', union___m128i * 2),
]

cl_uint8 = union_cl_uint8
class union_cl_uint16(Union):
    pass

class struct_cl_uint16_0(Structure):
    pass

struct_cl_uint16_0._pack_ = 1 # source:False
struct_cl_uint16_0._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
    ('w', ctypes.c_uint32),
    ('__spacer4', ctypes.c_uint32),
    ('__spacer5', ctypes.c_uint32),
    ('__spacer6', ctypes.c_uint32),
    ('__spacer7', ctypes.c_uint32),
    ('__spacer8', ctypes.c_uint32),
    ('__spacer9', ctypes.c_uint32),
    ('sa', ctypes.c_uint32),
    ('sb', ctypes.c_uint32),
    ('sc', ctypes.c_uint32),
    ('sd', ctypes.c_uint32),
    ('se', ctypes.c_uint32),
    ('sf', ctypes.c_uint32),
]

class struct_cl_uint16_1(Structure):
    pass

struct_cl_uint16_1._pack_ = 1 # source:False
struct_cl_uint16_1._fields_ = [
    ('s0', ctypes.c_uint32),
    ('s1', ctypes.c_uint32),
    ('s2', ctypes.c_uint32),
    ('s3', ctypes.c_uint32),
    ('s4', ctypes.c_uint32),
    ('s5', ctypes.c_uint32),
    ('s6', ctypes.c_uint32),
    ('s7', ctypes.c_uint32),
    ('s8', ctypes.c_uint32),
    ('s9', ctypes.c_uint32),
    ('sA', ctypes.c_uint32),
    ('sB', ctypes.c_uint32),
    ('sC', ctypes.c_uint32),
    ('sD', ctypes.c_uint32),
    ('sE', ctypes.c_uint32),
    ('sF', ctypes.c_uint32),
]

class struct_cl_uint16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_uint8),
    ('hi', cl_uint8),
     ]

union_cl_uint16._pack_ = 1 # source:False
union_cl_uint16._anonymous_ = ('_0', '_1', '_2',)
union_cl_uint16._fields_ = [
    ('s', ctypes.c_uint32 * 16),
    ('_0', struct_cl_uint16_0),
    ('_1', struct_cl_uint16_1),
    ('_2', struct_cl_uint16_2),
    ('v2', union___m64 * 8),
    ('v4', union___m128i * 4),
]

cl_uint16 = union_cl_uint16
class union_cl_long2(Union):
    pass

class struct_cl_long2_0(Structure):
    pass

struct_cl_long2_0._pack_ = 1 # source:False
struct_cl_long2_0._fields_ = [
    ('x', ctypes.c_int64),
    ('y', ctypes.c_int64),
]

class struct_cl_long2_1(Structure):
    pass

struct_cl_long2_1._pack_ = 1 # source:False
struct_cl_long2_1._fields_ = [
    ('s0', ctypes.c_int64),
    ('s1', ctypes.c_int64),
]

class struct_cl_long2_2(Structure):
    pass

struct_cl_long2_2._pack_ = 1 # source:False
struct_cl_long2_2._fields_ = [
    ('lo', ctypes.c_int64),
    ('hi', ctypes.c_int64),
]

union_cl_long2._pack_ = 1 # source:False
union_cl_long2._anonymous_ = ('_0', '_1', '_2',)
union_cl_long2._fields_ = [
    ('s', ctypes.c_int64 * 2),
    ('_0', struct_cl_long2_0),
    ('_1', struct_cl_long2_1),
    ('_2', struct_cl_long2_2),
    ('v2', globals()['__cl_long2']),
]

cl_long2 = union_cl_long2
class union_cl_long4(Union):
    pass

class struct_cl_long4_0(Structure):
    pass

struct_cl_long4_0._pack_ = 1 # source:False
struct_cl_long4_0._fields_ = [
    ('x', ctypes.c_int64),
    ('y', ctypes.c_int64),
    ('z', ctypes.c_int64),
    ('w', ctypes.c_int64),
]

class struct_cl_long4_1(Structure):
    pass

struct_cl_long4_1._pack_ = 1 # source:False
struct_cl_long4_1._fields_ = [
    ('s0', ctypes.c_int64),
    ('s1', ctypes.c_int64),
    ('s2', ctypes.c_int64),
    ('s3', ctypes.c_int64),
]

class struct_cl_long4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_long2),
    ('hi', cl_long2),
     ]

union_cl_long4._pack_ = 1 # source:False
union_cl_long4._anonymous_ = ('_0', '_1', '_2',)
union_cl_long4._fields_ = [
    ('s', ctypes.c_int64 * 4),
    ('_0', struct_cl_long4_0),
    ('_1', struct_cl_long4_1),
    ('_2', struct_cl_long4_2),
    ('v2', union___m128i * 2),
]

cl_long4 = union_cl_long4
cl_long3 = union_cl_long4
class union_cl_long8(Union):
    pass

class struct_cl_long8_0(Structure):
    pass

struct_cl_long8_0._pack_ = 1 # source:False
struct_cl_long8_0._fields_ = [
    ('x', ctypes.c_int64),
    ('y', ctypes.c_int64),
    ('z', ctypes.c_int64),
    ('w', ctypes.c_int64),
]

class struct_cl_long8_1(Structure):
    pass

struct_cl_long8_1._pack_ = 1 # source:False
struct_cl_long8_1._fields_ = [
    ('s0', ctypes.c_int64),
    ('s1', ctypes.c_int64),
    ('s2', ctypes.c_int64),
    ('s3', ctypes.c_int64),
    ('s4', ctypes.c_int64),
    ('s5', ctypes.c_int64),
    ('s6', ctypes.c_int64),
    ('s7', ctypes.c_int64),
]

class struct_cl_long8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_long4),
    ('hi', cl_long4),
     ]

union_cl_long8._pack_ = 1 # source:False
union_cl_long8._anonymous_ = ('_0', '_1', '_2',)
union_cl_long8._fields_ = [
    ('s', ctypes.c_int64 * 8),
    ('_0', struct_cl_long8_0),
    ('_1', struct_cl_long8_1),
    ('_2', struct_cl_long8_2),
    ('v2', union___m128i * 4),
]

cl_long8 = union_cl_long8
class union_cl_long16(Union):
    pass

class struct_cl_long16_0(Structure):
    pass

struct_cl_long16_0._pack_ = 1 # source:False
struct_cl_long16_0._fields_ = [
    ('x', ctypes.c_int64),
    ('y', ctypes.c_int64),
    ('z', ctypes.c_int64),
    ('w', ctypes.c_int64),
    ('__spacer4', ctypes.c_int64),
    ('__spacer5', ctypes.c_int64),
    ('__spacer6', ctypes.c_int64),
    ('__spacer7', ctypes.c_int64),
    ('__spacer8', ctypes.c_int64),
    ('__spacer9', ctypes.c_int64),
    ('sa', ctypes.c_int64),
    ('sb', ctypes.c_int64),
    ('sc', ctypes.c_int64),
    ('sd', ctypes.c_int64),
    ('se', ctypes.c_int64),
    ('sf', ctypes.c_int64),
]

class struct_cl_long16_1(Structure):
    pass

struct_cl_long16_1._pack_ = 1 # source:False
struct_cl_long16_1._fields_ = [
    ('s0', ctypes.c_int64),
    ('s1', ctypes.c_int64),
    ('s2', ctypes.c_int64),
    ('s3', ctypes.c_int64),
    ('s4', ctypes.c_int64),
    ('s5', ctypes.c_int64),
    ('s6', ctypes.c_int64),
    ('s7', ctypes.c_int64),
    ('s8', ctypes.c_int64),
    ('s9', ctypes.c_int64),
    ('sA', ctypes.c_int64),
    ('sB', ctypes.c_int64),
    ('sC', ctypes.c_int64),
    ('sD', ctypes.c_int64),
    ('sE', ctypes.c_int64),
    ('sF', ctypes.c_int64),
]

class struct_cl_long16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_long8),
    ('hi', cl_long8),
     ]

union_cl_long16._pack_ = 1 # source:False
union_cl_long16._anonymous_ = ('_0', '_1', '_2',)
union_cl_long16._fields_ = [
    ('s', ctypes.c_int64 * 16),
    ('_0', struct_cl_long16_0),
    ('_1', struct_cl_long16_1),
    ('_2', struct_cl_long16_2),
    ('v2', union___m128i * 8),
]

cl_long16 = union_cl_long16
class union_cl_ulong2(Union):
    pass

class struct_cl_ulong2_0(Structure):
    pass

struct_cl_ulong2_0._pack_ = 1 # source:False
struct_cl_ulong2_0._fields_ = [
    ('x', ctypes.c_uint64),
    ('y', ctypes.c_uint64),
]

class struct_cl_ulong2_1(Structure):
    pass

struct_cl_ulong2_1._pack_ = 1 # source:False
struct_cl_ulong2_1._fields_ = [
    ('s0', ctypes.c_uint64),
    ('s1', ctypes.c_uint64),
]

class struct_cl_ulong2_2(Structure):
    pass

struct_cl_ulong2_2._pack_ = 1 # source:False
struct_cl_ulong2_2._fields_ = [
    ('lo', ctypes.c_uint64),
    ('hi', ctypes.c_uint64),
]

union_cl_ulong2._pack_ = 1 # source:False
union_cl_ulong2._anonymous_ = ('_0', '_1', '_2',)
union_cl_ulong2._fields_ = [
    ('s', ctypes.c_uint64 * 2),
    ('_0', struct_cl_ulong2_0),
    ('_1', struct_cl_ulong2_1),
    ('_2', struct_cl_ulong2_2),
    ('v2', globals()['__cl_ulong2']),
]

cl_ulong2 = union_cl_ulong2
class union_cl_ulong4(Union):
    pass

class struct_cl_ulong4_0(Structure):
    pass

struct_cl_ulong4_0._pack_ = 1 # source:False
struct_cl_ulong4_0._fields_ = [
    ('x', ctypes.c_uint64),
    ('y', ctypes.c_uint64),
    ('z', ctypes.c_uint64),
    ('w', ctypes.c_uint64),
]

class struct_cl_ulong4_1(Structure):
    pass

struct_cl_ulong4_1._pack_ = 1 # source:False
struct_cl_ulong4_1._fields_ = [
    ('s0', ctypes.c_uint64),
    ('s1', ctypes.c_uint64),
    ('s2', ctypes.c_uint64),
    ('s3', ctypes.c_uint64),
]

class struct_cl_ulong4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ulong2),
    ('hi', cl_ulong2),
     ]

union_cl_ulong4._pack_ = 1 # source:False
union_cl_ulong4._anonymous_ = ('_0', '_1', '_2',)
union_cl_ulong4._fields_ = [
    ('s', ctypes.c_uint64 * 4),
    ('_0', struct_cl_ulong4_0),
    ('_1', struct_cl_ulong4_1),
    ('_2', struct_cl_ulong4_2),
    ('v2', union___m128i * 2),
]

cl_ulong4 = union_cl_ulong4
cl_ulong3 = union_cl_ulong4
class union_cl_ulong8(Union):
    pass

class struct_cl_ulong8_0(Structure):
    pass

struct_cl_ulong8_0._pack_ = 1 # source:False
struct_cl_ulong8_0._fields_ = [
    ('x', ctypes.c_uint64),
    ('y', ctypes.c_uint64),
    ('z', ctypes.c_uint64),
    ('w', ctypes.c_uint64),
]

class struct_cl_ulong8_1(Structure):
    pass

struct_cl_ulong8_1._pack_ = 1 # source:False
struct_cl_ulong8_1._fields_ = [
    ('s0', ctypes.c_uint64),
    ('s1', ctypes.c_uint64),
    ('s2', ctypes.c_uint64),
    ('s3', ctypes.c_uint64),
    ('s4', ctypes.c_uint64),
    ('s5', ctypes.c_uint64),
    ('s6', ctypes.c_uint64),
    ('s7', ctypes.c_uint64),
]

class struct_cl_ulong8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ulong4),
    ('hi', cl_ulong4),
     ]

union_cl_ulong8._pack_ = 1 # source:False
union_cl_ulong8._anonymous_ = ('_0', '_1', '_2',)
union_cl_ulong8._fields_ = [
    ('s', ctypes.c_uint64 * 8),
    ('_0', struct_cl_ulong8_0),
    ('_1', struct_cl_ulong8_1),
    ('_2', struct_cl_ulong8_2),
    ('v2', union___m128i * 4),
]

cl_ulong8 = union_cl_ulong8
class union_cl_ulong16(Union):
    pass

class struct_cl_ulong16_0(Structure):
    pass

struct_cl_ulong16_0._pack_ = 1 # source:False
struct_cl_ulong16_0._fields_ = [
    ('x', ctypes.c_uint64),
    ('y', ctypes.c_uint64),
    ('z', ctypes.c_uint64),
    ('w', ctypes.c_uint64),
    ('__spacer4', ctypes.c_uint64),
    ('__spacer5', ctypes.c_uint64),
    ('__spacer6', ctypes.c_uint64),
    ('__spacer7', ctypes.c_uint64),
    ('__spacer8', ctypes.c_uint64),
    ('__spacer9', ctypes.c_uint64),
    ('sa', ctypes.c_uint64),
    ('sb', ctypes.c_uint64),
    ('sc', ctypes.c_uint64),
    ('sd', ctypes.c_uint64),
    ('se', ctypes.c_uint64),
    ('sf', ctypes.c_uint64),
]

class struct_cl_ulong16_1(Structure):
    pass

struct_cl_ulong16_1._pack_ = 1 # source:False
struct_cl_ulong16_1._fields_ = [
    ('s0', ctypes.c_uint64),
    ('s1', ctypes.c_uint64),
    ('s2', ctypes.c_uint64),
    ('s3', ctypes.c_uint64),
    ('s4', ctypes.c_uint64),
    ('s5', ctypes.c_uint64),
    ('s6', ctypes.c_uint64),
    ('s7', ctypes.c_uint64),
    ('s8', ctypes.c_uint64),
    ('s9', ctypes.c_uint64),
    ('sA', ctypes.c_uint64),
    ('sB', ctypes.c_uint64),
    ('sC', ctypes.c_uint64),
    ('sD', ctypes.c_uint64),
    ('sE', ctypes.c_uint64),
    ('sF', ctypes.c_uint64),
]

class struct_cl_ulong16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_ulong8),
    ('hi', cl_ulong8),
     ]

union_cl_ulong16._pack_ = 1 # source:False
union_cl_ulong16._anonymous_ = ('_0', '_1', '_2',)
union_cl_ulong16._fields_ = [
    ('s', ctypes.c_uint64 * 16),
    ('_0', struct_cl_ulong16_0),
    ('_1', struct_cl_ulong16_1),
    ('_2', struct_cl_ulong16_2),
    ('v2', union___m128i * 8),
]

cl_ulong16 = union_cl_ulong16
class union_cl_float2(Union):
    pass

class struct_cl_float2_0(Structure):
    pass

struct_cl_float2_0._pack_ = 1 # source:False
struct_cl_float2_0._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
]

class struct_cl_float2_1(Structure):
    pass

struct_cl_float2_1._pack_ = 1 # source:False
struct_cl_float2_1._fields_ = [
    ('s0', ctypes.c_float),
    ('s1', ctypes.c_float),
]

class struct_cl_float2_2(Structure):
    pass

struct_cl_float2_2._pack_ = 1 # source:False
struct_cl_float2_2._fields_ = [
    ('lo', ctypes.c_float),
    ('hi', ctypes.c_float),
]

union_cl_float2._pack_ = 1 # source:False
union_cl_float2._anonymous_ = ('_0', '_1', '_2',)
union_cl_float2._fields_ = [
    ('s', ctypes.c_float * 2),
    ('_0', struct_cl_float2_0),
    ('_1', struct_cl_float2_1),
    ('_2', struct_cl_float2_2),
    ('v2', globals()['__cl_float2']),
]

cl_float2 = union_cl_float2
class union_cl_float4(Union):
    pass

class struct_cl_float4_0(Structure):
    pass

struct_cl_float4_0._pack_ = 1 # source:False
struct_cl_float4_0._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('z', ctypes.c_float),
    ('w', ctypes.c_float),
]

class struct_cl_float4_1(Structure):
    pass

struct_cl_float4_1._pack_ = 1 # source:False
struct_cl_float4_1._fields_ = [
    ('s0', ctypes.c_float),
    ('s1', ctypes.c_float),
    ('s2', ctypes.c_float),
    ('s3', ctypes.c_float),
]

class struct_cl_float4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_float2),
    ('hi', cl_float2),
     ]

union_cl_float4._pack_ = 1 # source:False
union_cl_float4._anonymous_ = ('_0', '_1', '_2',)
union_cl_float4._fields_ = [
    ('s', ctypes.c_float * 4),
    ('_0', struct_cl_float4_0),
    ('_1', struct_cl_float4_1),
    ('_2', struct_cl_float4_2),
    ('v2', union___m64 * 2),
    ('v4', globals()['__cl_float4']),
]

cl_float4 = union_cl_float4
cl_float3 = union_cl_float4
class union_cl_float8(Union):
    pass

class struct_cl_float8_0(Structure):
    pass

struct_cl_float8_0._pack_ = 1 # source:False
struct_cl_float8_0._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('z', ctypes.c_float),
    ('w', ctypes.c_float),
]

class struct_cl_float8_1(Structure):
    pass

struct_cl_float8_1._pack_ = 1 # source:False
struct_cl_float8_1._fields_ = [
    ('s0', ctypes.c_float),
    ('s1', ctypes.c_float),
    ('s2', ctypes.c_float),
    ('s3', ctypes.c_float),
    ('s4', ctypes.c_float),
    ('s5', ctypes.c_float),
    ('s6', ctypes.c_float),
    ('s7', ctypes.c_float),
]

class struct_cl_float8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_float4),
    ('hi', cl_float4),
     ]

union_cl_float8._pack_ = 1 # source:False
union_cl_float8._anonymous_ = ('_0', '_1', '_2',)
union_cl_float8._fields_ = [
    ('s', ctypes.c_float * 8),
    ('_0', struct_cl_float8_0),
    ('_1', struct_cl_float8_1),
    ('_2', struct_cl_float8_2),
    ('v2', union___m64 * 4),
    ('v4', union___m128 * 2),
]

cl_float8 = union_cl_float8
class union_cl_float16(Union):
    pass

class struct_cl_float16_0(Structure):
    pass

struct_cl_float16_0._pack_ = 1 # source:False
struct_cl_float16_0._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('z', ctypes.c_float),
    ('w', ctypes.c_float),
    ('__spacer4', ctypes.c_float),
    ('__spacer5', ctypes.c_float),
    ('__spacer6', ctypes.c_float),
    ('__spacer7', ctypes.c_float),
    ('__spacer8', ctypes.c_float),
    ('__spacer9', ctypes.c_float),
    ('sa', ctypes.c_float),
    ('sb', ctypes.c_float),
    ('sc', ctypes.c_float),
    ('sd', ctypes.c_float),
    ('se', ctypes.c_float),
    ('sf', ctypes.c_float),
]

class struct_cl_float16_1(Structure):
    pass

struct_cl_float16_1._pack_ = 1 # source:False
struct_cl_float16_1._fields_ = [
    ('s0', ctypes.c_float),
    ('s1', ctypes.c_float),
    ('s2', ctypes.c_float),
    ('s3', ctypes.c_float),
    ('s4', ctypes.c_float),
    ('s5', ctypes.c_float),
    ('s6', ctypes.c_float),
    ('s7', ctypes.c_float),
    ('s8', ctypes.c_float),
    ('s9', ctypes.c_float),
    ('sA', ctypes.c_float),
    ('sB', ctypes.c_float),
    ('sC', ctypes.c_float),
    ('sD', ctypes.c_float),
    ('sE', ctypes.c_float),
    ('sF', ctypes.c_float),
]

class struct_cl_float16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_float8),
    ('hi', cl_float8),
     ]

union_cl_float16._pack_ = 1 # source:False
union_cl_float16._anonymous_ = ('_0', '_1', '_2',)
union_cl_float16._fields_ = [
    ('s', ctypes.c_float * 16),
    ('_0', struct_cl_float16_0),
    ('_1', struct_cl_float16_1),
    ('_2', struct_cl_float16_2),
    ('v2', union___m64 * 8),
    ('v4', union___m128 * 4),
]

cl_float16 = union_cl_float16
class union_cl_double2(Union):
    pass

class struct_cl_double2_0(Structure):
    pass

struct_cl_double2_0._pack_ = 1 # source:False
struct_cl_double2_0._fields_ = [
    ('x', ctypes.c_double),
    ('y', ctypes.c_double),
]

class struct_cl_double2_1(Structure):
    pass

struct_cl_double2_1._pack_ = 1 # source:False
struct_cl_double2_1._fields_ = [
    ('s0', ctypes.c_double),
    ('s1', ctypes.c_double),
]

class struct_cl_double2_2(Structure):
    pass

struct_cl_double2_2._pack_ = 1 # source:False
struct_cl_double2_2._fields_ = [
    ('lo', ctypes.c_double),
    ('hi', ctypes.c_double),
]

union_cl_double2._pack_ = 1 # source:False
union_cl_double2._anonymous_ = ('_0', '_1', '_2',)
union_cl_double2._fields_ = [
    ('s', ctypes.c_double * 2),
    ('_0', struct_cl_double2_0),
    ('_1', struct_cl_double2_1),
    ('_2', struct_cl_double2_2),
    ('v2', globals()['__cl_double2']),
]

cl_double2 = union_cl_double2
class union_cl_double4(Union):
    pass

class struct_cl_double4_0(Structure):
    pass

struct_cl_double4_0._pack_ = 1 # source:False
struct_cl_double4_0._fields_ = [
    ('x', ctypes.c_double),
    ('y', ctypes.c_double),
    ('z', ctypes.c_double),
    ('w', ctypes.c_double),
]

class struct_cl_double4_1(Structure):
    pass

struct_cl_double4_1._pack_ = 1 # source:False
struct_cl_double4_1._fields_ = [
    ('s0', ctypes.c_double),
    ('s1', ctypes.c_double),
    ('s2', ctypes.c_double),
    ('s3', ctypes.c_double),
]

class struct_cl_double4_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_double2),
    ('hi', cl_double2),
     ]

union_cl_double4._pack_ = 1 # source:False
union_cl_double4._anonymous_ = ('_0', '_1', '_2',)
union_cl_double4._fields_ = [
    ('s', ctypes.c_double * 4),
    ('_0', struct_cl_double4_0),
    ('_1', struct_cl_double4_1),
    ('_2', struct_cl_double4_2),
    ('v2', struct___m128d * 2),
]

cl_double4 = union_cl_double4
cl_double3 = union_cl_double4
class union_cl_double8(Union):
    pass

class struct_cl_double8_0(Structure):
    pass

struct_cl_double8_0._pack_ = 1 # source:False
struct_cl_double8_0._fields_ = [
    ('x', ctypes.c_double),
    ('y', ctypes.c_double),
    ('z', ctypes.c_double),
    ('w', ctypes.c_double),
]

class struct_cl_double8_1(Structure):
    pass

struct_cl_double8_1._pack_ = 1 # source:False
struct_cl_double8_1._fields_ = [
    ('s0', ctypes.c_double),
    ('s1', ctypes.c_double),
    ('s2', ctypes.c_double),
    ('s3', ctypes.c_double),
    ('s4', ctypes.c_double),
    ('s5', ctypes.c_double),
    ('s6', ctypes.c_double),
    ('s7', ctypes.c_double),
]

class struct_cl_double8_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_double4),
    ('hi', cl_double4),
     ]

union_cl_double8._pack_ = 1 # source:False
union_cl_double8._anonymous_ = ('_0', '_1', '_2',)
union_cl_double8._fields_ = [
    ('s', ctypes.c_double * 8),
    ('_0', struct_cl_double8_0),
    ('_1', struct_cl_double8_1),
    ('_2', struct_cl_double8_2),
    ('v2', struct___m128d * 4),
]

cl_double8 = union_cl_double8
class union_cl_double16(Union):
    pass

class struct_cl_double16_0(Structure):
    pass

struct_cl_double16_0._pack_ = 1 # source:False
struct_cl_double16_0._fields_ = [
    ('x', ctypes.c_double),
    ('y', ctypes.c_double),
    ('z', ctypes.c_double),
    ('w', ctypes.c_double),
    ('__spacer4', ctypes.c_double),
    ('__spacer5', ctypes.c_double),
    ('__spacer6', ctypes.c_double),
    ('__spacer7', ctypes.c_double),
    ('__spacer8', ctypes.c_double),
    ('__spacer9', ctypes.c_double),
    ('sa', ctypes.c_double),
    ('sb', ctypes.c_double),
    ('sc', ctypes.c_double),
    ('sd', ctypes.c_double),
    ('se', ctypes.c_double),
    ('sf', ctypes.c_double),
]

class struct_cl_double16_1(Structure):
    pass

struct_cl_double16_1._pack_ = 1 # source:False
struct_cl_double16_1._fields_ = [
    ('s0', ctypes.c_double),
    ('s1', ctypes.c_double),
    ('s2', ctypes.c_double),
    ('s3', ctypes.c_double),
    ('s4', ctypes.c_double),
    ('s5', ctypes.c_double),
    ('s6', ctypes.c_double),
    ('s7', ctypes.c_double),
    ('s8', ctypes.c_double),
    ('s9', ctypes.c_double),
    ('sA', ctypes.c_double),
    ('sB', ctypes.c_double),
    ('sC', ctypes.c_double),
    ('sD', ctypes.c_double),
    ('sE', ctypes.c_double),
    ('sF', ctypes.c_double),
]

class struct_cl_double16_2(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('lo', cl_double8),
    ('hi', cl_double8),
     ]

union_cl_double16._pack_ = 1 # source:False
union_cl_double16._anonymous_ = ('_0', '_1', '_2',)
union_cl_double16._fields_ = [
    ('s', ctypes.c_double * 16),
    ('_0', struct_cl_double16_0),
    ('_1', struct_cl_double16_1),
    ('_2', struct_cl_double16_2),
    ('v2', struct___m128d * 8),
]

cl_double16 = union_cl_double16
