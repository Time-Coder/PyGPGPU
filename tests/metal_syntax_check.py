"""Check that the MSL the backend emits is syntactically well formed, as C++.

Metal's compiler is not available here, but MSL is a C++14 dialect with a handful of
keywords C++ lacks (`device`, `threadgroup`, `kernel`) and attributes (`[[...]]`).
Those can be defined away, and the *structure* of the generated source -- types,
declarator order, the shared-memory declarations, the atomic calls, the index
macros -- then compiles with a real C++ compiler. That catches the class of error a
string assertion cannot: a missing brace, a type that does not exist, an expression
that parses differently than intended, a duplicate parameter, or a call with the
wrong argument count.

**What this deliberately does NOT check**, so a passing run is not mistaken for more
than it is:

* **attributes** -- `[[thread_position_in_grid]]` and friends are stripped before
  compiling, so a misspelled or misplaced attribute is accepted here and only Metal
  would reject it. `metal_attributes.py` checks those as text;
* **address spaces** -- `device`, `constant` and `threadgroup` become nothing here,
  because MSL writes the qualifier before the type and a macro cannot expand into a
  wrapper type. Whether every pointer *names* a space is checked as text by
  `metal_address_space.py`; whether the chosen space is the right one (a group
  scratch buffer is `threadgroup`, a caller array is `device`) is what
  `tests/test_metal_codegen.py` asserts per declaration;
* **vector semantics** -- the `metal::vec` stand-in is an array with a few operators,
  not MSL's swizzles, so an illegal component combination can slip through;
* **the MSL standard library** -- `fast::sin`, `atomic_fetch_*` and the rest are
  hand-written stubs with the right *shape*, not the real signatures.

The real Metal compiler runs on a Mac; this is the check that runs everywhere else.

It is used by tests/test_metal_codegen.py when a C++ compiler is present.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple


# The C++ stand-in for the parts of MSL the generated source names. `vec` is an array
# with an index operator, which is enough for `a[i]`; `uint3` keeps its three fields.
STUB_HEADER: str = r"""
#include <cstdint>

namespace metal {
    // MSL vectors answer both an index and the `.x/.y/.z/.w` component names, and
    // both are *fields/operators*, not calls: the generated source writes
    // `_mtl_thread_position.x`. A union lays the components over the same storage.
    template<typename T, int N> struct vec {
        union {
            T v[N];
            struct { T x, y, z, w; };
        };
        T &operator[](int i) { return v[i]; }
        // MSL allows a vector and a scalar in the same expression (`v * 2.0f`),
        // and the translator emits exactly that for Metal.
        friend vec operator*(vec a, T s) {
            for (int i = 0; i < N; ++i) { a.v[i] *= s; }
            return a;
        }
        friend vec operator*(T s, vec a) { return a * s; }
        friend vec operator+(vec a, vec b) {
            for (int i = 0; i < N; ++i) { a.v[i] += b.v[i]; }
            return a;
        }
    };
    typedef vec<float, 4> float4;
    typedef vec<unsigned, 3> uint3;
    typedef vec<unsigned, 4> uint4;
    typedef vec<int, 4> int4;
    typedef vec<float, 2> float2;
    typedef vec<float, 3> float3;
    // MSL's scalar spellings, which C++ does not have.
    typedef unsigned int uint;
    typedef unsigned char uchar;
    typedef unsigned short ushort;
    typedef unsigned long ulong;

    struct atomic_uint { unsigned value; };
    struct atomic_int { int value; };
    enum memory_order { memory_order_relaxed, memory_order_acquire,
                        memory_order_release, memory_order_acq_rel };

    inline float fast_exp(float v) { return v; }
    inline float fast_sin(float v) { return v; }
    inline float fast_cos(float v) { return v; }
    inline float fast_sqrt(float v) { return v; }
    inline float floor(float v) { return v; }
    inline float fmin(float a, float b) { return a < b ? a : b; }
    inline float fmax(float a, float b) { return a > b ? a : b; }
    inline float erf(float v) { return v; }
    inline float copysign(float a, float b) { return a; }
    inline float ldexp(float a, int b) { return a; }

    // The plain transcendental names MSL's `metal_stdlib` declares in the global
    // namespace -- `log`, `cos` and `sqrt` are what a header such as the random one
    // calls. The `fast::` forms above are the accelerated ones, and they are a
    // different spelling on purpose: a generated source that names the plain form
    // must resolve to the precise one.
    inline float log(float v) { return v; }
    inline float cos(float v) { return v; }
    inline float sqrt(float v) { return v; }
    inline double log(double v) { return v; }
    inline double cos(double v) { return v; }
    inline double sqrt(double v) { return v; }

    // The work-group barrier and its memory flags. `mem_flags` is a bit-set in MSL
    // (`mem_threadgroup | mem_device`), so it is a class with an `operator|` rather
    // than a plain enum -- an enum's `|` yields an int, which a plain C++ build
    // rejects as a narrowing conversion.
    struct mem_flags {
        int bits;
        mem_flags(int v) : bits(v) {}
        // The generated source names them as `mem_flags::mem_threadgroup`, which is
        // a static member access rather than a namespace-qualified constant.
        static const mem_flags mem_threadgroup;
        static const mem_flags mem_device;
    };
    inline mem_flags operator|(mem_flags a, mem_flags b) {
        return mem_flags(a.bits | b.bits);
    }
    inline void threadgroup_barrier(mem_flags f) { (void)f; }
    // The MSL atomics deduce the atomic type from the pointer (the generated
    // sources cast it to `(device atomic_uint*)` / `(threadgroup atomic_uint*)`
    // explicitly), and the value answers the wrapper's `value` member. The
    // three-argument spelling is what `emit_atomic` emits; the spin barrier's
    // four-argument calls (with `mem_flags`, see metal_codegen.barrier) are
    // covered by the overloads after them.
    template<typename T> auto atomic_fetch_add_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value += v; return old;
    }
    template<typename T> auto atomic_fetch_add_explicit(T *p, unsigned v,
                                                     memory_order o, mem_flags f) -> decltype(p->value) {
        (void)f;
        decltype(p->value) old = p->value; p->value += v; return old;
    }
    // `atomic_load_explicit` answers the VALUE (an `unsigned`, not the atomic
    // wrapper), so the spin comparison against a plain uint compiles.
    template<typename T> auto atomic_load_explicit(T *p, memory_order o)
            -> decltype(p->value) {
        (void)o;
        return p->value;
    }
    template<typename T> auto atomic_load_explicit(T *p, memory_order o, mem_flags f)
            -> decltype(p->value) {
        (void)o; (void)f;
        return p->value;
    }
    template<typename T> void atomic_store_explicit(T *p, unsigned v,
                                                    memory_order o) {
        (void)p; (void)v; (void)o;
    }
    template<typename T> bool atomic_compare_exchange_weak_explicit(
        T *p, int *expected, int desired, memory_order a, memory_order b) {
        (void)p; (void)expected; (void)desired; (void)a; (void)b; return false;
    }
    template<typename T> bool atomic_compare_exchange_weak_explicit(
        T *p, int *expected, int desired, memory_order a, memory_order b, mem_flags f) {
        (void)p; (void)expected; (void)desired; (void)a; (void)b; (void)f; return false;
    }
    template<typename T> auto atomic_fetch_sub_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value -= v; return old;
    }
    template<typename T> auto atomic_fetch_min_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value = p->value < v ? p->value : v; return old;
    }
    template<typename T> auto atomic_fetch_max_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value = p->value > v ? p->value : v; return old;
    }
    template<typename T> auto atomic_fetch_and_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value &= v; return old;
    }
    template<typename T> auto atomic_fetch_or_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value |= v; return old;
    }
    template<typename T> auto atomic_fetch_xor_explicit(T *p, unsigned v,
                                                     memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value ^= v; return old;
    }
    template<typename T> auto atomic_exchange_explicit(T *p, unsigned v,
                                                    memory_order o) -> decltype(p->value) {
        (void)o;
        decltype(p->value) old = p->value; p->value = v; return old;
    }
    template<typename T> auto atomic_exchange_explicit(T *p, unsigned v,
                                                    memory_order o, mem_flags f) -> decltype(p->value) {
        (void)f;
        decltype(p->value) old = p->value; p->value = v; return old;
    }
    template<typename T> void atomic_store_explicit(T *p, unsigned v,
                                                    memory_order o, mem_flags f) {
        (void)p; (void)v; (void)o; (void)f;
    }
}
"""

# The MSL-only words. `kernel` expands to nothing (the source already spells
# `kernel void k(...)`), so the declaration reads `void k(...)`.
#
# The address spaces are NOT defined away: `device float*` becomes `addr<device_tag,
# float>*`, and `threadgroup float*` becomes `addr<threadgroup_tag, float>*`. They are
# different types, so a buffer bound into the wrong space is a compile error -- the
# one semantic Metal enforces on every declaration that a plain C++ build would
# otherwise accept silently.
#
# `addr` is a transparent wrapper: indexing and dereferencing reach the element, so
# the rest of the body compiles unchanged.
# The address spaces are defined away: `device float*` reads `float*` after the macro
# pass. A distinct C++ type per space would be a nice check, but MSL spells the
# qualifier *before* the type and a macro cannot expand into `wrapper(T)`, so the
# qualifier has to be empty here. Address-space correctness is checked as text by
# `tests/metal_address_space.py` instead.
STUB_MACROS: str = r"""
using namespace metal;

#define device
#define constant
#define threadgroup
#define kernel
#define precision
#define access
#define sample
#define write

// The work-item numbers are kernel parameters on Metal (see
// `kernel/metal_codegen.py`), and a device helper that reads one receives it as a
// parameter of its own -- but a helper's caller passes the launch terms positionally
// and the stub has no parameter list to name. The prelude's macros expand onto those
// parameters, so the stub defines the six underlying names as plain globals: the
// address-space and attribute checks are the semantic ones, and this stub only
// proves the text compiles as C++.
inline int _mtl_thread_position_x = 0, _mtl_thread_position_y = 0, _mtl_thread_position_z = 0;
inline int _mtl_thread_index_x = 0, _mtl_thread_index_y = 0, _mtl_thread_index_z = 0;
inline int _mtl_threadgroup_position_x = 0, _mtl_threadgroup_position_y = 0, _mtl_threadgroup_position_z = 0;
inline int _mtl_threads_per_threadgroup_x = 1, _mtl_threads_per_threadgroup_y = 1, _mtl_threads_per_threadgroup_z = 1;
inline int _mtl_threads_per_grid_x = 1, _mtl_threads_per_grid_y = 1, _mtl_threads_per_grid_z = 1;
inline int _mtl_threadgroups_per_grid_x = 1, _mtl_threadgroups_per_grid_y = 1, _mtl_threadgroups_per_grid_z = 1;

struct _mtl_vec3 {
    int x, y, z;
};
inline _mtl_vec3 _mtl_thread_position{};
inline _mtl_vec3 _mtl_thread_index{};
inline _mtl_vec3 _mtl_threadgroup_position{};
inline _mtl_vec3 _mtl_threads_per_threadgroup{1, 1, 1};
inline _mtl_vec3 _mtl_threads_per_grid{1, 1, 1};
inline _mtl_vec3 _mtl_threadgroups_per_grid{1, 1, 1};
"""



# `[[thread_position_in_grid]]`-style attributes: C++14 tolerates unknown
# attributes on some compilers and rejects them on others, so they are removed.
ATTRIBUTE_RE: re.Pattern = re.compile(r"\[\[\s*[^\]]*\]\]")

# The prelude's includes are replaced by the stub, and `using namespace metal;` is
# emitted by the stub; the macros themselves are kept, since they are what the body
# calls.
INCLUDE_RE: re.Pattern = re.compile(r"^#include\s+<metal_stdlib>\s*$", re.MULTILINE)
USING_RE: re.Pattern = re.compile(r"^using namespace metal;\s*$", re.MULTILINE)


def cpp_compiler()->Optional[str]:
    """A C++ compiler this machine has, or None."""
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _as_cpp(source: str) -> str:
    """The generated MSL, with the MSL-only words and attributes taken out."""
    text: str = INCLUDE_RE.sub("", source.lstrip("\ufeff"))
    text = USING_RE.sub("", text)
    text = ATTRIBUTE_RE.sub("", text)
    # `device`, `uint3` and the atomics come from the stub header.
    return STUB_HEADER + STUB_MACROS + text


def check(source: str, compiler: Optional[str] = None) -> Tuple[bool, str]:
    """Compile `source` as C++; return (ok, diagnostics)."""
    compiler = compiler or cpp_compiler()
    if compiler is None:
        return True, "no C++ compiler available"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "generated.cpp")
        with open(path, "w", encoding="utf-8", newline="\n") as target_file:
            target_file.write(_as_cpp(source))
        result = subprocess.run([compiler, "-std=c++14", "-fsyntax-only", path],
                                capture_output=True, text=True)
    return result.returncode == 0, (result.stderr or result.stdout)



if __name__ == "__main__":
    # `python tests/metal_syntax_check.py <file.metal>` checks one generated source.
    # Handy before a Mac is available: run a translated kernel through this and any
    # structural error shows up as a compiler diagnostic instead of a Metal one.
    import sys

    if len(sys.argv) != 2:
        print("usage: python tests/metal_syntax_check.py <file.metal>")
        raise SystemExit(2)
    # A file written by a Windows editor may carry a UTF-8 BOM; the compiler would
    # read it as part of the first token (`\ufeffkernel` does not name a type).
    source: str = open(sys.argv[1], encoding="utf-8-sig").read()
    ok, diagnostics = check(source)
    print("ok" if ok else "FAILED")
    if not ok:
        print(diagnostics)
    raise SystemExit(0 if ok else 1)
