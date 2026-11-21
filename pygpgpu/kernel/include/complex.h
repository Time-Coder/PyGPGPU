/*
 * Complex numbers, for generated kernels.
 *
 * A complex value is the one numeric type neither C nor OpenCL C, CUDA C or MSL has: numpy
 * carries it as a pair of reals, and device code has to do the same. The two structures are
 * declared here, with the member names numpy's own layout gives them, so a generated source
 * that includes this file needs no declaration of its own -- and so a `complex64` array binds
 * to a `c64*` parameter with no conversion at all (the two layouts are byte-identical).
 *
 * The names are fixed rather than content hashes (`c64` / `c128`), which is what makes that
 * possible: the layout is known in advance, so the header can spell the type.
 *
 * A generated source defines, before including this file:
 *   COMPLEX_QUAL          empty, or `__device__` where the backend demands it
 *   COMPLEX_NEED_FLOAT    when a complex64 value is used
 *   COMPLEX_NEED_DOUBLE   when a complex128 value is used
 *
 * Each block is guarded because its body is only legal where the width is: an OpenCL device
 * without `cl_khr_fp64` rejects a `double` variable, so the double block must not be compiled
 * merely because the header was included. The mixed block needs both, and a device without
 * fp64 cannot do either side of it.
 *
 * Every function here mirrors what numpy computes for the same operation on a complex scalar,
 * which is also what the interpreter backend computes -- the host side of these names is
 * `pygpgpu.vec_types.funcs` (pygf's own element-wise helpers, extended to take a complex
 * scalar), so the two have to agree. The formulas are the closed forms:
 *
 *   exp(z)   = exp(re) * (cos(im) + i sin(im))
 *   log(z)   = log|z| + i arg(z)
 *   sqrt(z)  = the principal root, from |z| and the sign of re
 *   sin(z)   = sin(re)cosh(im) + i cos(re)sinh(im)
 *   sinh(z)  = sinh(re)cos(im) + i cosh(re)sin(im)
 *   asin(z)  = -i log(iz + sqrt(1 - z^2))
 *   atan(z)  = (i/2) log((i + z) / (i - z))
 *   2^z      = exp2(re) * (cos(im ln 2) + i sin(im ln 2))
 *   z^w      = exp(w log z)
 *
 * The real functions are written under their plain C++ spellings (`sqrt`, not `sqrtf`):
 * OpenCL C 1.2 has only the overloaded form (there is no `sqrtf` there), while CUDA, HIP and
 * Metal are C++ and resolve `sqrt(float)` to the float overload. This is the same rule the
 * quaternion header follows, and it is why a `sqrtf` in this file would be a bug.
 */
#ifndef PYGPGPU_COMPLEX_H
#define PYGPGPU_COMPLEX_H

#ifndef COMPLEX_QUAL
#define COMPLEX_QUAL
#endif
/* Every function below is `static`, so a kernel that calls two of them leaves the rest
 * unreferenced and NVRTC reports `warning #177-D: function ... was declared but never
 * referenced` for each -- nineteen of them for a Mandelbrot kernel that uses four
 * operations. The attribute is what says "unused is expected here", and it is accepted by
 * every backend this header is compiled by: NVRTC and HIP's clang are C++, and OpenCL C
 * and MSL both take GNU-style function attributes. Overridable so a backend that spells it
 * differently can define its own. */
#ifndef COMPLEX_UNUSED
#define COMPLEX_UNUSED __attribute__((unused))
#endif
/* The magnitude helper. MSL has no `hypot` (metal_stdlib's math block does not
 * declare one), so the default is the sqrt-of-squares form every backend accepts;
 * a backend whose library has the accurate form can define COMPLEX_HYPOT itself. */
#ifndef COMPLEX_HYPOT
#define COMPLEX_HYPOT(a, b) sqrt((a) * (a) + (b) * (b))
#endif
/* The trig and exponential helpers. MSL spells the family only in the
 * `metal::fast` / `metal::precise` namespaces, and its global-namespace overloads
 * (which this header's callers rely on, C-style) compile to the *precise* forms
 * unless `__METAL_MATH_FP32_FUNCTIONS_FAST__` is defined -- so the bare spelling
 * already answers what numpy answers, and OpenCL C has only the bare spelling.
 * The macros therefore default to the plain name; Metal-compiled sources do not
 * need to (and must not) qualify, and no backend needs `precise::` here. */
#ifndef COMPLEX_SIN
#define COMPLEX_SIN(x) sin(x)
#endif
#ifndef COMPLEX_COS
#define COMPLEX_COS(x) cos(x)
#endif
#ifndef COMPLEX_EXP
#define COMPLEX_EXP(x) exp(x)
#endif
#ifndef COMPLEX_EXP2
#define COMPLEX_EXP2(x) exp2(x)
#endif
#ifndef COMPLEX_LOG
#define COMPLEX_LOG(x) log(x)
#endif
#ifndef COMPLEX_LOG2
#define COMPLEX_LOG2(x) log2(x)
#endif
#ifndef COMPLEX_LOG10
#define COMPLEX_LOG10(x) log10(x)
#endif
#ifndef COMPLEX_POW
#define COMPLEX_POW(a, b) pow(a, b)
#endif
#ifndef COMPLEX_SQRT
#define COMPLEX_SQRT(x) sqrt(x)
#endif
#ifndef COMPLEX_ATAN
#define COMPLEX_ATAN(x) atan(x)
#endif
#ifndef COMPLEX_ASIN
#define COMPLEX_ASIN(x) asin(x)
#endif
#ifndef COMPLEX_ACOS
#define COMPLEX_ACOS(x) acos(x)
#endif
#ifndef COMPLEX_SINH
#define COMPLEX_SINH(x) sinh(x)
#endif
#ifndef COMPLEX_COSH
#define COMPLEX_COSH(x) cosh(x)
#endif
#ifndef COMPLEX_TANH
#define COMPLEX_TANH(x) tanh(x)
#endif

/* The two structures a kernel can use. The member names are the ones numpy's layout has,
 * which is what lets a `complex64` array be bound to a `c64*` parameter unchanged. */
typedef struct { float re; float im; } c64;
/* The double-precision form sits behind its need macro: MSL has no `double`, so a
 * Metal kernel that never names a complex128 compiles this header cleanly, and a
 * kernel that does is refused by the translator before the compiler sees the
 * typedef (see `_python_complex_kind`'s refusal and `_convert_complex`'s). */
#ifdef COMPLEX_NEED_DOUBLE
typedef struct { double re; double im; } c128;
#endif

#define PYGPGPU_COMPLEX_QUAL_HELPER(name) name

/* The blocks below are the same arithmetic at two widths. They are written out rather than
 * generated by a macro so the file stays readable; a change to the maths belongs in both. */

#include "complex_float.h"
#ifdef COMPLEX_NEED_DOUBLE
#include "complex_double.h"
#include "complex_mixed.h"
#endif
#endif /* PYGPGPU_COMPLEX_H */

