#ifndef PYGPGPU_COMPLEX_MIXED_H
#define PYGPGPU_COMPLEX_MIXED_H

/* The mixed float/double operations. Every result is the double form,
 * which is the rule the rest of this package follows: a double
 * operand widens the whole operation, and nothing is ever rounded
 * down. */
#if defined(COMPLEX_NEED_FLOAT) && defined(COMPLEX_NEED_DOUBLE)

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_widen_c128(c64 a)
{
    c128 r;
    r.re = a.re;
    r.im = a.im;
    return r;
}

/* Narrowing exists so `np.complex64(z)` has something to call; it is the one conversion that
 * does lose precision, and the caller asked for it by name. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c128_narrow_c64(c128 a)
{
    c64 r;
    r.re = (float)a.re;
    r.im = (float)a.im;
    return r;
}

/* float complex * double complex, either order. */
COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_mul_c128(c64 a, c128 b)
{
    return c128_mul(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_mul_c64(c128 a, c64 b)
{
    return c128_mul(a, c64_widen_c128(b));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_add_c128(c64 a, c128 b)
{
    return c128_add(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_add_c64(c128 a, c64 b)
{
    return c128_add(a, c64_widen_c128(b));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_sub_c128(c64 a, c128 b)
{
    return c128_sub(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sub_c64(c128 a, c64 b)
{
    return c128_sub(a, c64_widen_c128(b));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_div_c128(c64 a, c128 b)
{
    return c128_div(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_div_c64(c128 a, c64 b)
{
    return c128_div(a, c64_widen_c128(b));
}

COMPLEX_QUAL static COMPLEX_UNUSED int c64_equal_c128(c64 a, c128 b)
{
    return c128_equal(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c128_equal_c64(c128 a, c64 b)
{
    return c128_equal(a, c64_widen_c128(b));
}

/* A float complex meeting a double *scalar* answers a double complex, and so does the
 * reverse; the widening is the same one the mixed pair uses. */
COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_scale_c128(c64 a, double s)
{
    return c128_scale(c64_widen_c128(a), s);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_add_real_c128(c64 a, double s)
{
    return c128_add_real(c64_widen_c128(a), s);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_sub_real_c128(c64 a, double s)
{
    return c128_sub_real(c64_widen_c128(a), s);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_scale_div_c128(c64 a, double s)
{
    return c128_scale_div(c64_widen_c128(a), s);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_div_c128_scalar(c64 a, double s)
{
    return c128_scale_div(c64_widen_c128(a), s);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_pow_c128(c64 a, c128 b)
{
    return c128_pow(c64_widen_c128(a), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_pow_c64(c128 a, c64 b)
{
    return c128_pow(a, c64_widen_c128(b));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c64_pow_real_c128(c64 a, double s)
{
    return c128_pow_real(c64_widen_c128(a), s);
}

#endif /* mixed widths */

#endif /* PYGPGPU_COMPLEX_MIXED_H */
