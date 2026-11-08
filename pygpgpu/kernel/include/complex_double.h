#ifndef PYGPGPU_COMPLEX_DOUBLE_H
#define PYGPGPU_COMPLEX_DOUBLE_H

/* The double half of the complex arithmetic: the same operations the
 * float half has, at the width numpy's complex128 carries. */
#ifdef COMPLEX_NEED_DOUBLE

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_make(double re, double im)
{
    c128 r;
    r.re = re;
    r.im = im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_from_real(double value)
{
    c128 r;
    r.re = value;
    r.im = 0.0;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED double c128_real(c128 value)
{
    return value.re;
}

COMPLEX_QUAL static COMPLEX_UNUSED double c128_imag(c128 value)
{
    return value.im;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_add(c128 a, c128 b)
{
    c128 r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sub(c128 a, c128 b)
{
    c128 r;
    r.re = a.re - b.re;
    r.im = a.im - b.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_neg(c128 a)
{
    c128 r;
    r.re = -a.re;
    r.im = -a.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_mul(c128 a, c128 b)
{
    c128 r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_div(c128 a, c128 b)
{
    double d = b.re * b.re + b.im * b.im;
    c128 r;
    r.re = (a.re * b.re + a.im * b.im) / d;
    r.im = (a.im * b.re - a.re * b.im) / d;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_scale(c128 a, double s)
{
    c128 r;
    r.re = a.re * s;
    r.im = a.im * s;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_scale_div(c128 a, double s)
{
    c128 r;
    r.re = a.re / s;
    r.im = a.im / s;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_rdiv(double s, c128 b)
{
    return c128_div(c128_from_real(s), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_add_real(c128 a, double s)
{
    c128 r;
    r.re = a.re + s;
    r.im = a.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sub_real(c128 a, double s)
{
    c128 r;
    r.re = a.re - s;
    r.im = a.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_rsub(double s, c128 b)
{
    return c128_sub(c128_from_real(s), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c128_equal(c128 a, c128 b)
{
    return (a.re == b.re) && (a.im == b.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED double c128_abs(c128 a)
{
    return hypot(a.re, a.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED double c128_angle(c128 a)
{
    return atan2(a.im, a.re);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_conjugate(c128 a)
{
    c128 r;
    r.re = a.re;
    r.im = -a.im;
    return r;
}

/* Ordering, the predicates, the rounding and the sign: the double-width copies of what the
 * `c64` block declares above, and those comments say what each one is and why the device
 * needs it (numpy gives a complex value an order, predicates and a rounding). */
COMPLEX_QUAL static COMPLEX_UNUSED int c128_less(c128 a, c128 b)
{
    return (a.re < b.re) || (a.re == b.re && a.im < b.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_min(c128 a, c128 b)
{
    return c128_less(a, b) ? a : b;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_max(c128 a, c128 b)
{
    return c128_less(a, b) ? b : a;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_clamp(c128 a, c128 lo, c128 hi)
{
    return c128_min(c128_max(a, lo), hi);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c128_isnan(c128 a)
{
    return (a.re != a.re) || (a.im != a.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c128_isinf(c128 a)
{
    return (a.re > 1.7976931348623157e308) || (a.re < -1.7976931348623157e308)
        || (a.im > 1.7976931348623157e308) || (a.im < -1.7976931348623157e308);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c128_isfinite(c128 a)
{
    return !c128_isnan(a) && !c128_isinf(a);
}

/* numpy's NaN-ignoring `fmin`/`fmax`; the `c64` block above says why they differ from
 * `c128_min`/`c128_max`. */
COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_fmin(c128 a, c128 b)
{
    if (c128_isnan(a))
    {
        return b;
    }
    if (c128_isnan(b))
    {
        return a;
    }
    return c128_min(a, b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_fmax(c128 a, c128 b)
{
    if (c128_isnan(a))
    {
        return b;
    }
    if (c128_isnan(b))
    {
        return a;
    }
    return c128_max(a, b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_rint(c128 a)
{
    c128 r;
    r.re = rint(a.re);
    r.im = rint(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sign(c128 a)
{
    double magnitude = c128_abs(a);
    return (magnitude == 0.0) ? c128_from_real(0.0) : c128_scale_div(a, magnitude);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_square(c128 a)
{
    return c128_mul(a, a);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_reciprocal(c128 a)
{
    double d = a.re * a.re + a.im * a.im;
    c128 r;
    r.re = a.re / d;
    r.im = -a.im / d;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sqrt(c128 a)
{
    c128 r;
    if (a.re == 0.0 && a.im == 0.0)
    {
        r.re = 0.0;
        r.im = 0.0;
        return r;
    }
    double magnitude = hypot(a.re, a.im);
    if (a.re >= 0.0)
    {
        double t = sqrt((magnitude + a.re) * 0.5);
        r.re = t;
        r.im = a.im / (2.0 * t);
        return r;
    }
    double t = sqrt((magnitude - a.re) * 0.5);
    r.re = (a.im < 0.0 ? -1.0 : 1.0) * fabs(a.im) / (2.0 * t);
    r.im = t;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_exp(c128 a)
{
    double magnitude = exp(a.re);
    c128 r;
    r.re = magnitude * cos(a.im);
    r.im = magnitude * sin(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_log(c128 a)
{
    c128 r;
    r.re = log(hypot(a.re, a.im));
    r.im = atan2(a.im, a.re);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_exp2(c128 a)
{
    double magnitude = exp2(a.re);
    double angle = a.im * 0.69314718055994530942;
    c128 r;
    r.re = magnitude * cos(angle);
    r.im = magnitude * sin(angle);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_log2(c128 a)
{
    c128 natural = c128_log(a);
    c128 r;
    r.re = natural.re / 0.69314718055994530942;
    r.im = natural.im / 0.69314718055994530942;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_log10(c128 a)
{
    c128 natural = c128_log(a);
    c128 r;
    r.re = natural.re / 2.30258509299404568402;
    r.im = natural.im / 2.30258509299404568402;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sin(c128 a)
{
    c128 r;
    r.re = sin(a.re) * cosh(a.im);
    r.im = cos(a.re) * sinh(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_cos(c128 a)
{
    c128 r;
    r.re = cos(a.re) * cosh(a.im);
    r.im = -sin(a.re) * sinh(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_tan(c128 a)
{
    return c128_div(c128_sin(a), c128_cos(a));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_sinh(c128 a)
{
    c128 r;
    r.re = sinh(a.re) * cos(a.im);
    r.im = cosh(a.re) * sin(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_cosh(c128 a)
{
    c128 r;
    r.re = cosh(a.re) * cos(a.im);
    r.im = sinh(a.re) * sin(a.im);
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_tanh(c128 a)
{
    return c128_div(c128_sinh(a), c128_cosh(a));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_asin(c128 a)
{
    c128 one;
    one.re = 1.0;
    one.im = 0.0;
    c128 inside = c128_sub(one, c128_square(a));
    c128 root = c128_sqrt(inside);
    c128 shifted;
    shifted.re = -a.im + root.re;
    shifted.im = a.re + root.im;
    c128 logarithm = c128_log(shifted);
    c128 r;
    r.re = logarithm.im;
    r.im = -logarithm.re;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_acos(c128 a)
{
    c128 arc = c128_asin(a);
    c128 r;
    r.re = 1.57079632679489661923 - arc.re;
    r.im = -arc.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_atan(c128 a)
{
    c128 i_value;
    i_value.re = 0.0;
    i_value.im = 1.0;
    c128 numerator = c128_add(i_value, a);
    c128 denominator = c128_sub(i_value, a);
    c128 logarithm = c128_log(c128_div(numerator, denominator));
    c128 r;
    r.re = -0.5 * logarithm.im;
    r.im = 0.5 * logarithm.re;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_pow(c128 a, c128 b)
{
    if (a.re == 0.0 && a.im == 0.0)
    {
        c128 one;
        one.re = 1.0;
        one.im = 0.0;
        if (b.re == 0.0 && b.im == 0.0)
        {
            return one;
        }
        c128 zero;
        zero.re = 0.0;
        zero.im = 0.0;
        return zero;
    }
    return c128_exp(c128_mul(b, c128_log(a)));
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_pow_real(c128 a, double s)
{
    c128 exponent;
    exponent.re = s;
    exponent.im = 0.0;
    return c128_pow(a, exponent);
}

COMPLEX_QUAL static COMPLEX_UNUSED c128 c128_rpow(double s, c128 b)
{
    return c128_pow(c128_from_real(s), b);
}

#endif /* COMPLEX_NEED_DOUBLE */

#endif /* PYGPGPU_COMPLEX_DOUBLE_H */
