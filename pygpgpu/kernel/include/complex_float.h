#ifndef PYGPGPU_COMPLEX_FLOAT_H
#define PYGPGPU_COMPLEX_FLOAT_H

/* The float half of the complex arithmetic: the same operations the
 * double half has, at the width numpy's complex64 carries. */
#ifdef COMPLEX_NEED_FLOAT

/* (re, im) as a value, and the two real conversions. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_make(float re, float im)
{
    c64 r;
    r.re = re;
    r.im = im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_from_real(float value)
{
    c64 r;
    r.re = value;
    r.im = 0.0f;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED float c64_real(c64 value)
{
    return value.re;
}

COMPLEX_QUAL static COMPLEX_UNUSED float c64_imag(c64 value)
{
    return value.im;
}

/* z + w */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_add(c64 a, c64 b)
{
    c64 r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

/* z - w */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sub(c64 a, c64 b)
{
    c64 r;
    r.re = a.re - b.re;
    r.im = a.im - b.im;
    return r;
}

/* -z */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_neg(c64 a)
{
    c64 r;
    r.re = -a.re;
    r.im = -a.im;
    return r;
}

/* z * w: the full product, not a per-component scale. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_mul(c64 a, c64 b)
{
    c64 r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* z / w, by the conjugate denominator. A zero divisor answers what the arithmetic gives:
 * a real division by zero, which is an infinity or a NaN rather than a raised exception. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_div(c64 a, c64 b)
{
    float d = b.re * b.re + b.im * b.im;
    c64 r;
    r.re = (a.re * b.re + a.im * b.im) / d;
    r.im = (a.im * b.re - a.re * b.im) / d;
    return r;
}

/* z * s and z / s, with a real scalar: what numpy does for `complex * float`. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_scale(c64 a, float s)
{
    c64 r;
    r.re = a.re * s;
    r.im = a.im * s;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_scale_div(c64 a, float s)
{
    c64 r;
    r.re = a.re / s;
    r.im = a.im / s;
    return r;
}

/* s / z, and s - z: the real operand widened, which the header spells out so the call site
 * does not have to build the value itself. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_rdiv(float s, c64 b)
{
    return c64_div(c64_from_real(s), b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_add_real(c64 a, float s)
{
    c64 r;
    r.re = a.re + s;
    r.im = a.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sub_real(c64 a, float s)
{
    c64 r;
    r.re = a.re - s;
    r.im = a.im;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_rsub(float s, c64 b)
{
    return c64_sub(c64_from_real(s), b);
}

/* z == w, as the number the device's bool is: numpy compares the two parts. */
COMPLEX_QUAL static COMPLEX_UNUSED int c64_equal(c64 a, c64 b)
{
    return (a.re == b.re) && (a.im == b.im);
}

/* The magnitude: numpy's `abs(complex)` is a real, not a complex. */
COMPLEX_QUAL static COMPLEX_UNUSED float c64_abs(c64 a)
{
    return hypot(a.re, a.im);
}

/* The phase, in [-pi, pi]: numpy's `np.angle` for a complex scalar. */
COMPLEX_QUAL static COMPLEX_UNUSED float c64_angle(c64 a)
{
    return atan2(a.im, a.re);
}

/* The conjugate, by value -- the same value `z.conjugate()` answers. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_conjugate(c64 a)
{
    c64 r;
    r.re = a.re;
    r.im = -a.im;
    return r;
}

/* Ordering: numpy gives a complex value one, and it is the real parts that decide, with the
 * imaginary parts breaking a tie. `np.minimum`, `np.maximum` and `np.clip` are built on it,
 * and so are the host's `min`/`max`/`clamp`, which is why the device needs it too. */
COMPLEX_QUAL static COMPLEX_UNUSED int c64_less(c64 a, c64 b)
{
    return (a.re < b.re) || (a.re == b.re && a.im < b.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_min(c64 a, c64 b)
{
    return c64_less(a, b) ? a : b;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_max(c64 a, c64 b)
{
    return c64_less(a, b) ? b : a;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_clamp(c64 a, c64 lo, c64 hi)
{
    return c64_min(c64_max(a, lo), hi);
}

/* numpy's predicates on a complex value: a NaN or an infinity in *either* part answers the
 * first two, and `isfinite` needs both parts finite. The infinities are compared against
 * FLT_MAX rather than spelled INFINITY, which the CUDA headers do not define (see the
 * translator's math constants); a NaN compares false to everything, so it is neither. */
COMPLEX_QUAL static COMPLEX_UNUSED int c64_isnan(c64 a)
{
    return (a.re != a.re) || (a.im != a.im);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c64_isinf(c64 a)
{
    return (a.re > 3.4028234663852886e38f) || (a.re < -3.4028234663852886e38f)
        || (a.im > 3.4028234663852886e38f) || (a.im < -3.4028234663852886e38f);
}

COMPLEX_QUAL static COMPLEX_UNUSED int c64_isfinite(c64 a)
{
    return !c64_isnan(a) && !c64_isinf(a);
}

/* The NaN-ignoring pair, numpy's `fmin`/`fmax`, which differ from `c64_min`/`c64_max` in
 * exactly this: an operand holding a NaN loses. `minimum`/`maximum` propagate it (a NaN has a
 * defined place in the order), `fmin`/`fmax` return the other operand -- the reason numpy has
 * both families, and why the translator picks one by the builtin's own name. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_fmin(c64 a, c64 b)
{
    if (c64_isnan(a))
    {
        return b;
    }
    if (c64_isnan(b))
    {
        return a;
    }
    return c64_min(a, b);
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_fmax(c64 a, c64 b)
{
    if (c64_isnan(a))
    {
        return b;
    }
    if (c64_isnan(b))
    {
        return a;
    }
    return c64_max(a, b);
}

/* Rounding to the nearest even integer, per part: numpy's `np.round` and `np.rint` on a
 * complex value, and the same rule the real path spells `rint`. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_rint(c64 a)
{
    c64 r;
    r.re = rint(a.re);
    r.im = rint(a.im);
    return r;
}

/* numpy's `sign` for a complex value: the direction on the unit circle. Only zero answers
 * zero -- `z / abs(z)` there is 0/0, a NaN numpy never answers. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sign(c64 a)
{
    float magnitude = c64_abs(a);
    return (magnitude == 0.0f) ? c64_from_real(0.0f) : c64_scale_div(a, magnitude);
}

/* z * z and 1 / z. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_square(c64 a)
{
    return c64_mul(a, a);
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_reciprocal(c64 a)
{
    float d = a.re * a.re + a.im * a.im;
    c64 r;
    r.re = a.re / d;
    r.im = -a.im / d;
    return r;
}

/* The principal square root.
 *
 * The closed form from |z| and the sign of the real part: for a positive real part the
 * root's real part is sqrt((|z| + re) / 2) and its imaginary part follows from the product,
 * and for a negative one the two swap roles with the sign of im carried through. The zero
 * case is answered directly, because both formulas divide by a zero there. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sqrt(c64 a)
{
    c64 r;
    if (a.re == 0.0f && a.im == 0.0f)
    {
        r.re = 0.0f;
        r.im = 0.0f;
        return r;
    }
    float magnitude = hypot(a.re, a.im);
    if (a.re >= 0.0f)
    {
        float t = sqrt((magnitude + a.re) * 0.5f);
        r.re = t;
        r.im = a.im / (2.0f * t);
        return r;
    }
    float t = sqrt((magnitude - a.re) * 0.5f);
    r.re = (a.im < 0.0f ? -1.0f : 1.0f) * fabs(a.im) / (2.0f * t);
    r.im = t;
    return r;
}

/* exp(z) = exp(re) * (cos(im) + i sin(im)) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_exp(c64 a)
{
    float magnitude = exp(a.re);
    c64 r;
    r.re = magnitude * cos(a.im);
    r.im = magnitude * sin(a.im);
    return r;
}

/* log(z) = log|z| + i arg(z) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_log(c64 a)
{
    c64 r;
    r.re = log(hypot(a.re, a.im));
    r.im = atan2(a.im, a.re);
    return r;
}

/* 2^z = exp2(re) * (cos(im ln 2) + i sin(im ln 2)) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_exp2(c64 a)
{
    float magnitude = exp2(a.re);
    float angle = a.im * 0.6931471805599453f;
    c64 r;
    r.re = magnitude * cos(angle);
    r.im = magnitude * sin(angle);
    return r;
}

/* log2(z) and log10(z): the natural logarithm rescaled, which is what numpy computes. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_log2(c64 a)
{
    c64 natural = c64_log(a);
    c64 r;
    r.re = natural.re / 0.6931471805599453f;
    r.im = natural.im / 0.6931471805599453f;
    return r;
}

COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_log10(c64 a)
{
    c64 natural = c64_log(a);
    c64 r;
    r.re = natural.re / 2.302585092994046f;
    r.im = natural.im / 2.302585092994046f;
    return r;
}

/* sin(z) = sin(re)cosh(im) + i cos(re)sinh(im) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sin(c64 a)
{
    c64 r;
    r.re = sin(a.re) * cosh(a.im);
    r.im = cos(a.re) * sinh(a.im);
    return r;
}

/* cos(z) = cos(re)cosh(im) - i sin(re)sinh(im) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_cos(c64 a)
{
    c64 r;
    r.re = cos(a.re) * cosh(a.im);
    r.im = -sin(a.re) * sinh(a.im);
    return r;
}

/* tan(z) = sin(z) / cos(z) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_tan(c64 a)
{
    return c64_div(c64_sin(a), c64_cos(a));
}

/* sinh(z) = sinh(re)cos(im) + i cosh(re)sin(im) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_sinh(c64 a)
{
    c64 r;
    r.re = sinh(a.re) * cos(a.im);
    r.im = cosh(a.re) * sin(a.im);
    return r;
}

/* cosh(z) = cosh(re)cos(im) + i sinh(re)sin(im) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_cosh(c64 a)
{
    c64 r;
    r.re = cosh(a.re) * cos(a.im);
    r.im = sinh(a.re) * sin(a.im);
    return r;
}

/* tanh(z) = sinh(z) / cosh(z) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_tanh(c64 a)
{
    return c64_div(c64_sinh(a), c64_cosh(a));
}

/* asin(z) = -i log(iz + sqrt(1 - z^2)) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_asin(c64 a)
{
    c64 one;
    one.re = 1.0f;
    one.im = 0.0f;
    c64 inside = c64_sub(one, c64_square(a));
    c64 root = c64_sqrt(inside);
    c64 shifted;
    shifted.re = -a.im + root.re;
    shifted.im = a.re + root.im;
    c64 logarithm = c64_log(shifted);
    c64 r;
    r.re = logarithm.im;
    r.im = -logarithm.re;
    return r;
}

/* acos(z) = pi/2 - asin(z) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_acos(c64 a)
{
    c64 arc = c64_asin(a);
    c64 r;
    r.re = 1.5707963267948966f - arc.re;
    r.im = -arc.im;
    return r;
}

/* atan(z) = (i/2) log((i + z) / (i - z)) */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_atan(c64 a)
{
    c64 i_value;
    i_value.re = 0.0f;
    i_value.im = 1.0f;
    c64 numerator = c64_add(i_value, a);
    c64 denominator = c64_sub(i_value, a);
    c64 logarithm = c64_log(c64_div(numerator, denominator));
    c64 r;
    r.re = -0.5f * logarithm.im;
    r.im = 0.5f * logarithm.re;
    return r;
}

/* z^w = exp(w log z). A zero base answers 1 for a zero exponent and 0 otherwise, which is
 * what numpy's `0 ** w` answers for a complex w of positive real part. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_pow(c64 a, c64 b)
{
    if (a.re == 0.0f && a.im == 0.0f)
    {
        c64 one;
        one.re = 1.0f;
        one.im = 0.0f;
        if (b.re == 0.0f && b.im == 0.0f)
        {
            return one;
        }
        c64 zero;
        zero.re = 0.0f;
        zero.im = 0.0f;
        return zero;
    }
    return c64_exp(c64_mul(b, c64_log(a)));
}

/* z^s for a real exponent, which avoids building the widened value at the call site. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_pow_real(c64 a, float s)
{
    c64 exponent;
    exponent.re = s;
    exponent.im = 0.0f;
    return c64_pow(a, exponent);
}

/* s^z for a real base: the base widened, then the general power. */
COMPLEX_QUAL static COMPLEX_UNUSED c64 c64_rpow(float s, c64 b)
{
    return c64_pow(c64_from_real(s), b);
}

#endif /* COMPLEX_NEED_FLOAT */

#endif /* PYGPGPU_COMPLEX_FLOAT_H */
