/*
 * Continuous distributions, for generated kernels.
 *
 * Included from random.h, after the elementary draws, so a body here can call
 * random_uniform / random_normal / random_standard_gamma at either width. The
 * algorithms follow numpy's legacy RandomState generators: a distribution is built
 * out of uniforms, exponentials and normals rather than a special function table,
 * which is what keeps the header plain C that OpenCL C 1.2, NVRTC, HIP's clang and
 * MSL all accept (none of them need lgamma or erf here).
 *
 * One body macro serves both precisions. RANDOM_DEFINE_DIST gives it the real type,
 * the uniform draw, the normal draw and the standard-gamma draw it may call; the
 * double form is emitted only when RANDOM_NEED_DOUBLE is set, for the same reason
 * random.h guards its own double block. The three parameters a, b, c are the
 * distribution's arguments in the order the emitter passes them; a body that takes
 * fewer simply ignores the rest.
 */
#ifndef PYGPGPU_RANDOM_CONTINUOUS_H
#define PYGPGPU_RANDOM_CONTINUOUS_H

#define RANDOM_DEFINE_DIST(NAME, SUFFIX, REAL, U, N, SG, BODY) \
RANDOM_QUAL static RANDOM_UNUSED REAL random_##NAME##SUFFIX( \
    random_state* st, REAL a, REAL b, REAL c) \
{ \
    (void)a; (void)b; (void)c; \
    BODY \
}

#define RANDOM_DIST_FLOAT(NAME, BODY) \
    RANDOM_DEFINE_DIST(NAME, , float, random_uniform, random_normal, \
        random_standard_gamma, \
        BODY(float, random_uniform, random_normal, random_standard_gamma))

#ifdef RANDOM_NEED_DOUBLE
#define RANDOM_DIST_DOUBLE(NAME, BODY) \
    RANDOM_DEFINE_DIST(NAME, _double, double, random_uniform_double, \
        random_normal_double, random_standard_gamma_double, \
        BODY(double, random_uniform_double, random_normal_double, \
             random_standard_gamma_double))
#else
#define RANDOM_DIST_DOUBLE(NAME, BODY)
#endif

#define RANDOM_DIST(NAME, BODY) \
    RANDOM_DIST_FLOAT(NAME, BODY) \
    RANDOM_DIST_DOUBLE(NAME, BODY)

/* Standard gamma: Marsaglia-Tsang for shape >= 1 and the boost-and-rejection pair
 * below 1, the same two branches numpy uses. `b` is scale; `a` is shape. */
#define RANDOM_BODY_STANDARD_GAMMA(REAL, U, N, SG) \
    REAL shape = a; \
    REAL bb, cc, uu, vv, xx, yy; \
    if (shape == 1) { uu = 1 - U(st); return -RANDOM_LOG(uu); } \
    if (shape <= 0) return 0; \
    if (shape < 1) { \
        for (;;) { \
            uu = U(st); \
            vv = -RANDOM_LOG(1 - U(st)); \
            if (uu <= 1 - shape) { \
                xx = RANDOM_POW(uu, 1 / shape); \
                if (xx <= vv) return xx; \
            } else { \
                yy = -RANDOM_LOG((1 - uu) / shape); \
                xx = RANDOM_POW(1 - shape + shape * yy, 1 / shape); \
                if (xx <= vv + yy) return xx; \
            } \
        } \
    } \
    bb = shape - 1.0 / 3.0; \
    cc = 1 / RANDOM_SQRT(9 * bb); \
    for (;;) { \
        do { xx = N(st, 0, 1); vv = 1 + cc * xx; } while (vv <= 0); \
        vv = vv * vv * vv; \
        uu = U(st); \
        if (uu < 1 - 0.0331 * xx * xx * xx * xx) return bb * vv; \
        if (RANDOM_LOG(uu) < 0.5 * xx * xx + bb * (1 - vv + RANDOM_LOG(vv))) \
            return bb * vv; \
    }

#define RANDOM_BODY_GAMMA(REAL, U, N, SG) return b * SG(st, a, 0, 0);

#define RANDOM_BODY_BETA(REAL, U, N, SG) \
    REAL ga, gb; \
    if (a <= 1 && b <= 1) { \
        for (;;) { \
            REAL uu = U(st); \
            REAL vv = U(st); \
            REAL xx = RANDOM_POW(uu, 1 / a); \
            REAL yy = RANDOM_POW(vv, 1 / b); \
            if (xx + yy <= 1 && xx + yy > 0) return xx / (xx + yy); \
        } \
    } \
    ga = SG(st, a, 0, 0); \
    gb = SG(st, b, 0, 0); \
    return ga / (ga + gb);

/* a is the rate for expovariate (python), the scale for exponential (numpy); the two
 * stems are separate so neither library's argument has to be inverted by the caller. */
#define RANDOM_BODY_EXPOVARIATE(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return -RANDOM_LOG(uu) / a;

#define RANDOM_BODY_EXPONENTIAL(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return -a * RANDOM_LOG(uu);

#define RANDOM_BODY_STANDARD_EXPONENTIAL(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return -RANDOM_LOG(uu);

#define RANDOM_BODY_LOGNORMAL(REAL, U, N, SG) \
    return RANDOM_EXP(a + b * N(st, 0, 1));

/* a = low, b = high, c = mode. The numpy spelling (left, mode, right) is reordered by
 * the emitter; the header always takes the book's (low, high, mode) order. */
#define RANDOM_BODY_TRIANGULAR(REAL, U, N, SG) \
    REAL uu = U(st); \
    REAL fc = (b > a) ? (c - a) / (b - a) : 0; \
    if (uu < fc) return a + RANDOM_SQRT(uu * (b - a) * (c - a)); \
    return b - RANDOM_SQRT((1 - uu) * (b - a) * (b - c));

/* Best-Fisher rejection. The angle is computed once, then each library spells its
 * range differently: numpy wraps to [-pi, pi] around mu, python to [0, 2*pi). */
#define RANDOM_VONMISES_ANGLE(REAL, U, N, SG) \
    REAL ss, uu, zz, ww, yy, vv, result, modd; \
    int neg; \
    if (b < 1e-8) result = RANDOM_PI * (2 * U(st) - 1) + a; \
    else { \
        if (b < 1e-5) ss = 1 / b + b; \
        else { \
            REAL rr = 1 + RANDOM_SQRT(1 + 4 * b * b); \
            REAL rho = (rr - RANDOM_SQRT(2 * rr)) / (2 * b); \
            ss = (1 + rho * rho) / (2 * rho); \
        } \
        for (;;) { \
            uu = U(st); \
            zz = RANDOM_COS(RANDOM_PI * uu); \
            ww = (1 + ss * zz) / (ss + zz); \
            yy = b * (ss - ww); \
            vv = U(st); \
            if ((yy * (2 - yy) - vv >= 0) || \
                (RANDOM_LOG(yy / vv) + 1 - yy >= 0)) break; \
        } \
        uu = U(st); \
        result = RANDOM_ACOS(ww); \
        if (uu < 0.5) result = -result; \
        result += a; \
        neg = (result < 0); \
        modd = RANDOM_ABS(result); \
        modd = RANDOM_FMOD(modd + RANDOM_PI, 2 * RANDOM_PI) - RANDOM_PI; \
        result = neg ? -modd : modd; \
    }

#define RANDOM_BODY_VONMISES(REAL, U, N, SG) \
    RANDOM_VONMISES_ANGLE(REAL, U, N, SG) \
    return result;

#define RANDOM_BODY_VONMISESVARIATE(REAL, U, N, SG) \
    RANDOM_VONMISES_ANGLE(REAL, U, N, SG) \
    return RANDOM_FMOD(result + 2 * RANDOM_PI, 2 * RANDOM_PI);

/* a is the shape for pareto and power; for weibull a is the shape and b the scale
 * (the numpy spelling passes 1 as b, python's weibullvariate passes alpha there). */
#define RANDOM_BODY_PARETO(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return RANDOM_EXP(-RANDOM_LOG(uu) / a) - 1;

/* python's `paretovariate` is the Pareto I form: the same power law shifted so the
 * support starts at one. numpy's `pareto` above is the Lomax (offset) form. */
#define RANDOM_BODY_PARETOVARIATE(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return RANDOM_EXP(-RANDOM_LOG(uu) / a);

#define RANDOM_BODY_WEIBULL(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return b * RANDOM_POW(-RANDOM_LOG(uu), 1 / a);

#define RANDOM_BODY_POWER(REAL, U, N, SG) \
    return RANDOM_POW(U(st), 1 / a);

#define RANDOM_BODY_RAYLEIGH(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return a * RANDOM_SQRT(-2 * RANDOM_LOG(uu));

#define RANDOM_BODY_GUMBEL(REAL, U, N, SG) \
    REAL uu = 1 - U(st); \
    return a - b * RANDOM_LOG(-RANDOM_LOG(uu));

#define RANDOM_BODY_LAPLACE(REAL, U, N, SG) \
    REAL uu = U(st) - 0.5; \
    return a - b * (uu >= 0 ? 1 : -1) * RANDOM_LOG(1 - 2 * RANDOM_ABS(uu));

#define RANDOM_BODY_LOGISTIC(REAL, U, N, SG) \
    REAL uu = U(st); \
    return a + b * RANDOM_LOG(uu / (1 - uu));

#define RANDOM_BODY_CAUCHY(REAL, U, N, SG) \
    return a + b * RANDOM_TAN(RANDOM_PI * (U(st) - 0.5));

/* The chi-square family, F and Student-t are all a standard-gamma or a normal away
 * from each other, which is why they need no special-function table. */
#define RANDOM_BODY_CHISQUARE(REAL, U, N, SG) return 2 * SG(st, a / 2, 0, 0);

#define RANDOM_BODY_NONCENTRAL_CHISQUARE(REAL, U, N, SG) \
    REAL cc; \
    if (b == 0) return 2 * SG(st, a / 2, 0, 0); \
    if (a > 1) { \
        REAL nn = N(st, 0, 1) + RANDOM_SQRT(b); \
        return 2 * SG(st, (a - 1) / 2, 0, 0) + nn * nn; \
    } \
    { \
        int kk = 0; \
        REAL prod = 1, limit = RANDOM_EXP(-b / 2); \
        while (prod > limit) { prod *= 1 - U(st); kk++; } \
        cc = 2 * SG(st, (a + 2 * kk) / 2, 0, 0); \
    } \
    return cc;

#define RANDOM_BODY_F(REAL, U, N, SG) \
    return (2 * SG(st, a / 2, 0, 0) / a) / (2 * SG(st, b / 2, 0, 0) / b);

#define RANDOM_BODY_NONCENTRAL_F(REAL, U, N, SG) \
    REAL num; \
    if (c == 0) num = 2 * SG(st, a / 2, 0, 0); \
    else if (a > 1) { \
        REAL nn = N(st, 0, 1) + RANDOM_SQRT(c); \
        num = 2 * SG(st, (a - 1) / 2, 0, 0) + nn * nn; \
    } else { \
        int kk = 0; \
        REAL prod = 1, limit = RANDOM_EXP(-c / 2); \
        while (prod > limit) { prod *= 1 - U(st); kk++; } \
        num = 2 * SG(st, (a + 2 * kk) / 2, 0, 0); \
    } \
    return (num / a) / ((2 * SG(st, b / 2, 0, 0)) / b);

#define RANDOM_BODY_STANDARD_T(REAL, U, N, SG) \
    REAL nn = N(st, 0, 1); \
    return RANDOM_SQRT(a / 2) * nn / RANDOM_SQRT(SG(st, a / 2, 0, 0));

/* The inverse-Gaussian sampler numpy's wald uses; a is mean, b is scale. */
#define RANDOM_BODY_WALD(REAL, U, N, SG) \
    REAL gg = N(st, 0, 1); \
    REAL yy = a * gg * gg; \
    REAL xx = a + (a / (2 * b)) * (yy - RANDOM_SQRT(4 * b * yy + yy * yy)); \
    REAL uu = U(st); \
    return (uu <= a / (a + xx)) ? xx : (a * a / xx);

RANDOM_DIST(standard_gamma, RANDOM_BODY_STANDARD_GAMMA)
RANDOM_DIST(gamma, RANDOM_BODY_GAMMA)
RANDOM_DIST(beta, RANDOM_BODY_BETA)
RANDOM_DIST(expovariate, RANDOM_BODY_EXPOVARIATE)
RANDOM_DIST(exponential, RANDOM_BODY_EXPONENTIAL)
RANDOM_DIST(standard_exponential, RANDOM_BODY_STANDARD_EXPONENTIAL)
RANDOM_DIST(lognormal, RANDOM_BODY_LOGNORMAL)
RANDOM_DIST(triangular, RANDOM_BODY_TRIANGULAR)
RANDOM_DIST(vonmises, RANDOM_BODY_VONMISES)
RANDOM_DIST(vonmisesvariate, RANDOM_BODY_VONMISESVARIATE)
RANDOM_DIST(pareto, RANDOM_BODY_PARETO)
RANDOM_DIST(paretovariate, RANDOM_BODY_PARETOVARIATE)
RANDOM_DIST(weibull, RANDOM_BODY_WEIBULL)
RANDOM_DIST(power, RANDOM_BODY_POWER)
RANDOM_DIST(rayleigh, RANDOM_BODY_RAYLEIGH)
RANDOM_DIST(gumbel, RANDOM_BODY_GUMBEL)
RANDOM_DIST(laplace, RANDOM_BODY_LAPLACE)
RANDOM_DIST(logistic, RANDOM_BODY_LOGISTIC)
RANDOM_DIST(cauchy, RANDOM_BODY_CAUCHY)
RANDOM_DIST(chisquare, RANDOM_BODY_CHISQUARE)
RANDOM_DIST(noncentral_chisquare, RANDOM_BODY_NONCENTRAL_CHISQUARE)
RANDOM_DIST(f, RANDOM_BODY_F)
RANDOM_DIST(noncentral_f, RANDOM_BODY_NONCENTRAL_F)
RANDOM_DIST(standard_t, RANDOM_BODY_STANDARD_T)
RANDOM_DIST(wald, RANDOM_BODY_WALD)

#endif /* PYGPGPU_RANDOM_CONTINUOUS_H */
