/*
 * Discrete distributions, for generated kernels.
 *
 * Included from random.h after the continuous distributions, so negative binomial
 * can build on the gamma draw there. Every function here answers an `int` and takes
 * its arguments at float width: a count or a rate does not become more correct by
 * being carried as a double, and the integer result is what the kernel declared.
 *
 * The algorithms are the inversion and rejection samplers numpy's legacy RandomState
 * uses: geometric inversion, Knuth's product test and PTRS for Poisson, the
 * probability-recursion inversion for binomial and hypergeometric, and the rejection
 * transforms for the logarithmic series and Zipf. They are loops over uniforms and
 * elementary math, so the header stays plain C on every backend.
 */
#ifndef PYGPGPU_RANDOM_DISCRETE_H
#define PYGPGPU_RANDOM_DISCRETE_H

#define RANDOM_DEFINE_IDIST(NAME, BODY) \
RANDOM_QUAL static RANDOM_UNUSED int random_##NAME( \
    random_state* st, float a, float b, float c) \
{ \
    (void)a; (void)b; (void)c; \
    BODY \
}

#define RANDOM_IDIST(NAME, BODY) \
    RANDOM_DEFINE_IDIST(NAME, BODY(float, random_uniform))

/* log(k!) to enough accuracy for an acceptance test: an exact sum for small k and
 * Stirling's series beyond it (the 1/(12k) term keeps the error below 1e-4). */
RANDOM_QUAL static RANDOM_UNUSED float random_log_fact(int kk)
{
    int i;
    float total = 0.0f;
    float n;
    if (kk <= 32) {
        for (i = 2; i <= kk; i++) total += RANDOM_LOG((float)i);
        return total;
    }
    n = (float)kk;
    return (n + 0.5f) * RANDOM_LOG(n) - n
           + 0.9189385332046727f + 1.0f / (12.0f * n);
}

#define RANDOM_IBODY_GEOMETRIC(REAL, U) \
    REAL uu = 1 - U(st); \
    return (int)RANDOM_FLOOR(RANDOM_LOG(uu) / RANDOM_LOG(1 - a)) + 1;

/* The product test for a small mean, PTRS (Hormann's transformed rejection) for a
 * large one. PTRS needs log(k!), which is the helper just above. */
#define RANDOM_IBODY_POISSON(REAL, U) \
    REAL limit, prod, bb, aa, inv_alpha, v_r; \
    int kk; \
    if (a < 30) { \
        kk = -1; \
        prod = 1; \
        limit = RANDOM_EXP(-a); \
        do { kk++; prod *= U(st); } while (prod > limit); \
        return kk; \
    } \
    bb = 0.931f + 2.53f * RANDOM_SQRT(a); \
    aa = -0.059f + 0.02483f * bb; \
    inv_alpha = 1.1239f + 1.1328f / (bb - 3.4f); \
    v_r = 0.9277f - 3.6224f / (bb - 2.0f); \
    for (;;) { \
        REAL uu = U(st) - 0.5f; \
        REAL vv = U(st); \
        REAL us = 0.5f - RANDOM_ABS(uu); \
        REAL xx = RANDOM_FLOOR((2 * aa / us + bb) * uu + a + 0.43f); \
        if (us >= 0.07f && vv <= v_r) return (int)xx; \
        if (xx < 0 || (us < 0.013f && vv > us)) continue; \
        if (RANDOM_LOG(vv * inv_alpha / (aa / (us * us) + bb)) \
                <= -a + xx * RANDOM_LOG(a) - random_log_fact((int)xx)) \
            return (int)xx; \
    }

/* The probability-recursion inversion: start at P(X=0) = q^n and walk the pmf until
 * the uniform draw is covered. Restarting at zero when the bound is passed keeps the
 * tail from being truncated for small p and large n. */
#define RANDOM_IBODY_BINOMIAL(REAL, U) \
    REAL qq, qn, npd, px, uu, bound; \
    int xx; \
    if (b <= 0) return 0; \
    if (b >= 1) return (int)a; \
    qq = 1 - b; \
    npd = a * b; \
    qn = RANDOM_EXP(a * RANDOM_LOG(qq)); \
    bound = RANDOM_FLOOR(npd + 10 * RANDOM_SQRT(npd * qq + 1)); \
    if (bound > a) bound = a; \
    xx = 0; \
    px = qn; \
    uu = U(st); \
    while (uu > px) { \
        xx++; \
        if (xx > bound) { xx = 0; px = qn; uu = U(st); } \
        else { \
            uu -= px; \
            px = ((a - xx + 1) * b * px) / (xx * qq); \
        } \
    } \
    return xx;

/* A negative binomial is a Poisson whose mean is gamma-distributed. */
#define RANDOM_IBODY_NEGATIVE_BINOMIAL(REAL, U) \
    REAL yy = random_gamma(st, a, (1 - b) / b, 0); \
    return random_poisson(st, yy, 0, 0);

/* Sample nsample times without replacement, shrinking the two populations as we go. */
#define RANDOM_IBODY_HYPERGEOMETRIC(REAL, U) \
    int good = (int)a, bad = (int)b, sample = (int)c, drawn = 0; \
    while (sample > 0) { \
        int total = good + bad; \
        if (total <= 0) break; \
        if (U(st) * total < good) { drawn++; good--; } \
        else bad--; \
        sample--; \
    } \
    return drawn;

/* The logarithmic series by the rejection transform of its own CDF. */
#define RANDOM_IBODY_LOGSERIES(REAL, U) \
    REAL rr = RANDOM_LOG(1 - a); \
    for (;;) { \
        REAL vv = U(st); \
        REAL uu, qq, result; \
        if (vv >= a) return 1; \
        uu = U(st); \
        qq = 1 - RANDOM_EXP(rr * uu); \
        if (vv <= qq * qq) { \
            result = RANDOM_FLOOR(1 + RANDOM_LOG(vv) / RANDOM_LOG(qq)); \
            if (result >= 1 && vv > 0) return (int)result; \
            continue; \
        } \
        if (vv >= qq) return 1; \
        return 2; \
    }

/* Zipf by the rejection method numpy uses. The support is bounded by INT_MAX; a
 * parameter at or above 1025 answers 1 with probability indistinguishable from one
 * (numpy makes the same shortcut). a <= 1 is refused by the emitter: the normalising
 * constant does not exist. */
#define RANDOM_IBODY_ZIPF(REAL, U) \
    REAL am1, bb, umin; \
    if (a >= 1025) return 1; \
    am1 = a - 1; \
    bb = RANDOM_POW(2.0f, am1); \
    umin = RANDOM_POW(2147483647.0f, -am1); \
    for (;;) { \
        REAL u01 = U(st); \
        REAL uu = u01 * umin + (1 - u01); \
        REAL vv = U(st); \
        REAL xx = RANDOM_FLOOR(RANDOM_POW(uu, -1 / am1)); \
        REAL tt; \
        if (xx < 1 || xx > 2147483647) continue; \
        tt = RANDOM_POW(1 + 1 / xx, am1); \
        if (vv * xx * (tt - 1) / (bb - 1) <= tt / bb) return (int)xx; \
    }

RANDOM_IDIST(geometric, RANDOM_IBODY_GEOMETRIC)
RANDOM_IDIST(poisson, RANDOM_IBODY_POISSON)
RANDOM_IDIST(binomial, RANDOM_IBODY_BINOMIAL)
RANDOM_IDIST(negative_binomial, RANDOM_IBODY_NEGATIVE_BINOMIAL)
RANDOM_IDIST(hypergeometric, RANDOM_IBODY_HYPERGEOMETRIC)
RANDOM_IDIST(logseries, RANDOM_IBODY_LOGSERIES)
RANDOM_IDIST(zipf, RANDOM_IBODY_ZIPF)

#endif /* PYGPGPU_RANDOM_DISCRETE_H */
