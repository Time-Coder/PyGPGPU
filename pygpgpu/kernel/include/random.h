/*
 * Random numbers, for generated kernels.
 *
 * A kernel body writes `random.random()` (or `np.random.random()`, and the rest of the
 * vocabulary `pygpgpu.kernel.rng` lists), and the generated device code calls the
 * functions declared here. The body never names a work item or a seed: the launch owns
 * both. The translator declares one `pyg_rng` per kernel invocation, seeds it from the
 * launch seed and the work item's position, and hands it to every draw -- so two work
 * items are two independent streams, two launches are two independent draws, and the
 * sequence of one work item is reproducible for a given seed.
 *
 * This is the same shape `include/quat.h` and `include/complex.h` have: one fixed
 * header, plain C that every backend compiles, and a `*_QUAL` macro the generated
 * source defines before including it (`__device__` where NVRTC demands it, empty
 * elsewhere). `PYG_RNG_WORD` is the backend's 64-bit unsigned spelling: `ulong` in
 * OpenCL C and MSL, `unsigned long long` in CUDA C, where `ulong` does not exist.
 *
 * The generator is PCG32 (O'Neill, 2014): a 64-bit LCG state with a 32-bit output
 * function. It is counter-based in the sense that matters here -- the state advances on
 * every draw and nothing else is kept -- so a work item's stream is exactly "the state
 * I started with, advanced once per draw", which is what makes a serial run and a
 * parallel run give the same numbers for the same work item.
 *
 * The double block sits behind `PYG_RNG_NEED_DOUBLE` for the reason the complex header
 * uses the same guard: a device without fp64 (and MSL, which has no `double` at all)
 * rejects a `double` variable even inside a function nothing calls.
 */
#ifndef PYGPGPU_RANDOM_H
#define PYGPGPU_RANDOM_H

#ifndef PYG_RNG_QUAL
#define PYG_RNG_QUAL
#endif
/* Every function below is `static`, so a kernel that draws once leaves the rest
 * unreferenced and NVRTC reports #177-D for each. The attribute says that is expected;
 * it is accepted by NVRTC, HIP's clang, OpenCL C and MSL alike, and mirrors the
 * complex and quaternion headers' *_UNUSED. */
#ifndef PYG_RNG_UNUSED
#define PYG_RNG_UNUSED __attribute__((unused))
#endif
#ifndef PYG_RNG_WORD
#define PYG_RNG_WORD unsigned long long
#endif
/* The transcendental helpers the normal draw needs.
 *
 * OpenCL C 1.2 has only the overloaded form (`log`, `cos`), CUDA and HIP have the
 * `f`-suffixed float form as well, and MSL's `metal_stdlib` declares the plain names
 * too -- so the plain spelling is what every backend here resolves, exactly as the
 * complex header's `COMPLEX_SIN`/`COMPLEX_COS` macros do. A backend whose library
 * wants the suffixed form can define its own before including this file. */
#ifndef PYG_RNG_LOG
#define PYG_RNG_LOG(x) log(x)
#endif
#ifndef PYG_RNG_COS
#define PYG_RNG_COS(x) cos(x)
#endif
#ifndef PYG_RNG_SQRT
#define PYG_RNG_SQRT(x) sqrt(x)
#endif

/* One work item's stream. `state` advances on every draw and `inc` selects the stream;
 * both are 64-bit so the period is the LCG's 2^64 rather than something a long kernel
 * could exhaust. */
typedef struct {
    PYG_RNG_WORD state;
    PYG_RNG_WORD inc;
} pyg_rng;

/* The splitmix64 finalizer: the one mixer both the seeding and the output use. Written
 * with the same constants as PCG's own seeding code so a stream is decorrelated from
 * its neighbours by the mixer rather than by a lucky multiplier. */
PYG_RNG_QUAL static PYG_RNG_UNUSED PYG_RNG_WORD pyg_rng_mix(PYG_RNG_WORD z)
{
    z = (z ^ (z >> 30)) * (PYG_RNG_WORD)0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * (PYG_RNG_WORD)0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

/* One draw: `state * MULT + inc`, with PCG's XSH-RR output function on the old state.
 * The shift amount is taken modulo 32 by the mask, which is what PCG specifies. */
PYG_RNG_QUAL static PYG_RNG_UNUSED unsigned int pyg_rng_next(pyg_rng* rng)
{
    PYG_RNG_WORD old = rng->state;
    unsigned int xorshifted;
    unsigned int rot;
    rng->state = old * (PYG_RNG_WORD)6364136223846793005UL + rng->inc;
    xorshifted = (unsigned int)(((old >> 18) ^ old) >> 27);
    rot = (unsigned int)(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

/* Seed one work item's stream from the launch's seed and the work item's position.
 *
 * The three axes are folded into the low half of the initial state and the launch seed
 * occupies the high half, so two work items of one launch and two launches of one
 * work item are both separate streams. PCG's own seeding procedure is followed
 * (`pcg32_srandom_r`): reset, advance, add the state, advance again -- a plain
 * `state = seed` leaves the first outputs of neighbouring streams correlated. */
PYG_RNG_QUAL static PYG_RNG_UNUSED void pyg_rng_init(pyg_rng* rng, unsigned int seed,
                                                     unsigned int x, unsigned int y,
                                                     unsigned int z)
{
    PYG_RNG_WORD index = ((PYG_RNG_WORD)x * (PYG_RNG_WORD)0x9E3779B1UL)
                       ^ ((PYG_RNG_WORD)y * (PYG_RNG_WORD)0x85EBCA77UL)
                       ^ ((PYG_RNG_WORD)z * (PYG_RNG_WORD)0xC2B2AE3DUL);
    rng->state = 0UL;
    rng->inc = (pyg_rng_mix(index + (PYG_RNG_WORD)0x9E3779B97F4A7C15UL) << 1) | 1UL;
    pyg_rng_next(rng);
    rng->state += ((PYG_RNG_WORD)seed << 32) ^ index;
    pyg_rng_next(rng);
}

/* A uniform draw in [0, 1): the top 24 bits, which is the whole mantissa of a `float`,
 * so every value is exactly representable and no draw is quantised below the width the
 * kernel asked for. */
PYG_RNG_QUAL static PYG_RNG_UNUSED float pyg_rng_uniform(pyg_rng* rng)
{
    return (float)(pyg_rng_next(rng) >> 8) * (1.0f / 16777216.0f);
}

/* A uniform draw in [low, high). `high < low` answers a value in [high, low), which is
 * python's rule for the same mistake rather than a refusal: a kernel cannot raise. */
PYG_RNG_QUAL static PYG_RNG_UNUSED float pyg_rng_range(pyg_rng* rng, float low,
                                                       float high)
{
    return low + (high - low) * pyg_rng_uniform(rng);
}

/* A normal draw, by the Box-Muller transform. `1 - u` is in (0, 1], so `log` never
 * sees zero; the turn's angle is `2*pi*u2` with pi written out rather than taken from
 * M_PI (NVRTC does not define it). Two draws per normal, which is the standard price. */
PYG_RNG_QUAL static PYG_RNG_UNUSED float pyg_rng_normal(pyg_rng* rng, float mean,
                                                        float deviation)
{
    float u1 = 1.0f - pyg_rng_uniform(rng);
    float u2 = pyg_rng_uniform(rng);
    return mean + deviation * PYG_RNG_SQRT(-2.0f * PYG_RNG_LOG(u1)) * PYG_RNG_COS(6.28318530717958647692f * u2);
}

/* An integer draw in [low, high): Lemire's multiply-shift over the full 32 bits, so
 * the bias is below 2^-32 rather than the `% span` bias a small range would show. An
 * empty range answers `low`, which is the only value a refusal could not reach. */
PYG_RNG_QUAL static PYG_RNG_UNUSED int pyg_rng_int(pyg_rng* rng, int low, int high)
{
    unsigned int span;
    unsigned int draw;
    if (high <= low) {
        return low;
    }
    span = (unsigned int)(high - low);
    draw = pyg_rng_next(rng);
    return low + (int)(((PYG_RNG_WORD)draw * (PYG_RNG_WORD)span) >> 32);
}

/* An integer draw in [low, high], which is python's `random.randint` (its high end is
 * inclusive; numpy's `randint` is not, and gets `pyg_rng_int` above). */
PYG_RNG_QUAL static PYG_RNG_UNUSED int pyg_rng_int_inclusive(pyg_rng* rng, int low,
                                                             int high)
{
    unsigned int span;
    unsigned int draw;
    if (high < low) {
        return low;
    }
    span = (unsigned int)high - (unsigned int)low + 1u;
    draw = pyg_rng_next(rng);
    return low + (int)(((PYG_RNG_WORD)draw * (PYG_RNG_WORD)span) >> 32);
}

#ifdef PYG_RNG_NEED_DOUBLE

/* The same generator at the width a double kernel asks for: 53 bits, the whole
 * mantissa of a `double`, assembled from two draws. */
PYG_RNG_QUAL static PYG_RNG_UNUSED double pyg_rng_uniform_double(pyg_rng* rng)
{
    unsigned int high = pyg_rng_next(rng) >> 5;
    unsigned int low = pyg_rng_next(rng) >> 6;
    return ((double)high * 67108864.0 + (double)low)
           * (1.0 / 9007199254740992.0);
}

PYG_RNG_QUAL static PYG_RNG_UNUSED double pyg_rng_range_double(pyg_rng* rng, double low,
                                                               double high)
{
    return low + (high - low) * pyg_rng_uniform_double(rng);
}

PYG_RNG_QUAL static PYG_RNG_UNUSED double pyg_rng_normal_double(pyg_rng* rng, double mean,
                                                                double deviation)
{
    double u1 = 1.0 - pyg_rng_uniform_double(rng);
    double u2 = pyg_rng_uniform_double(rng);
    return mean + deviation * PYG_RNG_SQRT(-2.0 * PYG_RNG_LOG(u1)) * PYG_RNG_COS(6.28318530717958647692 * u2);
}

#endif /* PYG_RNG_NEED_DOUBLE */

#endif /* PYGPGPU_RANDOM_H */