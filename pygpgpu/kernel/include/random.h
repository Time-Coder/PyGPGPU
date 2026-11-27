/*
 * Random numbers, for generated kernels.
 *
 * A kernel body writes `random.random()` (or `np.random.random()`, and the rest of the
 * vocabulary `pygpgpu.kernel.st` lists), and the generated device code calls the
 * functions declared here. The body never names a work item or a seed: the launch owns
 * both. The translator declares one `random_state` per kernel invocation, seeds it from the
 * launch seed and the work item's position, and hands it to every draw -- so two work
 * items are two independent streams, two launches are two independent draws, and the
 * sequence of one work item is reproducible for a given seed.
 *
 * This is the same shape `include/quat.h` and `include/complex.h` have: one fixed
 * header, plain C that every backend compiles, and a `*_QUAL` macro the generated
 * source defines before including it (`__device__` where NVRTC demands it, empty
 * elsewhere). `RANDOM_WORD` is the backend's 64-bit unsigned spelling: `ulong` in
 * OpenCL C and MSL, `unsigned long long` in CUDA C, where `ulong` does not exist.
 *
 * The generator is PCG32 (O'Neill, 2014): a 64-bit LCG state with a 32-bit output
 * function. It is counter-based in the sense that matters here -- the state advances on
 * every draw and nothing else is kept -- so a work item's stream is exactly "the state
 * I started with, advanced once per draw", which is what makes a serial run and a
 * parallel run give the same numbers for the same work item.
 *
 * The double block sits behind `RANDOM_NEED_DOUBLE` for the reason the complex header
 * uses the same guard: a device without fp64 (and MSL, which has no `double` at all)
 * rejects a `double` variable even inside a function nothing calls.
 */
#ifndef PYGPGPU_RANDOM_H
#define PYGPGPU_RANDOM_H

#ifndef RANDOM_QUAL
#define RANDOM_QUAL
#endif
/* Every function below is `static`, so a kernel that draws once leaves the rest
 * unreferenced and NVRTC reports #177-D for each. The attribute says that is expected;
 * it is accepted by NVRTC, HIP's clang, OpenCL C and MSL alike, and mirrors the
 * complex and quaternion headers' *_UNUSED. */
#ifndef RANDOM_UNUSED
#define RANDOM_UNUSED __attribute__((unused))
#endif
#ifndef RANDOM_WORD
#define RANDOM_WORD unsigned long long
#endif
/* The transcendental helpers the normal draw needs.
 *
 * OpenCL C 1.2 has only the overloaded form (`log`, `cos`), CUDA and HIP have the
 * `f`-suffixed float form as well, and MSL's `metal_stdlib` declares the plain names
 * too -- so the plain spelling is what every backend here resolves, exactly as the
 * complex header's `COMPLEX_SIN`/`COMPLEX_COS` macros do. A backend whose library
 * wants the suffixed form can define its own before including this file. */
#ifndef RANDOM_LOG
#define RANDOM_LOG(x) log(x)
#endif
#ifndef RANDOM_COS
#define RANDOM_COS(x) cos(x)
#endif
#ifndef RANDOM_SQRT
#define RANDOM_SQRT(x) sqrt(x)
#endif
#ifndef RANDOM_EXP
#define RANDOM_EXP(x) exp(x)
#endif
#ifndef RANDOM_POW
#define RANDOM_POW(x, y) pow((x), (y))
#endif
#ifndef RANDOM_FLOOR
#define RANDOM_FLOOR(x) floor(x)
#endif
#ifndef RANDOM_TAN
#define RANDOM_TAN(x) tan(x)
#endif
#ifndef RANDOM_ACOS
#define RANDOM_ACOS(x) acos(x)
#endif
#ifndef RANDOM_ABS
#define RANDOM_ABS(x) fabs(x)
#endif
#ifndef RANDOM_FMOD
#define RANDOM_FMOD(x, y) fmod((x), (y))
#endif
#ifndef RANDOM_PI
#define RANDOM_PI 3.14159265358979323846
#endif

/* One work item's stream. `state` advances on every draw and `inc` selects the stream;
 * both are 64-bit so the period is the LCG's 2^64 rather than something a long kernel
 * could exhaust. */
typedef struct {
    RANDOM_WORD state;
    RANDOM_WORD inc;
} random_state;

/* The splitmix64 finalizer: the one mixer both the seeding and the output use. Written
 * with the same constants as PCG's own seeding code so a stream is decorrelated from
 * its neighbours by the mixer rather than by a lucky multiplier. */
RANDOM_QUAL static RANDOM_UNUSED RANDOM_WORD random_mix(RANDOM_WORD z)
{
    z = (z ^ (z >> 30)) * (RANDOM_WORD)0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * (RANDOM_WORD)0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

/* One draw: `state * MULT + inc`, with PCG's XSH-RR output function on the old state.
 * The shift amount is taken modulo 32 by the mask, which is what PCG specifies. */
RANDOM_QUAL static RANDOM_UNUSED unsigned int random_next(random_state* st)
{
    RANDOM_WORD old = st->state;
    unsigned int xorshifted;
    unsigned int rot;
    st->state = old * (RANDOM_WORD)6364136223846793005UL + st->inc;
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
RANDOM_QUAL static RANDOM_UNUSED void random_init(random_state* st, unsigned int seed,
                                                     unsigned int x, unsigned int y,
                                                     unsigned int z)
{
    RANDOM_WORD index = ((RANDOM_WORD)x * (RANDOM_WORD)0x9E3779B1UL)
                       ^ ((RANDOM_WORD)y * (RANDOM_WORD)0x85EBCA77UL)
                       ^ ((RANDOM_WORD)z * (RANDOM_WORD)0xC2B2AE3DUL);
    st->state = 0UL;
    st->inc = (random_mix(index + (RANDOM_WORD)0x9E3779B97F4A7C15UL) << 1) | 1UL;
    random_next(st);
    st->state += ((RANDOM_WORD)seed << 32) ^ index;
    random_next(st);
}

/* A uniform draw in [0, 1): the top 24 bits, which is the whole mantissa of a `float`,
 * so every value is exactly representable and no draw is quantised below the width the
 * kernel asked for. */
RANDOM_QUAL static RANDOM_UNUSED float random_uniform(random_state* st)
{
    return (float)(random_next(st) >> 8) * (1.0f / 16777216.0f);
}

/* A uniform draw in [low, high). `high < low` answers a value in [high, low), which is
 * python's rule for the same mistake rather than a refusal: a kernel cannot raise. */
RANDOM_QUAL static RANDOM_UNUSED float random_range(random_state* st, float low,
                                                       float high)
{
    return low + (high - low) * random_uniform(st);
}

/* A normal draw, by the Box-Muller transform. `1 - u` is in (0, 1], so `log` never
 * sees zero; the turn's angle is `2*pi*u2` with pi written out rather than taken from
 * M_PI (NVRTC does not define it). Two draws per normal, which is the standard price. */
RANDOM_QUAL static RANDOM_UNUSED float random_normal(random_state* st, float mean,
                                                        float deviation)
{
    float u1 = 1.0f - random_uniform(st);
    float u2 = random_uniform(st);
    return mean + deviation * RANDOM_SQRT(-2.0f * RANDOM_LOG(u1)) * RANDOM_COS(6.28318530717958647692f * u2);
}

/* An integer draw in [low, high): Lemire's multiply-shift over the full 32 bits, so
 * the bias is below 2^-32 rather than the `% span` bias a small range would show. An
 * empty range answers `low`, which is the only value a refusal could not reach. */
RANDOM_QUAL static RANDOM_UNUSED int random_int(random_state* st, int low, int high)
{
    unsigned int span;
    unsigned int draw;
    if (high <= low) {
        return low;
    }
    span = (unsigned int)(high - low);
    draw = random_next(st);
    return low + (int)(((RANDOM_WORD)draw * (RANDOM_WORD)span) >> 32);
}

/* An integer draw in [low, high], which is python's `random.randint` (its high end is
 * inclusive; numpy's `randint` is not, and gets `random_int` above). */
RANDOM_QUAL static RANDOM_UNUSED int random_int_inclusive(random_state* st, int low,
                                                             int high)
{
    unsigned int span;
    unsigned int draw;
    if (high < low) {
        return low;
    }
    span = (unsigned int)high - (unsigned int)low + 1u;
    draw = random_next(st);
    return low + (int)(((RANDOM_WORD)draw * (RANDOM_WORD)span) >> 32);
}

#ifdef RANDOM_NEED_DOUBLE

/* The same generator at the width a double kernel asks for: 53 bits, the whole
 * mantissa of a `double`, assembled from two draws. */
RANDOM_QUAL static RANDOM_UNUSED double random_uniform_double(random_state* st)
{
    unsigned int high = random_next(st) >> 5;
    unsigned int low = random_next(st) >> 6;
    return ((double)high * 67108864.0 + (double)low)
           * (1.0 / 9007199254740992.0);
}

RANDOM_QUAL static RANDOM_UNUSED double random_range_double(random_state* st, double low,
                                                               double high)
{
    return low + (high - low) * random_uniform_double(st);
}

RANDOM_QUAL static RANDOM_UNUSED double random_normal_double(random_state* st, double mean,
                                                                double deviation)
{
    double u1 = 1.0 - random_uniform_double(st);
    double u2 = random_uniform_double(st);
    return mean + deviation * RANDOM_SQRT(-2.0 * RANDOM_LOG(u1)) * RANDOM_COS(6.28318530717958647692 * u2);
}

#endif /* RANDOM_NEED_DOUBLE */

/* The probability distributions are split by the value they answer -- real or integer
 * -- because the two groups have different algorithms and different widths. They are
 * included here, after the elementary draws, so a distribution body can build on
 * `random_uniform`, `random_normal` and their double forms. The quoted name resolves
 * relative to this file's own folder, which `install_header` fills alongside it. */
#include "random_continuous.h"
#include "random_discrete.h"

#endif /* PYGPGPU_RANDOM_H */