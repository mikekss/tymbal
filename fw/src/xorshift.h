/* xorshift.h — contract C-4: the same bit sequence as in
 * dsp/skeleton_a.py::xorshift32_block. u in [-1,1): (int32)s * 2^-31.
 * The _f32 variant: THE SAME bit sequence, f32 output (M55 hot paths). */
#ifndef N6_XORSHIFT_H
#define N6_XORSHIFT_H
#include <stdint.h>

static inline uint32_t n6_xs32_seed(uint32_t seed) { return seed ? seed : 1u; }

static inline double n6_xs32_next(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (double)(int32_t)x * (1.0 / 2147483648.0);
}

static inline float n6_xs32_next_f32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (float)(int32_t)x * (1.0f / 2147483648.0f);
}
#endif
