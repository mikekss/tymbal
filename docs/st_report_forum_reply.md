# Reply to Yanis — paste as is

Attach `repro_concat.py` (it is in `tools/`, checked).

Below is the post text. There are almost no headings and no lists, on
purpose: on the forum people write in running text, not in sections.

---

Hi Yanis, thanks for the quick reply. Everything is below, and I have to
correct myself on one point first.

In my original post I described the twelve expensive `Concat` blocks as
"channel-wise taps". That was wrong, and it would have sent you looking at
the wrong nodes. The expensive ones are the state-ring concatenations along
the width axis. Our graph does contain channel-axis concatenations as well —
the dilation taps — and those are fine, they take the fast path and cost
nothing worth mentioning. Both kinds sit in the same graph, same build, same
image, which is really the cleanest evidence I can offer.

The condition is `axis_is_leftmost` in `LL_ATON_LIB_Concat` (`ll_aton_lib.c`):
every dimension to the left of the concat axis has to be 1. On device our
tensors are laid out `[1, V=2, T, C=88]` and we concatenate along the width
axis, so the `V=2` sits to the left, the check fails, and the block drops into
the generic branch at the bottom of the function — four `memcpy` calls per
block, two per input, one per `V` row. The channel-axis concats have only `[1]`
to their left, so they pass.

The measurement is what convinced me it is a copy loop and not DMA setup.
Twelve blocks, widths 50/52/56/64/80/112, which is exactly `2*88*(48+2*d)` for
dilations 1 to 32. They move 145 728 bytes per inference — the same number as
the sum of the compiler's own `est`, since Concat is estimated at one element
per cycle. On the M55, with the DWT counter and per-epoch profiling
(`LL_ATON_EB_DBG_INFO`):

```
total    1 019 501 cycles / 145 728 bytes = 6.997 cycles/byte
by size  7.04  7.01  6.95  6.96  6.91  6.87
```

A constant price per byte with no fixed offset. If this were DMA setup there
would be an offset, and the small blocks would look relatively worse. They
don't.

I have attached `repro_concat.py` rather than the project itself. It needs
`onnx` and `numpy` and writes a graph containing only the offending pattern —
six `Concat` nodes on `[1, 88, 2, W]` along `axis=3`, no arithmetic at all —
so whatever the generated code spends on it is the copy. The real graph has
twelve such nodes because the stack is doubled; change `DILATIONS` in the
script if you want the full count. I'd rather not dump the whole project on
you: it is a trained audio model, and its topology would only make the Concat
harder to find. If the actual `.onnx` and the quantization JSON would help,
tell me and I'll send them privately.

About the model. It is a streaming causal TCN, int8 via `quantize_static`,
12 layers, 88 channels, dilations 1..32 in two stacks, input `x[1, 8, V, T]`
and output `y[1, 4, V, T]` with V=2 and T=48. One layer looks like this:

```
cat_i       = Concat(state_in_i[1,C,V,2d], h_i[1,C,V,T], axis=3)   <- slow one
d == 1: r_i = Conv(cat_i, W[C,C,1,3], B)
d  > 1: tap_j = Slice(cat_i, j*d, j*d+T, axis=3), j = 0,1,2
        g_i   = Concat(tap_0, tap_1, tap_2, axis=1)                <- fast one
        r_i   = Conv(g_i, W[C,3C,1,1], B)
state_out_i = Slice(cat_i, T, T+2d, axis=3)
h_{i+1}     = Add(h_i, Relu(r_i))
```

The explicit slices are our workaround for the `dilations > 1` limitation I
mentioned in the first post.

The `memcpy` override is a strong symbol in the application, so it displaces
the newlib-nano one and reaches the runtime too. MVE, four 16-byte loads
issued before the first store, tail by predication:

```c
void *memcpy(void *__restrict dst, const void *__restrict src, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;

    if ((((uintptr_t)d | (uintptr_t)s) & 3u) == 0u) {
        while (n >= 64u) {
            uint32x4_t a = vldrwq_u32((const uint32_t *)(s +  0));
            uint32x4_t b = vldrwq_u32((const uint32_t *)(s + 16));
            uint32x4_t c = vldrwq_u32((const uint32_t *)(s + 32));
            uint32x4_t e = vldrwq_u32((const uint32_t *)(s + 48));
            vstrwq_u32((uint32_t *)(d +  0), a);
            vstrwq_u32((uint32_t *)(d + 16), b);
            vstrwq_u32((uint32_t *)(d + 32), c);
            vstrwq_u32((uint32_t *)(d + 48), e);
            s += 64; d += 64; n -= 64u;
        }
        while (n >= 16u) {
            vstrwq_u32((uint32_t *)d, vldrwq_u32((const uint32_t *)s));
            s += 16; d += 16; n -= 16u;
        }
    } else {
        while (n >= 16u) {
            vstrbq_u8(d, vldrbq_u8(s));
            s += 16; d += 16; n -= 16u;
        }
    }
    if (n) {
        mve_pred16_t p = vctp8q((uint32_t)n);
        vstrbq_p_u8(d, vldrbq_z_u8(s, p), p);
    }
    return dst;
}
```

That took Concat from 1 019 501 to 343 459 cycles (6.997 to 2.36 per byte) and
the whole hop in silence from 2 826 500 to 2 104 000.

I want to be precise about why it helps, because "we wrote a faster memcpy" is
misleading. The newlib-nano one is not byte-wise — for aligned pointers it
already copies words, we checked the disassembly. What changes is the number
of outstanding misses on the path from the M55 to npuRAM, where a single
transaction costs roughly 66 cycles. One miss at a time gives you seven cycles
per byte no matter how the loop is written; four in flight give you 1.4 in a
plain copy and 2.36 through the runtime's per-row calls. So this is a memory
latency effect, and I don't think an application should have to redefine a
libc function to get at it.

We check the override at start-up against a byte-wise reference over lengths
0..300 on eight alignments. The reference needs `volatile` pointers, otherwise
GCC recognises the loop and calls `memcpy` — the very function under test.

Settings. The backend line is taken verbatim from
`network_generate_report.txt`:

```
atonn -i n6_gather2_qdq_OE_3_3_1.onnx \
      --json-quant-file n6_gather2_qdq_OE_3_3_1_Q.json \
      -g network.c \
      --load-mdesc stm32n6.mdesc \
      --load-mpool stm32n6_nucleo_app_safe.mpool \
      --load-cdesc cortex-m55.cdesc \
      --optimization 3 --all-buffers-info --cache-maintenance --Oauto-sched \
      --native-float --enable-virtual-mem-pools --Omax-ca-pipe 4 \
      --Ocache-opt --Os --enable-epoch-controller --generate-stai
```

On the frontend, `stedgeai generate` from ST Edge AI Core 4.0 with
user-allocated IO (`--no-inputs-allocation`, `--no-outputs-allocation`) and
channel positions left at their defaults. The memory pool is
`stm32n6_nucleo_full_onchip` with cpuRAM1/2 trimmed so they don't overlap our
own RAM; weights (273 kB) and activations are all on-chip.

For completeness, compiler options that did not move Concat at all:
`eliminate_concat_split`, `fuse_consecutive_concats_new`, `--ec-optimize`,
`-S`. And `--Ox` was clearly worse — 172 blocks instead of 70, estimate
2 725 403 against 1 263 056.

As for what would help us: ideally the fast path would apply to any
concatenation that is contiguous in the destination, not only when the axis is
leftmost. Failing that, having the runtime call its own optimized copy instead
of libc `memcpy` would remove the need for applications to do what we did. And
at the very least, documenting the `axis_is_leftmost` condition would be
worth a lot — nothing in the docs hints that a size-2 axis sitting to the left
of the concat axis triples its cost, and that note alone would have saved us
several days.

If it's useful I'm happy to run experiments on our board — the epoch profiler
is wired up and I can turn a build around quickly.
