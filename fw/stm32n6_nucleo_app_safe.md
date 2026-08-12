# stm32n6_nucleo_app_safe.mpool — a map kept clear of the application memory

Created 2 Aug 2026. Source — `stm32n6_nucleo_full_onchip.mpool`.

## Why

In `full_onchip` the pools `cpuRAM1` and `cpuRAM2` covered the FSBL `N6_m1`
regions:

| application region | addresses | size |
|---|---|---|
| `.ram2` (g_pipe 229.2K + g_ck4_buf 192K) | 0x340A4000..0x34114000 | 448K |
| ROM  | from 0x34180400 | 255K |
| RAM  | from 0x341C0000 | 256K |

That is, the `atonn` compiler treated memory occupied by the firmware as its
own. The overlap was 336 KB, and the FSBL ROM/RAM lay entirely inside
`cpuRAM2`.

## What was changed (and only this)

| pool | full_onchip | app_safe |
|---|---|---|
| `cpuRAM1` | 0x34064000, 512K | 0x34064000, **256K** — ends exactly where `.ram2` starts |
| `cpuRAM2` | 0x34100000, 1024K | **0x34114000, 432K** — moved into the free hole 0x34114000..0x34180000 |

The remaining pools (`flexMEM`, `npuRAM3..6`) and the `params` block are byte
for byte as in `full_onchip`. Verified by comparing the parsed JSON.

On-chip budget after the edit: flexMEM 400 + cpuRAM1 256 + cpuRAM2 432 +
npuRAM3..6 4x448 = **2880K**. For d44 (weights 0.23-0.28 MB int8, activations
and states ~0.3-0.5 MB) — with a large margin.

## FORMAT: a pitfall that cost one run

`atonn` parses the mpool with a **strict JSON parser**:

- **`//` comments are FORBIDDEN** — the file must start with `{`;
- **ASCII only** (a cp1252 reader).

Not to be confused with `neural_art.json`: there `//` comments are in fact
allowed, and so the temptation to comment the map the same way is strong. The
symptom of a violation:

```
Error: memory pool JSON ERROR:
INVALID_ARGUMENT:Expected a value.
// stm32n6_nucleo_ap
^
Warning: Oauto did not find valid compile options: aborting
```

The line `total bytes left unallocated=...` is meaningless in this case — it is
a rollback after the parse failed, not a report about the layout.

## Open

`t_call` = 2866 us was measured on the OLD map (`full_onchip`). The headroom
was 10%. After trimming `cpuRAM1` and moving `cpuRAM2` it has to be
**re-measured** — that is an M2 criterion.
