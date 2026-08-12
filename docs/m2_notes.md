# M2 — the NPU in the runtime. Journal, 3 Aug 2026

Milestone: replace `npu_stub.c` with the real LL_ATON driver, measure `t_call`,
cross-check the path numerically, load the weights. Status at end of day:

**Done.** The NPU runs on real weights (`peak` 980 → 63). In silence we fit the
budget with headroom (`underrun=0`), while playing we do not (4.18 against
3.2 Mcyc, analysis below). Milestone M2 is closed, apart from two time reserves.

## Timing summary

| configuration | inference, cycles | µs | hop, Mcyc | underrun |
|---|---|---|---|---|
| NPU 400 MHz, no EC | 3 879 418 | 4849 | 4.46 | grows +142/hb |
| NPU 800 MHz, no EC | 2 834 969 | 3543 | 3.41 | +31/hb |
| **NPU 800 MHz + EC** | **2 540 911** | **3176** | **3.115** | **0** |
| same, but with notes sounding | 3 601 216 | 4502 | 4.18 | grows |

The hop budget is 3.2 Mcyc (4 ms at 800 MHz). The M0 reference for d44 is
2866 µs.

Hop breakdown in silence: skb+submit 53 k · **inference 2 541 k** · PQMF 74 k ·
wow/flutter 133 k · `swap_states` ~320 k (it does not land in a PROF slot; it is
measured as the remainder).

## Eight pitfalls, each one failing with no message

The common pattern: **on this chip an init failure does not look like a
failure**. Not one of the errors listed here produced any diagnostics — only
silence in the terminal, a hard fault, or a quietly wrong result. The only
thing that really helped was `printf` breadcrumbs around every step in `main()`.

1. **`1f` instead of `1.0f`** in the generated `n6_npu_scales.h`. `%.9g` on a
   round number gives `1`, and concatenating that with `f` is not a
   floating-point constant. Fix it in the generators (`quantize_gather2.py`,
   `tools/export_fir.py`), which now have `_cfloat()`. Checked with the
   compiler both ways.

2. **`//` comments in the mpool.** `atonn` parses the map as strict JSON: the
   file must start with `{`, ASCII only. `neural_art.json`, on the other hand,
   does allow comments — hence the temptation. Details:
   `fw/stm32n6_nucleo_app_safe.md`.

3. **`n6_pipe_init` came before UART init.** Harmless with the stub; with the
   real driver the very first access to ATON went into a hard fault before the
   terminal came up. The order in `main()` was changed and breadcrumbs added.

4. **Banks AXISRAM3..6 are off after the bootloader.** The clock domain is not
   started (`RCC->MEMENR`), and the banks are in shutdown (`RAMCFG_CR_SRAMSD`).
   These are npuRAM3..6, where the weights and activations live. The first
   write there hangs the core. Fixed by step 0 in `npu_boot.c`.

5. **RIF and RISAF for the NPU were never configured anywhere.** CubeMX's
   `SystemIsolation_Config()` touches only GPDMA and GPIO. The NPU master
   attributes and `RISAF2..8,15` are needed — the main one is `RISAF6`
   (SRAM3..6).

6. **The graph buffer list holds more than the inputs.** In `network.c`, under
   `#if LL_ATON_DBG_BUFFER_INFO_EXCLUDED == 0`, all the weight and bias buffers
   (`is_param = 1`) land in the same array. They must not be counted together
   with the 13 inputs — the filter is strictly `is_param == 0`. Symptom:
   `prof_err=1`.

7. **The shape lives in `mem_shape`, not in `shape`.** For the input, `.shape` =
   `{1,2,48,8}` and `.mem_shape` = `{1,8,2,48}`. The `[1,C,V,T]` layout the
   caller works with is `mem_shape` (`chpos = CHPos_First`).

8. **IC6 and IC11 were set to divider 4.** These are the clocks for the NPU and
   its memory; in `SystemClock_Config` they were marked "not used" from back
   when the NPU was a stub. The NPU ran at 400 MHz instead of 800, and so did
   the memory. **This was the main reason for the 4849 against 2866
   discrepancy.**

9. **The file is in `.project` but does not reach the build.** CubeIDE keeps
   `Debug/.../subdir.mk` as a cache and does not always re-read `.project`.
   Symptom: `undefined reference` to a function from a file that is on disk and
   listed in the project. Fixed by refreshing the project in the IDE (F5), or
   by editing `subdir.mk` by hand: three lists (`C_SRCS`, `OBJS`, `C_DEPS`), the
   build rule and the `clean` recipe. The second path is safe: on the next
   regeneration the IDE arrives at the same result from `.project`.

10. **A HAL module is disabled in `stm32n6xx_hal_conf.h`.** We already hit this
   with `HAL_CACHEAXI_MODULE_ENABLED` and hit it again with
   `HAL_XSPI_MODULE_ENABLED`. CubeMX enables only the modules it sees in the
   `.ioc`; anything added by hand you have to enable yourself — and also add
   `stm32n6xx_hal_<module>.c` to `.project`. Three errors in a row from one
   cause: unknown type, then unknown function, then unresolved symbol.

## What the clock analysis showed

Doubling the NPU clock gave a speedup of only 1.37 times. Solving
`A + B = 3 879 418` (at 400 MHz) and `A/2 + B = 2 834 969` (at 800):
**A = 2.09 Mcyc on the NPU, B = 1.79 Mcyc on the M55**.

That is, more than half of the "inference" is the core executing `Concat` and
`Slice` in 54 of the 88 Hybrid epochs. The NPU clock does not fix this at all.
Hybrid epochs are the direct price of the dilation workaround (D-10, the
gather2 shape).

After enabling EC: 34 hardware epochs fused into **15 EC blobs** instead of
zero, inference −11.6%. The weights also shrank 279.9 → 273.3 kB.

`blocks=13217` per hop for 88 epochs: `RunEpochBlock` is called hundreds of
times per epoch, returning "not ready yet". Cooperative operation is declared
in `npu_iface` but not used — the skeleton is computed before the polling loop,
and between calls there is nothing for the M55 to do.

## AXI cache: off, and that is normal

Three failures in one day, all of them a hard fault in `HAL_CACHEAXI_Enable` on
the first read of `CACHEAXI->SR`. The first two: before and after RISAF setup,
identically, at 400 MHz. The third was already at 800 MHz, and it matters more
than the first two, because **the failure is intermittent**: with the same
`N6_NPU_USE_CACHE 1` several runs before it went through and printed "bus
open". So the cause is not the call order inside `n6_npu_boot` but a state
inherited from the previous run or from the bootloader.

Decision: **the cache is off for good** until it gets its own analysis. The
trade is obvious — a measured benefit of 0.07% (that is, zero) against an
intermittent failure at startup. Candidates for that analysis are written down
in the header of `npu_boot.c`: power for the NPU_CACHE domain
(`RCC_MEMENR_NPUCACHERAMEN`, not `CACHEAXIRAMEN`, which is what we set) and an
MPU that does not cover the CACHEAXI register window.

Important: **this is not a loss.** In ST's official mpools for the
NUCLEO-N657X0-Q, all the internal `npuRAM3..6` pools come without `cacheable`,
and `CACHEABLE_ON` is set only on the external flash pool `octoFlash`. The AXI
cache on the N6 is for weights read from NOR, not for internal SRAM. While
everything is on-chip, the cache is not needed. It will be needed if the
weights move to NOR (see below).

## Open

## Weights: loaded. Five causes in a row, all of them NOR path configuration

`network.c` holds only the epoch descriptors; the 273 kB of weights sit in a
separate file `network_atonbuf.AXISRAM5.raw` and do not fit into the FSBL image
(the ROM region is 255 kB, 215 used). Before 3 Aug the network ran on
uninitialized memory: the timing measurements are still valid — convolution
speed does not depend on the data — but there was no point measuring quality at
all. The indicator was `peak` in the heartbeat: **980/1000, that is, the output
almost clipping.**

**Result: `weights: 279840 B, CRC D8CF24D3 — OK`, and `peak` dropped 980 → 63.**
That is the real confirmation, not a checksum match: for the first time the
network runs on its own weights. Timing did not change: hop 3.116 Mcyc,
`underrun=0`, inference 2.54 Mcyc — loading costs about 3 ms once at startup.

### Why not ST's canonical path

ST puts the weights in the `xSPI2` pool with `constants_preferred`, and the NPU
reads them straight from flash through the AXI cache. For us that is a bad
trade: `t_call` sits right at the edge (3.115 of 3.2 Mcyc), `octoFlash` in the
compiler model has byteWidth 1 and latency HIGH against byteWidth 8 and LOW for
`npuRAM`, and we have memory to spare — 402 of 2880 kB of pools are used. We
copy the blob into AXISRAM5 at 0x342E0000 — where the graph is already
addressed — and touch neither the mpool nor the generation.

### Five causes, and the order in which they surfaced

Each one masked the next, so the order is what matters.

1. **The VDDIO3 power domain was not enabled.** The XSPI2 pins sit on it. I did
   call `HAL_PWREx_ConfigVddIORange`, but not `HAL_PWREx_EnableVddIO3`. In ST's
   examples it is not in `HAL_XSPI_MspInit` but in the **global** `HAL_MspInit`,
   which is why it did not show up in a line-by-line cross-check of the XSPI
   part. Our `msp.c` does not have it at all: the project started from a SAI
   example. Symptom: commands go out, the controller returns `HAL_OK`, the
   JEDEC ID reads back as zeros.

2. **My own XSPI init instead of the board's standard driver.** Once access to
   `Drivers/` opened up, a cross-check against `stm32n6xx_nucleo_xspi.c` turned
   up three more differences: no `FORCE_RESET` of the XSPI2 peripheral before
   configuration (and the programmer's external loader had been working with it
   before us), NCS without a pull-up, and configuration not done at a reduced
   clock. Plus `HAL_XSPIM_Config`, which the standard driver does not call at
   all. Conclusion: **hand the whole low-level part to the BSP**, keeping only
   what the BSP does not do — the XSPI2 kernel clock source.

3. **A 266 MHz kernel clock.** `BSP_XSPI_NOR_Init` configures the chip at
   prescaler 3, and its last line calls `HAL_XSPI_SetClockPrescaler(hxspi, 0)` —
   it moves the bus to the FULL kernel clock. So the kernel clock has to be the
   flash's working frequency. We had IC3 = PLL1/6 = 266.7 MHz (taken from an ST
   example where the application always sits at prescaler 1 and never gets down
   to zero). The log proves the diagnosis: `BSP_XSPI_NOR_Init` returned 0, and
   inside it `XSPI_NOR_EnterDOPIMode` **reads CR2 back from the chip** and
   compares it with `MX25UM51245G_CR2_DOPI` — that cross-check passed. What
   failed was exactly the calls AFTER the prescaler change.

4. **The dummy cycle count.** ST ships `mx25um51245g_conf.h` only as a template
   (`*_conf_template.h`), and in it `DUMMY_CYCLES_READ_OCTAL_DTR = 6`. But the
   board driver, in `XSPI_NOR_EnterDOPIMode`, programs CR2 register 3 in the
   chip with `MX25UM51245G_CR2_DC_20_CYCLES`. The template is not tied to the
   board; its values have to be checked against what is actually written to
   CR2. We now have 20.

5. **133 MHz is still too fast.** After the change to 133 the chip started
   answering (JEDEC ID `C2 C2 80` — the correct answer in octal DTR, where
   every byte is doubled and the driver takes the first three), but reads from
   non-zero offsets lied. A probe at 0 / 0x400000 / 0x400004 returned
   `324D5453 / 324D5453 / FFFFFFFF` instead of
   `324D5453 / 3157364E / 00044520` — the address phase was not getting
   through, and 0x400000 is one single bit (bit 22). The reference turned up in
   `Middlewares/ST/STM32_ExtMem_Manager/custom/memories/`: for this flash there
   is **exactly one** profile, `stm32_mx25um51245g_50Mhz.h`, and it is the one
   the programmer's external loader uses. The working point: start at 20 MHz,
   octal DTR **50 MHz**. We set IC3 = PLL1/32 = 50 MHz — it worked on the first
   try.

### The working configuration, in short

- XSPI2 kernel clock: **IC3 = PLL1/32 = 50 MHz** (`n6_weights.c`,
  `xspi_kernel_clock` — the only thing the BSP does not do itself);
- `mx25um51245g_conf.h`: **dummy octal = 20** (matched to what the BSP writes
  into CR2);
- the entire low level: `BSP_XSPI_NOR_Init(0, {OPI, DTR})` from
  `stm32n6xx_nucleo_xspi.c` + `mx25um51245g.c`;
- reads are **indirect**, `BSP_XSPI_NOR_Read` straight into AXISRAM5. The
  memory-mapped window is not used: it is for code that reads flash constantly,
  and we only have to move 273 kB once. It also takes the core cache, AXI
  prefetch and the `CMD_CFG` state out of the picture;
- the blob is built by `tools/pack_weights.py`: signature `N6W1`, length,
  CRC32. It is flashed at 0x70400000, separately from the image. The check
  happens before copying — ST's documentation warns outright that weights out
  of sync with the graph give wrong inference with NO diagnostics at all.

### What this teaches for next time

- **An init failure on this chip does not look like a failure.** `HAL_OK` from
  the controller does not mean anything reached the device at the other end of
  the bus. The only thing that really worked was printing raw values (the JEDEC
  ID, the first word at an address, a snapshot of `hxspi->ErrorCode`/`State`)
  and comparing them with what the programmer read independently.
- **ST's templates (`*_conf_template.h`) are not configuration, they are a
  blank.** Their values have to be checked against the code that consumes them.
- **A reference working configuration exists in ST's tree.**
  `STM32_ExtMem_Manager` describes every supported chip as a table: commands,
  dummy, frequency. That is the exact place where the vendor says what its
  hardware runs on. I should have gone there first, not fifth.

**Two time reserves, both on the M55:**

- **1.5 Mcyc** — the Hybrid epochs (`Concat`/`Slice`). This touches the graph
  shape, that is, D-10. The largest reserve and the most expensive in its
  consequences.
- **1.11 Mcyc** — the skeleton with notes sounding. There is an unfinished MVE
  loop here (275 cycles per harmonic against an expected 5–10), postponed
  "until after M2". That moment has come.

**~320 k cycles** — `swap_states`, copying the 43.7 kB state ring. A
consequence of generating with `allocate-outputs`; fixed by user-allocated IO.

**Inference grows by 42% with notes sounding** (2.54 → 3.60 Mcyc), even though
the NPU's work does not depend on the data. Hypothesis: skb flushes the M55
cache that the Hybrid epochs depend on. Not verified.
