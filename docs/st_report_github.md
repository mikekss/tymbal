# Report #2 — GitHub issue in STM32CubeN6 (config templates and memory)

**Where:** github.com/STMicroelectronics/STM32CubeN6 → the **Issues** tab →
*New issue*. Needs a GitHub account (not an ST one).
**Title:** from "Title" below.
**Labels:** you cannot set them (the maintainer does) — not a problem.
**Language:** English.

Two findings in one issue: both are about a template or a default disagreeing
with the real board. If they ask to split it, it cuts cleanly at the headings.

---

**Title:** `mx25um51245g_conf.h` template defaults do not match the
NUCLEO-N657X0-Q board (dummy cycles, XSPI kernel clock); AXISRAM3..6 left
disabled after boot

**Package:** STM32Cube_FW_N6 V1.4.0
**Board:** NUCLEO-N657X0-Q (MB1940), STM32N657X0H3Q
**Tools:** STM32CubeIDE 2.1.1

## 1. `mx25um51245g_conf.h`: dummy cycles 6 vs the 20 the BSP actually programs

The package ships this file only as `*_conf_template.h`, with

```c
#define MX25UM51245G_DUMMY_CYCLES_READ_OCTAL_DTR   6
```

The board driver, however, programs the flash itself: `XSPI_NOR_EnterDOPIMode`
writes CR2 register 3 with `MX25UM51245G_CR2_DC_20_CYCLES` — that is **20**
dummy cycles.

So an application that copies the template unchanged (which is what a template
invites) ends up with a host expecting 6 while the device is configured for 20.
Reads then return plausible-looking garbage rather than failing cleanly.

Suggestion: either ship a board-specific `mx25um51245g_conf.h` with 20, or add
a comment in the template stating that the value must match what the BSP writes
into CR2 and pointing at `XSPI_NOR_EnterDOPIMode`.

## 2. XSPI2 kernel clock: examples use 266 MHz, this flash on this board works at 50 MHz

`BSP_XSPI_NOR_Init` configures the device with prescaler 3, then — as its last
action — calls `HAL_XSPI_SetClockPrescaler(hxspi, 0)`, putting the bus at the
**full kernel clock**. The kernel clock therefore has to be the flash's actual
operating frequency, not merely a frequency at which initialisation succeeds.

Following the ST examples we had IC3 = PLL1/6 = 266.7 MHz. Initialisation
reported success: `BSP_XSPI_NOR_Init` returned 0, and internally
`XSPI_NOR_EnterDOPIMode` reads CR2 back from the device and compares it — the
comparison passed. Everything *after* the prescaler change failed.

At 133 MHz the device answered (JEDEC ID `C2 C2 80`, correct for octal DTR)
but reads from non-zero offsets were wrong: offsets 0 / 0x400000 / 0x400004
returned `324D5453 / 324D5453 / FFFFFFFF` instead of
`324D5453 / 3157364E / 00044520` — the address phase was not making it, and
0x400000 is a single bit (bit 22).

The authoritative value turned out to be elsewhere in the package:
`Middlewares/ST/STM32_ExtMem_Manager/custom/memories/` contains exactly one
profile for this part, `stm32_mx25um51245g_50Mhz.h`, and the programmer's
external loader uses it. With IC3 = PLL1/32 = **50 MHz** it worked first try.

Suggestion: state the supported XSPI kernel-clock range for this flash where
users will look — in the BSP or in the conf template — and ideally note that
`ExtMem_Manager/custom/memories/` is the source of truth for per-device timing.
It is not an obvious place to look when the symptom is "init succeeded, reads
are wrong".

## 3. AXISRAM3..6 are left disabled after the bootloader hands over

On this device the FSBL receives control with AXISRAM banks 3..6 **not
clocked** (`RCC->MEMENR`) and held in shutdown (`RAMCFG_CR_SRAMSD`). The first
write into those banks hard-faults.

This is presumably intentional (power), but it is not obvious from the
examples: a linker script that places a buffer there builds cleanly and fails
at run time. A line in the memory documentation, or a comment in the example
linker scripts, would help.

## Why this class of report

All three share a shape worth naming: **on this device a misconfiguration does
not look like a failure.** Initialisation returns success, the device answers
its ID, the build links — and the fault appears later, somewhere else. Vendor
templates and defaults are a starting point, not a configuration, and it would
help a lot if the packages said so where the default is known to be wrong for
the shipped board.

---

### Practice

1. The issue takes a minute to file, and there may be no answer for weeks —
   that is normal for ST firmware repositories.
2. If you want an answer for certain, duplicate it as an **online ticket**
   from your ST account and link to the issue from there.
3. Item 3 (AXISRAM) is the weakest of the three: it is documentation rather
   than a defect. If you want to shorten this, cut it first.
