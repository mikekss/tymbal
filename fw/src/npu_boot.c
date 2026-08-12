/* npu_boot.c — bringing the NPU up before the first call into LL_ATON.
 * Milestone M2.
 *
 * WHY. With the npu_stub.c stub none of this was needed, and n6_pipe_init sat
 * quite happily in main() BEFORE the UART was initialised. With the production
 * driver the very first LL_ATON_RT_RuntimeInit() reaches into the ATON
 * registers, and they are unreachable until three things have been done. The
 * symptom of failure is not an error message but SILENCE: the bus fault turns
 * into a hard fault before the terminal starts working (caught 3 Aug).
 *
 *   0. POWER FOR BANKS AXISRAM3..6. These are npuRAM3..6, where the graph's
 *      weights and activations live. After the bootloader they are OFF: the
 *      clock domain is not started (RCC->MEMENR) and the banks sit in shutdown
 *      (RAMCFG_CR_SRAMSD). The very first write there hangs the core — and it
 *      happens not in LL_ATON but in our own memset of the states inside
 *      n6_npu_create. While the shape contract did not match, create returned
 *      before that point and the symptom stayed hidden.
 *   1. Clocking the NPU and releasing it from reset.
 *   2. RIF: as a master the NPU has to be given secure/privileged attributes,
 *      otherwise the firewall cuts its memory accesses. SystemIsolation_Config(),
 *      which CubeMX generated, configures only GPDMA and GPIO — there is
 *      nothing about the NPU in it.
 *   3. Default RISAF for the regions the graph touches: SRAM3..6
 *      (RISAF6) — these are exactly our npuRAM3..6, where the weights and
 *      activations live — plus NPU MST0/MST1 (RISAF4/5), FLEXMEM (RISAF7) and
 *      SRAM1/2 (RISAF2/3).
 *
 * AXI CACHE: see N6_NPU_USE_CACHE below. The graph was built with
 * --cache-maintenance and the compiler placed LL_ATON_Cache_MCU_Invalidate_Range
 * calls into network.c, that is, it counted on a working cache — but so far we
 * have not managed to switch it on. Cache maintenance on the M55 side works
 * without it anyway.
 *
 * The sequence is modelled on misc_toolbox.c from the ST hello_world example
 * (STEdgeAI\4.0\Projects\STM32N6570-DK\Applications\hello_world).
 */
#include "stm32n6xx_hal.h"
#include "npu_cache.h"
#include "npu_boot.h"

/* Weak hooks that npu_cache.c calls. Without them enabling/disabling the
 * cache silently does nothing. */
void npu_cache_enable_clocks_and_reset(void)
{
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_FORCE_RESET();
    __HAL_RCC_CACHEAXI_RELEASE_RESET();
}

void npu_cache_disable_clocks_and_reset(void)
{
    __HAL_RCC_CACHEAXI_FORCE_RESET();
    __HAL_RCC_CACHEAXI_RELEASE_RESET();
    __HAL_RCC_CACHEAXI_CLK_DISABLE();
}

static uint32_t risaf_max_addr(RISAF_TypeDef *r)
{
    if (r == RISAF2_S) return RISAF2_LIMIT_ADDRESS_SPACE_SIZE;  /* SRAM1_AXI  */
    if (r == RISAF3_S) return RISAF3_LIMIT_ADDRESS_SPACE_SIZE;  /* SRAM2_AXI  */
    if (r == RISAF4_S) return RISAF4_LIMIT_ADDRESS_SPACE_SIZE;  /* NPU MST0   */
    if (r == RISAF5_S) return RISAF5_LIMIT_ADDRESS_SPACE_SIZE;  /* NPU MST1   */
    if (r == RISAF6_S) return RISAF6_LIMIT_ADDRESS_SPACE_SIZE;  /* SRAM3..6   */
    if (r == RISAF7_S) return RISAF7_LIMIT_ADDRESS_SPACE_SIZE;  /* FLEXMEM    */
    if (r == RISAF8_S) return RISAF8_LIMIT_ADDRESS_SPACE_SIZE;  /* NPU_CACHE  */
    if (r == RISAF15_S) return RISAF15_LIMIT_ADDRESS_SPACE_SIZE;/* its config */
    return 0U;
}

/* Two fully overlapping regions: one for secure requests, the second for
 * non-secure. Exactly as in ST's set_risaf_default(). */
static void risaf_open(RISAF_TypeDef *r)
{
    RISAF_BaseRegionConfig_t c;
    c.StartAddress   = 0x0U;
    c.EndAddress     = risaf_max_addr(r);
    c.Filtering      = RISAF_FILTER_ENABLE;
    c.PrivWhitelist  = RIF_CID_NONE;
    c.ReadWhitelist  = RIF_CID_MASK;
    c.WriteWhitelist = RIF_CID_MASK;
    c.Secure = RIF_ATTRIBUTE_SEC;
    HAL_RIF_RISAF_ConfigBaseRegion(r, 0, &c);
    c.Secure = RIF_ATTRIBUTE_NSEC;
    HAL_RIF_RISAF_ConfigBaseRegion(r, 1, &c);
}

/* The AXI cache is OFF. Two attempts to switch it on gave a hard fault right
 * in HAL_CACHEAXI_Enable: the first when it was called before RISAF was
 * configured, the second already after it (3 Aug, the same stack both times).
 * So the cause is not the order of initialisation, and it has to be looked at
 * separately: the likely candidates are the power of the NPU_CACHE domain
 * (RCC_MEMENR_NPUCACHERAMEN, not CACHEAXIRAMEN), or the MPU not covering the
 * CACHEAXI register window.
 * 3 Aug, second attempt: both earlier ones crashed while the NPU domain was
 * clocked at 400 MHz (the IC6/IC11 dividers were 4). With the frequencies
 * fixed we try again — CACHEAXI sits in the same domain, which is a plausible
 * cause.
 * If it faults again — set 0 and take the fallback route for the weights (a
 * copy into AXISRAM5 instead of reading from NOR).
 * The operating point without the cache is measured and works. Set 1 only
 * together with the investigation above. */
/* THIRD failure on 3 Aug, this time at 800 MHz: hard fault in
 * HAL_CACHEAXI_Enable (the first read of CACHEAXI->SR), stack npu_boot.c:145
 * -> npu_cache_enable -> npu_cache_init -> HAL_CACHEAXI_Init/Enable. The
 * important part: this is NOT deterministic — with the same define set to 1
 * several runs before it went through and printed "bus open". So it is about
 * state left over from the previous run or from the bootloader, not about the
 * order of calls inside n6_npu_boot.
 * The cache is off FOR GOOD until a separate investigation: the measured
 * benefit is 0.07%, that is zero, and the price is an intermittent failure at
 * startup. Candidates to investigate: the power of the NPU_CACHE domain
 * (RCC_MEMENR_NPUCACHERAMEN, not CACHEAXIRAMEN, which is the one we set now)
 * and the MPU not covering the CACHEAXI register window.
 * A reminder why the loss is small: in ST's official mpool files for this board
 * the internal npuRAM3..6 pools come WITHOUT cacheable, and CACHEABLE_ON is
 * set only on the external flash pool. We copy the weights into AXISRAM5, the
 * NPU reads on-chip. */
/* SWITCHED ON 4 Aug — now not "let us try", but out of necessity.
 * With user-allocated IO the runtime calls NPU cache maintenance for our
 * buffers (npu_cache_clean_invalidate_range), and that function starts with
 * assert(hcacheaxi_s.Instance == CACHEAXI). The handle is set ONLY in
 * npu_cache_init, which is called only from npu_cache_enable. That is, with
 * the cache off a Debug build stops on the assert, even though with the cache
 * off the operation itself correctly does nothing (there is also an if on the
 * same condition inside). We have no other levers: ST's functions cannot be
 * intercepted, and -DNDEBUG would switch off all asserts at once, including
 * the useful ones.
 *
 * At the same time one of the two candidates from the 3 Aug investigation is
 * closed: the register RCC_MEMENR_NPUCACHERAMEN IS NOT IN stm32n657xx.h AT
 * ALL, there is only CACHEAXIRAMEN, which is what we set. So the "wrong power
 * bit" hypothesis does not hold; the second one remains — coverage of the
 * CACHEAXI register window in the MPU.
 *
 * If it falls over in HAL_CACHEAXI_Enable again, it is visible immediately at
 * startup, before any sound, and rolling back costs one line. */
/* SWITCHED OFF AGAIN on 4 Aug, in the evening — the price is measured, and it
 * is large. Three builds, silence, everything else identical (cycles per hop):
 *
 *   before user-IO, cache off        swap 92 080  npu 2 559 700  hop 2 933 500
 *   user-IO, cache on, .ram2 bufs    swap    213  npu 2 902 000  hop 3 181 000
 *   user-IO, cache on, npuRAM6 bufs  swap    211  npu 2 894 500  hop 3 173 700
 *
 * State concatenation got 92 k cheaper, while the NPU stage got 335 k more
 * expensive. Moving the buffers from .ram2 to npuRAM6 gave back only 7 k — so
 * it is not about placement but about the cache itself. That is exactly what
 * ST's official mpool files say: the internal npuRAM3..6 pools come WITHOUT
 * cacheable.
 *
 * The assert the cache was switched on for is handled differently: -DNDEBUG
 * was added to the project properties. Yes, it silences all of ST's assert()s
 * at once — but the only ST assert that has ever fired for us is this very
 * one, and it is false: with the cache off the function itself correctly does
 * nothing (there is also an if on the same condition inside, see npu_cache.c).
 * Our own checks do not rest on asserts — we have n6_npu_prof_err and codes
 * 71..77, and they live under any NDEBUG.
 *
 * Maintenance of the PROCESSOR cache (LL_ATON_Cache_MCU_*, also known as
 * mcu_cache.c) is not affected by this define: on the N6 it is a separate pair
 * of functions and they are always compiled. Coherency of our buffers between
 * the M55 and the NPU stays in place.
 *
 * So both hypotheses from 3 Aug are closed, and with them the question "why
 * does the cache not help": it does not "not help" — it hurts. */
#ifndef N6_NPU_USE_CACHE
#define N6_NPU_USE_CACHE 0
#endif

void n6_npu_boot(void)
{
    /* 0. Power and release from shutdown for the banks the graph uses.
     * Strictly first: everything that follows touches this memory. */
    RCC->MEMENR |= RCC_MEMENR_AXISRAM3EN | RCC_MEMENR_AXISRAM4EN |
                   RCC_MEMENR_AXISRAM5EN | RCC_MEMENR_AXISRAM6EN |
                   RCC_MEMENR_CACHEAXIRAMEN;
    RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    __DSB();

    /* 1. Clock and reset for the NPU and for the CACHEAXI block.
     * IMPORTANT: npu_cache.c does NOT call npu_cache_enable_clocks_and_reset()
     * itself — the HAL reaches it only through MspInit, and not always then.
     * So we raise the CACHEAXI clock here explicitly, before any
     * HAL_CACHEAXI_*. */
    __HAL_RCC_NPU_CLK_ENABLE();
    __HAL_RCC_NPU_FORCE_RESET();
    __HAL_RCC_NPU_RELEASE_RESET();
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_FORCE_RESET();
    __HAL_RCC_CACHEAXI_RELEASE_RESET();
    __DSB();

    /* 2. The NPU as a master: secure + privileged, CID 1 domain */
    RIMC_MasterConfig_t m;
    m.MasterCID = RIF_CID_1;
    m.SecPriv   = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
    HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &m);
    HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU,
                                          RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);

    /* 3. RISAF — BEFORE touching the cache. The first attempt enabled the
     * cache ahead of the firewall and gave a hard fault right in
     * HAL_CACHEAXI_Enable (3 Aug). */
    risaf_open(RISAF2_S);   /* SRAM1_AXI                                   */
    risaf_open(RISAF3_S);   /* SRAM2_AXI                                   */
    risaf_open(RISAF4_S);   /* NPU MST0                                    */
    risaf_open(RISAF5_S);   /* NPU MST1                                    */
    risaf_open(RISAF6_S);   /* SRAM3..6 = npuRAM3..6: weights, activations */
    risaf_open(RISAF7_S);   /* FLEXMEM                                     */
    risaf_open(RISAF8_S);   /* NPU_CACHE                                   */
    risaf_open(RISAF15_S);  /* NPU_CACHE configuration                     */

    /* 4. The cache last, once everything is open. */
#if N6_NPU_USE_CACHE
    npu_cache_enable();
#else
    npu_cache_disable();
#endif
}
