/* n6_weights.c — loading the NPU weights from external NOR into AXISRAM5.
 * Milestone M2.
 *
 * WHY. The atonn compiler puts only the epoch descriptors into network.c, and
 * emits the weights (273 kB) as a separate file network_atonbuf.AXISRAM5.raw,
 * addressed absolutely at 0x342E0000. They do not fit into the FSBL image: the
 * ROM region is 255 kB, of which 207 are already taken. So the blob lives in
 * NOR and is copied here at startup.
 *
 * WHY WE COPY INSTEAD OF READING FROM FLASH. ST's canonical path is another
 * one: an xSPI2 pool with "constants_preferred", and the NPU reads the weights
 * from NOR through the AXI cache. For us that is a bad trade: t_call sits
 * right at the edge (3.115 out of 3.2 Mcyc), octoFlash in the compiler's model
 * has byteWidth 1 and latency HIGH against byteWidth 8 and LOW for npuRAM, and
 * memory is plentiful — 402 kB out of 2880 kB of pools are taken. On top of
 * that, copying requires neither changing the mpool nor regenerating the
 * graph: it is already addressed into AXISRAM5.
 *
 * WHY THERE IS NO XSPI INITIALISATION OF OUR OWN HERE (rewritten 3 Aug).
 * There was one: carried over from the ST example XSPI_NOR_MemoryMapped_DTR.
 * Two runs in a row gave "commands go out, no answer" — the JEDEC ID read as
 * zeros. When access to Drivers/ opened up, three divergences from the board's
 * STOCK driver turned up at once
 * (Drivers/BSP/STM32N6xx_Nucleo/stm32n6xx_nucleo_xspi.c):
 *
 *   1. the VDDIO3 power domain was not enabled (HAL_PWREx_EnableVddIO3). In
 *      the examples that call sits in the GLOBAL HAL_MspInit, not in
 *      HAL_XSPI_MspInit — which is why it escaped me while I was comparing
 *      the XSPI part line by line. In our msp.c it is missing altogether: the
 *      project grew out of an SAI example;
 *   2. no FORCE_RESET/RELEASE_RESET of the XSPI2 peripheral was done before
 *      configuring it, and before us the programmer's external loader had
 *      been working with it;
 *   3. the NCS pin was driven high without a pull-up (ST uses GPIO_PULLUP).
 *
 * Plus the reset of the chip itself in all three protocols and "configure at
 * the slow clock, read at the fast one" — also in the driver. There is no
 * point repeating all of that by hand: the board driver is the stock one,
 * debugged by ST for exactly this NOR and exactly this board, and the external
 * loader uses the very same logic. So the whole low-level part is now two BSP
 * calls, and what is left here is only what is genuinely ours: the blob
 * layout, the integrity check and the copy.
 *
 * Exactly one thing is left to the application that the BSP does not do: the
 * source of the XSPI2 kernel clock (IC3 from PLL1). That is configured below.
 *
 * PROTECTION AGAINST GOING OUT OF SYNC. The weights are rigidly tied to the
 * graph layout in the firmware; ST's documentation warns outright that a
 * mismatch gives wrong inference WITHOUT any diagnostics. So a header with a
 * signature, a length and a CRC32 sits in front of the blob — they are written
 * by tools/pack_weights.py, and here they are checked before copying.
 */
#include <string.h>
#include "stm32n6xx_hal.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "n6_weights.h"

/* --- external NOR layout --------------------------------------------------
 * The FSBL is flashed at 0x70000000 (h0_notes: standalone flash boot, signed).
 * The blob is moved 4 MB further on — safely outside the image. */
#define W_NOR_OFFSET     0x00400000u          /* offset inside the NOR      */
#define W_NOR_ADDR       (XSPI2_BASE + W_NOR_OFFSET)
#define W_DST            0x342E0000u          /* npuRAM5, addr from network.c */
#define W_MAGIC          0x3157364Eu          /* "N6W1" little-endian        */
#define W_MAX_BYTES      (448u * 1024u)       /* size of the AXISRAM5 bank   */

typedef struct {                 /* exactly what tools/pack_weights.py writes */
    uint32_t magic;
    uint32_t bytes;              /* length of the payload, header excluded */
    uint32_t crc32;              /* over the payload */
    uint32_t reserved;
} w_hdr_t;

uint32_t n6_weights_bytes = 0;   /* how much was copied, for the heartbeat */
uint32_t n6_weights_crc   = 0;   /* the computed CRC32, for checking by eye */
uint32_t n6_weights_id    = 0;   /* JEDEC ID, we expect 0xC2803A */
int32_t  n6_weights_step  = 0;   /* return code of BSP_XSPI_NOR_Init */
uint32_t n6_weights_hal   = 0;   /* (ErrorCode << 8) | State — HAL snapshot on failure */
uint32_t n6_weights_seen  = 0;   /* what was actually read at W_NOR_ADDR */
uint32_t n6_weights_p[3]  = {0, 0, 0};   /* addressing probe: see n6_weights.h */

/* --- CRC32 (IEEE 802.3, as in tools/ck4_compare.py) -----------------------
 * Nibble table: 16 words against 1 kB, and over 273 kB the speed difference
 * is imperceptible next to the startup time. */
static const uint32_t crc_nib[16] = {
    0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu,
    0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
    0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
    0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu
};

static uint32_t crc32_buf(const uint8_t *p, uint32_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        c ^= p[i];
        c = (c >> 4) ^ crc_nib[c & 0x0Fu];
        c = (c >> 4) ^ crc_nib[c & 0x0Fu];
    }
    return ~c;
}

/* Source of the XSPI2 kernel clock. The BSP does not do this — it is the
 * application's duty, and it is easy to get wrong here.
 *
 * DIVIDER 12, NOT 6 (fix of 3 Aug, failure with code 7).
 * The BSP configures the flash at prescaler 3, and at the very end
 * BSP_XSPI_NOR_Init does HAL_XSPI_SetClockPrescaler(hxspi, 0) — that is, it
 * puts the bus at the FULL kernel frequency. So the kernel frequency must be
 * the working frequency of the flash, not four times higher. We had
 * IC3 = PLL1/6 = 266.7 MHz (taken from the ST example, where the application
 * always sat at prescaler 1 and never came down to 0) — and after
 * SetClockPrescaler(0) the bus ran away to 266.7 MHz.
 *
 * That this is exactly the cause is visible from the log: BSP_XSPI_NOR_Init
 * returned 0, and inside it XSPI_NOR_EnterDOPIMode READS CR2 back from the
 * chip and compares it with MX25UM51245G_CR2_DOPI. That check passed — so the
 * path is alive and the chip answers. What failed was exactly the two calls
 * that come AFTER the prescaler change: ReadID returned zeros, and
 * EnableMemoryMappedMode returned an error.
 *
 * DIVIDER 32 = 50 MHz (fix of 3 Aug, the second one). At first I set 12
 * (133 MHz) by analogy with the ST example — the chip started answering, but
 * reads from non-zero offsets gave garbage: the probe of 0/0x400000/0x400004
 * returned 324D5453 / 324D5453 / FFFFFFFF instead of
 * 324D5453 / 3157364E / 00044520. That is, the address phase was not getting
 * through.
 *
 * The reference turned up in Middlewares/ST/STM32_ExtMem_Manager: for THIS
 * flash ST keeps the profile stm32_mx25um51245g_50Mhz.h, and there is no other
 * one for it. That is exactly what the programmer's external loader uses — the
 * very one that reads 0x400000 correctly. Its operating point is: octal DTR at
 * .OptionalConfig.Frequency = 50000000, starting at 20 MHz. We go there too.
 *
 * PLL1 = 1600 MHz (PLLM=3, PLLN=100 in SystemClock_Config), divider 32 gives
 * 50 MHz. The BSP configures at prescaler 3, that is 12.5 MHz, and puts the
 * bus at 50 MHz at the end. 273 kB read in ~3 ms — negligible at startup. */
static int xspi_kernel_clock(void)
{
    RCC_PeriphCLKInitTypeDef pc = {0};
    pc.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
    pc.Xspi2ClockSelection  = RCC_XSPI2CLKSOURCE_IC3;
    pc.ICSelection[RCC_IC3].ClockSelection = RCC_ICCLKSOURCE_PLL1;
    pc.ICSelection[RCC_IC3].ClockDivider   = 32;   /* 1600/32 = 50 MHz */
    return (HAL_RCCEx_PeriphCLKConfig(&pc) == HAL_OK) ? N6_W_OK : N6_W_ERR_CLK;
}

/* ------------------------------------------------------------------ the load
 * Call AFTER n6_npu_boot() (the AXISRAM banks must be powered) and BEFORE
 * n6_pipe_init(), which creates the graph and does the first inference. */
int n6_weights_load(void)
{
    BSP_XSPI_NOR_Init_t init;

    int rc = xspi_kernel_clock();
    if (rc) return rc;

    /* The BSP does it all: VDDIO3 power, XSPI2 clock and reset, pins
     * PN0..PN11, chip reset in all three protocols, switch to octal DTR. */
    init.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
    init.TransferRate  = BSP_XSPI_NOR_DTR_TRANSFER;
    n6_weights_step    = BSP_XSPI_NOR_Init(0, &init);
    if (n6_weights_step != BSP_ERROR_NONE) return N6_W_ERR_XSPI_INIT;

    /* WE READ INDIRECTLY, WITHOUT MEMORY MAPPING (fix of 3 Aug, code 8).
     * The mapping switched on successfully (HAL=0000), but a read at
     * 0x70400000 returned the contents of flash address zero: the firmware
     * saw the signature of the signed image there, 324D5453 ("STM2"), while
     * the programmer reads our blob 3157364E at that same address. That is,
     * the mapping window loses the high address bits. There is no point
     * digging into that separately: the window is for those who execute or
     * read from flash constantly, and we need to move 273 kB across ONCE. An
     * indirect read takes an OFFSET in the flash, not an address in the
     * window, so the question disappears entirely — together with the core
     * cache, the AXI prefetch and the state of CMD_CFG.
     *
     * We read straight into AXISRAM5, without an intermediate buffer: that is
     * where the graph needs it. */
    /* ADDRESSING PROBE. One word from three offsets — that is enough to
     * separate "the address is ignored entirely" from "the address gets
     * through partially". What is in the chip is known exactly, the
     * programmer read it on 3 Aug:
     *   0x000000 -> 324D5453 ("STM2", header of the signed image)
     *   0x400000 -> 3157364E ("N6W1", our blob)
     *   0x400004 -> 00044520 (279840 — length of the payload)
     * So a healthy answer is exactly this triple. If all three are the same,
     * the address phase does not work at all. If p[2] equals the second word
     * at zero (00000000), the address is ignored but the offset within the
     * page still works. */
    BSP_XSPI_NOR_Read(0, (uint8_t *)&n6_weights_p[0], 0x000000u,        4);
    BSP_XSPI_NOR_Read(0, (uint8_t *)&n6_weights_p[1], W_NOR_OFFSET,     4);
    BSP_XSPI_NOR_Read(0, (uint8_t *)&n6_weights_p[2], W_NOR_OFFSET + 4, 4);

    w_hdr_t h;
    if (BSP_XSPI_NOR_Read(0, (uint8_t *)&h, W_NOR_OFFSET, sizeof h) != BSP_ERROR_NONE) {
        n6_weights_hal = ((uint32_t)hxspi_nor[0].ErrorCode << 8) |
                         (uint32_t)hxspi_nor[0].State;
        return N6_W_ERR_NOR_READ;
    }
    n6_weights_seen = h.magic;            /* always printed, for diagnostics */
    if (h.magic != W_MAGIC)               return N6_W_ERR_MAGIC;
    if (h.bytes == 0 || h.bytes > W_MAX_BYTES) return N6_W_ERR_SIZE;

    if (BSP_XSPI_NOR_Read(0, (uint8_t *)W_DST, W_NOR_OFFSET + sizeof(w_hdr_t),
                          h.bytes) != BSP_ERROR_NONE) {
        n6_weights_hal = ((uint32_t)hxspi_nor[0].ErrorCode << 8) |
                         (uint32_t)hxspi_nor[0].State;
        return N6_W_ERR_NOR_READ;
    }

    n6_weights_bytes = h.bytes;
    n6_weights_crc   = crc32_buf((const uint8_t *)W_DST, h.bytes);
    if (n6_weights_crc != h.crc32)        return N6_W_ERR_CRC;

    /* The NPU reads this memory past the core cache — push the lines out. */
    SCB_CleanDCache_by_Addr((uint32_t *)W_DST, (int32_t)h.bytes);
    __DSB();
    return N6_W_OK;
}
