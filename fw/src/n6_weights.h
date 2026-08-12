/* n6_weights.h — loading the NPU weights from the external NOR into AXISRAM5.
 *
 * Call order in main():
 *     n6_npu_boot();          // AXISRAM banks powered, RIF/RISAF opened
 *     n6_weights_load();      // <-- here
 *     n6_pipe_init(...);      // creates the graph, first inference
 *
 * The blob is prepared on the host: python tools/pack_weights.py
 * and flashed into the NOR at offset 0x00400000 (absolute 0x70400000).
 *
 * The low level is done by the STOCK board driver (BSP_XSPI_NOR_*), see the
 * header of n6_weights.c — it also explains why our own XSPI initialization
 * never took off. So the build must include two ST files:
 *     Drivers/BSP/STM32N6xx_Nucleo/stm32n6xx_nucleo_xspi.c
 *     Drivers/BSP/Components/mx25um51245g/mx25um51245g.c
 */
#ifndef N6_WEIGHTS_H
#define N6_WEIGHTS_H

#include <stdint.h>

enum {
    N6_W_OK = 0,
    N6_W_ERR_CLK = 1,          /* XSPI2 kernel clock did not come up    */
    N6_W_ERR_XSPI_INIT,        /* 2  BSP_XSPI_NOR_Init, details in      */
                               /*    n6_weights_step                    */
    N6_W_ERR_XSPI_MGR,         /* 3  not used                           */
    N6_W_ERR_NOR_WREN,         /* 4  not used                           */
    N6_W_ERR_NOR_CR2,          /* 5  not used                           */
    N6_W_ERR_NOR_READ,         /* 6  BSP_XSPI_NOR_Read, details in      */
                               /*    n6_weights_hal                     */
    N6_W_ERR_NOR_MMAP,         /* 7  not used: the mapping window       */
                               /*    is not engaged, we read indirectly */
    N6_W_ERR_MAGIC,            /* 8  no blob at that address (sig)      */
    N6_W_ERR_SIZE,             /* 9  length outside sane limits         */
    N6_W_ERR_CRC               /* 10 blob corrupt / from other graph    */
};

/* 0 = the weights are in place at 0x342E0000. Otherwise a code from the enum
 * above. */
int n6_weights_load(void);

extern uint32_t n6_weights_bytes;  /* how much was copied */
extern uint32_t n6_weights_crc;    /* the computed CRC32  */

/* --- diagnostics (printed in main whatever the outcome) -------------------
 *
 * n6_weights_id — not used (the JEDEC ID was confirmed on 3 Aug: C2 C2 80, in
 *   octal DTR every byte is doubled and the driver takes the first three).
 *
 * n6_weights_hal — a HAL snapshot on failure: (ErrorCode << 8) | State.
 *   ErrorCode: 1 TIMEOUT, 2 TRANSFER, 8 INVALID_PARAM, 0x10 INVALID_SEQUENCE.
 *
 * n6_weights_step — the return code of BSP_XSPI_NOR_Init (0 = BSP_ERROR_NONE,
 *   negative ones come from stm32n6xx_nucleo_errno.h: -1 WRONG_PARAM,
 *   -3 PERIPH_FAILURE, -4 COMPONENT_FAILURE).
 */
extern uint32_t n6_weights_id;
extern uint32_t n6_weights_hal;

/* n6_weights_seen — the first word at 0x70400000, as it actually read back.
 *   3157364E — the signature is in place; FFFFFFFF — no blob there (erased);
 *   00000000 — the read returns zeros; anything else — we read the wrong
 *   place. */
extern uint32_t n6_weights_seen;

/* n6_weights_p[0..2] — an addressing probe: the words at offsets 0x000000,
 * 0x400000 and 0x400004. A healthy answer (cross-checked with a programmer on
 * 3 Aug):
 *     324D5453  3157364E  00044520
 * All three identical -> the address phase does not work at all.
 * p[1] == p[0]        -> the high address bits are being lost. */
extern uint32_t n6_weights_p[3];
extern int32_t  n6_weights_step;

#endif
