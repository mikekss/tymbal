/* npu_boot.h — bringing up the NPU before the first call into LL_ATON.
 * Call it in main() BEFORE n6_pipe_init() and, preferably, AFTER the UART is
 * initialized: then a failure shows up as a message, not as silence. */
#ifndef N6_NPU_BOOT_H
#define N6_NPU_BOOT_H

/* NPU clocking and reset, RIF attributes for the NPU master, default RISAF
 * for SRAM1..6, FLEXMEM and NPU MST0/MST1. Idempotent. */
void n6_npu_boot(void);

#endif
