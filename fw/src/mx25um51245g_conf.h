/* mx25um51245g_conf.h — configuration of the stock NOR driver for our project.
 *
 * The driver Drivers/BSP/Components/mx25um51245g/mx25um51245g.h includes this
 * file by name, while ST itself ships only *_conf_template.h next to it — a
 * blank with no include path for a specific MCU. Here it has been brought in
 * line with our project: stm32xxxx_hal.h -> stm32n6xx_hal.h, the dummy-cycle
 * values left as ST has them (the same ones are baked into the octal DTR read
 * commands).
 *
 * Lives in fw/src and is copied into FSBL/DSP/src — that directory is already
 * in -I.
 */
#ifndef MX25UM51245G_CONF_H
#define MX25UM51245G_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal.h"

/* CONF_OSPI_ODS from ST's template was dropped deliberately: the driver never
 * uses it, and the constant MX25UM51245G_CR_ODS_24 it refers to is missing
 * from mx25um51245g.h (there is MX25UM51245G_CR1_ODS instead). Keeping it
 * would mean leaving a mine armed in case someone ever expands it. */

/* THE NUMBER OF DUMMY CYCLES MUST MATCH WHAT THE DRIVER WRITES INTO THE CHIP.
 * ST's template had 6 here, and that was the cause of the 3 Aug failure (code
 * 8, a read from any offset returned one and the same thing). The board driver
 * in XSPI_NOR_EnterDOPIMode programs CR2 register 3 with
 * MX25UM51245G_CR2_DC_20_CYCLES, that is, the chip waits 20 cycles, while the
 * read command substituted 6 here — the data was picked up 14 cycles earlier
 * than the chip started to put it out. The *_conf_template.h template is not
 * tied to a board; its values have to be cross-checked against what is
 * actually written into CR2, not carried over as is.
 *
 * Register reads (REG_OCTAL*) do not depend on CR2 register 3, they have their
 * own fixed count — their template values are correct, which was confirmed:
 * the CR2 cross-check after switching to octal DTR passed. */
#define DUMMY_CYCLES_READ            8U    /* 1-line fast read, not used          */
#define DUMMY_CYCLES_READ_OCTAL      20U   /* octal STR, CR2[3] = DC_20_CYCLES    */
#define DUMMY_CYCLES_READ_OCTAL_DTR  20U   /* octal DTR — our path                */
#define DUMMY_CYCLES_REG_OCTAL       4U
#define DUMMY_CYCLES_REG_OCTAL_DTR   5U

#ifdef __cplusplus
}
#endif

#endif /* MX25UM51245G_CONF_H */
