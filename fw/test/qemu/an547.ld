/* an547.ld — the MPS3/AN547 (Cortex-M55) memory map, QEMU model.
 * For the host profiler only: to count the hot-path instructions on a real
 * M55 decoder without tying up the board. Unrelated to the firmware. */
MEMORY
{
  ITCM  (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
  SRAM  (rwx) : ORIGIN = 0x01000000, LENGTH = 2M     /* SSRAM1: code and data */
  DTCM  (rw)  : ORIGIN = 0x20000000, LENGTH = 512K
  SRAM2 (rw)  : ORIGIN = 0x21000000, LENGTH = 4M     /* SSRAM2/3: heap and bss */
}
ENTRY(Reset_Handler)
SECTIONS
{
  .vectors : { KEEP(*(.vectors)) } > ITCM
  .text : {
    *(.text*) *(.rodata*) *(.glue_7) *(.glue_7t)
    *(.ARM.extab* .gnu.linkonce.armextab.*)
    . = ALIGN(4);
  } > ITCM
  .ARM.exidx : {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
    . = ALIGN(4);
    __etext = .;
  } > ITCM
  .data : AT (__etext) {
    __data_start__ = .;
    *(.data*)
    . = ALIGN(4);
    __data_end__ = .;
  } > SRAM2
  /* The hot skeleton arrays go into DTCM (on the board they are there too,
   * section .dtcm). On the AN547 DTCM is at 0x20000000, on the STM32N657 in
   * the secure world it is 0x30000000; the section name is the same, the
   * address comes from the linker script. */
  .dtcm (NOLOAD) : {
    __dtcm_start__ = .;
    *(.dtcm) *(.dtcm.*)
    . = ALIGN(8);
    __dtcm_end__ = .;
  } > DTCM
  .bss (NOLOAD) : {
    __bss_start__ = .;
    *(.bss*) *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > SRAM2
  .heap (NOLOAD) : {
    . = ALIGN(8);
    __heap_start__ = .;
    . = . + 1M;
    __heap_end__ = .;
  } > SRAM2
  __stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
}
