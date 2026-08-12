/*
******************************************************************************
**
** @file        : STM32N6XX_AXISRAM2_fsbl.ld
**
** @author      : GPM Application Team
**
** @brief       : Linker script for STM32N6XX Device from STM32N6 series
**                      512 KBytes RAM
**
**                Set heap size, stack size and stack location according
**                to application requirements.
**
**                Set memory bank area and size if external memory is used
**
**  Target      : STMicroelectronics STM32
**
**  Distribution: The file is distributed as is, without any warranty
**                of any kind.
**
******************************************************************************
** @attention
**
** Copyright (c) 2026 STMicroelectronics.
** All rights reserved.
**
** This software is licensed under terms that can be found in the LICENSE file
** in the root directory of this software component.
** If no LICENSE file comes with this software, it is provided AS-IS.
**
******************************************************************************
*/

/* Entry Point */
ENTRY(Reset_Handler)

/* Highest address of the user mode stack */
_estack = ORIGIN(RAM) + LENGTH(RAM); /* end of "RAM" Ram type memory */
_sstack = _estack - _Min_Stack_Size;

_Min_Heap_Size = 0x2000; /* required amount of heap */
_Min_Stack_Size = 0x4000;  /* 16K: STKOF at 0x800 — DSP render + nested ISRs with FP frames */ /* required amount of stack */

/* Memories definition */
MEMORY
{
  ROM    (xrw)    : ORIGIN = 0x34180400,   LENGTH = 255K
  RAM    (xrw)    : ORIGIN = 0x341C0000,   LENGTH = 256K
  RAM2   (xrw)    : ORIGIN = 0x340A4000,   LENGTH = 448K   /* g_pipe ~198K + CK4 buffer 192K (.ram2); end 0x34114000 < ROM 0x34180400 */
  /* DTCM Cortex-M55: 128 kB right next to the core, zero wait states, past
     the cache and past AXI. The FSBL lives in the SECURE world (AXISRAM at
     0x34...), so DTCM is at 0x30000000 here, not at the non-secure
     0x20000000.
     WARNING: the TCM has ECC; after reset the ECC bits do not match the
     random contents, and THE VERY FIRST read gives a hard fault (PECC,
     AFSR bit 17). The area must be written once in words before any
     access — n6_dtcm_init() does that at the start of main. */
  DTCM   (xrw)    : ORIGIN = 0x30000000,   LENGTH = 128K
  /* npuRAM6 — the bank the NPU reads over its own short path. The graph does
     not occupy it at all (used_size_bytes = 0 in network_c_info.json), while
     we need 45.5 kB for the user-allocated IO. Putting them into cpuRAM2
     (.ram2) was a mistake: the NPU reads these buffers every epoch, and the
     stage got 343 k cycles more expensive with the processor's work
     unchanged. npu_boot brings up the bank power and RISAF6. The address
     comes from fw/stm32n6_nucleo_app_safe.mpool. */
  NPURAM6 (xrw)   : ORIGIN = 0x34350000,   LENGTH = 448K
}

/* Sections */
SECTIONS
{
  /* The startup code into "ROM" Rom type memory */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >ROM

  /* The program code and other data into "ROM" Rom type memory */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)
    *(.RamFunc)        /* .RamFunc sections */
    *(.RamFunc*)       /* .RamFunc* sections */

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >ROM

  /* Constant data into "ROM" Rom type memory */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >ROM

  .ARM.extab   (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
   {
    . = ALIGN(4);
    *(.ARM.extab* .gnu.linkonce.armextab.*)
    . = ALIGN(4);
  } >ROM

  .ARM (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
   {
    . = ALIGN(4);
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
    . = ALIGN(4);
  } >ROM

  .preinit_array     (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
  } >ROM

  .init_array (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
  } >ROM

  .fini_array (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
    . = ALIGN(4);
  } >ROM

  /* Used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* Initialized data sections into "RAM" Ram type memory */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */

  } >RAM AT> ROM

  .noncacheable :
  {
    . = ALIGN(8);
    __snoncacheable = .;/* create symbol for start of section */
    KEEP(*(.noncacheable))
    . = ALIGN(8);
    __enoncacheable = .;  /* create symbol for end of section */
  } > RAM

  /* N6 M1: the large DSP state (g_pipe) outside the main RAM; NOLOAD —
     it is cleared in n6_pipe_init, startup does not touch it */
  .ram2 (NOLOAD) :
  {
    . = ALIGN(8);
    *(.ram2)
    *(.ram2*)
  } >RAM2


  /* N6: the hot skeleton arrays (g_br8 30.0 kB, the sine table, w_re/w_im,
     p_s/q_s/dA) — ~40 kB out of 128. NOLOAD: the contents are built at
     runtime, clearing is done by n6_dtcm_init() (ECC, see above), startup
     does not go here. */
  /* Graph IO buffers: see the NPURAM6 region above. */
  .npuram (NOLOAD) :
  {
    . = ALIGN(32);
    *(.npuram)
    *(.npuram*)
  } >NPURAM6

  .dtcm (NOLOAD) :
  {
    . = ALIGN(8);
    __dtcm_start__ = .;
    *(.dtcm)
    *(.dtcm*)
    . = ALIGN(8);
    __dtcm_end__ = .;
  } >DTCM

  .gnu.sgstubs :
  {
    . = ALIGN(4);
    *(.gnu.sgstubs*)   /* Secure Gateway stubs */
    . = ALIGN(4);
  } >ROM
  /* Uninitialized data section into "RAM" Ram type memory */
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss section */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >RAM

  /* User_heap_stack section, used to check that there is enough "RAM" Ram  type memory left */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >RAM

  /* Remove information from the compiler libraries */
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
