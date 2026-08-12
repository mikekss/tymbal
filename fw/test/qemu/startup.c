/* startup.c — a minimal startup for QEMU/AN547: the vector table, copying
 * .data, clearing .bss, enabling FPU/MVE (CPACR), starting SysTick as an
 * instruction counter (with -icount virtual cycles == instructions). */
#include <stdint.h>
extern int  main(void);
extern uint32_t __etext, __data_start__, __data_end__, __bss_start__, __bss_end__;
extern uint32_t __stack_top;

void Reset_Handler(void);
static void Default_Handler(void) { for (;;) {} }

__attribute__((section(".vectors"), used))
void (* const g_vectors[])(void) = {
    (void (*)(void))&__stack_top, Reset_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler,
};

#define SCB_CPACR   (*(volatile uint32_t *)0xE000ED88u)
#define NVIC_CPPWR  (*(volatile uint32_t *)0xE000E00Cu)
#define SYST_CSR    (*(volatile uint32_t *)0xE000E010u)
#define SYST_RVR    (*(volatile uint32_t *)0xE000E014u)
#define SYST_CVR    (*(volatile uint32_t *)0xE000E018u)

void Reset_Handler(void)
{
    SCB_CPACR |= (0xFu << 20);          /* CP10/CP11 full access: FPU + MVE */
    NVIC_CPPWR  = 0u;                   /* power for the MVE coproc registers */
    __asm volatile ("dsb; isb");

    uint32_t *s = &__etext, *d = &__data_start__;
    while (d < &__data_end__) *d++ = *s++;
    for (d = &__bss_start__; d < &__bss_end__; ) *d++ = 0u;

    SYST_RVR = 0x00FFFFFFu;             /* free-running 24-bit count */
    SYST_CVR = 0u;
    SYST_CSR = 5u;                      /* enabled, source — the core clock */
    main();
    for (;;) {}
}

/* A "cycle" counter for SKB_P: SysTick counts DOWN, hence the inversion and
 * the stitching of the 24-bit wraps. With -icount shift=0 one cycle == one
 * instruction, so the number is comparable with the board only by STRUCTURE,
 * not in absolute terms: QEMU models neither the cache nor dual issue. */
uint32_t n6_cyc_now(void)
{
    static uint32_t hi, prev;
    uint32_t cur = 0x00FFFFFFu - (SYST_CVR & 0x00FFFFFFu);
    if (cur < prev) hi += 0x01000000u;
    prev = cur;
    return hi + cur;
}
