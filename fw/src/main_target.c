/* main_target.c — the N6 firmware SCAFFOLD (builds only with -DN6_TARGET).
 * HARD RULE (guide §5.1): the clock tree, MPU/cache, the linker script and the
 * SAI/DMA/LL_ATON initialization must be PORTED from the CubeN6 package
 * example / AI template, not invented. Here — only structure and order
 * (§6, §8).
 * Pitfall no. 1: D-Cache and DMA (§6.1.6) — either the buffers sit in a
 * non-cacheable MPU region OR SCB_CleanDCache_by_Addr before handing over a
 * half. Pick one and write a comment saying why. Printing from an ISR is
 * forbidden (§8.3). */
#ifdef N6_TARGET
#include "n6_config.h"
#include "pipeline.h"
#include "midi.h"

static n6_midi_fifo_t g_mf;
static n6_midi_parser_t g_mp;
static n6_pipe_t g_pipe;

/* DMA double buffering: two half-buffers of one hop (§6.1), stereo 24-in-32 */
#define AUDIO_HALF (N6_HOP48)
static int32_t g_dma_buf[2][AUDIO_HALF * 2];     /* L/R interleaved */
static volatile int g_half_ready = -1;

/* --- ISR (vector names — from the CubeN6 example) ----------------------- */
void SAI_DMA_HalfComplete_ISR(void) { g_half_ready = 0; }
void SAI_DMA_Complete_ISR(void)     { g_half_ready = 1; }
void UART_MIDI_RX_ISR(void)
{
    /* uint8_t b = LL_USART_ReceiveData8(MIDI_USART); */
    /* n6_mf_push(&g_mf, b);  -- no parsing in the ISR (§6.2.2) */
}

static void render_into_half(int half)
{
    static float out48[N6_HOP48];
    uint8_t b; n6_midi_ev_t ev;
    while (n6_mf_pop(&g_mf, &b))                 /* drain the FIFO in the tick */
        if (n6_midi_parse_byte(&g_mp, b, &ev))
            n6_vm_event(&g_pipe.vm, &ev);
    n6_pipe_hop(&g_pipe, out48);
    for (int i = 0; i < N6_HOP48; ++i) {         /* f32 -> 24-in-32, mono->LR */
        int32_t s = (int32_t)(out48[i] * 8388607.0f) << 8;
        g_dma_buf[half][2 * i] = s;
        g_dma_buf[half][2 * i + 1] = s;
    }
    /* CACHE: SCB_CleanDCache_by_Addr(g_dma_buf[half], sizeof g_dma_buf[0]); */
}

int main(void)
{
    /* 1) SystemInit/clock/MPU: PORT FROM THE CubeN6 EXAMPLE (see header)   */
    /* 2) DWT_CYCCNT for measurements (guide §5.1.3)                        */
    /* 3) SAI master TX 48k/24-in-32, MCLK NOT output (PCM5102A SCK->GND),  */
    /*    DMA circular half/complete (§6.1)                                 */
    /* 4) USART MIDI 31250 8N1, IRQ per byte (§6.2)                         */
    /* 5) LL_ATON_RT_RuntimeInit + Init_Network (npu_neuralart.c)           */
    n6_params_t prm = N6_PARAMS_DEFAULT;
    n6_mf_init(&g_mf);
    n6_midi_parser_init(&g_mp);
    n6_pipe_init(&g_pipe, &prm);
    /* 6) DAC sequence: clocks stable -> ~10 ms -> XSMT high (§6.1.2)       */
    for (;;) {
        int h = g_half_ready;
        if (h >= 0) { g_half_ready = -1; render_into_half(h); }
        /* super-loop: telemetry once a second (§8.3), watchdog kick        */
    }
}
#endif /* N6_TARGET */
