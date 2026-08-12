/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdio.h>
#include <string.h>
#include "midi.h"
#include "pipeline.h"
#ifdef N6_CK4
#include "ck4.h"                       /* checkpoint 4: MVE validation (docs/ck4_notes.md) */
#endif
#include "stm32n6xx_nucleo_bus.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define GPIO_SET(led) HAL_GPIO_WritePin(led##_GPIO_Port, led##_Pin, GPIO_PIN_SET);
#define GPIO_RESET(led) HAL_GPIO_WritePin(led##_GPIO_Port, led##_Pin, GPIO_PIN_RESET);
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

SAI_HandleTypeDef hsai_BlockB1;
SAI_HandleTypeDef hsai_BlockB2;
DMA_NodeTypeDef Node_GPDMA1_Channel1 __NON_CACHEABLE;
DMA_QListTypeDef List_GPDMA1_Channel1;
DMA_HandleTypeDef handle_GPDMA1_Channel1;
DMA_NodeTypeDef Node_GPDMA1_Channel0 __NON_CACHEABLE;
DMA_QListTypeDef List_GPDMA1_Channel0;
DMA_HandleTypeDef handle_GPDMA1_Channel0;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_GPDMA1_Init(void);
static void MX_SAI1_Init(void);
static void MX_SAI2_Init(void);
static void SystemIsolation_Config(void);
/* USER CODE BEGIN PFP */

void MPU_Config(void);
int memcmp32 (const uint32_t *p1, const uint32_t *p2, uint32_t sz);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

#define APP_SAI_MODE APP_SAI_MODE_I2S

/* SAI Mode: 8 channel free protocol TDM */
#define APP_SAI_MODE_TDM 1

/* SAI Mode: stereo standard I2S */
#define APP_SAI_MODE_I2S 2

#if APP_SAI_MODE == APP_SAI_MODE_TDM
#define APP_SAI_SLOTS_AMOUNT 8
#else /* APP_SAI_MODE_I2S */
#define APP_SAI_SLOTS_AMOUNT 2
#endif /* APP_SAI_MODE_I2S */

#define APP_SAI_SAMPLING_FREQ 48000
/* half-buffer == pipeline hop: 4 ms = N6_HOP48 stereo frames 24-in-32 (§6.1) */
#define APP_SAI_HALF_WORDS (N6_HOP48 * 2)
#define APP_SAI_1MS_BUFFER_SIZE APP_SAI_HALF_WORDS   /* name from the example == half-buffer */
#define APP_SAI_DOUBLE_BUFFER_SIZE (APP_SAI_HALF_WORDS * 2u)
uint32_t sai_tx_double_buffer[APP_SAI_DOUBLE_BUFFER_SIZE] __NON_CACHEABLE;
uint32_t sai_rx_double_buffer[APP_SAI_DOUBLE_BUFFER_SIZE] __NON_CACHEABLE;

/* --- N6 M1: MIDI + DSP pipeline (skeleton plays from the keys) --------- */
UART_HandleTypeDef huart1;   /* VCP printf / telemetry */
UART_HandleTypeDef huart3;   /* MIDI 31250, PD9 = D0 = USART3_RX */
static n6_midi_fifo_t g_mf;
static n6_midi_parser_t g_mp;
/* g_pipe ~198KB (1 s wow ring) — a separate RAM2 region (see .ld);
 * NOLOAD: zeroed in n6_pipe_init (memset), startup does not touch it */
static n6_pipe_t g_pipe __attribute__((section(".ram2")));
static volatile int g_half_ready = -1;
static volatile uint32_t g_underrun, g_rx_bytes, g_fifo_drop, g_uart_err;
static uint32_t g_hops, g_cyc_min = 0xFFFFFFFFu, g_cyc_max, g_out_clip;
static uint32_t g_out_nan;               /* review item 5: NaN/Inf at the output */
static uint64_t g_cyc_sum;
static float g_peak;
static char g_hb[512];   /* heartbeat line: skb counters, hop tail, epochs and Cyrillic */
extern uint32_t n6_prof[8];          /* [7] = FIR + voice mix (4 Aug, evening) */
extern uint32_t n6_skb_prof[5];  /* wtab / prologue / body / decoder / phase */              /* skb/npu/pqmf/wf (pipeline.c) */
/* peaks |skeleton| / |net residual| / |FIR| before the mix, in thousandths
 * (pipeline.c) — diagnosing peak=980, agenda item (b): what pins the output
 * against the limiter */
extern uint32_t n6_dbg_sk_pk, n6_dbg_rs_pk, n6_dbg_fir_pk;
/* NPU driver (npu_boot.c / npu_neuralart.c) */
void n6_npu_boot(void);
extern uint32_t n6_npu_prof_err, n6_npu_prof_cycles, n6_npu_prof_epochs;
/* per-epoch profile (npu_neuralart.c, only with LL_ATON_EB_DBG_INFO) */
extern uint32_t n6_npu_cpu_hyb, n6_npu_cpu_ec, n6_npu_ep_hops;
extern uint32_t n6_npu_swap_copy, n6_npu_swap_cache;
/* 1 = the NPU buffers are ours and the state input/output are concatenated
 * (see npu_neuralart.c). Then the state ring swap is not done at all, and the
 * copy/cache counters above stay zero — no point printing them, we print the
 * flag itself. */
extern int n6_npu_user_io;
void n6_npu_ep_arm(void);
int  n6_npu_ep_dump_line(char *dst, int cap);
/* DTCM: bounds of the .dtcm section from the linker script */
extern uint32_t __dtcm_start__, __dtcm_end__;

/* ECC ON THE TCM (4 Aug). After reset the contents of DTCM are random while
   the ECC bits do not match them, so the VERY FIRST READ gives a precise hard
   fault (PECC, AFSR bit 17). The cure is a single write over the whole area
   BEFORE any access. We write in exactly 32-bit words: a byte-wise write into
   ECC memory is a read-modify-write, that is, the same read again. The stack
   and the heap live in AXISRAM (_estack = end of RAM), they are not in DTCM —
   wiping the whole area is safe. */
#define N6_DTCM_BASE  0x30000000u        /* secure alias */
#define N6_DTCM_SIZE  (128u * 1024u)
static void n6_dtcm_init(void)
{
  volatile uint32_t *p = (volatile uint32_t *)N6_DTCM_BASE;
  for (uint32_t i = 0; i < N6_DTCM_SIZE / 4u; ++i) p[i] = 0u;
}
int n6_weights_load(void);               /* fw/src/n6_weights.h */
extern uint32_t n6_weights_bytes, n6_weights_crc;
extern uint32_t n6_weights_hal, n6_weights_seen, n6_weights_p[3];
extern int32_t n6_weights_step;
uint32_t n6_cyc_now(void) { return DWT->CYCCNT; }
static void MX_USART1_UART_Init(void);
static void MX_USART3_UART_Init(void);
static void render_into_half(int half);
/* D-24: live A/B of the net on button B1 (bodies next to render_into_half) */
static volatile uint8_t g_net_on = 1u, g_net_msg = 0u;
static void net_ab_init(void);
static void net_ab_poll(void);

/* VDDCORE 0.9 V through the regulator on I2C2 (address 0x49, reg 0x01,
 * LSB=5mV) — BEFORE PLL1 is pushed to 800 MHz; distilled from
 * N6_shmoo/n6_ai_init.c */
static void upscale_vddcore_level(void)
{
  uint8_t tmp = 0x64;                       /* 0x64 = 900 mV */
  BSP_I2C2_Init();
  BSP_I2C2_WriteReg(0x49 << 1, 0x01, &tmp, 1);
  HAL_Delay(1);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  n6_dtcm_init();       /* BEFORE all else: the first DTCM read faults hard */
  MPU_Config();

  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* VDDCORE 0.9 V strictly BEFORE the push to 800 MHz */
  upscale_vddcore_level();

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_GPDMA1_Init();
  MX_SAI1_Init();
  MX_SAI2_Init();
  SystemIsolation_Config();
  /* USER CODE BEGIN 2 */

  /* DWT for measuring the render cycle (guide §5.1.3) */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  n6_params_t prm = N6_PARAMS_DEFAULT;
#ifdef N6_CK4
  /* CK4 = worst-case DSP, not the runtime canon: the ck4.c score is written
   * for a TRIO of low keys, and host_ck4.c builds the reference with exactly
   * 3 voices. Since 2 Aug the runtime canon is a duet (D-2), so for the
   * cross-check we set 3 explicitly: otherwise the board plays 2 voices
   * against 3 in the reference and CK4 fails with rel RMS ~1.0 (caught
   * 2 Aug). */
  prm.n_voices = 3;
#endif
#ifndef N6_CK4
  /* The "analogueness" preset (D-19): unison/cents, bloom, drift/cents.
   * To switch it off for A/B, comment out the three lines. CK4 always runs
   * with zeros (the reference), so the preset lives outside N6_CK4. */
  prm.uni_cents   = 7.0f;
  prm.bloom_k     = 0.35f;
  prm.drift_cents = 3.0f;
  /* D-22, settled 8 Aug: master level BEFORE the limiter. Two reasons, one
   * knob. (1) The limiter: raw peak 1.77 on a vel=127 chord against a
   * threshold of 0.98 — at 1.0 it worked almost all the time and killed
   * the velocity dynamics above 90. (2) Sensitivity: AKG K512 32 ohm /
   * 109 dB SPL/V — there are ~30 dB too many in the chain, and the whole
   * travel of the linear B50K fitted into the first tenth of a turn,
   * where its channel imbalance is worst.
   * 0.079 = −22 dB (8 Aug, by ear: −24 dB is too quiet for my
   * chain — the synth is not plugged into headphones yet). Worst peak
   * 0.140, the limiter stays silent. The price is 2.1 bits out of 24; the
   * noise floor of the chain is set by the skeleton's white noise (−19 dB)
   * anyway, not by the word length. */
  prm.out_gain    = 0.079f;
  /* D-23 (8 Aug): tilt of the skeleton noise across subbands, dB per band.
   * White noise at one level in all four bands gave almost as much up top
   * as the tones did (-2.4 dB above 4 kHz) — that was the "grit".
   * Checked on the host: 6 dB/band removes ~7 dB of skeleton HF noise and
   * at the same time IMPROVES the metric (line +23.79 -> +24.36 dB); 12 dB
   * gives nothing more. Zero = previous behaviour, references do not move. */
  prm.noise_tilt_db = 6.0f;
#endif
  /* UART FIRST. With the production NPU every failure below has to be
   * VISIBLE: n6_pipe_init used to stand before the terminal was initialised,
   * and LL_ATON falling into a hard fault looked like total silence (caught
   * 3 Aug). */
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();

  printf("\r\n=== N6_m1: skeleton from the keys (MIDI->voice->skb->PQMF->wow->SAI) ===\r\n");
  printf("hop=%d (4 ms), voices=%d, SAI1B master 48k/24-in-32, no DAC yet\r\n",
         N6_HOP48, prm.n_voices);
  printf("CPU 800 MHz (VDDCORE 0.9 V), hop budget = 3.2 Mcyc\r\n");
  printf("DTCM: cleared %lu kB at 0x%08lX; skeleton took %lu B\r\n",
         (unsigned long)(N6_DTCM_SIZE / 1024u), (unsigned long)N6_DTCM_BASE,
         (unsigned long)((uintptr_t)&__dtcm_end__ - (uintptr_t)&__dtcm_start__));
  printf("clocks: CPU %lu MHz, NPU %lu MHz, NPU-RAM %lu MHz\r\n",
         (unsigned long)(HAL_RCC_GetCpuClockFreq() / 1000000u),
         (unsigned long)(HAL_RCC_GetNPUClockFreq() / 1000000u),
         (unsigned long)(HAL_RCC_GetNPURAMSClockFreq() / 1000000u));

  /* Breadcrumbs: if the board goes silent, the last printed line points at
   * the exact place of the failure. It is not worth more than one printf. */
  printf("NPU: clock, reset, RIF/RISAF...\r\n");
  n6_npu_boot();
  printf("NPU: bus open\r\n");

  /* The weights live in NOR and are copied into AXISRAM5: they do not fit
   * into the FSBL image (ROM 255 kB, 207 taken). Strictly after n6_npu_boot —
   * the banks must be powered — and strictly before n6_pipe_init, which does
   * the first inference. */
  {
    int wrc = n6_weights_load();
    /* The path diagnostics are printed ALWAYS: from the failure code alone
     * you cannot tell whether the chip is silent at all or simply did not
     * understand a single-line command. */
    printf("NOR: BSP_Init=%ld, HAL=%04lX, magic=%08lX (expect 3157364E)\r\n",
           (long)n6_weights_step, (unsigned long)n6_weights_hal,
           (unsigned long)n6_weights_seen);
    printf("NOR: address probe 0/400000/400004 = %08lX %08lX %08lX"
           " (expect 324D5453 3157364E 00044520)\r\n",
           (unsigned long)n6_weights_p[0], (unsigned long)n6_weights_p[1],
           (unsigned long)n6_weights_p[2]);
    if (wrc == 0)
      printf("weights: %lu B, CRC %08lX — OK\r\n",
             (unsigned long)n6_weights_bytes, (unsigned long)n6_weights_crc);
    else
      printf("weights: NOT LOADED, code %d (see n6_weights.h). The net computes garbage.\r\n", wrc);
  }

  n6_mf_init(&g_mf);
  n6_midi_parser_init(&g_mp);
  printf("NPU: initialising the LL_ATON graph...\r\n");
  n6_pipe_init(&g_pipe, &prm);

  printf("NPU: prof_err=%lu (0 = the graph shape contract matched)\r\n",
         (unsigned long)n6_npu_prof_err);
  printf("FIR: %s\r\n", g_pipe.fir_c
         ? "production coefficients — stage is in the chain"
         : "zero stub — stage is OFF (waiting for n6_fir_coeffs.h)");
  /* print as INTEGERS: nano-printf without -u _printf_float silently skips %f
   * (caught 5 Aug — the banner came out empty; see chip_findings) */
  printf("Skeleton: unison %d c, bloom 0.%02d, drift %d c (D-19), level 0.%03d (D-22)\r\n",
         (int)prm.uni_cents, (int)(prm.bloom_k * 100.0f + 0.5f),
         (int)prm.drift_cents, (int)(prm.out_gain * 1000.0f + 0.5f));

  memset(sai_tx_double_buffer, 0, sizeof sai_tx_double_buffer);
  (void)sai_rx_double_buffer;   /* the loopback RX path is not started in M1 */

  if (HAL_SAI_Transmit_DMA(&hsai_BlockB1, (uint8_t *)sai_tx_double_buffer,
                           APP_SAI_DOUBLE_BUFFER_SIZE) != HAL_OK)
  {
    Error_Handler();
  }

  net_ab_init();                    /* D-24: button B1 = A/B of the net */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint32_t t_last = 0;
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    int h = g_half_ready;
    if (h >= 0) { g_half_ready = -1; render_into_half(h); }

#ifdef N6_CK4
    /* dump: one line per iteration, only while UART-IT is free (do not
       collide with the heartbeat); printed from the super-loop, not the ISR */
    if (huart1.gState == HAL_UART_STATE_READY) {
      static char ck4_l[160];
      int ck4_n = n6_ck4_dump_line(ck4_l, (int)sizeof ck4_l - 2);
      if (ck4_n > 0) {
        ck4_l[ck4_n] = '\r'; ck4_l[ck4_n + 1] = '\n';
        HAL_UART_Transmit_IT(&huart1, (uint8_t *)ck4_l, (uint16_t)(ck4_n + 2));
      }
    }
#endif

    /* The per-epoch table is printed ONCE, when enough hops have piled up:
       70 lines of 70 characters is ~0.4 s of UART, pouring that out in a
       loop is not on. After that only two aggregates go into the heartbeat,
       and the accumulators are zeroed every tick so the numbers cover the
       last second. */
    static int ep_dumped = 0;
    if (!ep_dumped && n6_npu_ep_hops >= 500u
        && huart1.gState == HAL_UART_STATE_READY) {
      static char ep_l[160];
      int ep_n = n6_npu_ep_dump_line(ep_l, (int)sizeof ep_l - 2);
      if (ep_n > 0) {
        ep_l[ep_n] = '\r'; ep_l[ep_n + 1] = '\n';
        HAL_UART_Transmit_IT(&huart1, (uint8_t *)ep_l, (uint16_t)(ep_n + 2));
      } else {
        ep_dumped = 1;
        n6_npu_ep_arm();
      }
    }

    uint32_t now = HAL_GetTick();
    if (now - t_last >= 2000u) {
      t_last = now;
      BSP_LED_Toggle(LED_BLUE);
      uint32_t avg = g_hops ? (uint32_t)(g_cyc_sum / g_hops) : 0u;
      /* hop budget @600 MHz = 2.4 Mcyc; printing is non-blocking (IT) */
      uint32_t hp = g_hops ? g_hops : 1u;
    if (g_net_msg && huart1.gState == HAL_UART_STATE_READY) {
      g_net_msg = 0u;
      printf("NET: %s\r\n", g_net_on ? "ON — skeleton+FIR+residual"
                                       : "OFF — only skeleton+FIR");
    }
      snprintf(g_hb, sizeof g_hb,
        "[hb] hops=%lu cyc(min/avg/max)=%lu/%lu/%lu underrun=%lu "
        "midi=%lu err=%lu drop=%lu peak=%lu/1000 clip=%lu nan=%lu "
        "prof(skb/npu/pq/wf/fir)=%lu/%lu/%lu/%lu/%lu skb(wt/pro/bod/dec/ph)=%lu/%lu/%lu/%lu/%lu "
        "tail(xc/notes/lim)=%lu/%lu/%lu peaks(sk/rs/fir)=%lu/%lu/%lu userIO=%d "
        "| npu: inference=%lu cyc (%lu us) blocks=%lu m55(hyb/EC)=%lu/%lu\r\n",
        (unsigned long)g_hops, (unsigned long)g_cyc_min, (unsigned long)avg,
        (unsigned long)g_cyc_max, (unsigned long)g_underrun,
        (unsigned long)g_rx_bytes, (unsigned long)g_uart_err,
        (unsigned long)g_fifo_drop, (unsigned long)(g_peak * 1000.0f),
        (unsigned long)g_out_clip, (unsigned long)g_out_nan,
        (unsigned long)(n6_prof[0] / hp), (unsigned long)(n6_prof[1] / hp),
        (unsigned long)(n6_prof[2] / hp), (unsigned long)(n6_prof[3] / hp),
        (unsigned long)(n6_prof[7] / hp),
        (unsigned long)(n6_skb_prof[0] / hp), (unsigned long)(n6_skb_prof[1] / hp),
        (unsigned long)(n6_skb_prof[2] / hp), (unsigned long)(n6_skb_prof[3] / hp),
        (unsigned long)(n6_skb_prof[4] / hp),
        (unsigned long)(n6_prof[4] / hp), (unsigned long)(n6_prof[5] / hp),
        (unsigned long)(n6_prof[6] / hp),
        (unsigned long)n6_dbg_sk_pk, (unsigned long)n6_dbg_rs_pk,
        (unsigned long)n6_dbg_fir_pk,
        n6_npu_user_io,
        (unsigned long)n6_npu_prof_cycles,
        (unsigned long)(n6_npu_prof_cycles / 800u),
        (unsigned long)n6_npu_prof_epochs,
        (unsigned long)(n6_npu_cpu_hyb / (n6_npu_ep_hops ? n6_npu_ep_hops : 1u)),
        (unsigned long)(n6_npu_cpu_ec  / (n6_npu_ep_hops ? n6_npu_ep_hops : 1u)));
      HAL_UART_Transmit_IT(&huart1, (uint8_t *)g_hb, (uint16_t)strlen(g_hb));
      g_cyc_min = 0xFFFFFFFFu; g_cyc_max = 0u; g_cyc_sum = 0u; g_hops = 0u;
      n6_prof[0] = n6_prof[1] = n6_prof[2] = n6_prof[3] = 0u;
      n6_prof[4] = n6_prof[5] = n6_prof[6] = n6_prof[7] = 0u;
      n6_dbg_sk_pk = n6_dbg_rs_pk = n6_dbg_fir_pk = 0u;
      n6_npu_swap_copy = n6_npu_swap_cache = 0u;
      n6_skb_prof[0] = n6_skb_prof[1] = n6_skb_prof[2] = 0u;
      n6_skb_prof[3] = n6_skb_prof[4] = 0u;
      if (ep_dumped) n6_npu_ep_arm();      /* aggregates cover the last second */
      g_peak = 0.0f;
    }
  }
  /* USER CODE END 3 */
}
/* USER CODE BEGIN CLK 1 */
/* USER CODE END CLK 1 */

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the System Power Supply
  */
  if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /* Enable HSI */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /* Wait HSE stabilization time before its selection as PLL source. */
  HAL_Delay(HSE_STARTUP_TIMEOUT);

  /** Get current CPU/System buses clocks configuration and if necessary switch
 to intermediate HSI clock to ensure target clock can be set
  */
  HAL_RCC_GetClockConfig(&RCC_ClkInitStruct);
  if ((RCC_ClkInitStruct.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
     (RCC_ClkInitStruct.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11))
  {
    RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK);
    RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
    {
      /* Initialization Error */
      Error_Handler();
    }
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL1.PLLM = 3;      /* 48/3*100 = 1600 MHz (was 1200) */
  RCC_OscInitStruct.PLL1.PLLN = 100;
  RCC_OscInitStruct.PLL1.PLLFractional = 0;
  RCC_OscInitStruct.PLL1.PLLP1 = 1;
  RCC_OscInitStruct.PLL1.PLLP2 = 1;
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL4.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL4.PLLM = 1;
  RCC_OscInitStruct.PLL4.PLLN = 24;
  RCC_OscInitStruct.PLL4.PLLFractional = 9663677;
  RCC_OscInitStruct.PLL4.PLLP1 = 1;
  RCC_OscInitStruct.PLL4.PLLP2 = 1;

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_HCLK
                              |RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1
                              |RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_PCLK5
                              |RCC_CLOCKTYPE_PCLK4;
  RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;
  RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC1Selection.ClockDivider = 2;   /* CPU 1600/2 = 800 MHz */
  RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC2Selection.ClockDivider = 4;   /* SYSCLK 400 MHz */
  /* IC6 = NPU clock, IC11 = AXISRAM3..6 clock (NPU memory). Until 3 Aug the
   * dividers were 4 with a note "not used" — written when the NPU was a
   * stub. With the production driver that gave the NPU 400 MHz instead of
   * 1000 and the memory 400 instead of 900; t_call came out at 4849 us
   * against the expected 2866. PLL1 = 1600 MHz, divider 2 -> 800 MHz. That is
   * the ceiling at VDDCORE 0.9 V; ST's 1 GHz comes off a separate PLL2 and
   * needs a check on the supply. */
  RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC6Selection.ClockDivider = 2;   /* NPU 1600/2 = 800 MHz */
  RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC11Selection.ClockDivider = 2;  /* NPU-RAM 800 MHz */

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPDMA1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPDMA1_Init(void)
{

  /* USER CODE BEGIN GPDMA1_Init 0 */

  /* USER CODE END GPDMA1_Init 0 */

  /* Peripheral clock enable */
  __HAL_RCC_GPDMA1_CLK_ENABLE();

  /* GPDMA1 interrupt Init */
    HAL_NVIC_SetPriority(GPDMA1_Channel0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel0_IRQn);
    HAL_NVIC_SetPriority(GPDMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel1_IRQn);

  /* USER CODE BEGIN GPDMA1_Init 1 */

  /* USER CODE END GPDMA1_Init 1 */
  /* USER CODE BEGIN GPDMA1_Init 2 */

  /* USER CODE END GPDMA1_Init 2 */

}

/**
  * @brief RIF Initialization Function
  * @param None
  * @retval None
  */
  static void SystemIsolation_Config(void)
{

  /* USER CODE BEGIN RIF_Init 0 */

  /* USER CODE END RIF_Init 0 */

  /* set all required IPs as secure privileged */
  __HAL_RCC_RIFSC_CLK_ENABLE();

  /* RIF-Aware IPs Config */

  /* set up GPDMA configuration */
  /* set GPDMA1 channel 0 used by SAI2 */
  if (HAL_DMA_ConfigChannelAttributes(&handle_GPDMA1_Channel0,DMA_CHANNEL_SEC|DMA_CHANNEL_PRIV|DMA_CHANNEL_SRC_SEC|DMA_CHANNEL_DEST_SEC)!= HAL_OK )
  {
    Error_Handler();
  }
  /* set GPDMA1 channel 1 used by SAI1 */
  if (HAL_DMA_ConfigChannelAttributes(&handle_GPDMA1_Channel1,DMA_CHANNEL_SEC|DMA_CHANNEL_PRIV|DMA_CHANNEL_SRC_SEC|DMA_CHANNEL_DEST_SEC)!= HAL_OK )
  {
    Error_Handler();
  }

  /* set up GPIO configuration */
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);

  /* USER CODE BEGIN RIF_Init 1 */

  /* USER CODE END RIF_Init 1 */
  /* USER CODE BEGIN RIF_Init 2 */

  /* USER CODE END RIF_Init 2 */

}

/**
  * @brief SAI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SAI1_Init(void)
{

  /* USER CODE BEGIN SAI1_Init 0 */

  /* USER CODE END SAI1_Init 0 */

  /* USER CODE BEGIN SAI1_Init 1 */

#if APP_SAI_MODE == APP_SAI_MODE_TDM

  hsai_BlockB1.Instance = SAI1_Block_B;
  hsai_BlockB1.Init.Protocol = SAI_FREE_PROTOCOL;
  hsai_BlockB1.Init.AudioMode = SAI_MODEMASTER_TX;
  hsai_BlockB1.Init.DataSize = SAI_DATASIZE_32;
  hsai_BlockB1.Init.FirstBit = SAI_FIRSTBIT_MSB;
  hsai_BlockB1.Init.ClockStrobing = SAI_CLOCKSTROBING_FALLINGEDGE;
  hsai_BlockB1.Init.Synchro = SAI_ASYNCHRONOUS;
  hsai_BlockB1.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
  hsai_BlockB1.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai_BlockB1.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
  hsai_BlockB1.Init.AudioFrequency = SAI_AUDIO_FREQUENCY_48K;
  hsai_BlockB1.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai_BlockB1.Init.MckOutput = SAI_MCK_OUTPUT_DISABLE;
  hsai_BlockB1.Init.MonoStereoMode = SAI_STEREOMODE;
  hsai_BlockB1.Init.CompandingMode = SAI_NOCOMPANDING;
  hsai_BlockB1.Init.TriState = SAI_OUTPUT_NOTRELEASED;
  hsai_BlockB1.Init.PdmInit.Activation = DISABLE;
  hsai_BlockB1.Init.PdmInit.MicPairsNbr = 1;
  hsai_BlockB1.Init.PdmInit.ClockEnable = SAI_PDM_CLOCK1_ENABLE;
  hsai_BlockB1.FrameInit.FrameLength = 256;
  hsai_BlockB1.FrameInit.ActiveFrameLength = 1;
  hsai_BlockB1.FrameInit.FSDefinition = SAI_FS_STARTFRAME;
  hsai_BlockB1.FrameInit.FSPolarity = SAI_FS_ACTIVE_LOW;
  hsai_BlockB1.FrameInit.FSOffset = SAI_FS_FIRSTBIT;
  hsai_BlockB1.SlotInit.FirstBitOffset = 0;
  hsai_BlockB1.SlotInit.SlotSize = SAI_SLOTSIZE_DATASIZE;
  hsai_BlockB1.SlotInit.SlotNumber = 8;
  hsai_BlockB1.SlotInit.SlotActive = 0x0000FFFF;
  if (HAL_SAI_Init(&hsai_BlockB1) != HAL_OK)
  {
    Error_Handler();
  }

#else /* APP_SAI_MODE_I2S */

  /* USER CODE END SAI1_Init 1 */
  hsai_BlockB1.Instance = SAI1_Block_B;
  hsai_BlockB1.Init.AudioMode = SAI_MODEMASTER_TX;
  hsai_BlockB1.Init.Synchro = SAI_ASYNCHRONOUS;
  hsai_BlockB1.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
  hsai_BlockB1.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai_BlockB1.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
  hsai_BlockB1.Init.AudioFrequency = SAI_AUDIO_FREQUENCY_48K;
  hsai_BlockB1.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai_BlockB1.Init.MckOutput = SAI_MCK_OUTPUT_DISABLE;
  hsai_BlockB1.Init.MonoStereoMode = SAI_STEREOMODE;
  hsai_BlockB1.Init.CompandingMode = SAI_NOCOMPANDING;
  hsai_BlockB1.Init.TriState = SAI_OUTPUT_NOTRELEASED;
  if (HAL_SAI_InitProtocol(&hsai_BlockB1, SAI_I2S_STANDARD, SAI_PROTOCOL_DATASIZE_32BIT, 2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SAI1_Init 2 */

#endif  /* APP_SAI_MODE_I2S */

  /* USER CODE END SAI1_Init 2 */

}

/**
  * @brief SAI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SAI2_Init(void)
{

  /* USER CODE BEGIN SAI2_Init 0 */

  /* USER CODE END SAI2_Init 0 */

  /* USER CODE BEGIN SAI2_Init 1 */

#if APP_SAI_MODE == APP_SAI_MODE_TDM

  hsai_BlockB2.Instance = SAI2_Block_B;
  hsai_BlockB2.Init.Protocol = SAI_FREE_PROTOCOL;
  hsai_BlockB2.Init.AudioMode = SAI_MODESLAVE_RX;
  hsai_BlockB2.Init.DataSize = SAI_DATASIZE_32;
  hsai_BlockB2.Init.FirstBit = SAI_FIRSTBIT_MSB;
  hsai_BlockB2.Init.ClockStrobing = SAI_CLOCKSTROBING_FALLINGEDGE;
  hsai_BlockB2.Init.Synchro = SAI_ASYNCHRONOUS;
  hsai_BlockB2.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
  hsai_BlockB2.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai_BlockB2.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
  hsai_BlockB2.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai_BlockB2.Init.MckOutput = SAI_MCK_OUTPUT_ENABLE;
  hsai_BlockB2.Init.MonoStereoMode = SAI_STEREOMODE;
  hsai_BlockB2.Init.CompandingMode = SAI_NOCOMPANDING;
  hsai_BlockB2.Init.TriState = SAI_OUTPUT_NOTRELEASED;
  hsai_BlockB2.Init.PdmInit.Activation = DISABLE;
  hsai_BlockB2.Init.PdmInit.MicPairsNbr = 1;
  hsai_BlockB2.Init.PdmInit.ClockEnable = SAI_PDM_CLOCK1_ENABLE;
  hsai_BlockB2.FrameInit.FrameLength = 256;
  hsai_BlockB2.FrameInit.ActiveFrameLength = 1;
  hsai_BlockB2.FrameInit.FSDefinition = SAI_FS_STARTFRAME;
  hsai_BlockB2.FrameInit.FSPolarity = SAI_FS_ACTIVE_LOW;
  hsai_BlockB2.FrameInit.FSOffset = SAI_FS_FIRSTBIT;
  hsai_BlockB2.SlotInit.FirstBitOffset = 0;
  hsai_BlockB2.SlotInit.SlotSize = SAI_SLOTSIZE_DATASIZE;
  hsai_BlockB2.SlotInit.SlotNumber = 8;
  hsai_BlockB2.SlotInit.SlotActive = 0x0000FFFF;
  if (HAL_SAI_Init(&hsai_BlockB2) != HAL_OK)
  {
    Error_Handler();
  }

#else /* APP_SAI_MODE_I2S */

  /* USER CODE END SAI2_Init 1 */
  hsai_BlockB2.Instance = SAI2_Block_B;
  hsai_BlockB2.Init.AudioMode = SAI_MODESLAVE_RX;
  hsai_BlockB2.Init.Synchro = SAI_ASYNCHRONOUS;
  hsai_BlockB2.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
  hsai_BlockB2.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai_BlockB2.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
  hsai_BlockB2.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai_BlockB2.Init.MckOutput = SAI_MCK_OUTPUT_ENABLE;
  hsai_BlockB2.Init.MonoStereoMode = SAI_STEREOMODE;
  hsai_BlockB2.Init.CompandingMode = SAI_NOCOMPANDING;
  hsai_BlockB2.Init.TriState = SAI_OUTPUT_NOTRELEASED;
  if (HAL_SAI_InitProtocol(&hsai_BlockB2, SAI_I2S_STANDARD, SAI_PROTOCOL_DATASIZE_32BIT, 2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SAI2_Init 2 */

#endif  /* APP_SAI_MODE_I2S */

  /* USER CODE END SAI2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, D1_Pin|D2_Pin|D3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, LED3_Pin|LED2_Pin|LED1_Pin, GPIO_PIN_SET);

  /*Configure GPIO pins : D1_Pin D2_Pin D3_Pin */
  GPIO_InitStruct.Pin = D1_Pin|D2_Pin|D3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : LED3_Pin LED2_Pin LED1_Pin */
  GPIO_InitStruct.Pin = LED3_Pin|LED2_Pin|LED1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void USART3_IRQHandler(void)
{
  uint32_t isr = USART3->ISR;
  if (isr & USART_ISR_RXNE_RXFNE) {
    uint8_t b = (uint8_t)USART3->RDR;
    g_rx_bytes++;
    if (!n6_mf_push(&g_mf, b)) g_fifo_drop++;   /* parsing NOT in ISR (§6.2.2) */
  }
  if (isr & (USART_ISR_ORE | USART_ISR_FE | USART_ISR_NE)) {
    USART3->ICR = USART_ICR_ORECF | USART_ICR_FECF | USART_ICR_NECF;
    g_uart_err++;
  }
}

void USART1_IRQHandler(void) { HAL_UART_IRQHandler(&huart1); }

void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
  GPIO_InitTypeDef g = {0};
  if (huart->Instance == USART1) {
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    g.Pin = GPIO_PIN_5 | GPIO_PIN_6;
    g.Mode = GPIO_MODE_AF_PP; g.Pull = GPIO_PULLUP;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH; g.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOE, &g);
    HAL_NVIC_SetPriority(USART1_IRQn, 10, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  } else if (huart->Instance == USART3) {
    __HAL_RCC_USART3_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    g.Pin = GPIO_PIN_9;                       /* PD9 = D0 = USART3_RX */
    g.Mode = GPIO_MODE_AF_PP; g.Pull = GPIO_PULLUP;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH; g.Alternate = GPIO_AF7_USART3;
    HAL_GPIO_Init(GPIOD, &g);
    HAL_NVIC_SetPriority(USART3_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(USART3_IRQn);
  }
}

static void MX_USART1_UART_Init(void)
{
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK) { Error_Handler(); }
}

static void MX_USART3_UART_Init(void)
{
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 31250;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK) { Error_Handler(); }
  SET_BIT(USART3->CR1, USART_CR1_RXNEIE_RXFNEIE);   /* bytes via IRQ */
}

int __io_putchar(int ch)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xFFFF);
  return ch;
}

/* Render a hop into the free half of the DMA buffer. The buffers are
 * __NON_CACHEABLE (MPU region from the example) — no cache maintenance
 * is needed (§6.1.6). */
/* --- D-24: live A/B of the net on button B1 ----------------------------
 * PC13, pressed = HIGH (UM3417 §7.6). When switched off, the net KEEPS
 * being computed — only the mixing-in of the residual is muted: the graph
 * state stays coherent, the cycles are the same, switching back is instant,
 * and exactly one factor is compared.
 * Polling once per hop (250 Hz), printing from the super-loop: the UART is
 * blocking, printing inside the audio tick is not allowed. A click when
 * switching on a sounding note is expected (there is no smoothing). */
static void net_ab_init(void)
{
  GPIO_InitTypeDef g = {0};
  __HAL_RCC_GPIOC_CLK_ENABLE();
  g.Pin = GPIO_PIN_13; g.Mode = GPIO_MODE_INPUT; g.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(GPIOC, &g);
}
static void net_ab_poll(void)
{
  static uint8_t prev = 0u, cnt = 0u;
  uint8_t now = (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_SET);
  if (now == prev) { cnt = 0u; return; }
  if (++cnt < 5u) return;                      /* 5 hops = 20 ms of debounce */
  cnt = 0u; prev = now;
  if (!now) return;                            /* we react to the press */
  g_net_on = (uint8_t)n6_pipe_set_net(&g_pipe, !g_net_on);
  g_net_msg = 1u;
}

static void render_into_half(int half)
{
  static float out48[N6_HOP48];
  uint8_t b; n6_midi_ev_t ev;
#ifdef N6_CK4
  n6_ck4_pre_hop(&g_mf);                       /* the score — BEFORE the drain */
#endif
  while (n6_mf_pop(&g_mf, &b))                 /* drain the FIFO in the tick */
    if (n6_midi_parse_byte(&g_mp, b, &ev))
      n6_vm_event(&g_pipe.vm, &ev);

  net_ab_poll();                               /* D-24: button B1 */

  uint32_t c0 = DWT->CYCCNT;
  n6_pipe_hop(&g_pipe, out48);
  uint32_t dc = DWT->CYCCNT - c0;
  g_cyc_sum += dc; g_hops++;
  if (dc < g_cyc_min) g_cyc_min = dc;
  if (dc > g_cyc_max) g_cyc_max = dc;
#ifdef N6_CK4
  n6_ck4_post_hop(out48, N6_HOP48);            /* capture — AFTER pipe_hop */
#endif

  uint32_t *dst = &sai_tx_double_buffer[half ? APP_SAI_HALF_WORDS : 0u];
  for (int i = 0; i < N6_HOP48; ++i) {         /* f32 -> 24-in-32, mono->LR */
    float x = out48[i];
    /* review item 5, 2 Aug. NaN was passing straight through: all three
     * comparisons below are FALSE for NaN, the clamp does not catch it, and
     * (int32_t)NaN is UB, so garbage goes out to the DAC. A check that
     * negates the range catches both NaN and both Infs. */
    if (!(x >= -1.0e3f && x <= 1.0e3f)) { x = 0.0f; g_out_nan++; }
    float ax = (x < 0.0f) ? -x : x;
    if (ax > g_peak) g_peak = ax;
    if (x >  0.999969f) { x =  0.999969f; g_out_clip++; }
    if (x < -0.999969f) { x = -0.999969f; g_out_clip++; }
    /* shifting a NEGATIVE int32 is UB (C11 6.5.7p4); we assemble the bit
     * pattern in unsigned, where the shift is defined modulo 2^32. The
     * format is the same: 24 bits, pushed left in a 32-bit slot. */
    int32_t smp = (int32_t)(x * 8388607.0f);
    uint32_t w = (uint32_t)smp << 8;
    dst[2 * i]     = w;
    dst[2 * i + 1] = w;
  }
}

/**
  * @brief  MPU configuration
  * @param  None
  * @retval None
  */
void MPU_Config(void)
{
  MPU_Region_InitTypeDef default_config = {0};
  MPU_Attributes_InitTypeDef attr_config = {0};
  uint32_t primask_bit = __get_PRIMASK();
  __disable_irq();

  /* disable the MPU */
  HAL_MPU_Disable();

  /* create an attribute configuration for the MPU */
  attr_config.Attributes = INNER_OUTER(MPU_NOT_CACHEABLE);
  attr_config.Number = MPU_ATTRIBUTES_NUMBER0;

  HAL_MPU_ConfigMemoryAttributes(&attr_config);

  /* Create a non cacheable region */
  /*Normal memory type, code execution allowed */
  default_config.Enable = MPU_REGION_ENABLE;
  default_config.Number = MPU_REGION_NUMBER0;
  default_config.BaseAddress = __NON_CACHEABLE_SECTION_BEGIN;
  default_config.LimitAddress = __NON_CACHEABLE_SECTION_END;
  default_config.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
  default_config.AccessPermission = MPU_REGION_ALL_RW;
  default_config.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
  default_config.AttributesIndex = MPU_ATTRIBUTES_NUMBER0;
  HAL_MPU_ConfigRegion(&default_config);

  /* enable the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

  /* Exit critical section to lock the system and avoid any issue around MPU mechanisme */
  __set_PRIMASK(primask_bit);
}

/**
  * @brief Tx Transfer completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
  if (hsai == &hsai_BlockB1)
  {
    GPIO_RESET(D1);
    if (g_half_ready >= 0) g_underrun++;   /* we missed the previous half */
    g_half_ready = 1;                      /* the second half went to SAI */
  }
}

/**
  * @brief Tx Transfer Half completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  if (hsai == &hsai_BlockB1)
  {
    GPIO_SET(D1);
    if (g_half_ready >= 0) g_underrun++;
    g_half_ready = 0;                      /* the first half went to SAI */
  }
}

/**
  * @brief Rx Transfer completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
  if (hsai == &hsai_BlockB2)
  {
    GPIO_RESET(D2);

    int res = memcmp32(&sai_tx_double_buffer[APP_SAI_1MS_BUFFER_SIZE], &sai_rx_double_buffer[APP_SAI_1MS_BUFFER_SIZE], sizeof(uint32_t) * APP_SAI_1MS_BUFFER_SIZE);
    if (res != 0)
    {
      GPIO_SET(D3);
      BSP_LED_On(LED_RED);
      GPIO_RESET(D3);
    }

    GPIO_SET(D2);
    GPIO_RESET(D2);
  }
}

/**
  * @brief Rx Transfer half completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  if (hsai == &hsai_BlockB2)
  {
    GPIO_SET(D2);

    int res = memcmp32(&sai_tx_double_buffer[0], &sai_rx_double_buffer[0], sizeof(uint32_t) * APP_SAI_1MS_BUFFER_SIZE);
    if (res != 0)
    {
      GPIO_SET(D3);
      BSP_LED_On(LED_RED);
      GPIO_RESET(D3);
    }

    GPIO_RESET(D2);
    GPIO_SET(D2);
  }
}

/**
  * @brief SAI error callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai)
{
  HAL_GPIO_WritePin(D3_GPIO_Port, D3_Pin, GPIO_PIN_SET);

  BSP_LED_On(LED_RED);

  HAL_SAI_DMAStop(hsai);
  HAL_SAI_Abort(hsai);

  HAL_GPIO_WritePin(D3_GPIO_Port, D3_Pin, GPIO_PIN_RESET);
}

#define CMP_STEP_INC()   res |= (*p1++ ^ *p2++);
#define CMP_STEP8_INC()  CMP_STEP_INC() CMP_STEP_INC() CMP_STEP_INC() CMP_STEP_INC() \
                         CMP_STEP_INC() CMP_STEP_INC() CMP_STEP_INC() CMP_STEP_INC()
#define CMP_STEP32_INC() CMP_STEP8_INC() CMP_STEP8_INC() CMP_STEP8_INC() CMP_STEP8_INC()

/**
  * @brief  Compares two 32-bit aligned memory buffers.
  * @param  p1 Pointer to the first memory buffer
  * @param  p2 Pointer to the second memory buffer
  * @param  sz Size of the memory block to compare, in bytes
  * @retval int 0 if the buffers are identical, 1 if a difference is found
  */
int memcmp32(const uint32_t *p1, const uint32_t *p2, uint32_t sz)
{
  uint32_t res = 0u;

  while (sz >= 128) {
    CMP_STEP32_INC()
    sz -= 128;
  }

  while (sz >= 4) {
    CMP_STEP_INC()
    sz -= 4;
  }

  return (res == 0) ? 0 : 1;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @param None
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
