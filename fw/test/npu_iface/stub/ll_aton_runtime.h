#ifndef STUB_LL_ATON_RUNTIME_H
#define STUB_LL_ATON_RUNTIME_H
#include <stdint.h>
typedef enum { LL_ATON_RT_NO_WFE = 0, LL_ATON_RT_WFE, LL_ATON_RT_DONE } LL_ATON_RT_RetValues_t;
typedef enum { CHPOS_NA = 0 } Buffer_CHPos_TypeDef;
typedef enum { DT_NA = 0 } Buffer_DataType_TypeDef;
typedef struct {                       /* fields 1:1 from ll_aton_NN_interface.h */
  const char *name; unsigned char *addr_base;
  uint32_t offset_start, offset_end, offset_limit;
  uint8_t is_user_allocated, is_param; uint16_t epoch; uint32_t batch;
  const uint32_t *mem_shape; uint16_t mem_ndims;
  Buffer_CHPos_TypeDef chpos; Buffer_DataType_TypeDef type;
  int8_t Qm, Qn; uint8_t Qunsigned, ndims, nbits, per_channel;
  const uint32_t *shape; const float *scale; const int16_t *offset;
} LL_Buffer_InfoTypeDef;
static inline unsigned char *LL_Buffer_addr_start(const LL_Buffer_InfoTypeDef *b){return b->addr_base+b->offset_start;}
static inline uint32_t LL_Buffer_len(const LL_Buffer_InfoTypeDef *b){return b->offset_end-b->offset_start;}
typedef struct { int dummy; } NN_Instance_TypeDef;
#define LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(nm) static NN_Instance_TypeDef NN_Instance_##nm;
void LL_ATON_RT_RuntimeInit(void);
void LL_ATON_RT_Init_Network(NN_Instance_TypeDef *i);
void LL_ATON_RT_Reset_Network(NN_Instance_TypeDef *i);
LL_ATON_RT_RetValues_t LL_ATON_RT_RunEpochBlock(NN_Instance_TypeDef *i);
const LL_Buffer_InfoTypeDef *LL_ATON_Input_Buffers_Info(const NN_Instance_TypeDef *i);
const LL_Buffer_InfoTypeDef *LL_ATON_Output_Buffers_Info(const NN_Instance_TypeDef *i);
#endif
