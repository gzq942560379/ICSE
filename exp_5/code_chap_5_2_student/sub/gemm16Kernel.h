typedef unsigned short half;




#ifdef __cplusplus
extern "C" {
#endif
// 补齐函数声明，对应gemm/gemm_SRAM.mlu
void gemm16Kernel(half *outputDDR,half *input1DDR,half *input2DDR,uint32_t m,uint32_t k,uint32_t n);
#ifdef __cplusplus
}
#endif
