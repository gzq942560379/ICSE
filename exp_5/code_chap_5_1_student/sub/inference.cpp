#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>



namespace StyleTransfer{

Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){
    // when generating an offline model, u need cnml and cnrt both
    // when running an offline model, u need cnrt only
    cnrtInit(0);

    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());

    // TODO:set current device
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);

    // TODO:load extract function
    cnrtFunction_t function;
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, "subnet0");

    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);

    // TODO:prepare data on cpu
    void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

    // TODO:allocate I/O data memory on MLU
    void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));

    void **inputHalf = (void**)malloc(inputNum * sizeof(void*));
    void **outputHalf = (void**)malloc(outputNum * sizeof(void*));

    // TODO:prepare input buffer
    cnrtQuantizedParam_t quantizedParam;
    cnrtCreateQuantizedParam(&quantizedParam,0,1,0);
    for (int i = 0; i < inputNum; i++) {
          // converts data format when using new interface model
          inputCpuPtrS[i] = (void*)malloc(inputSizeS[i]*2);
          int value[4] = {1,3,256,256};
          int order[4] = {0,2,3,1};
          cnrtTransDataOrder(DataT->input_data,CNRT_FLOAT32,inputCpuPtrS[i],4,value,order); 
          inputHalf[i] = (void*)malloc(inputSizeS[i]);
          cnrtCastDataType(inputCpuPtrS[i],CNRT_FLOAT32,inputHalf[i],CNRT_FLOAT16,inputSizeS[i]/2,NULL);
          // malloc mlu memory
          cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
          cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i], inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    // TODO:prepare output buffer
    for (int i = 0; i < outputNum; i++) {
          outputCpuPtrS[i] = (void*)malloc(outputSizeS[i]*2);
          outputHalf[i] = (void*)malloc(outputSizeS[i]);
          // malloc mlu memory
          cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
    }

    // prepare parameters for cnrtInvokeRuntimeContext
    void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
    for (int i = 0; i < inputNum; ++i) {
          param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; ++i) {
          param[inputNum + i] = outputMluPtrS[i];
    }

    // setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL);

    // bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0);
    cnrtInitRuntimeContext(ctx, NULL);

    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue);

    struct timeval tpend,tpstart;
    double time_use = 0.;

    gettimeofday(&tpstart,NULL);
    srand((unsigned)time(NULL));
    for(int i = 0;i<inputNum;++i){
        cnrtMemcpy(inputMluPtrS[i],inputHalf[i],inputSizeS[i],CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
    cnrtInvokeRuntimeContext(ctx, param, queue, NULL);
    cnrtSyncQueue(queue);
    for (int i = 0; i < outputNum; i++) {
          cnrtMemcpy(outputHalf[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
    }

    gettimeofday(&tpend,NULL);
    time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_usec - tpstart.tv_usec);
    printf("get data cost time %lfms \n",time_use/1000);
    int value[4] = {1,256,256,3};
    int order[4] = {0,3,1,2};
    cnrtCastDataType(outputHalf[0],CNRT_FLOAT16,outputCpuPtrS[0],CNRT_FLOAT32,outputSizeS[0]/2,NULL);
    DataT->output_data = (float*)malloc(outputSizeS[0]*2);
    cnrtTransDataOrder(outputCpuPtrS[0],CNRT_FLOAT32,DataT->output_data,4,value,order); 
    // TODO:free memory spac
    for(int i = 0;i<inputNum;i++){
        free(inputCpuPtrS[i]);
        free(inputHalf[i]);
        cnrtFree(inputMluPtrS[i]);
    }
    for(int i = 0;i<outputNum;i++){
        free(outputCpuPtrS[i]);
        free(outputHalf[i]);
        cnrtFree(outputMluPtrS[i]);
    }
    free(inputCpuPtrS);
    free(outputCpuPtrS);
    free(inputMluPtrS);
    free(outputMluPtrS);
    free(inputHalf);
    free(outputHalf);
    free(param);

    cnrtDestroyQueue(queue);
    cnrtDestroyRuntimeContext(ctx);
    cnrtDestroyFunction(function);
    cnrtDestroyQuantizedParam(quantizedParam);
    cnrtDestroy();

}

} // namespace StyleTransfer
