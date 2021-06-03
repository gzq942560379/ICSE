#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
#if CAMBRICON_MLU
#include <string>
#include <iostream>
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/mlu_op_kernel.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"

namespace tensorflow {
template <typename T>
class MLUSBCOp : public MLUOpKernel {
 public:
  explicit MLUSBCOp(OpKernelConstruction* ctx) :
          MLUOpKernel(ctx) {}

  void ComputeOnMLU(OpKernelContext* ctx) override {

    if (!ctx->ValidateInputsAreSameShape(this)) return;
    se::mlu::MLUStream* stream = static_cast<se::mlu::MLUStream*>(
        ctx->op_device_context()->stream()->implementation());

    Tensor input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    OP_REQUIRES_OK(ctx, stream->SBC(ctx,
            const_cast<Tensor *>(&input), output, input.shape().dim_size(0)));
  }
};

}  // namespace tensorflow


#endif  // CAMBRICON_MLU
#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
