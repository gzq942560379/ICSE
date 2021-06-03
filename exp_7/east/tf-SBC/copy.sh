#/bin/bash

tensorflow_root=/opt/code_chap_7_student/env/tensorflow-v1.10

cp ./cwise_op_* $tensorflow_root/tensorflow/core/kernels/
cp ./BUILD $tensorflow_root/tensorflow/core/kernels/
cp ./mlu_stream.h $tensorflow_root/tensorflow/stream_executor/mlu/
cp ./mlu_lib_ops.* $tensorflow_root/tensorflow/stream_executor/mlu/mlu_api/lib_ops/
cp ./mlu_ops.h $tensorflow_root/tensorflow/stream_executor/mlu/mlu_api/ops/
cp ./sbc.cc $tensorflow_root/tensorflow/stream_executor/mlu/mlu_api/ops/
cp ./math_ops.cc $tensorflow_root/tensorflow/core/ops/