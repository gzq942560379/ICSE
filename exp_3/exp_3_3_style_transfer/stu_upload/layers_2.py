# coding=utf-8
import numpy as np
import struct
import os
import time

def im2col(im,kernel_size,stride):
    assert im.ndim == 4
    N,channel,height_in,width_in = im.shape
    height_out = (height_in - kernel_size) / stride + 1
    width_out = (width_in - kernel_size) / stride + 1
    col = np.zeros([N,height_out,width_out,channel,kernel_size,kernel_size])
    for idxn in range(N):
        for idxh in range(height_out):
            for idxw in range(width_out):
                col[idxn,idxh,idxw,:,:,:] = im[idxn,:,
                                                idxh*stride:idxh*stride+kernel_size,
                                                idxw*stride:idxw*stride+kernel_size,]
    col = col.reshape([N*height_out*width_out,channel*kernel_size*kernel_size]).transpose([1,0])
    return col

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (
            self.kernel_size, self.channel_in, self.channel_out))

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(
            self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])



    def forward_speedup(self, input):
        # print("input shape:",input.shape)
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input
        N,channel_in,height_in,width_in = self.input.shape
        assert channel_in == self.channel_in
        height_in = height_in + self.padding * 2
        width_in = width_in + self.padding * 2
        height_out = (height_in - self.kernel_size) / self.stride + 1
        width_out = (width_in - self.kernel_size) / self.stride + 1

        self.input_pad = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_in, width_in])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2],
                       self.padding:self.padding+self.input.shape[3]] = self.input

        # print("input_pad shape:",self.input_pad.shape)

        # [channel_in*self.kernel_size*self.kernel_size,N,self.channel_out,N*height_out*width_out]
        self.input_col = im2col(self.input_pad,self.kernel_size,self.stride)
        
        # weight [self.channel_in, self.kernel_size, self.kernel_size, self.channel_out]
        # weight_col [self.channel_out, channel_in*self.kernel_size*self.kernel_size]
        weight_col = self.weight.transpose([3,0,1,2]).reshape([self.channel_out,self.channel_in*self.kernel_size*self.kernel_size])
        
        # output_col [N,self.channel_out,N*height_out*width_out]
        self.output_col = np.matmul(weight_col,self.input_col) + self.bias.reshape([self.channel_out,1])

        self.output = self.output_col.reshape([self.channel_out,N,height_out,width_out]).transpose([1,0,2,3])

        self.forward_time = time.time() - start_time
        return self.output

    def backward_speedup(self, top_diff):

        start_time = time.time()
        # self.d_weight = np.zeros(self.weight.shape)
        # self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        # top_diff [N,self.channel_out,N*height_out*width_out]
        N,channel_out,top_diff_height,top_diff_width = top_diff.shape
        top_diff_height_pad = top_diff_height + 2 * self.padding
        top_diff_width_pad = top_diff_width + 2 * self.padding
        bottom_diff_height = (top_diff_height_pad - self.kernel_size) / self.stride + 1
        bottom_diff_width = (top_diff_width_pad - self.kernel_size) / self.stride + 1
        top_diff_pad = np.zeros(
            [top_diff.shape[0], top_diff.shape[1], top_diff_height_pad, top_diff_width_pad])
        top_diff_pad[:, :,
                     self.padding:self.padding+top_diff.shape[2],
                     self.padding:self.padding+top_diff.shape[3]] = top_diff

        # d_weight , d_bias need not compute

        # top_diff [N*height_out*width_out,channel_out]
        # top_diff_reshape = top_diff.transpose([0,2,3,1]).reshape([N*top_diff_height*top_diff_width,channel_out])
        # # [channel_in*self.kernel_size*self.kernel_size ,channel_out]
        # self.d_weight = np.matmul(self.input_col,top_diff_reshape).reshape([self.channel_in,self.kernel_size,self.kernel_size,self.channel_out])
        # # [self.channel_out]
        # self.d_bias = np.sum(top_diff_reshape,axis = 0)
        # weight_col [channel_in,channel_out,kernel_size,kernel_size]
        
        weight_col = np.flip(self.weight.transpose([0,3,1,2]),(2,3)).reshape([self.channel_in,self.channel_out*self.kernel_size*self.kernel_size])
        # weight_col = self.weight.transpose([0,3,1,2]).reshape([self.channel_in,self.channel_out*self.kernel_size*self.kernel_size])
        
        # [channel_out,kernel_size,kernel_size,N,height_in,width_in]
        top_diff_col = im2col(top_diff_pad,self.kernel_size,self.stride)
        # [channel_in,N,height_in,width_in]
        bottom_diff = np.matmul(weight_col,top_diff_col)

        bottom_diff = bottom_diff.reshape([self.channel_in,N,bottom_diff_height,bottom_diff_width]).transpose([1,0,2,3])
        
        self.backward_time = time.time() - start_time

        return bottom_diff

    def forward_raw(self, input):
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros(
            [self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2],
                       self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw]  = np.sum(self.input_pad[idxn, :,
                                                                    idxh*self.stride:idxh*self.stride+self.kernel_size,
                                                                    idxw*self.stride:idxw*self.stride+self.kernel_size] *
                                                                    self.weight[:, :, :, idxc])+self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output

    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input.shape)
        top_diff_height = top_diff.shape[2]
        top_diff_width = top_diff.shape[3]
        top_diff_height_pad = top_diff_height + 2 * self.padding
        top_diff_width_pad = top_diff_width + 2 * self.padding
        bottom_diff_height = (top_diff_height_pad - self.kernel_size) / self.stride + 1
        bottom_diff_width = (top_diff_width_pad - self.kernel_size) / self.stride + 1
        top_diff_pad = np.zeros(
            [top_diff.shape[0], top_diff.shape[1], top_diff_height_pad, top_diff_width_pad])
        top_diff_pad[:, :,
                     self.padding:self.padding+top_diff.shape[2],
                     self.padding:self.padding+top_diff.shape[3]] = top_diff
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += self.input_pad[idxn, :,
                                                                       idxh * self.stride: idxh*self.stride+self.kernel_size,
                                                                       idxw * self.stride: idxw*self.stride+self.kernel_size] * top_diff[idxn, idxc, idxh, idxw]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
        
        weight = np.flip(self.weight.transpose([3,1,2,0]),(1,2))

        for idxn in range(top_diff.shape[0]):
            for idxc in range(self.channel_in):
                for idxh in range(bottom_diff_height):
                    for idxw in range(bottom_diff_width):
                        bottom_diff[idxn, idxc, idxh, idxw] = np.sum(top_diff_pad[idxn, :,
                                                                    idxh*self.stride:idxh*self.stride+self.kernel_size,
                                                                    idxw*self.stride:idxw*self.stride+self.kernel_size] *
                                                                    weight[:, :, :, idxc])
        self.backward_time = time.time() - start_time
        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias

    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def get_forward_time(self):
        return self.forward_time

    def get_backward_time(self):
        return self.backward_time


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' %
              (self.kernel_size, self.stride))

    def forward_raw(self, input):
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc,
                                                                                idxh*self.stride:idxh*self.stride+self.kernel_size,
                                                                                idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.argmax(
                            self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(
                            curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride +
                                       curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output

    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input

        N ,channel_in,height_in,width_in = input.shape
        height_out = (height_in - self.kernel_size) / self.stride + 1
        width_out = (width_in - self.kernel_size) / self.stride + 1

        self.input_col = im2col(self.input,self.kernel_size,self.stride)
        assert self.input_col.shape == (channel_in*self.kernel_size*self.kernel_size,N*height_out*width_out)

        self.input_col = self.input_col.reshape([channel_in,self.kernel_size*self.kernel_size,N,height_out,width_out]).transpose([2,0,3,4,1])
        assert self.input_col.shape == (N,channel_in,height_out,width_out,self.kernel_size*self.kernel_size)

        self.max_index = np.argmax(self.input_col,axis=4)
        assert self.max_index.shape == (N,channel_in,height_out,width_out)

        self.output = np.max(self.input_col,axis=4)

        return self.output
    
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速

        N ,channel_in,height_out,width_out = top_diff.shape
        assert self.kernel_size == self.stride
        height_in = height_out * self.kernel_size
        width_in = width_out * self.kernel_size
        bottom_diff = np.zeros(self.input.shape)
        assert bottom_diff.shape == (N,channel_in,height_in,width_in)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        max_index = self.max_index[idxn,idxc,idxh,idxw]
                        tmp = np.zeros(self.kernel_size*self.kernel_size)
                        tmp[max_index] = top_diff[idxn,idxc,idxh,idxw]
                        bottom_diff[idxn,idxc,
                                    idxh*self.stride:idxh*self.stride+self.kernel_size,
                                    idxw*self.stride:idxw*self.stride+self.kernel_size] = tmp.reshape(self.kernel_size,self.kernel_size)
        
        return bottom_diff

    def backward_raw(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        # max_index = _______________________
                        # bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = _______________________
                        bottom_diff[idxn, idxc,
                                    idxh*self.stride:idxh*self.stride+self.kernel_size,
                                    idxw*self.stride:idxw*self.stride+self.kernel_size] = self.max_index[idxn,
                                                                                                         idxc,
                                                                                                         idxh*self.stride:idxh*self.stride+self.kernel_size,
                                                                                                         idxw*self.stride:idxw*self.stride+self.kernel_size]*top_diff[idxn, idxc, idxh, idxw]

        return bottom_diff


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' %
              (str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape(
            [self.input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape(
            [top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
