# coding:utf-8
import numpy as np
import struct
import os
import scipy.io
import time

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    def forward(self, input_layer, content_layer):
         # TODO： 计算风格迁移图像和目标内容图像的内容损失
        loss = np.sum((input_layer-content_layer)**2) / (2*input_layer.size)
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        bottom_diff = (input_layer-content_layer)/input_layer.size
        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵
        self.style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        self.gram_style = np.zeros([style_layer.shape[0],style_layer.shape[1],style_layer.shape[1]])
        for indn in range(style_layer.shape[0]):
            for indi in range(style_layer.shape[1]):
                for indj in range(style_layer.shape[1]):
                    self.gram_style[indn,indi,indj] = np.sum(style_layer[indn,indi,:,:] * style_layer[indn,indj,:,:])

        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])

        for indn in range(input_layer.shape[0]):
            for indi in range(input_layer.shape[1]):
                for indj in range(input_layer.shape[1]):
                    self.gram_input[indn,indi,indj] = np.sum(input_layer[indn,indi,:,:] * input_layer[indn,indj,:,:])
        
        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        self.style_diff = self.gram_input - self.gram_style
        loss =  np.sum(self.style_diff**2)/(4*input_layer[0]*self.div)
        return loss


    def backward(self, input_layer, style_layer):
        input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
            for idxci in range(input_layer.shape[1]):
                bottom_diff[idxn, idxci, :] = np.sum(input_layer_reshape[idxn,:,:] * np.reshape(self.style_diff[idxn,:,idxci],[-1,1]),axis=0)
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
