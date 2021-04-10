import torch
import numpy as np
from stu_upload.layers_2 import *
import unittest

def mse(data1,data2):
    assert data1.shape == data2.shape
    return np.mean((data1 - data2)**2)

def check_mse(data1,data2,error = 1e-6):
    assert data1.shape == data2.shape
    return mse(data1,data2) < error


class Layers2Test(unittest.TestCase):
    
    def test_convolutional_layer(self):
        np.random.seed(1)
        # setting
        N,channel_in,height_in,width_in = (1,3,48,80)
        channel_out = 64
        kernel_size = 3
        padding = 1
        stride = 1
        input_shape = (N,channel_in,height_in,width_in)
        # torch
        torch_input = torch.randn(input_shape,requires_grad=True)
        conv2d = torch.nn.Conv2d(channel_in,channel_out,kernel_size,stride=stride,padding=padding,bias=True)
        torch_output = conv2d(torch_input)
        torch_top_grad = torch.randn(torch_output.shape)
        torch_output.backward(torch_top_grad)
        # convert tensor to ndarray
        torch_output = torch_output.detach().numpy()
        torch_bottom_gard = torch_input.grad.numpy()
        torch_weight_grad = conv2d.weight.grad.numpy().transpose([1,2,3,0])
        torch_bias_grad = conv2d.bias.grad.numpy()
        # my input
        input = torch_input.detach().numpy()
        top_grad = torch_top_grad.numpy()
        weight = conv2d.weight.detach().numpy().transpose([1,2,3,0])
        bias = conv2d.bias.detach().numpy()
        # run my code
        my_conv2d = ConvolutionalLayer(kernel_size,channel_in,channel_out,padding,stride,type=1)
        my_conv2d.init_param()
        my_conv2d.load_param(weight,bias)
        output = my_conv2d.forward(input)
        bottom_gard = my_conv2d.backward(top_grad)
        # assert
        self.assertTrue(check_mse(torch_output,output))
        self.assertTrue(check_mse(torch_bottom_gard,bottom_gard))

        # self.assertTrue(check_mse(torch_weight_grad,my_conv2d.d_weight))
        # self.assertTrue(check_mse(torch_bias_grad,my_conv2d.d_bias))


    def test_maxpooling_layer(self):
        np.random.seed(1)
        # setting
        N,channel_in,height_in,width_in = (1,1,4,4)
        kernel_size = 2
        stride = 2
        input_shape = (N,channel_in,height_in,width_in)
        # torch
        torch_input = torch.randn(input_shape,requires_grad=True)
        maxpooling = torch.nn.MaxPool2d(kernel_size,stride=stride)
        torch_output = maxpooling(torch_input)
        torch_top_grad = torch.randn(torch_output.shape)
        torch_output.backward(torch_top_grad)
        # convert tensor to ndarray
        torch_output = torch_output.detach().numpy()
        torch_bottom_gard = torch_input.grad.numpy()
        # my input
        input = torch_input.detach().numpy()
        top_grad = torch_top_grad.numpy()
        # run my code
        my_maxpooling = MaxPoolingLayer(kernel_size,stride,type=1)
        output = my_maxpooling.forward(input)
        bottom_gard = my_maxpooling.backward(top_grad)
        # assert
        self.assertTrue(check_mse(torch_output,output))
        self.assertTrue(check_mse(torch_bottom_gard,bottom_gard))

if __name__ == '__main__':
    unittest.main()

