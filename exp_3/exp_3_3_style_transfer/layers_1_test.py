import torch
import numpy as np
from stu_upload.layers_1 import *
import unittest

def check_mse(data1,data2,error = 1e-12):
    assert data1.shape == data2.shape
    return np.mean((data1 - data2)**2) < error


class Layers1Test(unittest.TestCase):
    
    def test_fully_connected_layer(self):
        np.random.seed(1)
        # setting
        input_size = 200
        output_size = 100
        use_bias = True
        # torch prepare
        torch_input = torch.randn((input_size,),requires_grad=True)
        torch_top_grad = torch.randn((output_size,))
        linear = torch.nn.Linear(input_size,output_size,use_bias)
        # my input
        input = torch_input.detach().numpy().reshape([1,-1])
        top_grad = torch_top_grad.numpy().reshape([1,-1])
        weight = linear.weight.detach().numpy().transpose([1,0])
        bais = linear.bias.detach().numpy().reshape([1,-1])
        # run torch
        torch_output = linear(torch_input)
        torch_output.backward(torch_top_grad)
        # run my code
        my_linear = FullyConnectedLayer(input_size,output_size)
        my_linear.init_param()
        my_linear.load_param(weight,bais)
        output = my_linear.forward(input)
        bottom_gard = my_linear.backward(top_grad)
        # convert tensor to ndarray
        torch_output = torch_output.detach().numpy().reshape([1,-1])
        torch_bottom_gard = torch_input.grad.numpy().reshape([1,-1])
        torch_weight_grad = linear.weight.grad.numpy().transpose([1,0])
        torch_bias_grad = linear.bias.grad.numpy()
        # assert
        self.assertTrue(check_mse(torch_output,output))
        self.assertTrue(check_mse(torch_bottom_gard,bottom_gard))
        self.assertTrue(check_mse(torch_weight_grad,my_linear.d_weight))
        self.assertTrue(check_mse(torch_bias_grad,my_linear.d_bias))


    def test_relu(self):
        np.random.seed(1)
        # setting
        input_shape = (200,100)
        # torch prepare
        torch_input = torch.randn(input_shape,requires_grad=True)
        torch_top_grad = torch.randn(input_shape)
        relu = torch.nn.ReLU()
        # my input
        input = torch_input.detach().numpy()
        top_grad = torch_top_grad.numpy()
        my_relu = ReLULayer()
        # run torch
        torch_output = relu(torch_input)
        torch_output.backward(torch_top_grad)
        # run my code
        output = my_relu.forward(input)
        bottom_gard = my_relu.backward(top_grad)
        # convert tensor to ndarray
        torch_output = torch_output.detach().numpy()
        torch_bottom_gard = torch_input.grad.numpy()
        # assert
        self.assertTrue(check_mse(torch_output,output))
        self.assertTrue(check_mse(torch_bottom_gard,bottom_gard))

        

if __name__ == '__main__':
    unittest.main()

