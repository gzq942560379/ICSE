import pycnml
import numpy as np
import time
import os
import struct

def load_mnist_image(file_path):
    with open(file_path,"rb") as f:
        bin_data = f.read()
        fmt_header = ">iiii"
        magic,num_images,num_rows,num_cols = struct.unpack_from(fmt_header,bin_data,0)
        data_size = num_images*num_rows*num_cols
        mat_data=struct.unpack_from('>'+str(data_size)+'B',bin_data,struct.calcsize(fmt_header))
        mat_data=np.reshape(mat_data,[num_images,num_rows*num_cols])
        return mat_data


def load_mnist_label(file_path):
    with open(file_path,"rb") as f:
        bin_data = f.read()
        fmt_header = ">ii"
        magic,num_images = struct.unpack_from(fmt_header,bin_data,0)
        data_size = num_images
        mat_data=struct.unpack_from('>'+str(data_size)+'B',bin_data,struct.calcsize(fmt_header))
        mat_data=np.reshape(mat_data,[num_images,1])
        return mat_data

class MNIST_MLP(object):
    
    def __init__(self):
        self.net = pycnml.CnmlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

    def build_model(self,input_size=784,hidden1=32,hidden2=16,output_classes=10,batch_size=100,quant_params_path="/opt/code_chap_2_3/code_chap_2_3_student/mnist_mlp_quant_param.npz"):

        self.batch_size = batch_size
        self.output_classes = output_classes

        params = np.load(quant_params_path)
        input_params = params["input"]
        filter_params = params["filter"]
        
        for i in range(0,len(input_params),2):
            self.input_quant_params.append(pycnml.QuantParam(int(input_params[i]),float(input_params[i+1])))
        for i in range(0,len(filter_params),2):
            self.filter_quant_params.append(pycnml.QuantParam(int(filter_params[i]),float(filter_params[i+1])))

        self.net.setInputShape(batch_size,input_size,1,1)
        self.net.createMlpLayer("fc1",hidden1,self.input_quant_params[0])    
        self.net.createReLuLayer("relu1")    
        self.net.createMlpLayer("fc2",hidden2,self.input_quant_params[1]) 
        self.net.createReLuLayer("relu2")    
        self.net.createMlpLayer("fc3",output_classes,self.input_quant_params[2])        
        self.net.createSoftmaxLayer("softmax",1)    


    def load_model(self,param_path):
        params = np.load(param_path).item()
        w1 = np.transpose(params["w1"],[1,0]).flatten().astype(np.float)
        b1 = params["b1"].flatten().astype(np.float)
        self.net.loadParams(0,w1,b1,self.filter_quant_params[0])

        w2 = np.transpose(params["w2"],[1,0]).flatten().astype(np.float)
        b2 = params["b2"].flatten().astype(np.float)
        self.net.loadParams(2,w2,b2,self.filter_quant_params[1])

        w3 = np.transpose(params["w3"],[1,0]).flatten().astype(np.float)
        b3 = params["b3"].flatten().astype(np.float)
        self.net.loadParams(4,w3,b3,self.filter_quant_params[2])


    def forward(self):
        return self.net.forward()
    
    def evaluate(self,test_data,test_label):
        
        test_label = np.reshape(test_label,[-1])
        pred_results = np.zeros([test_data.shape[0]])

        num_batch = test_data.shape[0]/self.batch_size
        for idx in range(num_batch):
            batch_images = test_data[idx * self.batch_size:(idx+1)*self.batch_size]
            data = batch_images.flatten().tolist()
            
            self.net.setInputData(data)

            start = time.time()

            self.forward()

            end = time.time()
            print("inferencing time : %f" % (end - start))
            prob = self.net.getOutputData()
            prob = np.array(prob).reshape((self.batch_size,self.output_classes))
            pred_labels = np.argmax(prob,axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == test_label)
        print('Accuracy in test set: %f' % accuracy)



if __name__ == "__main__":
    batch_size = 10000
    model_path = "experiment1/model/model_512_256_9827.npy"
    
    basename = os.path.basename(model_path)
    file_name = os.path.splitext(basename)[0]
    hidden1 = int(file_name.split("_")[1])
    hidden2 = int(file_name.split("_")[2])

    mlp = MNIST_MLP()
    mlp.build_model(input_size=784,hidden1=hidden1,hidden2=hidden2,output_classes=10,batch_size=batch_size)

    test_image_path="data/t10k-images-idx3-ubyte"
    test_label_path="data/t10k-labels-idx1-ubyte"

    test_image = load_mnist_image(test_image_path)
    test_label = load_mnist_label(test_label_path)

    mlp.load_model(model_path)
    for i in range(3):
        mlp.evaluate(test_image,test_label)