"""
author: Alex Baker @alexjetb
program: Neural network nodes.
Description: Part 01 of 03 for the implementation of a neural network
             as set in Weeks 06-08 of CS450
"""
import numpy as np
import pandas as pd
import math

# May remove, good reference for now
#class Node:
#    weight = 0.0 # Some small random number
#    input = 0     # Some given input default=0
#    bias = False
#
#    # Create node with given random weight
#    def __init__(self, bias=False, weight=0):
#        self.weight=weight
#        if (bias==True):
#            self.set_input(-1)
#            self.bias=True
#
#    def set_input(self, input):
#        self.input = input
#
#    # Consider several inputs using alg in future
#    def set_weight(self, weight):
#        self.weight = weight

class Node_Network:
    nodeArray = [] # empty list
    num_layers = 0
    num_inputs = 0
    num_nodes = 0
    num_outputs = 0
    maxHeight = 0
    learning_rate = 0.0
    # A node numpy matrix is created with a specified number of layers
    # (length/columns) and a specified number of inputs (height/rows)
    def __init__(self,num_inputs=2,num_layers=1,learning_rate=0.1,max_height=2,
                 num_outputs=2):
#        print("working")
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_height = max_height
        self.num_outputs = num_outputs
        #Generate our random weights...
        weights = np.random.uniform(-1,0,(max_height,max_height,max_height))
        layerPattern = np.random.randint(1, max_height, num_layers)
#        print(weights)
        # Construct empty jagged array
        self.nodeArray.append([dict()]*(num_inputs+1))
        with (np.nditer(layerPattern, flags=['c_index', 'multi_index'])) as it:
            for layer in it:
                blank_hidden = [dict()]*(layer.item()+1)
                self.nodeArray.append(blank_hidden)
        self.nodeArray.append([dict()]*num_outputs)
        # Build each layer and initialize each node, including biases
        i_ind = 0
        totalLayers = len(self.nodeArray)-1
        for layer in self.nodeArray:
            j_ind = 0
            width = len(layer)-1
            print(width)
            for node in layer:
                print(node)
                if i_ind==totalLayers:
                    outputs=[]
                    weight = []
                if i_ind>=totalLayers-1:
                    outputs =[0]*num_outputs
                    weight = weights[i_ind][j_ind][:num_outputs]
                else:
                    outputs=[0]*layerPattern[i_ind]
                    weight = weights[i_ind][j_ind][:layerPattern[i_ind]]

                if j_ind==0:
                    self.nodeArray[i_ind][j_ind] = {'bias':True,
                                                    'input':-1,
                                                    'weights':weight,
                                                    'outputs':outputs}
                else:
                    self.nodeArray[i_ind][j_ind] = {'bias':False,
                                                    'input':0,
                                                    'weights':weight,
                                                    'outputs':outputs}
                j_ind+=1
            i_ind+=1

    # "Fitting" function for neural network
    def train(self, num_times=1):
#        self.predictOne()
        print("Training!")

    # Predict a row
    def predictOne(self, dataTest=np.array([]), targetTest=np.array([])):
        if dataTest.size-1!=self.num_inputs: #temp workaround, off by 1 due to bias
            print('dataTest num of inputs must equal num_inputs!')
            return
        if dataTest.size==0:
            print('No data was given! Must give data to predict.')
            return
        if targetTest.size==0:
            print('No targets were given! Accuracy will not be predicted')

        # Initialize inputs
        with np.nditer(self.nodeArray[:,0], flags=['c_index', 'refs_ok'],
                       op_flags=['readwrite']) as it:
            for node in it:
                if(node.item()['bias']!=True):
                    index = it.index
#                    print(it.multi_index)
                    node.item()['input']=dataTest[index]

        with np.nditer(self.nodeArray, flags=['multi_index', 'refs_ok'],
                       op_flags=['readwrite'], order='F') as it:
            for node in it:
                column = it.multi_index[1]
                row = it.multi_index[0]
                if column==0:
                    node.item()['outputs']=node.item()['input']*node.item()['weights']
                if column>0 and node.item()['bias']==False:
#                    print("Working!")
                    nInput = 0;
                    for pLayNode in self.nodeArray[:,column-1]:
                        nInput += pLayNode['outputs'][row-1]
                    nInput = 1 / (1 + math.exp(-nInput))
                    node.item()['input']=nInput
                    node.item()['outputs']=node.item()['input']*node.item()['weights']
                if column>0 and node.item()['bias']==True:
                    node.item()['outputs'] = node.item()['weights']*node.item()['input']

        outputs = self.nodeArray[0:,self.nodeArray.shape[1]-1]

        for output in outputs:
            print("OUTPUT", output)

n_net = Node_Network(3,5,0.1,10,4)

n_net.predictOne(np.array([-1,1,2]))

print(n_net.nodeArray)